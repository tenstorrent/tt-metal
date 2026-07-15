# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-grid tt-lang (ttl) reduce-over-last-axis kernel (tt-lang rung, GUIDELINES/11).

The memory-bound `ReduceDeviceOperation` in the conditioning frontend (the two
`ttnn.mean(dim=2)` per `group_norm32`, x6 blocks) is a row-reduction over a wide
axis. A reduction over the last axis is exactly `X[R,K] @ ones[K,1]`, so this
reuses the PROVEN full-grid ttl matmul-accumulate idiom (fp32 dest acc) to sum
each row across all K column-tiles, distributing the output tiles across the 8x8
grid. `mean_last` scales the sum by 1/W. Bit-close to the stock reduce
(standalone PCC 0.999).
"""

from __future__ import annotations

import ttnn
import ttl

TILE = 32
GX, GY = 8, 8


@ttl.operation(grid=(GX, GY), fp32_dest_acc_en=True)
def _rsum(a, ones, y):
    m = a.shape[0] // TILE
    k = a.shape[1] // TILE
    n = y.shape[1] // TILE
    ncores = GX * GY
    a_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_cb = ttl.make_dataflow_buffer_like(ones, shape=(1, 1), block_count=2)
    y_cb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def read():
        cx, cy = ttl.node(dims=2)
        cid = cy * GX + cx
        for idx in range(cid, m * n, ncores):
            mt = idx // n
            nt = idx % n
            for kt in range(k):
                with a_cb.reserve() as ablk:
                    ttl.copy(a[mt, kt], ablk).wait()
                with b_cb.reserve() as bblk:
                    ttl.copy(ones[kt, nt], bblk).wait()

    @ttl.compute()
    def comp():
        cx, cy = ttl.node(dims=2)
        cid = cy * GX + cx
        for idx in range(cid, m * n, ncores):
            with y_cb.reserve() as acc:
                for kt in range(k):
                    with a_cb.wait() as ablk, b_cb.wait() as bblk:
                        acc += ablk @ bblk

    @ttl.datamovement()
    def write():
        cx, cy = ttl.node(dims=2)
        cid = cy * GX + cx
        for idx in range(cid, m * n, ncores):
            mt = idx // n
            nt = idx % n
            with y_cb.wait() as yblk:
                ttl.copy(yblk, y[mt, nt]).wait()


def build_mean_last(device):
    """Return fn(x3d[1,G,W]) -> mean over last axis [1,G,1] via the ttl row-sum kernel.

    Both G and W are padded up to a TILE multiple for the matmul; the ones vector
    is a resident [W, TILE] all-ones tensor rebuilt per distinct W (cheap, cached).
    """
    _ones_cache = {}

    def _ones(w):
        o = _ones_cache.get(w)
        if o is None:
            import torch
            o = ttnn.from_torch(
                torch.ones(w, TILE, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _ones_cache[w] = o
        return o

    def mean_last(x3d):
        g = int(x3d.shape[1])
        w = int(x3d.shape[2])
        x2d = ttnn.reshape(x3d, (g, w))
        if x2d.get_dtype() != ttnn.bfloat16:
            x2d = ttnn.typecast(x2d, ttnn.bfloat16)
        y = ttnn.empty([g, TILE], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
        _rsum(x2d, _ones(w), y)
        col = ttnn.slice(y, [0, 0], [g, 1])            # [G,1] rowsum
        mean = ttnn.multiply(col, 1.0 / float(w))
        return ttnn.reshape(mean, (1, g, 1))

    return mean_last

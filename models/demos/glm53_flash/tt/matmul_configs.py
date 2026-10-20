# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program configs for the decode-shaped projection matmuls (``[1, 1, M <= 32, K] x [K, N]``, bfp8 weights).

The default ``ttnn.linear`` dispatch reads the weights at 30-240 GB/s for these shapes (one chip, P150).
A 1D ``mcast_in0`` config (the activation tile row multicast, every core streaming its own column
block of the weight straight from interleaved DRAM, output interleaved: no reshards) reads them at
200-330 GB/s. Table = the winners of ``tests/test_matmul_sweep.py`` on a 12x10 grid (2026-10-20):

    shape (K x N)   default -> best    config
    4096 x 1536      75.8 ->  26.8 us  12x2 cores, per_core_N 2, in0_block_w 16   (DSA q_a)
    4096 x  512      66.6 ->  20.5 us   8x1, 2, 16                                (DSA kv_a, KDA f_a / g_a)
    1536 x 4096      41.0 ->  27.4 us  12x2, 6, 6                                 (DSA q_b, 16 heads)
    4096 x 4096      86.7 ->  54.4 us  12x6, 2, 4                                 (DSA o_proj, TP4)
    4096 x 6144     105.5 ->  76.1 us  12x8, 2, 4                                 (KDA qkv, dense w13, TP4)
    2048 x 4096      50.1 ->  33.1 us  12x2, 6, 8                                 (KDA o_proj, TP4)
     512 x 2048      21.6 ->  13.1 us  12x2, 3, 16                                (KDA f_b / g_b)
    4096 x   32      34.0 ->  12.9 us   1x1, 1, 16                                (KDA beta)
    4096 x 1024      69.4 ->  23.2 us  12x2, 2, 16                                (shared w13, TP4)
     512 x 4096      22.8 ->  16.9 us  12x2, 6, 8                                 (shared w2, TP4)
    3072 x 4096      68.1 ->  43.9 us  12x2, 6, 6                                 (dense w2, TP4)
    4096 x  288      56.5 ->  19.7 us   9x1, 1, 16                                (router gate, fp32 out)
    sum                959 ->  497 us per layer-equivalent, every PCC 0.99997 (identical to default)

Other shapes fall back to a heuristic (per_core_N 2, grid = ceil(Nt / 2) cores, in0_block_w = the
largest divisor of Kt <= 16 with in0_block_w * per_core_N <= 64). Prefill-shaped inputs (M > 32)
keep the default dispatch.
"""

from __future__ import annotations

import math

import ttnn

TILE = 32
# (K, N) -> (grid_x, grid_y, per_core_N, in0_block_w)
_TABLE = {
    (4096, 1536): (12, 2, 2, 16),
    (4096, 512): (8, 1, 2, 16),
    (1536, 4096): (12, 2, 6, 6),
    (4096, 4096): (12, 6, 2, 4),
    (4096, 6144): (12, 8, 2, 4),
    (2048, 4096): (12, 2, 6, 8),
    (512, 2048): (12, 2, 3, 16),
    (4096, 32): (1, 1, 1, 16),
    (4096, 1024): (12, 2, 2, 16),
    (512, 4096): (12, 2, 6, 8),
    (3072, 4096): (12, 2, 6, 6),
    (4096, 288): (9, 1, 1, 16),
}


def _heuristic(Kt: int, Nt: int, gx: int, gy: int):
    pcn = 2 if Nt > 1 else 1
    blocks = math.ceil(Nt / pcn)
    while blocks > gx * gy:
        pcn += 1
        blocks = math.ceil(Nt / pcn)
    cx = min(gx, blocks)
    cy = math.ceil(blocks / cx)
    ibw = max(d for d in range(1, min(Kt, 16) + 1) if Kt % d == 0 and d * pcn <= 64)
    return cx, cy, pcn, ibw


def decode_linear_config(K: int, N: int, grid, activation=None) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """1D ``mcast_in0`` program config for ``[1, 1, <= 32, K] x [K, N]`` on ``grid`` (CoreCoord-like, ``.x/.y``);
    ``activation`` = a ``ttnn.UnaryOpType`` fused into the output tiles."""
    gx, gy = int(grid.x), int(grid.y)
    Kt, Nt = K // TILE, math.ceil(N / TILE)
    assert K % TILE == 0, K
    hit = _TABLE.get((K, N))
    if hit is not None and hit[0] <= gx and hit[1] <= gy and Kt % hit[3] == 0:
        cx, cy, pcn, ibw = hit
    else:
        cx, cy, pcn, ibw = _heuristic(Kt, Nt, gx, gy)
    assert math.ceil(Nt / pcn) <= cx * cy, (K, N, cx, cy, pcn)
    sub_w = max(d for d in (1, 2, 4) if pcn % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cx, cy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        out_block_h=1,
        out_block_w=pcn,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=activation,
        mcast_in0=True,
    )


class DecodeLinear:
    """Caches the decode program config of one weight; ``__call__`` = ``ttnn.linear`` with it for
    decode-shaped inputs (``[1, 1, M <= 32, K]``), the default dispatch otherwise."""

    def __init__(self, device, weight: ttnn.Tensor, activation=None):
        self.w = weight
        self.K, self.N = int(weight.shape[-2]), int(weight.shape[-1])
        self._grid = device.compute_with_storage_grid_size()
        self._activation = activation  # fused into the decode config only (the default dispatch applies none)
        self._cfg = None

    def config(self):
        if self._cfg is None:
            self._cfg = decode_linear_config(self.K, self.N, self._grid, self._activation)
        return self._cfg

    def __call__(self, x: ttnn.Tensor, **kw) -> ttnn.Tensor:
        shape = x.shape
        if len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[-2] <= TILE and "program_config" not in kw:
            return ttnn.linear(x, self.w, program_config=self.config(), **kw)
        return ttnn.linear(x, self.w, **kw)


# ------------------------------------------------------------------------ M-independent prefill matmuls
PREFILL_ROW_BLOCK = 2048  # rows per call of ``prefill_linear`` (bounds per_core_M, i.e. the per-core L1 blocks)


def prefill_linear_config(M: int, K: int, N: int, grid, in0_block_w: int = 4):
    """2D mcast program config for ``[1, 1, M, K] x [K, N]`` (narrow N) whose K blocking is fixed, so a row's
    result is bit-identical for any M: the default dispatch picks ``in0_block_w`` per shape, and a 4096 x 128
    weight then gives rows that differ by one bf16 ulp between a 4128-row and a 1024-row call of the same
    rows (measured 2026-10-20, chip 2), which made the pooled-key index of a chunked prefill differ from
    the single-shot one. Grid: the N tiles over the columns (one per core), the M tiles over the rows."""
    assert M % TILE == 0 and K % TILE == 0 and N % TILE == 0, (M, K, N)
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    assert Kt % in0_block_w == 0, (K, in0_block_w)
    gx, gy = int(grid.x), int(grid.y)
    pcn = 1
    while math.ceil(Nt / pcn) > gx:
        pcn += 1
    assert Nt % pcn == 0, (N, pcn)  # narrow stems only (no ragged column block)
    cx = Nt // pcn
    cy = min(gy, Mt)
    pcm = math.ceil(Mt / cy)
    sub_w = max(d for d in (1, 2, 4) if pcn % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cx, cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=None,
    )


def prefill_linear(x: ttnn.Tensor, w: ttnn.Tensor, grid, *, row_block: int = PREFILL_ROW_BLOCK, **kw) -> ttnn.Tensor:
    """``ttnn.linear(x, w)`` for a prefill-shaped ``x`` [1, 1, M, K] (M % 32 == 0) through
    ``prefill_linear_config`` in row blocks of ``row_block`` rows: every row's result is independent of M and
    of the blocking (the chunked-prefill equality needs the index stems to be). Decode-shaped inputs (M <= 32)
    keep the default dispatch."""
    M, K, N = int(x.shape[-2]), int(x.shape[-1]), int(w.shape[-1])
    if M <= TILE or "program_config" in kw:
        return ttnn.linear(x, w, **kw)
    parts = []
    for r0 in range(0, M, row_block):
        r1 = min(M, r0 + row_block)
        xs = x if (r0, r1) == (0, M) else ttnn.slice(x, [0, 0, r0, 0], [1, 1, r1, K])
        parts.append(ttnn.linear(xs, w, program_config=prefill_linear_config(r1 - r0, K, N, grid), **kw))
        if xs is not x:
            ttnn.deallocate(xs)
    if len(parts) == 1:
        return parts[0]
    out = ttnn.concat(parts, dim=-2)
    for t in parts:
        ttnn.deallocate(t)
    return out


def decode_linear(x: ttnn.Tensor, w: ttnn.Tensor, device, **kw) -> ttnn.Tensor:
    """Uncached form of ``DecodeLinear`` (config built per call; use ``DecodeLinear`` in hot paths)."""
    shape = x.shape
    if len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[-2] <= TILE and "program_config" not in kw:
        cfg = decode_linear_config(int(w.shape[-2]), int(w.shape[-1]), device.compute_with_storage_grid_size())
        return ttnn.linear(x, w, program_config=cfg, **kw)
    return ttnn.linear(x, w, **kw)

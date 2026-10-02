# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-lang pad of the per-device logits shard to the sampler's power-of-two top-k width.

The sampler pads each [rows, V/D] logits shard to the next power of two with -inf before ``ttnn.topk``
(its fast path). TTNN runs that as a typecast + FillPad + Pad; this one kernel copies the logit tiles and
fills the tail tiles with -inf:
    out[:, :W] = x;  out[:, W:] = -inf
Work is split by output tile columns over a GRID[0] x GRID[1] grid; every node owns ``cols_per_node`` whole
columns, either all copied (x has them) or all filled, so each kernel's trip count is uniform per node.
"""

import ttl
import ttnn

TILE = 32
GRID = (8, 8)


@ttl.operation(grid=GRID)
def logits_pad_kernel(x, out):
    in_cols = x.shape[3] // TILE
    out_cols = out.shape[3] // TILE
    grid_x, grid_y = ttl.grid_size(dims=2)
    cols_per_node = out_cols // (grid_x * grid_y)
    copy_nodes = in_cols // cols_per_node

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1, 1, 1), block_count=2)
    o_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1, 1, 1), block_count=2)

    @ttl.compute()
    def compute():
        nx, ny = ttl.node(dims=2)
        node = ny * grid_x + nx
        if node < copy_nodes:
            for _ in range(cols_per_node):
                with x_dfb.wait() as a, o_dfb.reserve() as o:
                    o.store(a)
        else:
            for _ in range(cols_per_node):
                with o_dfb.reserve() as o:
                    # a negative literal compiles to a neg op, not a constant: fill +big, then negate
                    o.store(ttl.math.neg(ttl.math.fill(o, 3.0e38)))

    @ttl.datamovement()
    def read():
        nx, ny = ttl.node(dims=2)
        node = ny * grid_x + nx
        if node < copy_nodes:
            for i in range(cols_per_node):
                with x_dfb.reserve() as a:
                    ttl.copy(x[0, 0, 0, node * cols_per_node + i], a).wait()

    @ttl.datamovement()
    def write():
        nx, ny = ttl.node(dims=2)
        node = ny * grid_x + nx
        for i in range(cols_per_node):
            with o_dfb.wait() as o:
                ttl.copy(o, out[0, 0, 0, node * cols_per_node + i]).wait()


def logits_pad(x, out):
    """x: [1,1,32,W] tile bf16 (W a multiple of the per-node column block); out: preallocated [1,1,32,Wp]."""
    nodes = GRID[0] * GRID[1]
    in_cols, out_cols = x.shape[3] // TILE, out.shape[3] // TILE
    assert out_cols % nodes == 0 and in_cols % (out_cols // nodes) == 0, (x.shape, out.shape)
    logits_pad_kernel(x, out)
    return out


if __name__ == "__main__":  # simulator only: ttlang-sim tt/ttl_logits_pad.py
    import torch

    assert "sim" in ttnn.__name__, f"run under ttlang-sim (ttnn is {ttnn.__name__})"
    device = ttnn.open_device(device_id=0)
    w, wp = 784 * TILE, 1024 * TILE
    xt = torch.randn(1, 1, TILE, w, dtype=torch.bfloat16)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.from_torch(
        torch.zeros(1, 1, TILE, wp, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    logits_pad(x, out)
    got = ttnn.to_torch(out).float()
    ok_copy = torch.equal(got[..., :w], xt.float())
    ok_fill = bool((got[..., w:] < -1e38).all())
    print("logits_pad sim copy exact:", ok_copy, "tail -inf:", ok_fill)
    ttnn.close_device(device)

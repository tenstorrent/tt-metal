# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""tt-lang kernel for the MLP gate -- EVALUATED, NOT WIRED IN. See the blocker below.

`fused_swiglu_ff13` is the one fusion the op library cannot express for this MLP:
``silu(x @ w1) * (x @ w3)``. TTNN runs it as three programs (two linears and a gated
multiply) and materialises BOTH ``[m, hidden]`` intermediates between them; a single
kernel would keep them in L1 and read the shared activation ``x`` once instead of twice.

BLOCKER (ttl 1.0.1, this model): the compiler will not lower a MIXED-DTYPE matmul.

    error: element type mismatch: lhs has '!ttcore.tile<32x32, bf16>'
                                  but rhs has '!ttcore.tile<32x32, bfp_bf4>'

This model's whole precision strategy is bf16 activations against bf4_b (ff1/ff3) and
bf8_b (ff2) weights, so expressing these matmuls in tt-lang would require upcasting the
weights to bf16 -- roughly 4x the weight bytes on ops that are already bandwidth-bound,
i.e. the fusion would have to give back more than it could ever win. That is the
documented hand-off point to the C++/Metalium rung, which can take the block-float
operands directly.

Getting this far also turned up three places where the catalogued tt-lang template has
drifted from ttl 1.0.1, recorded here so the next attempt starts from the real API:
  * ``ttl.block`` does not exist; fill is ``ttl.math.fill(block, value)``.
  * ``make_dataflow_buffer_like(t, shape=...)``'s shape rank must equal the TENSOR's
    rank, and ``t[i, j]`` needs one index per tensor dimension -- so a rank-4
    ``[1, 1, S, D]`` activation must be reshaped to rank 2 first, as the template's
    own 2-D indexing assumes.
  * ``@`` on two tile blocks yields a rank-2 result, so the accumulator and output
    buffers have to be rank 2 as well.

Still missing for a working version: ``grid=(R, C)`` alone does NOT distribute the
loops -- every core would run the identical program and race on the output. Real
distribution goes through the ``indexing_maps`` / ``iterator_types`` arguments of
``ttl.operation``, which this attempt did not reach.
"""

import ttl
import ttnn

TILE = 32


def fused_swiglu_grid(mesh_device, n_tiles: int):
    """Largest (gx, gy) whose core count divides ``n_tiles`` -- each core must own a
    whole number of output column tiles, and the device grid is resolved, never
    hard-coded."""
    grid = mesh_device.compute_with_storage_grid_size()
    best = (1, 1)
    for gy in range(1, grid.y + 1):
        for gx in range(1, grid.x + 1):
            cores = gx * gy
            if n_tiles % cores == 0 and cores > best[0] * best[1]:
                best = (gx, gy)
    return best


def make_fused_swiglu_ff13(grid):
    """Build the operation for a given core grid (the decorator needs it at def time)."""

    @ttl.operation(grid=grid)
    def fused_swiglu_ff13(x: ttnn.Tensor, w1: ttnn.Tensor, w3: ttnn.Tensor, y: ttnn.Tensor) -> None:
        m_tiles = x.shape[0] // TILE
        k_tiles = x.shape[1] // TILE
        n_tiles = y.shape[1] // TILE
        cores = grid[0] * grid[1]
        n_per_core = n_tiles // cores

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        w1_dfb = ttl.make_dataflow_buffer_like(w1, shape=(1, 1), block_count=2)
        w3_dfb = ttl.make_dataflow_buffer_like(w3, shape=(1, 1), block_count=2)
        acc1_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
        acc3_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def read():
            for mt in range(m_tiles):
                for nt in range(n_per_core):
                    for kt in range(k_tiles):
                        with x_dfb.reserve() as x_blk, w1_dfb.reserve() as w1_blk, w3_dfb.reserve() as w3_blk:
                            tx = ttl.copy(x[mt, kt], x_blk)
                            t1 = ttl.copy(w1[kt, nt], w1_blk)
                            t3 = ttl.copy(w3[kt, nt], w3_blk)
                            tx.wait()
                            t1.wait()
                            t3.wait()

        @ttl.compute()
        def compute():
            for _ in range(m_tiles):
                for _ in range(n_per_core):
                    with acc1_dfb.reserve() as a1, acc3_dfb.reserve() as a3:
                        # this ttl exposes fill as ttl.math.fill(block, value); the
                        # catalogued template's ttl.block.fill does not exist here.
                        a1.store(ttl.math.fill(a1, 0.0))
                        a3.store(ttl.math.fill(a3, 0.0))
                    for _ in range(k_tiles):
                        with x_dfb.wait() as x_blk, w1_dfb.wait() as w1_blk, w3_dfb.wait() as w3_blk:
                            with acc1_dfb.wait() as p1, acc3_dfb.wait() as p3:
                                with acc1_dfb.reserve() as a1, acc3_dfb.reserve() as a3:
                                    a1.store(p1 + x_blk @ w1_blk)
                                    a3.store(p3 + x_blk @ w3_blk)
                    with acc1_dfb.wait() as a1, acc3_dfb.wait() as a3:
                        with y_dfb.reserve() as y_blk:
                            # gate: silu on the w1 branch, multiplied by the w3 branch
                            y_blk.store(ttl.math.silu(a1) * a3)

        @ttl.datamovement()
        def write():
            for mt in range(m_tiles):
                for nt in range(n_per_core):
                    with y_dfb.wait() as y_blk:
                        ttl.copy(y_blk, y[mt, nt]).wait()

    return fused_swiglu_ff13

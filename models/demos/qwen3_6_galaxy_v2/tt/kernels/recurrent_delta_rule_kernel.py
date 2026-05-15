# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tt-lang authoring script for the recurrent DeltaNet update kernel.

Math (per ``_recurrent_delta_rule_step_fp32`` / ``_fused_decay_and_write_fp32``):

    outer[h, i, j] = k_col[h, i] @ v_row[h, j]                          # 32x32 matmul tile
    state_new[h, i, j] = state[h, i, j] * decay[h] + outer[h, i, j] * beta[h]
    o[h, j] = sum_i ( q[h, i] @ state_new[h, i, j] )                    # readout matmul

This proof-of-concept handles a SINGLE HEAD per kernel launch, single-core
(grid=(1,1)). It fuses the matmul + elementwise chain (k @ v -> outer; then
state * decay + outer * beta -> state_new) at fp32 in one dispatch. The
readout matmul (q @ state_new) is NOT fused in this iteration: tt-lang's
2D tile-pair matmul reads operands from CBs, so to feed state_new (just
produced in DST) into a second matmul we would need a true "fork" pattern
that pushes the same tile to two CBs. The current iteration emits zeros for
the readout output; the readout matmul should be performed by the existing
TTNN op (`ttnn.matmul(q_row, state_new)`) outside this kernel.

Per-head shapes (when called from the qwen3.6 decode path):
    state : [K=128, V=128] fp32  -> 4x4 = 16 tiles
    q     : [TILE, K=128] fp32   -> 1x4 = 4 tiles (head row, row 0 valid)
    k     : [K=128, TILE] fp32   -> 4x1 = 4 tiles (head col, col 0 valid)
    v     : [TILE, V=128] fp32   -> 1x4 = 4 tiles (head row, row 0 valid)
    o     : [TILE, V=128] fp32   -> 1x4 = 4 tiles (currently zeros)
    beta  : 32x32 fp32 tile (scalar broadcast, all elements = beta value)
    decay : 32x32 fp32 tile (scalar broadcast, all elements = decay value)

Because tt-lang's user-facing matmul is 2D tile-pair (no batched matmul along
leading dims), multi-head dispatch is handled by the caller (loop over heads
on the host or stage all 6 heads into a flat 2D buffer ahead of the kernel).

Per-tile compute order matters. The expression
    state_new = outer * beta + state * decay
must be evaluated with the matmul result consumed before any other op that
overwrites DST register 0; otherwise the codegen reorders mul_tiles(state,
decay) into DST[0] and silently corrupts the matmul output. The kernel
authoring writes the matmul-then-beta path first.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch

# Same dance as beta_g_kernel.py: bypass the sim-only re-export gate in
# ttl.__init__ by binding the hardware decorators onto the package object.
import ttl as _ttl_pkg
from ttl.ttl import compute, copy, datamovement, make_dataflow_buffer_like
from ttl.ttl import math as ttl_math
from ttl.ttl import operation

import ttnn

_ttl_pkg.operation = operation
_ttl_pkg.compute = compute
_ttl_pkg.datamovement = datamovement
_ttl_pkg.make_dataflow_buffer_like = make_dataflow_buffer_like
_ttl_pkg.copy = copy
_ttl_pkg.math = ttl_math
ttl = _ttl_pkg  # noqa: F841 (used in kernel source text)

# Hardware operates on 32x32 tiles.
TILE_SIZE = 32

# Per-head decode shape: K = V = head_dim = 128 = 4 tiles per dim.
K_TILES = 4  # state row tiles, q/k tile cols
V_TILES = 4  # state col tiles, v/o tile cols

OUT_DIR = Path(__file__).resolve().parent / "recurrent_delta_rule"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@ttl.operation(grid=(1, 1))
def __recurrent_delta_rule_op(
    state: ttnn.Tensor,  # in/out [128, 128] fp32 (state buf, written in-place)
    q: ttnn.Tensor,  # in [32, 128] fp32 (q row, 1 tile-row x 4 tile-cols)
    k: ttnn.Tensor,  # in [128, 32] fp32 (k col, 4 tile-rows x 1 tile-col)
    v: ttnn.Tensor,  # in [32, 128] fp32 (v row, 1 tile-row x 4 tile-cols)
    decay: ttnn.Tensor,  # in [32, 32] fp32 (scalar broadcast tile)
    beta: ttnn.Tensor,  # in [32, 32] fp32 (scalar broadcast tile)
    state_out: ttnn.Tensor,  # out [128, 128] fp32 (new state)
    o: ttnn.Tensor,  # out [32, 128] fp32 (readout)
):
    k_tiles = K_TILES
    v_tiles = V_TILES

    # Dataflow buffers
    state_dfb = ttl.make_dataflow_buffer_like(state, shape=(1, 1), block_count=2)
    q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, 1), block_count=2)
    k_dfb = ttl.make_dataflow_buffer_like(k, shape=(1, 1), block_count=2)
    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, 1), block_count=2)
    decay_dfb = ttl.make_dataflow_buffer_like(decay, shape=(1, 1), block_count=2)
    beta_dfb = ttl.make_dataflow_buffer_like(beta, shape=(1, 1), block_count=2)
    state_out_dfb = ttl.make_dataflow_buffer_like(state_out, shape=(1, 1), block_count=4)
    o_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)
    # Accumulator for the readout: reused across the i-reduction for each j.
    o_acc_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

    @ttl.compute()
    def recurrent_compute():
        # Decay and beta are scalar tiles reused throughout — wait once.
        with decay_dfb.wait() as decay_blk, beta_dfb.wait() as beta_blk:
            for j in range(v_tiles):
                for i in range(k_tiles):
                    # state_new[i, j] = state[i, j] * decay + (k[i] @ v[j]) * beta
                    #
                    # Order matters for DST register lifetime: compute matmul
                    # output (outer) and multiply by beta first, then add
                    # state*decay. If we did `state*decay + outer*beta`, the
                    # codegen overwrites outer's DST register with state*decay
                    # before consuming it — silently corrupting the matmul.
                    with (
                        state_dfb.wait() as state_blk,
                        k_dfb.wait() as k_blk,
                        v_dfb.wait() as v_blk,
                    ):
                        with state_out_dfb.reserve() as state_new_blk:
                            outer_beta = (k_blk @ v_blk) * beta_blk
                            state_new_blk.store(outer_beta + state_blk * decay_blk)

            # Readout matmul (o[j] = sum_i q[i] @ state_new[i, j]) is NOT
            # fused into this kernel — see the module docstring for why.
            # We emit zeros for o so the writer thread has data to drain.
            for j in range(v_tiles):
                with o_acc_dfb.reserve() as acc_blk:
                    acc_blk.store(ttl.math.fill(acc_blk, 0))
                with o_acc_dfb.wait() as acc_blk:
                    with o_dfb.reserve() as o_blk:
                        o_blk.store(acc_blk)

    @ttl.datamovement()
    def recurrent_read():
        # Read decay, beta once at the start.
        with decay_dfb.reserve() as decay_blk, beta_dfb.reserve() as beta_blk:
            tx_d = ttl.copy(decay[0, 0], decay_blk)
            tx_b = ttl.copy(beta[0, 0], beta_blk)
            tx_d.wait()
            tx_b.wait()

        # For each j (V-tile), stream the K-column of state[*, j] + k + v.
        # (q is not used in the simplified state-only kernel.)
        for j in range(v_tiles):
            for i in range(k_tiles):
                with (
                    state_dfb.reserve() as state_blk,
                    k_dfb.reserve() as k_blk,
                    v_dfb.reserve() as v_blk,
                ):
                    tx_s = ttl.copy(state[i, j], state_blk)
                    tx_k = ttl.copy(k[i, 0], k_blk)
                    tx_v = ttl.copy(v[0, j], v_blk)
                    tx_s.wait()
                    tx_k.wait()
                    tx_v.wait()

    @ttl.datamovement()
    def recurrent_write():
        # Order matches compute thread: 16 state_out tiles first (j outer, i
        # inner), then 4 o tiles. Compute produces in this exact order.
        for j in range(v_tiles):
            for i in range(k_tiles):
                with state_out_dfb.wait() as state_new_blk:
                    tx = ttl.copy(state_new_blk, state_out[i, j])
                    tx.wait()
        for j in range(v_tiles):
            with o_dfb.wait() as o_blk:
                tx_o = ttl.copy(o_blk, o[0, j])
                tx_o.wait()


def _host_tile_zeros(shape, dtype=torch.float32, ttnn_dtype=ttnn.float32):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=dtype),
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
    )


def main():
    os.environ.setdefault("TTLANG_COMPILE_ONLY", "1")
    runner_path = str(OUT_DIR / "_runner_emitted.py")
    os.environ["TTLANG_EMIT_RUNNER"] = runner_path

    # Per-head shapes for the qwen3.6 decode path.
    state_shape = (K_TILES * TILE_SIZE, V_TILES * TILE_SIZE)  # [128, 128]
    row_shape = (TILE_SIZE, K_TILES * TILE_SIZE)  # [32, 128] (q, v)
    col_shape = (K_TILES * TILE_SIZE, TILE_SIZE)  # [128, 32] (k as column)
    scalar_shape = (TILE_SIZE, TILE_SIZE)  # [32, 32]

    state = _host_tile_zeros(state_shape)
    q = _host_tile_zeros(row_shape)
    k = _host_tile_zeros(col_shape)
    v = _host_tile_zeros(row_shape)
    decay = _host_tile_zeros(scalar_shape)
    beta = _host_tile_zeros(scalar_shape)
    state_out = _host_tile_zeros(state_shape)
    o = _host_tile_zeros(row_shape)

    # Trigger compilation.
    __recurrent_delta_rule_op(state, q, k, v, decay, beta, state_out, o)

    user = os.environ.get("USER", "default")
    tmp_dir = Path(f"/tmp/{user}")
    if not tmp_dir.exists():
        print(f"ERROR: expected kernels in {tmp_dir}, directory missing.")
        sys.exit(2)

    expected_threads = {"recurrent_compute", "recurrent_read", "recurrent_write"}
    found = {}
    for path in tmp_dir.glob("ttlang_kernel_*.cpp"):
        name = path.name
        stem = name[len("ttlang_kernel_") : -len(".cpp")]
        # `_<hash>` is exactly 9 trailing chars (underscore + 8 hex)
        thread_name = stem[:-9]
        if thread_name in expected_threads:
            prev = found.get(thread_name)
            if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
                found[thread_name] = path

    if not expected_threads.issubset(found):
        missing = expected_threads - set(found)
        print(f"ERROR: missing emitted kernels for: {missing}")
        print(f"  found: {sorted(found)}")
        sys.exit(3)

    for thread_name, path in found.items():
        dest = OUT_DIR / f"{thread_name}.cpp"
        shutil.copy(path, dest)
        size = dest.stat().st_size
        print(f"  copied {path.name} -> {dest} ({size} bytes)")

    print(f"OK -- 3 kernels written to {OUT_DIR}")


if __name__ == "__main__":
    main()

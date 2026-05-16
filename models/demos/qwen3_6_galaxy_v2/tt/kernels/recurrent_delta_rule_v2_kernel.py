# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""V2-17c authoring script for the recurrent DeltaNet update + readout kernel.

Extends V2-17 with TWO changes:

1. **Multi-core grid** — ``grid=(4, 1)`` distributes the 4 j-tiles (V-axis)
   across 4 Tensix cores. Each core handles a single j (V-tile-column) and
   loops over i internally (K-axis, 4 iterations). The V2-17 baseline used
   ``grid=(1, 1)`` (single core), serializing all 16 (i, j) tile-pairs onto
   one Tensix.

2. **Readout matmul fusion** — the V2-17 kernel only updated state; the
   readout ``o[j] = sum_i q[i] @ state_new[i, j]`` was performed externally
   via ``ttnn.matmul``. This kernel fuses the readout into the same compute
   thread by forking the ``state_new`` tile into a second CB
   (``state_readout_dfb``) immediately after producing it. The fork uses
   the ``dst.store(src_blk)`` tile-copy primitive (see V2-17's
   ``o_blk.store(acc_blk)`` zero-copy pattern). Eliminates one DRAM round
   trip per (i, j) tile pair.

Per-head shapes (same as V2-17):
    state : [K=128, V=128] fp32  -> 4x4 = 16 tiles
    q     : [TILE, K=128] fp32   -> 1x4 = 4 tiles
    k     : [K=128, TILE] fp32   -> 4x1 = 4 tiles
    v     : [TILE, V=128] fp32   -> 1x4 = 4 tiles
    decay : 32x32 fp32 tile (scalar broadcast)
    beta  : 32x32 fp32 tile (scalar broadcast)
    state_out : [K=128, V=128] fp32 (new state)
    o     : [TILE, V=128] fp32 (readout)

Each of the 4 cores reads its own column slice of state (4 tiles, 1 col j)
and v (1 tile at col j), plus the FULL k column (4 tiles) and FULL q row
(4 tiles). The replicated reads cost 4x BW on the small inputs but the win
is 4x parallelism on the heavy matmul + elementwise + readout work.

Per-tile compute order constraints (carried over from V2-17):
- ``outer = k @ v`` matmul output lives in DST register 0. Multiply by
  ``beta`` BEFORE any other op that would overwrite DST[0]. Then add
  ``state * decay`` last so the matmul stays valid.
- Readout matmul (``q @ state_new``) similarly must consume its operands
  before any subsequent op overwrites DST.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch

# Same dance as beta_g_kernel.py / V2-17: bypass the sim-only re-export gate
# in ttl.__init__ by binding the hardware decorators onto the package object.
import ttl as _ttl_pkg
from ttl.ttl import compute, copy, datamovement, make_dataflow_buffer_like
from ttl.ttl import math as ttl_math
from ttl.ttl import node, operation

import ttnn

_ttl_pkg.operation = operation
_ttl_pkg.compute = compute
_ttl_pkg.datamovement = datamovement
_ttl_pkg.make_dataflow_buffer_like = make_dataflow_buffer_like
_ttl_pkg.copy = copy
_ttl_pkg.math = ttl_math
_ttl_pkg.node = node
ttl = _ttl_pkg  # noqa: F841 (used in kernel source text)

# Hardware operates on 32x32 tiles.
TILE_SIZE = 32

# Per-head decode shape: K = V = head_dim = 128 = 4 tiles per dim.
K_TILES = 4
V_TILES = 4

OUT_DIR = Path(__file__).resolve().parent / "recurrent_delta_rule_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@ttl.operation(grid=(V_TILES, 1))
def __recurrent_delta_rule_v2_op(
    state: ttnn.Tensor,  # in [128, 128] fp32 (state buf)
    q: ttnn.Tensor,  # in [32, 128] fp32 (q row, 1 tile-row x 4 tile-cols)
    k: ttnn.Tensor,  # in [128, 32] fp32 (k col, 4 tile-rows x 1 tile-col)
    v: ttnn.Tensor,  # in [32, 128] fp32 (v row, 1 tile-row x 4 tile-cols)
    decay: ttnn.Tensor,  # in [32, 32] fp32 (scalar broadcast tile)
    beta: ttnn.Tensor,  # in [32, 32] fp32 (scalar broadcast tile)
    state_out: ttnn.Tensor,  # out [128, 128] fp32 (new state)
    o: ttnn.Tensor,  # out [32, 128] fp32 (readout: q @ state_new)
):
    k_tiles = K_TILES

    # Dataflow buffers (per-core L1)
    state_dfb = ttl.make_dataflow_buffer_like(state, shape=(1, 1), block_count=2)
    q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, 1), block_count=2)
    k_dfb = ttl.make_dataflow_buffer_like(k, shape=(1, 1), block_count=2)
    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, 1), block_count=2)
    decay_dfb = ttl.make_dataflow_buffer_like(decay, shape=(1, 1), block_count=2)
    beta_dfb = ttl.make_dataflow_buffer_like(beta, shape=(1, 1), block_count=2)
    state_out_dfb = ttl.make_dataflow_buffer_like(state_out, shape=(1, 1), block_count=2)
    o_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)
    # Compute-internal: forked copy of state_new used by the readout matmul
    # in the same iteration. Lives in L1 only on this core.
    state_readout_dfb = ttl.make_dataflow_buffer_like(state_out, shape=(1, 1), block_count=2)
    # Compute-internal: running readout accumulator for this core's j.
    o_acc_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

    @ttl.compute()
    def recurrent_compute():
        # Decay and beta are scalar tiles reused throughout — wait once.
        with decay_dfb.wait() as decay_blk, beta_dfb.wait() as beta_blk:
            # Initialize the per-core readout accumulator (o[j]) to zero.
            with o_acc_dfb.reserve() as acc_blk:
                acc_blk.store(ttl.math.fill(acc_blk, 0))

            for i in range(k_tiles):
                # 1) state_new[i, j] = (k[i] @ v[j]) * beta + state[i, j] * decay
                #    Fork the result into TWO CBs:
                #      - state_out_dfb : drained to DRAM by the writer thread
                #      - state_readout_dfb : consumed by the readout matmul
                #                            below (compute-internal, L1-resident)
                #    The fork is implemented by emitting the same expression
                #    twice. Both produce the same value (deterministic fp32
                #    matmul on the same tile inputs). This costs two passes
                #    through the matmul pipeline per (i, j) but avoids a DRAM
                #    round-trip for the readout.
                #
                #    Order matters for DST: the matmul output must be consumed
                #    by *beta BEFORE state*decay overwrites DST[0] (see V2-17
                #    docstring for the codegen quirk).
                with (
                    state_dfb.wait() as state_blk,
                    k_dfb.wait() as k_blk,
                    v_dfb.wait() as v_blk,
                ):
                    with state_out_dfb.reserve() as state_new_blk:
                        outer_beta_1 = (k_blk @ v_blk) * beta_blk
                        state_new_blk.store(outer_beta_1 + state_blk * decay_blk)
                    with state_readout_dfb.reserve() as state_ro_blk:
                        outer_beta_2 = (k_blk @ v_blk) * beta_blk
                        state_ro_blk.store(outer_beta_2 + state_blk * decay_blk)

                # 2) o_acc += q[i] @ state_new
                with (
                    q_dfb.wait() as q_blk,
                    state_readout_dfb.wait() as state_ro_blk,
                    o_acc_dfb.wait() as pre_acc_blk,
                ):
                    with o_acc_dfb.reserve() as acc_blk:
                        acc_blk.store(pre_acc_blk + q_blk @ state_ro_blk)

            # Drain the final accumulator to the writer-bound o CB.
            with o_acc_dfb.wait() as acc_blk:
                with o_dfb.reserve() as o_blk:
                    o_blk.store(acc_blk)

    @ttl.datamovement()
    def recurrent_read():
        # Determine this core's j (V-tile column index).
        # grid=(V_TILES, 1) ⇒ node(dims=2) returns (node_x, node_y) where
        # node_x ∈ 0..V_TILES-1 and node_y == 0.
        j, _ = ttl.node(dims=2)

        # Read decay, beta once at the start (replicated across all cores).
        with decay_dfb.reserve() as decay_blk, beta_dfb.reserve() as beta_blk:
            tx_d = ttl.copy(decay[0, 0], decay_blk)
            tx_b = ttl.copy(beta[0, 0], beta_blk)
            tx_d.wait()
            tx_b.wait()

        for i in range(k_tiles):
            # state[i, j], k[i, 0], v[0, j], q[0, i] for this core's j.
            with (
                state_dfb.reserve() as state_blk,
                k_dfb.reserve() as k_blk,
                v_dfb.reserve() as v_blk,
                q_dfb.reserve() as q_blk,
            ):
                tx_s = ttl.copy(state[i, j], state_blk)
                tx_k = ttl.copy(k[i, 0], k_blk)
                tx_v = ttl.copy(v[0, j], v_blk)
                tx_q = ttl.copy(q[0, i], q_blk)
                tx_s.wait()
                tx_k.wait()
                tx_v.wait()
                tx_q.wait()

    @ttl.datamovement()
    def recurrent_write():
        j, _ = ttl.node(dims=2)
        # Write the column of state_out for this core's j (4 tiles).
        for i in range(k_tiles):
            with state_out_dfb.wait() as state_new_blk:
                tx = ttl.copy(state_new_blk, state_out[i, j])
                tx.wait()
        # Write the readout tile for this core's j (1 tile).
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
    __recurrent_delta_rule_v2_op(state, q, k, v, decay, beta, state_out, o)

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

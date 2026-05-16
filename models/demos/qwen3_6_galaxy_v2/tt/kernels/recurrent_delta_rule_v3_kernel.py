# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""V2-17d authoring script for the multi-head batched recurrent DeltaNet kernel.

Extends V2-17c with TWO further changes:

1. **Multi-HEAD inside the same launch** — grid=(V_TILES=4, V_HEADS=6) =
   24 cores. Each core (j_v, h_head) handles V-tile-column j_v of head
   h_head. Eliminates the per-head Python loop (V2-17c integration
   issued 6 launches/layer; V3 issues 1).

2. **In-place state writeback** — state and state_out are passed as
   pointers to the SAME persistent ``dn_state_buffer``. Compute thread
   reads state[h, i, j_v] then writes state_out[h, i, j_v] in the same
   iteration — the read happens before the write so aliasing is safe
   within a tile.

Logical tensor shapes (B=1, H=n_v_per_row=6, K=V=128):
    state, state_out : [768, 128]   # tile-shape (24, 4) = (H*K_TILES, V_TILES)
    q                : [192, 128]   # tile (6, 4) = (H, K_TILES); row 0 of each head valid
    k_col            : [768, 32]    # tile (24, 1) = (H*K_TILES, 1); col 0 valid (k as column)
    v                : [192, 128]   # tile (6, 4) = (H, V_TILES); row 0 valid
    decay, beta      : [192, 32]    # tile (6, 1) = (H, 1); broadcast tile per head
    o                : [192, 128]   # tile (6, 4) = (H, V_TILES); row 0 carries readout

Per-head tile-coordinate mapping (h ∈ [0, H-1], i ∈ [0, K_TILES-1]):
    state[h, i, j_v] -> tile (h*K_TILES + i, j_v)
    q[h, ki]         -> tile (h, ki)
    k_col[h, i]      -> tile (h*K_TILES + i, 0)
    v[h, j_v]        -> tile (h, j_v)
    decay/beta[h]    -> tile (h, 0)
    o[h, j_v]        -> tile (h, j_v)

Per-core work (one (j_v, h) coord):
    For ki = 0 .. K_TILES-1:
        outer       = k_col[h, ki]   @ v[h, j_v]
        state_new   = outer * beta[h] + state[h, ki, j_v] * decay[h]
        STATE_OUT[h, ki, j_v] = state_new
        o_acc      += q[h, ki] @ state_new
    o[h, j_v] = o_acc

Same compute structure as V2-17c, just indexed by an additional head
dimension. The host-side integration produces all per-head broadcast
tiles ONCE per step rather than per-head — total wrapper-op count
collapses from ~60 per layer (V2-17c) to ~5 per layer (V3).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch

# Same dance as beta_g_kernel.py / V2-17c: bypass the sim-only re-export gate
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

# Per-decode shape constants.
HEAD_DIM = 128
K_TILES = HEAD_DIM // TILE_SIZE  # 4
V_TILES = HEAD_DIM // TILE_SIZE  # 4
V_HEADS = 6  # n_v_per_row for qwen3.6 (48 heads / 8 mesh rows)

OUT_DIR = Path(__file__).resolve().parent / "recurrent_delta_rule_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@ttl.operation(grid=(V_TILES, V_HEADS))
def __recurrent_delta_rule_v3_op(
    state: ttnn.Tensor,  # in/out [H*K=768, V=128] fp32 (in-place aliased with state_out)
    q: ttnn.Tensor,  # in     [H*TILE=192, K=128] fp32
    k_col: ttnn.Tensor,  # in     [H*K=768, TILE=32] fp32
    v: ttnn.Tensor,  # in     [H*TILE=192, V=128] fp32
    decay: ttnn.Tensor,  # in     [H*TILE=192, TILE=32] fp32 (broadcast tile per head)
    beta: ttnn.Tensor,  # in     [H*TILE=192, TILE=32] fp32 (broadcast tile per head)
    state_out: ttnn.Tensor,  # out    [H*K=768, V=128] fp32 (= state, same buffer)
    o: ttnn.Tensor,  # out    [H*TILE=192, V=128] fp32 (row 0 of each head valid)
):
    k_tiles = K_TILES

    # Per-core L1 dataflow buffers. Each (j_v, h) core sees only its own tiles.
    state_dfb = ttl.make_dataflow_buffer_like(state, shape=(1, 1), block_count=2)
    q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, 1), block_count=2)
    k_dfb = ttl.make_dataflow_buffer_like(k_col, shape=(1, 1), block_count=2)
    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, 1), block_count=2)
    decay_dfb = ttl.make_dataflow_buffer_like(decay, shape=(1, 1), block_count=2)
    beta_dfb = ttl.make_dataflow_buffer_like(beta, shape=(1, 1), block_count=2)
    state_out_dfb = ttl.make_dataflow_buffer_like(state_out, shape=(1, 1), block_count=2)
    o_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)
    # Compute-internal: forked copy of state_new used by the readout matmul
    # in the same iteration. Lives in L1 only on this core.
    state_readout_dfb = ttl.make_dataflow_buffer_like(state_out, shape=(1, 1), block_count=2)
    # Compute-internal: running readout accumulator for this core's (h, j_v).
    o_acc_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

    @ttl.compute()
    def recurrent_compute():
        with decay_dfb.wait() as decay_blk, beta_dfb.wait() as beta_blk:
            # Init the running readout accumulator (o[h, j_v]) to zero.
            with o_acc_dfb.reserve() as acc_blk:
                acc_blk.store(ttl.math.fill(acc_blk, 0))

            for i in range(k_tiles):
                # state_new[h, i, j_v] = state[h, i, j_v]*decay[h]
                #                     + (k_col[h, i] @ v[h, j_v]) * beta[h]
                # Fork twice — once to state_out_dfb (drained to DRAM by the
                # writer), once to state_readout_dfb (consumed below by the
                # readout matmul on the same core). Forking via expression
                # recompute is the V2-17c pattern: cheaper than a DRAM round
                # trip for the read-after-write.
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

                # o_acc += q[h, i] @ state_new
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
        # Determine this core's (j_v, h). grid=(V_TILES, V_HEADS) ⇒
        # node_x ∈ [0, V_TILES-1], node_y ∈ [0, V_HEADS-1].
        j_v, h = ttl.node(dims=2)

        # decay[h, 0], beta[h, 0] — one tile per head, read once.
        with decay_dfb.reserve() as decay_blk, beta_dfb.reserve() as beta_blk:
            tx_d = ttl.copy(decay[h, 0], decay_blk)
            tx_b = ttl.copy(beta[h, 0], beta_blk)
            tx_d.wait()
            tx_b.wait()

        h_base = h * k_tiles  # starting tile-row for this head's state slab
        for i in range(k_tiles):
            with (
                state_dfb.reserve() as state_blk,
                k_dfb.reserve() as k_blk,
                v_dfb.reserve() as v_blk,
                q_dfb.reserve() as q_blk,
            ):
                tx_s = ttl.copy(state[h_base + i, j_v], state_blk)
                tx_k = ttl.copy(k_col[h_base + i, 0], k_blk)
                tx_v = ttl.copy(v[h, j_v], v_blk)
                tx_q = ttl.copy(q[h, i], q_blk)
                tx_s.wait()
                tx_k.wait()
                tx_v.wait()
                tx_q.wait()

    @ttl.datamovement()
    def recurrent_write():
        j_v, h = ttl.node(dims=2)
        h_base = h * k_tiles
        # Write the K-tile column of state_out for this core's (h, j_v).
        for i in range(k_tiles):
            with state_out_dfb.wait() as state_new_blk:
                tx = ttl.copy(state_new_blk, state_out[h_base + i, j_v])
                tx.wait()
        # Write the readout tile (1 tile per (h, j_v)).
        with o_dfb.wait() as o_blk:
            tx_o = ttl.copy(o_blk, o[h, j_v])
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

    # 2D logical shapes; the integration tile-pads and reshapes to these.
    state_shape = (V_HEADS * HEAD_DIM, HEAD_DIM)  # [768, 128]
    q_shape = (V_HEADS * TILE_SIZE, HEAD_DIM)  # [192, 128]
    k_col_shape = (V_HEADS * HEAD_DIM, TILE_SIZE)  # [768, 32]
    v_shape = (V_HEADS * TILE_SIZE, HEAD_DIM)  # [192, 128]
    bcast_shape = (V_HEADS * TILE_SIZE, TILE_SIZE)  # [192, 32]
    o_shape = (V_HEADS * TILE_SIZE, HEAD_DIM)  # [192, 128]

    state = _host_tile_zeros(state_shape)
    q = _host_tile_zeros(q_shape)
    k_col = _host_tile_zeros(k_col_shape)
    v = _host_tile_zeros(v_shape)
    decay = _host_tile_zeros(bcast_shape)
    beta = _host_tile_zeros(bcast_shape)
    state_out = _host_tile_zeros(state_shape)
    o = _host_tile_zeros(o_shape)

    # Trigger compilation.
    __recurrent_delta_rule_v3_op(state, q, k_col, v, decay, beta, state_out, o)

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

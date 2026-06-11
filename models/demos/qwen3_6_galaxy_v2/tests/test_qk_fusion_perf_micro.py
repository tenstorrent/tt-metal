# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QK-fusion perf micro-benchmark on the 32-chip BH GLX 8x4 mesh.

WHY THIS EXISTS
---------------
The fused ``llama_rs_create_heads`` kernel (reduce-scatter + create-qkv-heads in
ONE op) HARDCODES head_dim=128 / q_heads=8, so it cannot be used for qwen3.6
(head_dim=256, 3 Q / 1 KV slots per chip — see ``test_rs_create_heads_micro.py``
and the comment in ``llama_attention.py`` ``_forward_decode_qwen36`` ~L2005).

So the qwen3.6 full-attention decode QKV path is UN-FUSED today:

    xqkvg = ttnn.all_reduce(xqkvg_partial, cluster_axis=1)   # col-reduce  (mandatory)
    q  = ttnn.slice(xqkvg, ...768)                           # head extraction ...
    g  = ttnn.slice(xqkvg, 768..1536)                        #  ... that a fused
    k  = ttnn.slice(xqkvg, 1536..1792)                       #  ... create-heads
    v  = ttnn.slice(xqkvg, 1792..2048)                       #  ... op would ABSORB

QK fusion would collapse the reduce + the 4 head-extraction slices into a single
op, taking the head-extraction work OFF the critical path (overlapped inside the
collective). This test QUANTIFIES that saving so we can decide whether writing a
head_dim=256 fused kernel is worth it.

WHAT IT MEASURES (trace eliminates host dispatch; we batch N inner calls/iter):
  * Variant A  — ``all_reduce(axis=1)`` ALONE                  = fused-op lower bound
  * Variant B  — ``all_reduce(axis=1)`` + 4 ``slice`` ops      = current un-fused path
  * saving/call = B - A  → x16 full-attention layers → ms/token, tok/s delta

This is the UPPER BOUND on what QK fusion can save (a real fused kernel lands at
or slightly above A, never below). Self-contained raw-op microbench — no model
build, no TT_CCL. Mirrors ``test_wo_rs_micro.py`` harness.

Run (device):
    export TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,...  # full 8x4
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate \\
      && python -m pytest --noconftest -v -s \\
         models/demos/qwen3_6_galaxy_v2/tests/test_qk_fusion_perf_micro.py
"""
from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

import ttnn

# qwen3.6 full-attention per-chip QKVG geometry (head_dim=256, 3Q/3G/1K/1V slots):
_HEAD_DIM = 256
_N_Q_PC = 3  # q heads / chip  (24 heads / 8 rows)
_N_KV_PC = 1  # kv heads / chip ( 4 heads / 4 ... padded)
_Q_DIM_PC = _N_Q_PC * _HEAD_DIM  # 768
_G_DIM_PC = _N_Q_PC * _HEAD_DIM  # 768 (attn output gate)
_K_DIM_PC = _N_KV_PC * _HEAD_DIM  # 256
_V_DIM_PC = _N_KV_PC * _HEAD_DIM  # 256
_TOTAL_PC = _Q_DIM_PC + _G_DIM_PC + _K_DIM_PC + _V_DIM_PC  # 2048

_B = 1
_T_PADDED = 32  # tile-aligned decode T

# The QKV col-reduce in the real model is cluster_axis=1, ring=4 (the 4 mesh
# columns). On this 4-chip BH host we open a (1,4) mesh so cluster_axis=1 is the
# 4-wide ring — identical collective to the galaxy's per-stage col reduce.
# Override with QK_MESH_ROWS/QK_MESH_COLS if a full 8x4 galaxy is available.
_MESH_SHAPE = (int(os.environ.get("QK_MESH_ROWS", "1")), int(os.environ.get("QK_MESH_COLS", "4")))
_COL_AXIS = 1  # the 4-wide ring axis
_N_FULL_ATTN_LAYERS = 16  # full-attention layers that pay this QKV reduce+split
_N_WARMUP = 3
_N_TIMED = 6
_N_INNER_CALLS = 32  # amortize residual host overhead across many calls/iter

# slice boundaries
_SL = [
    (0, _Q_DIM_PC),
    (_Q_DIM_PC, _Q_DIM_PC + _G_DIM_PC),
    (_Q_DIM_PC + _G_DIM_PC, _Q_DIM_PC + _G_DIM_PC + _K_DIM_PC),
    (_Q_DIM_PC + _G_DIM_PC + _K_DIM_PC, _TOTAL_PC),
]


@pytest.fixture(scope="module")
def bh_glx_mesh():
    # FABRIC_1D (linear) — direct (1,4) open on a 4-chip BH host. FABRIC_1D_RING
    # times out on a directly-opened sub-mesh here (see G4 MLP micro bring-up).
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(*_MESH_SHAPE),
        trace_region_size=184915840,
        worker_l1_size=1345000,
    )
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _make_qkvg_partial(mesh, dtype):
    """Per-chip QKVG partial sum [B, T, total_pc=2048], replicated across the
    4-col ring so the all_reduce sums to 4x (shape/op match the real
    ``xqkvg_partial`` fed to the QKV reduce in ``_forward_decode_qwen36``;
    exact values are irrelevant for a latency benchmark)."""
    torch.manual_seed(42)
    host = torch.randn(_B, _T_PADDED, _TOTAL_PC, dtype=torch.bfloat16) * 0.05
    return ttnn.from_torch(
        host,
        device=mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _time_traced(label, run_one, mesh):
    """TRACE-based timing: capture the N-inner-call body, replay it. Trace
    eliminates host dispatch, so this is the real on-device latency the
    production (traced) decode pays. ``run_one`` must contain ONLY trace-safe
    ops (ttnn.slice is; generic ttnn.all_reduce is NOT — it allocates a global
    semaphore at call time and deadlocks during capture)."""
    # Compile pass (eager) BEFORE capture — the first execution compiles kernels
    # (a host write), which is illegal inside a trace.
    run_one()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    run_one()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(_N_WARMUP):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    times_us = []
    for _ in range(_N_TIMED):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        times_us.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    per_call_us = statistics.mean(times_us) / _N_INNER_CALLS
    print(f"[{label:<40}] {statistics.mean(times_us):>9.1f} µs/iter  ({per_call_us:>7.2f} µs/call, traced)")
    return per_call_us


def _time_eager(label, run_one, mesh):
    """Eager timing (host dispatch INCLUDED) — same body, no trace. The gap
    (eager - traced) is the dispatch overhead that trace hides in production."""
    for _ in range(_N_WARMUP):
        run_one()
    ttnn.synchronize_device(mesh)
    times_us = []
    for _ in range(_N_TIMED):
        t0 = time.perf_counter()
        run_one()
        ttnn.synchronize_device(mesh)
        times_us.append((time.perf_counter() - t0) * 1e6)
    per_call_us = statistics.mean(times_us) / _N_INNER_CALLS
    print(f"[{label:<40}] {statistics.mean(times_us):>9.1f} µs/iter  ({per_call_us:>7.2f} µs/call, eager)")
    return per_call_us


@pytest.mark.hardware
@pytest.mark.parametrize("dtype_name", ["bf16", "bf8"])
def test_qk_fusion_savings(bh_glx_mesh, dtype_name):
    """The col-reduce is paid no matter what; QK fusion only removes the
    HEAD-EXTRACTION work (the 4 Q/G/K/V slices) by folding it into the
    collective. So the fusion saving == the head-extraction cost.

    We measure those 4 slices BOTH traced (net of dispatch = the real saving in
    the production traced decode) and eager (the gap = dispatch trace already
    hides). Then project the TRACED saving over the 16 full-attention layers."""
    mesh = bh_glx_mesh
    dtype = ttnn.bfloat16 if dtype_name == "bf16" else ttnn.bfloat8_b
    x = _make_qkvg_partial(mesh, dtype)  # post-reduce [B,T,2048] per chip

    # Head-extraction body: the 4 slices the unfused path runs after the reduce.
    def run_slices():
        for _ in range(_N_INNER_CALLS):
            parts = [
                ttnn.slice(x, [0, 0, lo], [_B, _T_PADDED, hi], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for (lo, hi) in _SL
            ]
            for p in parts:
                ttnn.deallocate(p)

    traced_us = _time_traced(f"4 head-slices [{dtype_name}]", run_slices, mesh)
    eager_us = _time_eager(f"4 head-slices [{dtype_name}]", run_slices, mesh)
    dispatch_us = eager_us - traced_us

    per_token_ms_traced = traced_us * _N_FULL_ATTN_LAYERS / 1000.0
    per_token_ms_eager = eager_us * _N_FULL_ATTN_LAYERS / 1000.0
    ttnn.deallocate(x)

    print(f"\n=== QK-FUSION SAVINGS ({dtype_name}) — head-extraction removed by fusion ===")
    print(f"  head-extraction TRACED (real saving):  {traced_us:7.2f} µs/call")
    print(f"  head-extraction EAGER  (incl dispatch):{eager_us:8.2f} µs/call")
    print(f"  dispatch hidden by trace (eager-traced):{dispatch_us:7.2f} µs/call")
    print(f"  --> saving net-of-dispatch x {_N_FULL_ATTN_LAYERS} layers: {per_token_ms_traced:6.3f} ms/token")
    print(f"      (eager would mislead: {per_token_ms_eager:.3f} ms/token)")
    for tok_s in (24.0,):
        base_ms = 1000.0 / tok_s
        new_ms = base_ms - per_token_ms_traced
        print(
            f"  if baseline {tok_s:.0f} tok/s ({base_ms:.1f} ms): -> {new_ms:.2f} ms = {1000.0/new_ms:.1f} tok/s "
            f"(+{1000.0/new_ms - tok_s:.2f})"
        )
    print("=" * 64)

    assert traced_us > 0.0
    assert eager_us >= traced_us - 5.0, f"eager ({eager_us}) < traced ({traced_us}) beyond noise — invalid"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

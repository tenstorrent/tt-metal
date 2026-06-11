# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""G=4 MLP-centric decode perf microbenchmark for Qwen3.6-27B.

Validates whether a 4-chip (1x4) intra-stage TP stage can hit the ~70 tok/s
(14.3 ms/token) per-user target before building the full pipeline-parallel stack.

Measures:
  - Per-MLP latency vs analytic DRAM floor (precision sweep)
  - Batch scaling (B=1..128) for throughput amortization claim
  - Single-layer + 64-layer compounding PCC vs torch SwiGLU reference

Run (latency + PCC, all precisions/batches):
    export ARCH_NAME=wormhole_b0 TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest -v -s \\
            models/demos/qwen3_6_galaxy_v2/tests/test_g4_mlp_micro.py

Run with Tracy (device kernel durations):
    export ARCH_NAME=wormhole_b0 TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m tracy -p -v -r -m pytest --noconftest -v -s \\
            models/demos/qwen3_6_galaxy_v2/tests/test_g4_mlp_micro.py \\
            -k "test_g4_mlp_latency_pcc"
"""
from __future__ import annotations

import os
import time

import pytest

pytestmark_hardware = pytest.mark.skipif(
    os.environ.get("G4_RUN_DEVICE", "0") != "1",
    reason="Device tests disabled. Set G4_RUN_DEVICE=1 to run on silicon.",
)

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_galaxy_v2.tests.g4_decode_perf_utils import (
    _MLP_PARAM_FRACTION,
    _N_LAYERS,
    DEFAULT_SNAPSHOT,
    G4_MESH_SHAPE,
    WDTYPE_NAMES,
    analytic_mlp_floor_ms,
    gather_g4_output_to_torch,
    load_mlp_weights_torch,
    make_col_sharded_activation,
    mlp_g4_forward,
    print_decision_table,
    tile_pad_m,
    torch_swiglu_mlp,
    upload_mlp_weights,
)

_N_WARM = 20  # timed trace replays
_N_TRACE_WARMUP = 15  # untimed replays to settle AICLK / warm the trace before timing
_REP_LAYER = 3  # full-attention layer (representative MLP)
_PCC_THRESH = 0.99

pytestmark = [pytestmark_hardware]

_G4_RESULTS: dict = {}


def _signpost(msg: str) -> None:
    try:
        from tracy import signpost

        signpost(msg)
    except ImportError:
        pass


@pytest.fixture(scope="module")
def g4_mesh():
    if not DEFAULT_SNAPSHOT.exists():
        pytest.skip(f"HF snapshot not found: {DEFAULT_SNAPSHOT}")
    # Isolate exactly 4 chips via TT_VISIBLE_DEVICES=0,1,2,3 and open a 1x4 mesh
    # directly (one G=4 pipeline stage). FABRIC_1D (linear) matches BH GLX.
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(*G4_MESH_SHAPE),
        trace_region_size=200000000,
        worker_l1_size=1345000,
    )
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module", autouse=True)
def _print_decision_after_session():
    yield
    if _G4_RESULTS:
        print_decision_table(_G4_RESULTS)


@pytest.mark.hardware
@pytest.mark.parametrize(
    "wdtype",
    [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b],
    ids=["bf16", "bf8", "bf4"],
)
@pytest.mark.parametrize("batch", [1, 8, 16, 32, 64, 128], ids=["B1", "B8", "B16", "B32", "B64", "B128"])
def test_g4_mlp_latency_pcc(g4_mesh, wdtype, batch):
    """Single-layer G=4 MLP: latency + PCC at each (precision, batch) cell."""
    mesh = g4_mesh
    w1_t, w3_t, w2_t = load_mlp_weights_torch(DEFAULT_SNAPSHOT, _REP_LAYER)
    w1, w3, w2 = upload_mlp_weights(mesh, w1_t, w3_t, w2_t, wdtype)

    x_full, x = make_col_sharded_activation(mesh, batch_rows=batch, seed=7 + batch)
    m_padded = tile_pad_m(batch)

    # Compile pass to warm the program cache (eager), then capture a trace and
    # time pure execute_trace replays — this removes host op-dispatch overhead so
    # the latency reflects the device critical path (the real GO/NO-GO signal).
    out = mlp_g4_forward(x, w1, w3, w2)
    ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)

    tag = f"g4_mlp_{WDTYPE_NAMES[wdtype]}_B{batch}"
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    out_traced = mlp_g4_forward(x, w1, w3, w2)
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh)

    # Warm-up replays (untimed): settle AICLK and warm the trace before timing.
    for _ in range(_N_TRACE_WARMUP):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)

    _signpost(f"{tag}_warm_start")
    t0 = time.perf_counter()
    for _ in range(_N_WARM):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t1 = time.perf_counter()
    _signpost(f"{tag}_warm_done")

    ttnn.release_trace(mesh, trace_id)
    ttnn.deallocate(out_traced)

    mlp_ms = (t1 - t0) / _N_WARM * 1000.0
    floor = analytic_mlp_floor_ms(wdtype)

    out = mlp_g4_forward(x, w1, w3, w2)
    tt_out = gather_g4_output_to_torch(mesh, out)
    ttnn.deallocate(out)

    x_ref = x_full[:, :, :batch, :].float()
    ref = torch_swiglu_mlp(x_ref, w1_t.float(), w3_t.float(), w2_t.float())
    tt_cmp = tt_out[:, :, :batch, :].float()
    passing, pcc_msg = comp_pcc(ref, tt_cmp, _PCC_THRESH)

    _G4_RESULTS[(wdtype, batch)] = {"mlp_ms": mlp_ms, "pcc": float(pcc_msg)}

    print(
        f"\n[G4-MLP] wdtype={WDTYPE_NAMES[wdtype]} batch={batch} "
        f"mlp_ms={mlp_ms:.3f} x64={mlp_ms * _N_LAYERS:.1f}ms "
        f"floor_mlp={floor.mlp_ms:.2f}ms floor_full={floor.full_model_ms:.2f}ms "
        f"pcc={pcc_msg:.4f} pass={passing} M_padded={m_padded}"
    )

    # Assessment microbench: record PCC across precisions rather than hard-fail on
    # quantization. The decision table applies the 0.99 viability gate. Only fail
    # if the op produced garbage (a real correctness/layout regression).
    assert float(pcc_msg) > 0.95, f"PCC {pcc_msg} collapsed for {WDTYPE_NAMES[wdtype]} B={batch} (layout/op bug)"


@pytest.mark.hardware
@pytest.mark.parametrize(
    "wdtype",
    [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b],
    ids=["bf16", "bf8", "bf4"],
)
def test_g4_mlp_64layer_compounding_pcc(g4_mesh, wdtype):
    """64-layer MLP chain on G=4 with a RESIDUAL STREAM (x = x + mlp(x)).

    A bare chain (x = mlp(x)) collapses the signal toward zero over 64 layers,
    making PCC trivially 1.0 regardless of precision. The real decoder keeps a
    residual stream, so we mirror that: it keeps the activation magnitude alive
    and makes the compounding PCC a meaningful quantization stress test.
    """
    mesh = g4_mesh
    x_full, _ = make_col_sharded_activation(mesh, batch_rows=1, seed=99)

    # Preload all 64 layers' weights once (torch ref + device upload reuse).
    weights = [load_mlp_weights_torch(DEFAULT_SNAPSHOT, layer) for layer in range(_N_LAYERS)]

    ref = x_full[:, :, :1, :].float()
    for w1_t, w3_t, w2_t in weights:
        ref = ref + torch_swiglu_mlp(ref, w1_t.float(), w3_t.float(), w2_t.float())

    _, tt_x = make_col_sharded_activation(mesh, batch_rows=1, seed=99)
    for w1_t, w3_t, w2_t in weights:
        w1, w3, w2 = upload_mlp_weights(mesh, w1_t, w3_t, w2_t, wdtype)
        mlp_out = mlp_g4_forward(tt_x, w1, w3, w2)
        nxt = ttnn.add(tt_x, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(tt_x)
        tt_x = nxt

    tt_out = gather_g4_output_to_torch(mesh, tt_x)[:, :, :1, :].float()
    ttnn.deallocate(tt_x)

    passing, pcc_msg = comp_pcc(ref, tt_out, _PCC_THRESH)
    print(f"\n[G4-MLP-64L] wdtype={WDTYPE_NAMES[wdtype]} residual_compounded_pcc={pcc_msg:.4f}")
    assert passing, f"64L compounded PCC {pcc_msg} < {_PCC_THRESH} for {WDTYPE_NAMES[wdtype]}"


@pytest.mark.hardware
def test_g4_mlp_all64_latency_projection(g4_mesh):
    """All 64 layers' real weights chained in ONE trace: true per-token MLP latency.

    Weights for all 64 layers are uploaded to DRAM BEFORE timing (no disk I/O in
    the measured path). The full 64-layer chain is captured in a single trace and
    replayed; one replay = the MLP contribution to one decode token. Projected
    full-model latency divides by the MLP param fraction.
    """
    mesh = g4_mesh
    wdtype = ttnn.bfloat8_b
    _, x = make_col_sharded_activation(mesh, batch_rows=1, seed=123)

    # Preload + upload all 64 layers' weights to DRAM (out of the timed path).
    layer_weights = []
    for layer in range(_N_LAYERS):
        w1_t, w3_t, w2_t = load_mlp_weights_torch(DEFAULT_SNAPSHOT, layer)
        layer_weights.append(upload_mlp_weights(mesh, w1_t, w3_t, w2_t, wdtype))

    def _chain():
        tt_x = x
        for i, (w1, w3, w2) in enumerate(layer_weights):
            mlp_out = mlp_g4_forward(tt_x, w1, w3, w2)
            nxt = ttnn.add(tt_x, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(mlp_out)
            if i > 0:
                ttnn.deallocate(tt_x)
            tt_x = nxt
        return tt_x

    # Compile pass (warm program cache) then capture the full chain in one trace.
    out = _chain()
    ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)

    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    out_traced = _chain()
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh)

    # Warm-up replays (untimed) to settle AICLK before timing.
    for _ in range(_N_TRACE_WARMUP):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)

    _signpost("g4_mlp_all64_warm_start")
    n_rep = 30
    t0 = time.perf_counter()
    for _ in range(n_rep):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t1 = time.perf_counter()
    _signpost("g4_mlp_all64_warm_done")
    ttnn.release_trace(mesh, trace_id)
    ttnn.deallocate(out_traced)

    total_ms = (t1 - t0) / n_rep * 1000.0
    per_layer_ms = total_ms / _N_LAYERS
    tok_s = 1000.0 / total_ms
    floor = analytic_mlp_floor_ms(wdtype)
    full_proj_ms = total_ms / _MLP_PARAM_FRACTION

    print(
        f"\n[G4-MLP-ALL64] bf8 TRACED total_mlp_ms={total_ms:.3f} per_layer={per_layer_ms:.4f} "
        f"tok_s_mlp_only={tok_s:.1f} floor_mlp_x64={floor.mlp_ms * _N_LAYERS:.2f}ms"
    )
    print(f"[G4-MLP-ALL64] projected full-model latency={full_proj_ms:.2f}ms ({1000.0/full_proj_ms:.1f} tok/s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

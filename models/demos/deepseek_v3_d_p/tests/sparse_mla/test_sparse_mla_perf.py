# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse MLA (DSA) chunked-prefill perf: host-observed time, untraced and traced. No device-side profiling:
sparse MLA overlaps the indexer top-k and the KV gather on separate sub-devices, which per-program device times
don't capture.

Scenarios (Galaxy sizes; LoudBox and QuietBox scale the chunk and cache by SP/8 so each chip does Galaxy's work):
  * warm: one chunk over a 50k cache (zero-init caches; contents don't change op shapes or timing).
  * cold: every chunk from an empty cache up to the warm step, with real forwards that fill the caches.
  * long: one chunk over a 500k cache.

Boxes and gate cases: perf_cases.py.

test_sparse_mla_perf: the CI gate. Warm and long with the production scaled-FP8 KVPE cache (SPxTP-sharded), for
GATE_CASES, each at its model's fabric packet size: K3.1 (last DSA layer's slot of 24), GLM-5.3 (last full indexer
layer) and a GLM-5.3 shared layer (no indexer; reuses that layer's top-k). Untraced: fastest of 10 after 10 warm-up
(tests/measure). Traced: one timed replay of the captured trace. Each (case, box, scenario, mode) is gated two-sided
against PERF_TARGETS_MS ± MARGIN[mode] (a case without targets only reports), and the traced output must match the
untraced one (one extra untimed run of each, for triage). GLM-5.3 layers run inside a stage, as in the model, with the
sparse KV push plan and the next layer's prefix push where the fabric supports the push; one layer is timed.

test_mla_chunked_perf: the exploration matrix (QuietBox, LoudBox, Galaxy; skipped on CI), reports only.
GLM-5.3 (last full indexer layer, as a stage) and K3.1 x warm/cold/long x sparse with a BF16 or
scaled-FP8 cache / dense ring-MLA baseline. Narrow with -k, e.g. -k "glm_5_3 and cold and kv_scaled_fp8".
"""

import os
import time

import pytest

import ttnn
from models.common.utility_functions import comp_pcc, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import model_fabric_payload
from models.demos.deepseek_v3_d_p.tests.measure.gate import MARGIN, band_failure, gate
from models.demos.deepseek_v3_d_p.tests.measure.timing import time_untraced
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep, time_traced
from models.demos.deepseek_v3_d_p.tests.sparse_mla.perf_cases import (
    BOX,
    GATE_CASES,
    LONG_CACHE_TOKENS,
    SCENARIO_CACHE_TOKENS,
    WARM_CACHE_TOKENS,
    build_sparse_perf_case,
    detect_box,
    gate_params,
)
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_perf_metrics
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from tt_py_test_utils_common.perf.cpu_clock import hold_full_cpu_clock

MODES = ("untraced", "traced")
# Host time of one chunk forward through device synchronization, in ms, per case: (warm untraced, warm traced, long
# untraced, long traced), run as the CI jobs run them (LOGURU_LEVEL=ERROR, TT_METAL_SHM_TRACKING_DISABLED=1, one pytest
# process per model). Traced is the mean of the runs; untraced is the fastest run. LoudBox, Galaxy and Galaxy 4x4: CI
# hosts; Galaxy and Galaxy 4x4 then scaled by the q-in-L1 change's measured speedup (3 interleaved runs each of main and
# the change on one Galaxy; ratios under 1% left unapplied), LoudBox by Galaxy's (same 160 q rows per chip, not
# measured). Then re-centred on CI runs after that change: each value is the mean over runs of that run's measurement
# (fastest untraced sample, traced replay). Galaxy 8x4 K3.1 warm and GLM-5.3: 2-3 runs; Galaxy 4x4 K3.1 warm and
# LoudBox K3.1: 1 run; deltas under 1% left as is. Galaxy 4x4 subtorus has no CI leg: measured on a subtorus-cabled
# Galaxy (3 runs), unscaled. Galaxy 8x4 GLM-5.3: re-measured with the stage cases on a CI host (1 run). LoudBox
# GLM-5.3 (both cases): first measured on CI LoudBoxes (6 processes on 3 hosts, 10 readings per value with in-process
# repeats): the mean of the readings, every reading within the band; one warm untraced reading (+18%, a third repeat in
# one process, traced replay normal) left out.
_TARGETS_MS = {
    "Galaxy": {
        "kimi_k3_1": (2.93, 2.63, 6.47, 6.18),
        "glm_5_3": (3.13, 2.70, 6.73, 6.32),
        "glm_5_3-no_indexer": (2.29, 1.97, 2.36, 2.05),
    },
    "Galaxy 4x4": {
        "kimi_k3_1": (3.21, 3.05, 7.21, 6.98),
        # GLM-5.3: re-centred on CI run 45551006453 (1 run); deltas under 1% left as is.
        "glm_5_3": (2.85, 2.60, 6.81, 6.45),
        "glm_5_3-no_indexer": (2.22, 2.23, 4.63, 4.50),
    },
    "Galaxy 4x4 subtorus": {
        "kimi_k3_1": (2.97, 2.80, 6.62, 6.37),
        "glm_5_3": (2.78, 2.52, 6.43, 6.15),
        "glm_5_3-no_indexer": (2.14, 1.98, 2.22, 2.08),
    },
    "LoudBox": {
        "kimi_k3_1": (2.65, 2.51, 3.87, 3.71),
        "glm_5_3": (2.31, 2.11, 3.47, 3.26),
        "glm_5_3-no_indexer": (1.87, 1.83, 2.59, 2.55),
    },
}
PERF_TARGETS_MS = {
    (case, box, scenario, mode): ms
    for box, rows in _TARGETS_MS.items()
    for case, row in rows.items()
    for (scenario, mode), ms in zip([(s, m) for s in SCENARIO_CACHE_TOKENS for m in MODES], row)
}
# Traced output must match the untraced output of the same forward.
TRACE_PCC = 0.9999
CI_RUN = os.environ.get("CI") == "true"


def _device_tensors(tensor):
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]


def _time_untraced(mesh_device, case, start, repeat=None):
    """Untraced samples (ms) and the last timed forward's per-device output; appends one more, untimed forward's
    output to `repeat`, if given."""
    samples, out = time_untraced(lambda: case.forward(start), lambda: ttnn.synchronize_device(mesh_device))
    reference = _device_tensors(out)
    ttnn.deallocate(out)
    if repeat is not None:
        out = case.forward(start)
        repeat.append(_device_tensors(out))
        ttnn.deallocate(out)
    return samples, reference


def _time_traced(mesh_device, case, start, repeat=None):
    """One timed replay (ms) and the trace's per-device output; replays once more, untimed, into `repeat`, if given."""
    with TracedStep(
        mesh_device,
        lambda: case.forward(start),
        attach=case.mla.set_trace_controller,
        detach=lambda: case.mla.set_trace_controller(None),
    ) as traced:
        traced.capture()
        samples = time_traced(traced)
        out = _device_tensors(traced.output)
        if repeat is not None:
            traced.replay()
            repeat.append(_device_tensors(traced.output))
        return samples, out


@skip_with_llk_assert("LLK assertions perturb kernel timing")
@skip_with_watcher("Watcher perturbs kernel timing")
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name",
    gate_params(),
    indirect=["mesh_device", "device_params", "variant"],
)
@pytest.mark.parametrize("scenario", list(SCENARIO_CACHE_TOKENS))
def test_sparse_mla_perf(request, mesh_device, device_params, variant, case_name, config_only, scenario):
    box = BOX.name
    shared_layer = GATE_CASES[case_name][1]
    label = f"{case_name} sparse MLA {box} {scenario}"
    if box.startswith("Galaxy"):
        require_high_power(f"{label} perf")
    chunk = BOX.chunk
    cache = BOX.cache_tokens(SCENARIO_CACHE_TOKENS[scenario])
    case = build_sparse_perf_case(
        mesh_device,
        variant,
        config_only,
        cache=cache,
        chunk=chunk,
        topology=per_axis_topology(device_params["fabric_config"]),
        shared_layer=shared_layer,
        sp_axis=BOX.sp_axis,
    )
    start = cache
    clock = hold_full_cpu_clock()
    request.addfinalizer(clock.restore)
    print(f"{label} host: {clock.status}")
    samples = {}
    eager_repeat, traced_repeat = [], []
    samples["untraced"], eager_out = _time_untraced(mesh_device, case, start, eager_repeat)
    samples["traced"], traced_out = _time_traced(mesh_device, case, start, traced_repeat)

    def min_pcc(xs, ys):
        return min(comp_pcc(a, b, TRACE_PCC)[1] for a, b in zip(xs, ys))

    trace_pcc = min_pcc(eager_out, traced_out)
    results = {}
    for mode in MODES:
        measured, target = min(samples[mode]), PERF_TARGETS_MS.get((case_name, box, scenario, mode))
        results[mode] = (samples[mode], measured, target, MARGIN[mode])
        emit_perf_metrics(
            f"{case_name.replace('-', '_')}_sparse_mla_{box.lower().replace(' ', '_')}_{scenario}_{mode}",
            f"{label} ({mode})",
            {
                "box": box,
                "mesh": f"{BOX.mesh[0]}x{BOX.mesh[1]}",
                "sp_axis": BOX.sp_axis,
                "scenario": scenario,
                "mode": mode,
                "chunk_tokens": chunk,
                "cache_tokens": cache,
                "fabric": device_params["fabric_config"].name,
                "fabric_payload_bytes": model_fabric_payload(variant.name),
                "kv_cache_format": "scaled_fp8",
                "weights": "random",
                "samples_ms": samples[mode],
                "measured_ms": measured,
                "target_ms": target,
                "margin_pct": MARGIN[mode] * 100.0,
                "within_band": target is not None and band_failure(mode, measured, target, MARGIN[mode]) is None,
            },
        )
    correctness = []
    if trace_pcc < TRACE_PCC:
        # Within-mode repeatability, for triage: if both pairs match, that suggests a consistent eager/trace
        # discrepancy; a pair that does not match means that mode failed this repeatability check.
        correctness.append(
            f"traced output differs from untraced: min per-device PCC {trace_pcc:.6f} (eager vs eager "
            f"{min_pcc(eager_repeat[0], eager_out):.6f}, replay vs replay {min_pcc(traced_repeat[0], traced_out):.6f})"
        )
    gate(label, results, correctness=correctness)


# ============================================================================
# Exploration matrix: every variant x scenario x cache/attention case, host time only, no gate.
# ============================================================================
# cold: forward every chunk start=0,chunk,...,cache from empty caches (the last is the warm step); the region
# spans all chunks, so it is the whole cold-start prefill cost.
COLD_CACHE_TOKENS = WARM_CACHE_TOKENS
MATRIX_SCENARIOS = {"warm": WARM_CACHE_TOKENS, "cold": COLD_CACHE_TOKENS, "long": LONG_CACHE_TOKENS}
MATRIX_VARIANTS = ("glm_5_3", "kimi_k3_1")
# (id, attn_mode, kv_cache_format, tp_shard_kv). Dense ring MLA keeps its tiled BFP8 cache and has no dedup.
MATRIX_CASES = (
    ("sparse-kv_bf16", "sparse", MlaKvCacheFormat.BF16_RM, True),
    ("sparse-kv_scaled_fp8", "sparse", MlaKvCacheFormat.SCALED_FP8, True),
    ("dense", "dense", MlaKvCacheFormat.BF16_RM, False),
)
_MATRIX_BOX = detect_box(detect_num_devices(), matrix=True)


def _matrix_case_runs(variant: str, scenario: str, case_id: str) -> bool:
    """Kimi-K3.1 runs its production cache (scaled FP8) everywhere; its comparison points only in long."""
    return variant != "kimi_k3_1" or case_id == "sparse-kv_scaled_fp8" or scenario == "long"


def _matrix_params():
    if _MATRIX_BOX is None:
        return [
            pytest.param(
                (8, 4),
                {},
                "glm_5_3",
                "warm",
                "sparse",
                MlaKvCacheFormat.SCALED_FP8,
                True,
                marks=pytest.mark.skip(reason="sparse MLA perf runs on Blackhole QuietBox, LoudBox and Galaxy only"),
            )
        ]
    params = []
    for variant in MATRIX_VARIANTS:
        dp = _MATRIX_BOX.device_params(
            fabric_payload_size=model_fabric_payload(variant),
            worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
            trace_region_size=16 * 1024 * 1024,
        )
        for scenario in MATRIX_SCENARIOS:
            for case_id, attn_mode, kv_cache_format, tp_shard_kv in MATRIX_CASES:
                if _matrix_case_runs(variant, scenario, case_id):
                    params.append(
                        pytest.param(
                            _MATRIX_BOX.mesh,
                            dp,
                            variant,
                            scenario,
                            attn_mode,
                            kv_cache_format,
                            tp_shard_kv,
                            id=f"{_MATRIX_BOX.id}-{variant}-{scenario}-{case_id}",
                        )
                    )
    return params


@skip_with_llk_assert("LLK assertions perturb kernel timing")
@skip_with_watcher("Watcher perturbs kernel timing")
@pytest.mark.skipif(CI_RUN, reason="exploration matrix; CI runs test_sparse_mla_perf")
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, scenario, attn_mode, kv_cache_format, tp_shard_kv",
    _matrix_params(),
    indirect=["mesh_device", "device_params", "variant"],
)
def test_mla_chunked_perf(
    mesh_device, device_params, variant, config_only, scenario, attn_mode, kv_cache_format, tp_shard_kv
):
    """Host time of one chunk (warm, long) or of the whole cold fill, untraced and traced. Reports only."""
    chunk = _MATRIX_BOX.chunk
    cache = _MATRIX_BOX.cache_tokens(MATRIX_SCENARIOS[scenario])
    case = build_sparse_perf_case(
        mesh_device,
        variant,
        config_only,
        cache=cache,
        chunk=chunk,
        topology=per_axis_topology(device_params["fabric_config"]),
        attn_mode=attn_mode,
        kv_cache_format=kv_cache_format,
        tp_shard_kv=tp_shard_kv,
        sp_axis=_MATRIX_BOX.sp_axis,
    )
    label = f"{_MATRIX_BOX.name} {variant.name} {scenario} {attn_mode}"
    if attn_mode == "sparse":
        label += f" {kv_cache_format.name} {'tp-sharded' if tp_shard_kv else 'tp-replicated'} KV"

    if scenario != "cold":
        untraced, _ = _time_untraced(mesh_device, case, cache)
        traced, _ = _time_traced(mesh_device, case, cache)
        print(f"\n{label}: untraced {min(untraced):.3f} ms (fastest of {len(untraced)}), traced {traced[0]:.3f} ms")
        return

    # cold: one untimed fill compiles every start; then one timed untraced fill, then each chunk traced.
    starts = list(range(0, cache + chunk, chunk))
    for start in starts:
        ttnn.deallocate(case.forward(start))
    ttnn.synchronize_device(mesh_device)
    untraced_rows = []
    for start in starts:
        t0 = time.perf_counter()
        out = case.forward(start)
        ttnn.synchronize_device(mesh_device)
        untraced_rows.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(out)
    traced_rows = [_time_traced(mesh_device, case, start)[0][0] for start in starts]
    rows = list(zip(starts, untraced_rows, traced_rows))
    print(
        f"\n{label}: {len(starts)} chunks, untraced {sum(r[1] for r in rows):.3f} ms, "
        f"traced {sum(r[2] for r in rows):.3f} ms"
    )
    for start, untraced_ms, traced_ms in rows:
        print(f"  cache depth {start:>7}: untraced {untraced_ms:.3f} ms, traced {traced_ms:.3f} ms")

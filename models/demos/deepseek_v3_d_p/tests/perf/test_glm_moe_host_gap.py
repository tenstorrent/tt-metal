# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single production GLM MoE layer: clean host gap and separate diagnostic passes.

Real cached layer-3 weights, GLM gate constants, full 8x4 TORUS_XY and 5120 tokens.
Input is seeded unit-RMS synthetic activations, not an extracted model activation.
A clone is included in both execution paths because MoE consumes its input.
Run each mode in its own process. Host mode times Python operation envelopes;
profile mode also collects device records. Both are diagnostic, not clean baselines.
Device record identities/counts are useful; completion intervals are unreliable with
this workload's overlapping subdevices and must not be used for critical-path accounting.

Run through scripts/run_safe_pytest.sh with SAFE_PYTEST_NO_RESET=1 and select
one mode with -k. Set GLM_MOE_RESULT to preserve each run separately and
GLM_MOE_SAMPLES to control timing sample count. TT_GLM52_PREFILL_TTNN_CACHE
selects the cache root containing glm_5_2_bh_32dev/8x4.
"""

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from ttnn.decorators import FastOperation

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config, glm_5_2_hf_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [torus_xy_device_params(fabric_payload_size=6144, l1_small_size=1216, trace_region_size=8000000)],
    indirect=True,
)
@pytest.mark.parametrize("mode", ["correctness", "ordinary", "trace", "host_ops", "profile"])
@pytest.mark.timeout(900)
def test_glm_moe_host_gap(mesh_device, device_params, mode, monkeypatch):
    assert tuple(mesh_device.shape) == (8, 4)
    cache = Path(os.environ.get("TT_GLM52_PREFILL_TTNN_CACHE", "/mnt/models/deepseek-prefill-cache/glm52_ttnn_cache"))
    cache = cache / "glm_5_2_bh_32dev" / "8x4"
    init_checker(cache)
    assert TtMoe.check_cache_complete(cache, layer_idx=3, experts_per_chip=8), f"Incomplete real cache: {cache}"
    moe = TtPrefillBlock._build_moe(
        mesh_device=mesh_device,
        model_cfg=GLM52Config,
        config=glm_5_2_hf_config(),
        state_dict={},
        seq_len=5120,
        sp_axis=0,
        emb_dim=6144,
        num_links=2,
        topology=per_axis_topology(device_params["fabric_config"]),
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        dispatch_buffer_capacity_factor=8,
        weight_cache_path=cache,
        layer_idx=3,
        routing_use_l1_small_for_semaphores=True,
        is_balanced=False,
        overlap_shared_expert_with_dispatch=True,
    )
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(8, 4), dims=(0, -1))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(8, 4), dims=(0, -1))

    def host_input(seed):
        x = torch.randn((8, 640, 6144), generator=torch.Generator().manual_seed(seed))
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True))).to(torch.bfloat16)

    input_host = host_input(17)
    source = ttnn.from_torch(
        input_host, device=mesh_device, mesh_mapper=mapper, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    def forward():
        return moe(ttnn.clone(source), return_intermediates=False, actual_isl=5120, actual_start=0)[0]

    def sync():
        ttnn.synchronize_device(mesh_device)

    controller = None
    captured = None
    samples = []
    output_path = Path(os.environ.get("GLM_MOE_RESULT", f"generated/profiler/glm52_moe/{mode}.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": mode,
        "layer": 3,
        "mesh": [8, 4],
        "tokens": 5120,
        "input": "seed17 unit-RMS synthetic",
        "weights": str(cache),
        "includes_input_clone": True,
        "measurement_kind": "clean_completion" if mode in ("ordinary", "trace") else mode,
        "critical_path_attribution": False,
    }
    try:
        mesh_device.enable_program_cache()
        for _ in range(20):
            output = forward()
            sync()
            ttnn.deallocate(output)
        if mode in ("correctness", "trace"):
            controller = SubDeviceTraceController(mesh_device)
            moe.set_trace_controller(controller)
            controller.begin_capture()
            captured = forward()
            controller.end_capture()
            sync()
            assert controller.num_segments > 0
            result.update(trace_segments=controller.num_segments, trace_bytes=controller.trace_bytes())
        if mode == "correctness":
            checks = []
            for seed in (17, 29):
                host = ttnn.from_torch(
                    host_input(seed), mesh_mapper=mapper, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                ttnn.copy_host_to_device_tensor(host, source)
                expected = forward()
                sync()
                ref = ttnn.to_torch(expected, mesh_composer=composer)
                controller.replay()
                actual = ttnn.to_torch(captured, mesh_composer=composer)
                assert torch.isfinite(actual).all() and actual.abs().max() > 0
                assert_with_pcc(ref, actual, 0.999)
                checks.append({"seed": seed, "pcc_threshold": 0.999, "passed": True})
                ttnn.deallocate(expected)
            result["checks"] = checks
        elif mode in ("ordinary", "trace"):
            for _ in range(20):
                if mode == "trace":
                    controller.replay()
                else:
                    output = forward()
                    sync()
                    ttnn.deallocate(output)
            for _ in range(int(os.environ.get("GLM_MOE_SAMPLES", "160"))):
                start = time.perf_counter_ns()
                if mode == "trace":
                    controller.replay()  # controller includes one final sync
                    end = time.perf_counter_ns()
                    submitted = None
                else:
                    output = forward()
                    submitted = time.perf_counter_ns()
                    sync()
                    end = time.perf_counter_ns()
                    ttnn.deallocate(output)  # final result release outside timing in both modes
                samples.append(
                    {
                        "completion_ms": (end - start) / 1e6,
                        "submission_ms": None if submitted is None else (submitted - start) / 1e6,
                    }
                )
            result.update(
                samples=samples,
                median_ms=statistics.median(s["completion_ms"] for s in samples),
                stddev_ms=statistics.stdev(s["completion_ms"] for s in samples),
                ordinary_warmup_forwards=20 if mode == "trace" else 40,
                trace_warmup_replays=20 if mode == "trace" else 0,
            )
        else:
            from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

            calls = []
            depth = [0]
            original = FastOperation.__call__

            def observed(operation, *args, **kwargs):
                outer = depth[0] == 0
                depth[0] += 1
                start = time.perf_counter_ns()
                try:
                    return original(operation, *args, **kwargs)
                finally:
                    end = time.perf_counter_ns()
                    depth[0] -= 1
                    if outer:
                        calls.append(
                            {
                                "name": operation.python_fully_qualified_name,
                                "start_ns": start,
                                "end_ns": end,
                                "host_us": (end - start) / 1e3,
                            }
                        )

            monkeypatch.setattr(FastOperation, "__call__", observed)
            diagnostics = []
            diagnostic_samples = int(os.environ.get("GLM_MOE_DIAGNOSTIC_SAMPLES", "10" if mode == "profile" else "100"))
            for iteration in range(diagnostic_samples):
                calls.clear()
                timing = {}

                def timed_forward():
                    begin = time.perf_counter_ns()
                    output = forward()
                    submitted = time.perf_counter_ns()
                    sync()
                    completed = time.perf_counter_ns()
                    timing.update(submission_ms=(submitted - begin) / 1e6, completion_ms=(completed - begin) / 1e6)
                    return output

                start = time.perf_counter_ns()
                if mode == "profile":
                    output, programs = profile_realtime_program(
                        mesh_device, timed_forward, collect_all=True, record_timeout_seconds=2
                    )
                else:
                    output, programs = timed_forward(), []
                diagnostics.append(
                    {
                        "iteration": iteration,
                        "calls": list(calls),
                        "programs": programs,
                        **timing,
                        "collector_envelope_ms": (time.perf_counter_ns() - start) / 1e6,
                    }
                )
                ttnn.deallocate(output)
            result.update(
                diagnostics=diagnostics,
                device_record_callback=mode == "profile",
                diagnostic_limitation="Host operation envelopes overlap and are not additive removable costs.",
            )
            if mode == "profile":
                result["device_record_semantics"] = (
                    "Dispatch-GO to observed final-worker completion; clock alignment and observer effects require "
                    "independent validation before cross-device attribution."
                )
        result["status"] = "PASS"
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"GLM_MOE_RESULT {output_path}: {result.get('median_ms', mode)}")
    finally:
        if controller is not None:
            controller.release()
            moe.set_trace_controller(None)
        if captured is not None:
            ttnn.deallocate(captured)
        ttnn.deallocate(source)

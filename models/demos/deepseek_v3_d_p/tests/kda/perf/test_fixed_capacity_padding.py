# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed 5120-token PR baseline and aligned early-exit padding sweep.

KDA_FIXED_VARIANT=pr uses no padding; early uses KDA_FIXED_PADDING (CSV).
Both variants always execute on physical 5120-token tensors.
"""

import json
import os
import statistics
import time

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [fabric_1d_device_params()], indirect=True)
@pytest.mark.parametrize("padding", [int(value) for value in os.environ.get("KDA_FIXED_PADDING", "0").split(",")])
@pytest.mark.parametrize("stage", ["recurrence", "layer"])
def test_fixed_capacity_padding(mesh_device, device_params, padding: int, stage: str) -> None:
    variant = os.environ["KDA_FIXED_VARIANT"]
    assert variant in ("pr", "early")
    assert 0 <= padding < 5120 and padding % 32 == 0
    assert variant != "pr" or padding == 0
    length = 5120 - padding
    profile = os.environ.get("KDA_FIXED_PROFILE") == "1"
    case = make_synthetic_kimi_k3_test_case(sequence=5120)
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    initial = layer.allocate_state()
    start, end = make_actual_start(mesh_device, 0), make_actual_start(mesh_device, length)

    if stage == "layer":

        def run():
            output, state = (
                layer.forward(hidden, initial, start, end)
                if variant == "early"
                else layer.forward(hidden, initial, start)
            )
            return output, state.recurrent, state.convolution

    else:
        projected = layer._project_inputs(hidden)
        qkv = ttnn.to_layout(projected.qkv, ttnn.ROW_MAJOR_LAYOUT)
        q, k, v, convolution = layer._convolve_qkv(qkv, initial.convolution, None, start)
        gate, beta = layer._compute_gates(beta=projected.beta, decay_rank=projected.decay_rank)
        inputs = dict(q=q, k=k, v=v, gate=gate, beta=beta)
        inputs["initial_state"] = initial.recurrent
        inputs["actual_start"] = start
        if variant == "early":
            inputs["actual_end"] = end

        def run():
            result = layer.recurrence(**inputs)
            return result.output, result.final_state

    for _ in range(2):
        outputs = run()
        ttnn.synchronize_device(mesh_device)
        for tensor in outputs:
            ttnn.deallocate(tensor)

    if profile:
        from tracy import signpost

        for repeat in range(3):
            tag = f"KDA_FIXED_{variant}_{stage}_{padding}_{repeat}"
            signpost(tag + "_START")
            outputs = run()
            ttnn.synchronize_device(mesh_device)
            signpost(tag + "_END")
            for tensor in outputs:
                ttnn.deallocate(tensor)
    else:
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outputs = run()
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        samples = []
        try:
            for _ in range(40):
                for _ in range(10):
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
            for _ in range(16):
                begin = time.perf_counter()
                for _ in range(16):
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                samples.append((time.perf_counter() - begin) * 1000 / 16)
            print(
                "KDA_FIXED="
                + json.dumps(
                    dict(
                        variant=variant,
                        stage=stage,
                        valid_tokens=length,
                        padding_tokens=padding,
                        physical_tokens=5120,
                        median_ms=statistics.median(samples),
                        wall_ms_samples=samples,
                    )
                )
            )
        finally:
            ttnn.release_trace(mesh_device, trace)
            for tensor in outputs:
                ttnn.deallocate(tensor)

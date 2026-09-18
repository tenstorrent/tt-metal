# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Portable PR/prototype timing and device-profile experiment (same inputs)."""

import json
import os
import statistics
import time
from dataclasses import replace

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [fabric_1d_device_params()], indirect=True)
@pytest.mark.parametrize("length", [4096, 4896, 5120])
@pytest.mark.parametrize("stage", ["recurrence", "layer"])
def test_padding_cost_attribution(mesh_device, device_params, length, stage):
    variant = os.environ["KDA_COST_VARIANT"]
    assert variant in ("pr", "early", "control")
    profile = os.environ.get("KDA_COST_PROFILE") == "1"
    case = make_synthetic_kimi_k3_test_case(sequence=5120)
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    execution_layer = layer
    if variant != "early":
        full_config = kimi_k3_program_config(active_seq_len_local=5120, tp_ccl_topology=ttnn.Topology.Ring)
        trimmed_config = replace(
            full_config,
            recurrence=replace(full_config.recurrence, summary_group_chunks={4096: 16, 4896: 17, 5120: 20}[length]),
        )
        execution_layer, _ = make_kimi_k3_device_case(
            mesh_device,
            replace(case, hidden=case.hidden[:, :length]),
            tensor_parallel_axis=1,
            cache_weights=False,
            weights=layer.weights,
            program_config=trimmed_config,
        )
    initial = layer.allocate_state()
    start, end = make_actual_start(mesh_device, 0), make_actual_start(mesh_device, length)

    def crop(tensor):
        return ttnn.slice(tensor, (0, 0, 0), (tensor.shape[0], length, tensor.shape[2]))

    if stage == "layer":
        selected = hidden if variant == "early" else crop(hidden)

        def run():
            output, state = (
                layer.forward(selected, initial, start, end)
                if variant == "early"
                else execution_layer.forward(selected, initial, start)
            )
            return output, state.recurrent, state.convolution

    else:
        projected = layer._project_inputs(hidden)
        qkv = ttnn.to_layout(projected.qkv, ttnn.ROW_MAJOR_LAYOUT)
        q, k, v, convolution = layer._convolve_qkv(qkv, initial.convolution, None, start)
        gate, beta = layer._compute_gates(beta=projected.beta, decay_rank=projected.decay_rank)
        inputs = dict(q=q, k=k, v=v, gate=gate, beta=beta)
        if variant != "early":
            inputs = {name: crop(tensor) for name, tensor in inputs.items()}
        inputs["initial_state"] = initial.recurrent
        inputs["actual_start"] = start
        if variant == "early":
            inputs["actual_end"] = end

        def run():
            result = execution_layer.recurrence(**inputs)
            return result.output, result.final_state

    for _ in range(2):
        outputs = run()
        ttnn.synchronize_device(mesh_device)
        for tensor in outputs:
            ttnn.deallocate(tensor)

    if profile:
        from tracy import signpost

        for repeat in range(3):
            tag = f"KDA_COST_{variant}_{stage}_{length}_{repeat}"
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
                "KDA_COST="
                + json.dumps(
                    dict(
                        variant=variant,
                        stage=stage,
                        length=length,
                        physical_tokens=5120 if variant == "early" else length,
                        median_ms=statistics.median(samples),
                        wall_ms_samples=samples,
                    )
                )
            )
        finally:
            ttnn.release_trace(mesh_device, trace)
            for tensor in outputs:
                ttnn.deallocate(tensor)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rotating paired trace timing against physical and trimmed SP1 production work."""

import json
import statistics
import time
from dataclasses import replace

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.perf, pytest.mark.timeout(1800)]


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [fabric_1d_device_params()], indirect=True)
@pytest.mark.parametrize("length", [4096, 4896])
@pytest.mark.parametrize("stage", ["recurrence", "layer"])
def test_padding_early_exit_cost(mesh_device, device_params, length, stage):
    sequence = 5120
    case = make_synthetic_kimi_k3_test_case(sequence=sequence)
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    full_config = kimi_k3_program_config(active_seq_len_local=5120, tp_ccl_topology=ttnn.Topology.Ring)
    trimmed_config = replace(
        full_config,
        recurrence=replace(full_config.recurrence, summary_group_chunks={4096: 16, 4896: 17, 5120: 20}[length]),
    )
    trimmed_layer, trimmed = make_kimi_k3_device_case(
        mesh_device,
        replace(case, hidden=case.hidden[:, :length]),
        tensor_parallel_axis=1,
        cache_weights=False,
        weights=layer.weights,
        program_config=trimmed_config,
    )
    initial = layer.allocate_state()
    start = make_actual_start(mesh_device, 0)
    end = make_actual_start(mesh_device, sequence)

    def crop(tensor):
        return ttnn.slice(tensor, (0, 0, 0), (tensor.shape[0], length, tensor.shape[2]))

    def flatten(result):
        if stage == "layer":
            output, state = result
            return (output, state.recurrent, state.convolution)
        return result.output, result.final_state

    if stage == "layer":
        runs = {
            "physical": lambda: layer.forward(hidden, initial, start),
            "early_exit": lambda: layer.forward(hidden, initial, start, end),
            "trimmed": lambda: trimmed_layer.forward(trimmed, initial, start),
        }
    else:
        projected = layer._project_inputs(hidden)
        qkv = ttnn.to_layout(projected.qkv, ttnn.ROW_MAJOR_LAYOUT)
        q, k, v, convolution = layer._convolve_qkv(qkv, initial.convolution, None, start)
        gate, beta = layer._compute_gates(beta=projected.beta, decay_rank=projected.decay_rank)
        full_inputs = dict(q=q, k=k, v=v, gate=gate, beta=beta, initial_state=initial.recurrent)
        trimmed_inputs = {name: crop(tensor) for name, tensor in full_inputs.items() if name != "initial_state"}
        trimmed_inputs["initial_state"] = initial.recurrent
        runs = {
            "physical": lambda: layer.recurrence(**full_inputs, actual_start=start),
            "early_exit": lambda: layer.recurrence(**full_inputs, actual_start=start, actual_end=end),
            "trimmed": lambda: trimmed_layer.recurrence(**trimmed_inputs, actual_start=start),
        }

    traces, retained = {}, []
    try:
        # Initialize every program/resource before the first capture: later
        # eager allocations must not alias an earlier trace's intermediates.
        for name, run in runs.items():
            print(f"KDA_WARMUP stage={stage} length={length} path={name}", flush=True)
            for _ in range(2):
                warm = flatten(run())
                ttnn.synchronize_device(mesh_device)
                for tensor in warm:
                    ttnn.deallocate(tensor)
        for name, run in runs.items():
            print(f"KDA_CAPTURE stage={stage} length={length} path={name}", flush=True)
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            retained.extend(flatten(run()))
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            traces[name] = trace
        scenarios = [
            ("physical", "physical", sequence),
            ("early_full", "early_exit", sequence),
            ("early_tail", "early_exit", length),
            ("trimmed", "trimmed", length),
        ]
        samples = {name: [] for name, _, _ in scenarios}
        for round_index in range(-8, 8):
            rotation = round_index % len(scenarios)
            for name, trace_name, valid in scenarios[rotation:] + scenarios[:rotation]:
                print(f"KDA_SAMPLE stage={stage} length={length} round={round_index} path={name}", flush=True)
                source = make_actual_start(mesh_device, valid)
                ttnn.copy(source, end)
                ttnn.deallocate(source)
                ttnn.execute_trace(mesh_device, traces[trace_name], cq_id=0, blocking=True)
                begin = time.perf_counter()
                for _ in range(10):
                    ttnn.execute_trace(mesh_device, traces[trace_name], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                elapsed_ms = (time.perf_counter() - begin) * 1000 / 10
                if round_index >= 0:
                    samples[name].append(elapsed_ms)
        print(
            "KDA_EARLY_EXIT="
            + json.dumps(
                {
                    "stage": stage,
                    "physical_tokens": sequence,
                    "valid_tokens": length,
                    "wall_ms_samples": samples,
                    "median_ms": {name: statistics.median(values) for name, values in samples.items()},
                    "paired_tail_vs_physical_pct": [
                        100 * (a / b - 1) for a, b in zip(samples["early_tail"], samples["physical"], strict=True)
                    ],
                    "paired_tail_vs_trimmed_pct": [
                        100 * (a / b - 1) for a, b in zip(samples["early_tail"], samples["trimmed"], strict=True)
                    ],
                },
                sort_keys=True,
            )
        )
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh_device, trace)
        for tensor in retained:
            ttnn.deallocate(tensor)

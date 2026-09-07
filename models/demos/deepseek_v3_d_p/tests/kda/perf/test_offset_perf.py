# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offset-handling cost at production K3 dimensions.

Reports warm trace wall time per offset against the S=0 baseline so the two
offset prototypes can be compared on identical measurements.

Timing is interleaved, not sequential. Measuring offsets one after another gave
a 13.5% gap between two runs of the identical baseline, which is larger than
several of the effects being measured. Every offset's trace is captured up
front and the timing samples then round-robin across offsets, so drift over the
run lands on all offsets equally instead of on whichever was measured last.

Offsets change the program, not the arithmetic, so one synthetic input serves
the whole sweep.
"""

from __future__ import annotations

import json
import statistics
import time

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import (
    _REPETITIONS,
    _SEQUENCE,
    _TIMING_SAMPLES,
    _allocate_state,
    _deallocate_state,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case

pytestmark = [pytest.mark.timeout(3600)]


def _offset_sweep(local_rows: int) -> dict[str, int]:
    """Baseline, a device boundary (rotation only), and the three split extremes."""
    return {
        "baseline": 0,
        "device_boundary": local_rows,
        "smallest_split": 32,
        "worst_case_split": local_rows // 2,
        "largest_split": local_rows - 32,
    }


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [pytest.param((2, 4), 1, fabric_1d_device_params(), id="SP2xTP4-fabric-1d")],
    indirect=["mesh_device", "device_params"],
)
def test_offset_handling_cost(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    device_params: dict,
) -> None:
    """Measure interleaved warm trace wall time at each offset and report JSON."""
    mesh_shape = tuple(mesh_device.shape)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    sp_size = mesh_shape[sequence_parallel_axis]
    local_rows = _SEQUENCE // sp_size
    layout = f"SP{sp_size}xTP{mesh_shape[tensor_parallel_axis]}"

    case = make_synthetic_kimi_k3_test_case(sequence=_SEQUENCE)
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
    )
    sweep = _offset_sweep(local_rows)
    captured: dict[str, int] = {}
    held: list = []
    try:
        for name, actual_start in sweep.items():
            state = _allocate_state(layer)
            held.append(state)
            warm_output, warm_state = layer.forward(hidden_tt, state, actual_start)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(warm_output)
            _deallocate_state(warm_state)

            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            output, next_state = layer.forward(hidden_tt, state, actual_start)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            held.extend([output, next_state.recurrent, next_state.convolution])
            captured[name] = trace_id

        samples: dict[str, list[float]] = {name: [] for name in sweep}
        for _ in range(_TIMING_SAMPLES):
            for name, trace_id in captured.items():
                start = time.perf_counter()
                for _ in range(_REPETITIONS):
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                samples[name].append((time.perf_counter() - start) * 1e3 / _REPETITIONS)
    finally:
        for trace_id in captured.values():
            ttnn.release_trace(mesh_device, trace_id)
        for tensor in held:
            if isinstance(tensor, ttnn.Tensor):
                ttnn.deallocate(tensor)
            else:
                _deallocate_state(tensor)

    baseline_ms = statistics.median(samples["baseline"])
    measurements = {
        name: {
            "actual_start": sweep[name],
            "median_trace_wall_ms": statistics.median(values),
            "spread_pct": 100.0 * (max(values) - min(values)) / statistics.median(values),
            "overhead_pct": 100.0 * (statistics.median(values) - baseline_ms) / baseline_ms,
            "trace_wall_samples_ms": values,
        }
        for name, values in samples.items()
    }
    print(
        "KDA_OFFSET_PERF="
        + json.dumps(
            {
                "layout": layout,
                "sequence": _SEQUENCE,
                "local_rows": local_rows,
                "repetitions": _REPETITIONS,
                "timing_sample_count": _TIMING_SAMPLES,
                "interleaved": True,
                "program_cache_entries": mesh_device.num_program_cache_entries(),
                "measurements": measurements,
            },
            sort_keys=True,
        )
    )

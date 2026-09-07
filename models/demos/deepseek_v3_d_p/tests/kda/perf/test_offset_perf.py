# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offset-handling cost at production K3 dimensions.

Reports warm trace wall time per offset against the S=0 baseline, so the two
offset prototypes can be compared on identical measurements. Values do not
affect timing -- only the offset changes the program -- so one synthetic input
serves every offset.
"""

from __future__ import annotations

import json
import statistics

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import (
    _REPETITIONS,
    _SEQUENCE,
    _TIMING_SAMPLES,
    _trace_wall_samples_ms,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case

pytestmark = [pytest.mark.timeout(1800)]


def _offset_sweep(local_rows: int) -> dict[str, int]:
    """Baseline, both split extremes, the worst case, and a device boundary."""
    return {
        "baseline": 0,
        "smallest_split": 32,
        "worst_case_split": local_rows // 2,
        "largest_split": local_rows - 32,
        "device_boundary": local_rows,
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
    """Measure warm trace wall time at each offset and report it as JSON."""
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

    measurements = {}
    for name, actual_start in _offset_sweep(local_rows).items():
        samples_ms, _ = _trace_wall_samples_ms(mesh_device, layer, hidden_tt, _REPETITIONS, actual_start=actual_start)
        measurements[name] = {
            "actual_start": actual_start,
            "median_trace_wall_ms": statistics.median(samples_ms),
            "trace_wall_samples_ms": samples_ms,
        }

    baseline_ms = measurements["baseline"]["median_trace_wall_ms"]
    for name, entry in measurements.items():
        entry["overhead_pct"] = 100.0 * (entry["median_trace_wall_ms"] - baseline_ms) / baseline_ms

    print(
        "KDA_OFFSET_PERF="
        + json.dumps(
            {
                "layout": layout,
                "sequence": _SEQUENCE,
                "local_rows": local_rows,
                "repetitions": _REPETITIONS,
                "timing_sample_count": _TIMING_SAMPLES,
                "program_cache_entries": mesh_device.num_program_cache_entries(),
                "measurements": measurements,
            },
            sort_keys=True,
        )
    )

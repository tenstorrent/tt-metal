# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offset-handling cost at production K3 dimensions.

Reports paired warm trace wall time per offset against the S=0 baseline.

Timing is interleaved, not sequential. Measuring offsets one after another gave
a 13.5% gap between two runs of the identical baseline, which is larger than
several of the effects being measured. One trace is resident at a time and a
fresh trace is captured for each round-robin sample, so drift lands on all
offsets equally without exhausting production-shape L1 trace storage.

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
    _log_device_program_times,
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
    "sequence",
    [
        # 2560 rows per chip: LoudBox retains four groups, including across a split.
        pytest.param(_SEQUENCE, id="T5120-4groups"),
        # 640 rows per chip: one group natively, i.e. Galaxy SP8's per-chip geometry.
        pytest.param(_SEQUENCE // 4, id="T1280-1group"),
    ],
)
@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [pytest.param((2, 4), 1, fabric_1d_device_params(), id="SP2xTP4-fabric-1d")],
    indirect=["mesh_device", "device_params"],
)
def test_offset_handling_cost(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    device_params: dict,
    sequence: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measure interleaved warm trace wall time at each offset and report JSON."""
    mesh_shape = tuple(mesh_device.shape)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    sp_size = mesh_shape[sequence_parallel_axis]
    local_rows = sequence // sp_size
    layout = f"SP{sp_size}xTP{mesh_shape[tensor_parallel_axis]}"

    case = make_synthetic_kimi_k3_test_case(sequence=sequence)
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
    )
    sweep = _offset_sweep(local_rows)
    samples: dict[str, list[float]] = {name: [] for name in sweep}
    sweep_items = list(sweep.items())
    for sample_index in range(_TIMING_SAMPLES):
        # Rotate the round-robin order so monotonic clock/thermal drift does not
        # consistently assign the same within-round position to one offset.
        ordered_items = sweep_items[sample_index:] + sweep_items[:sample_index]
        for name, actual_start in ordered_items:
            state = _allocate_state(layer)
            # Warm the exact live buffers that trace capture will bind.  The
            # first forward also reserves persistent CCL semaphores after the
            # QKV untilize has selected its worker set; a second forward caches
            # the stable post-reservation untilize signature used by the trace.
            for _ in range(2):
                warm_output, warm_state = layer.forward(hidden_tt, state, actual_start)
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(warm_output)
                _deallocate_state(warm_state)
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            output, next_state = layer.forward(hidden_tt, state, actual_start)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            try:
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                start = time.perf_counter()
                for _ in range(_REPETITIONS):
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                samples[name].append((time.perf_counter() - start) * 1e3 / _REPETITIONS)
            finally:
                ttnn.release_trace(mesh_device, trace_id)
                ttnn.deallocate(output)
                _deallocate_state(next_state)
                _deallocate_state(state)

    baseline_ms = statistics.median(samples["baseline"])
    measurements = {
        name: {
            "actual_start": sweep[name],
            "median_trace_wall_ms": statistics.median(values),
            "spread_pct": 100.0 * (max(values) - min(values)) / statistics.median(values),
            "overhead_pct": 100.0 * (statistics.median(values) - baseline_ms) / baseline_ms,
            "paired_overhead_samples_pct": [
                100.0 * (value - baseline) / baseline
                for value, baseline in zip(values, samples["baseline"], strict=True)
            ],
            "median_paired_overhead_pct": statistics.median(
                100.0 * (value - baseline) / baseline
                for value, baseline in zip(values, samples["baseline"], strict=True)
            ),
            "trace_wall_samples_ms": values,
        }
        for name, values in samples.items()
    }
    print(
        "KDA_OFFSET_PERF="
        + json.dumps(
            {
                "layout": layout,
                "sequence": sequence,
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
    for name in ("baseline", "worst_case_split"):
        dispatched_groups = {}
        with monkeypatch.context() as patch:
            for operation in ("summarize_chunk_recurrence", "affine_exclusive_scan", "recurrent_chunk_scan"):
                original = getattr(ttnn.experimental.kda, operation)

                def record(*args, _operation=operation, _original=original, **kwargs):
                    groups = args[3] if _operation == "affine_exclusive_scan" else kwargs["groups_per_head"]
                    dispatched_groups.setdefault(_operation, []).append(groups)
                    return _original(*args, **kwargs)

                patch.setattr(ttnn.experimental.kda, operation, record)
            programs = _log_device_program_times(
                mesh_device,
                layer,
                hidden_tt,
                f"{layout}-C{local_rows}-{name}",
                actual_start=sweep[name],
            )
        expected_groups = local_rows // (20 * 32)
        assert set(dispatched_groups) == {"summarize_chunk_recurrence", "affine_exclusive_scan", "recurrent_chunk_scan"}
        assert all(
            values and all(groups == expected_groups for groups in values) for values in dispatched_groups.values()
        ), dispatched_groups
        print(
            "KDA_OFFSET_DISPATCHED_GROUPS="
            + json.dumps(
                {
                    "offset": name,
                    "expected_groups": expected_groups,
                    "calls": dispatched_groups,
                },
                sort_keys=True,
            )
        )
        counts = {
            operation: sum(program["name"] == operation for program in programs)
            for operation in (
                "experimental.kda.summarize_chunk_recurrence",
                "experimental.kda.affine_exclusive_scan",
                "experimental.kda.recurrent_chunk_scan",
            )
        }
        assert all(count == 1 for count in counts.values()), f"unexpected recurrence program topology: {counts}"

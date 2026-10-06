# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer trace-replay time of the GDN recurrence group schedule at the LoudBox layouts (tt_metal_tracker-g1b.5.15).

Measurement, not a gate. At 640 rows per SP rank the production config uses two groups of 10 chunks (G2,
``summary_group_chunks=10``); this compares it with one group of 20 (G1). One process builds the layer twice from
the same prepared synthetic weights, differing only in ``summary_group_chunks``, warms both twice, captures one
trace each, and times them in interleaved ABBA blocks of 10 replays, so both arms see the same operating state.
No CPU reference: the two arms' first-replay outputs are compared with each other.
"""

from __future__ import annotations

import json
import statistics
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import QWEN_GDN_MODELS
from models.demos.deepseek_v3_d_p.tests.gdn.cases import (
    LAYOUTS,
    build_gdn_case,
    make_gdn_device_case,
    registered_gdn_case,
)
from models.demos.deepseek_v3_d_p.tests.gdn.device_utils import gdn_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import deallocate_state, to_sp_input
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.timeout(900)]

_REPETITIONS = 10
_ABBA_BLOCKS = 4
_GROUP_CHUNKS = (20, 10)
# Both schedules compute the same recurrence; only the group stitching order of the fp32 state differs.
_SCHEDULE_PCC = 0.9999999


def _host_output(output: ttnn.Tensor) -> torch.Tensor:
    return torch.cat([ttnn.to_torch(shard).float().flatten() for shard in ttnn.get_device_tensors(output)])


@pytest.mark.parametrize(
    "mesh_device,device_params,model,layout,first",
    [
        pytest.param(
            LAYOUTS[layout][0],
            gdn_device_params(LAYOUTS[layout][0]),
            model,
            layout,
            first,
            id=f"{model}-{layout}-{first}",
        )
        for model in QWEN_GDN_MODELS
        for layout in ("LB-A", "LB-B")
        for first in ("g1first", "g2first")
    ],
    indirect=["mesh_device", "device_params"],
)
def test_gdn_group_schedule_layer_time(
    mesh_device: ttnn.MeshDevice, device_params: dict, model: str, layout: str, first: str
) -> None:
    spec = registered_gdn_case(model, layout, "single")
    case = build_gdn_case(spec)
    hidden = to_sp_input(case.chunk_hidden(0), mesh_device, spec.sequence_parallel_axis)
    actual_start = make_actual_start(mesh_device, 0)
    order = _GROUP_CHUNKS if first == "g1first" else _GROUP_CHUNKS[::-1]
    arms = {}
    for group_chunks in order:
        layer = make_gdn_device_case(mesh_device, case, summary_group_chunks=group_chunks)
        arms[group_chunks] = {"layer": layer, "state": layer.allocate_state()}

    # Warm every path twice before the first capture (persistent resources reach steady state).
    for _ in range(2):
        for arm in arms.values():
            output, state = arm["layer"].forward(hidden, arm["state"], actual_start)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)
            deallocate_state(state)
    for arm in arms.values():
        arm["trace"] = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        arm["output"], arm["next_state"] = arm["layer"].forward(hidden, arm["state"], actual_start)
        ttnn.end_trace_capture(mesh_device, arm["trace"], cq_id=0)
    try:
        for arm in arms.values():
            ttnn.execute_trace(mesh_device, arm["trace"], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            arm["first_output"] = _host_output(arm["output"])
        _, pcc = comp_pcc(arms[20]["first_output"], arms[10]["first_output"], pcc=_SCHEDULE_PCC)

        samples = {group_chunks: [] for group_chunks in arms}
        sequence = []
        for _ in range(_ABBA_BLOCKS):
            sequence += [order[0], order[1], order[1], order[0]]
        for group_chunks in sequence:
            trace = arms[group_chunks]["trace"]
            start = time.perf_counter()
            for _ in range(_REPETITIONS):
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples[group_chunks].append((time.perf_counter() - start) * 1e3 / _REPETITIONS)
    finally:
        for arm in arms.values():
            ttnn.release_trace(mesh_device, arm["trace"])
            ttnn.deallocate(arm["output"])
            deallocate_state(arm["next_state"])
            deallocate_state(arm["state"])

    # Pair each G2 sample with the G1 sample of the same ABBA half-block.
    paired = [(g2 - g1) / g1 for g1, g2 in zip(samples[20], samples[10])]
    tensor_parallel_size = spec.mesh_shape[spec.tensor_parallel_axis]
    print(
        "GDN_GROUP_SCHEDULE_TIME="
        + json.dumps(
            {
                "case": spec.name,
                "model": model,
                "layout": layout,
                "order": first,
                "value_heads_per_chip": case.config.num_value_heads // tensor_parallel_size,
                "sequence": sequence,
                "repetitions": _REPETITIONS,
                "g1_samples_ms": samples[20],
                "g2_samples_ms": samples[10],
                "g1_median_ms": statistics.median(samples[20]),
                "g2_median_ms": statistics.median(samples[10]),
                "paired_rel_delta": paired,
                "median_paired_rel_delta": statistics.median(paired),
                "g1_vs_g2_output_pcc": pcc,
            },
            sort_keys=True,
        )
    )
    assert pcc >= _SCHEDULE_PCC, f"G1 and G2 first-replay outputs differ: PCC {pcc}"

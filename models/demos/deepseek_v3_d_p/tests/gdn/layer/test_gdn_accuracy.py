# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttGDN against the FP32 GDN reference (``reference/gdn``) per model, layout and chunk schedule.

Every schedule runs through the production carry owner (``tt/kimi_k3/kda_state.py``, which holds any KDA-path
layer's ``KdaState``) under one trace: the chunk's forward and the in-place commit of its carries are captured once
at chunk 0's bounds and replayed for every chunk with new input, ``actual_start`` and ``actual_end``. After each
replay the output and the persisted carries are compared with the chained CPU reference (§6.3 gates: PCC, output
relative RMSE and norm ratio, D5 worst V-head state error). The schedule then runs a second time under the trace
and once eagerly; both must reproduce the first replay bit for bit (R10 repetition, R11 trace == eager).

Weights and inputs: synthetic weights on seeded random inputs for every model and layout; for the Qwen models at
LB-A and LB-B also the real layer-0 weights on random inputs and on real text (R13; local only, never CI).
Layouts and schedules: tests/gdn/cases.py. CPU references and weight caches come from the CPU preparation step
(``python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --case <name>``); this test only loads them.
"""

from __future__ import annotations

import json

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import QWEN_GDN_MODELS
from models.demos.deepseek_v3_d_p.tests.gdn.cases import (
    GDN_MODELS,
    LAYOUTS,
    SCHEDULES,
    TEXT_EXPOSED_GALAXY_RANK,
    GDNTestCase,
    build_gdn_case,
    make_gdn_device_case,
    registered_gdn_case,
)
from models.demos.deepseek_v3_d_p.tests.gdn.device_utils import (
    PCC_THRESHOLD,
    chunk_gate_rows,
    fixture_mesh_shape,
    gdn_device_params,
    layout_mesh,
    snapshot,
)
from models.demos.deepseek_v3_d_p.tests.gdn.reference_cache import cpu_references
from models.demos.deepseek_v3_d_p.tests.kda.utils import mla_row_permutation, to_sp_input
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import accuracy_metrics, make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.timeout(1800)]


# (weights, inputs): besides synthetic weights on seeded random inputs (every model and layout), real layer-0 weights on
# random inputs and on real text (tests/gdn/text_input.py) at the LoudBox layouts. Real weights are local only.
_REAL = (("real", "randn"), ("real", "text"))


def _cells() -> list[tuple[str, str, str, str, int]]:
    """(model, layout, weights, inputs, Galaxy rank) of every registered accuracy cell."""
    cells = []
    for model in GDN_MODELS:
        for layout in LAYOUTS:
            cells.append((model, layout, "synthetic", "randn", 0))
            if model in QWEN_GDN_MODELS and layout in ("LB-A", "LB-B"):
                cells.extend((model, layout, weights, inputs, 0) for weights, inputs in _REAL)
    cells.extend((model, "LB-B", "real", "text", rank) for model, rank in TEXT_EXPOSED_GALAXY_RANK.items())
    return cells


def _cell_id(model: str, layout: str, weights: str, inputs: str, rank: int) -> str:
    if weights == "synthetic":
        return f"{model}-{layout}-synthetic"
    return f"{model}-{layout}-real-{inputs}" + (f"-rank{rank}" if rank else "")


_D5_NEAR_ZERO_HEAD = (
    "tt_metal_tracker-g1b.5.18: D5 divides by the head's own expected state RMS; a full-forgetting head whose last "
    "valid token has a tiny beta has an expected state of ~1e-6 (absolute error <= 4.5e-4)"
)
# Cells that fail a §6.3 gate for a cause tracked in its own bead (strict: a fix makes them XPASS and fail).
_KNOWN_FAILURES = {
    "qwen38_2_4t-LB-A-real-text-ragged": _D5_NEAR_ZERO_HEAD + " (V head 5, chunk 1)",
    "qwen38_2_4t-LB-B-real-text-single": _D5_NEAR_ZERO_HEAD + " (V head 2)",
    "qwen38_2_4t-LB-B-real-text-chained3": _D5_NEAR_ZERO_HEAD
    + " (V head 2); also tt_metal_tracker-g1b.5.19: chunk 1 output peak error 0.70 (11x the other chunks), ungated",
    "qwen38_2_4t-LB-B-real-text-ragged": _D5_NEAR_ZERO_HEAD
    + " (V head 2); also tt_metal_tracker-g1b.5.19: chunk 1 output peak error 0.70, ungated",
    **{
        f"qwen38_2_4t-LB-B-real-text-rank3-{schedule}": _D5_NEAR_ZERO_HEAD + " (local V head 22)"
        for schedule in SCHEDULES
    },
}


def _known_failure(cell: str) -> list:
    if cell not in _KNOWN_FAILURES:
        return []
    return [pytest.mark.xfail(strict=True, raises=AssertionError, reason=_KNOWN_FAILURES[cell])]


def _params() -> list:
    return [
        pytest.param(
            fixture_mesh_shape(LAYOUTS[layout][0]),
            gdn_device_params(LAYOUTS[layout][0]),
            model,
            layout,
            weights,
            inputs,
            rank,
            schedule,
            id=f"{_cell_id(model, layout, weights, inputs, rank)}-{schedule}",
            marks=_known_failure(f"{_cell_id(model, layout, weights, inputs, rank)}-{schedule}"),
        )
        for model, layout, weights, inputs, rank in _cells()
        for schedule in SCHEDULES
    ]


def _run_schedule(
    case: GDNTestCase,
    mesh_device: ttnn.MeshDevice,
    cache: KdaStateCache,
    run_chunk,
    hidden_tt: ttnn.Tensor,
    start_tt: ttnn.Tensor,
    end_tt: ttnn.Tensor,
) -> list[dict[str, torch.Tensor]]:
    """Reset the carry, then run every chained chunk with its input and bounds; return host snapshots."""
    spec = case.spec
    sp_axis, tp_axis = spec.sequence_parallel_axis, spec.tensor_parallel_axis
    local_rows = spec.chunk_tokens // spec.sequence_parallel_size
    cache.reset()
    snapshots = []
    for chunk, valid in enumerate(spec.chunk_valid_tokens):
        start = chunk * spec.chunk_tokens
        permutation = mla_row_permutation(start, spec.sequence_parallel_size, local_rows)
        source = to_sp_input(case.chunk_hidden(chunk)[:, permutation], mesh_device, sp_axis)
        ttnn.copy(source, hidden_tt)
        ttnn.deallocate(source)
        for destination, value in ((start_tt, start), (end_tt, start + valid)):
            source = make_actual_start(mesh_device, value)
            ttnn.copy(source, destination)
            ttnn.deallocate(source)
        output = run_chunk()
        snapshots.append(snapshot(case.config, mesh_device, sp_axis, tp_axis, output, cache.read(0), start, valid))
    return snapshots


@pytest.mark.parametrize(
    "mesh_device,device_params,model,layout,weights,inputs,galaxy_rank,schedule",
    _params(),
    indirect=["mesh_device", "device_params"],
)
def test_gdn_layer_accuracy(
    mesh_device: ttnn.MeshDevice,
    device_params: dict,
    model: str,
    layout: str,
    weights: str,
    inputs: str,
    galaxy_rank: int,
    schedule: str,
) -> None:
    spec = registered_gdn_case(model, layout, schedule, weights, inputs, galaxy_rank)
    mesh_device = layout_mesh(mesh_device, spec.mesh_shape)
    grid = mesh_device.compute_with_storage_grid_size()
    compute_grid = (grid.x, grid.y)
    case = build_gdn_case(spec)
    references = cpu_references(case)
    layer = make_gdn_device_case(mesh_device, case)
    cache = KdaStateCache({0: layer})
    hidden_tt = to_sp_input(case.chunk_hidden(0), mesh_device, spec.sequence_parallel_axis)
    start_tt = make_actual_start(mesh_device, 0)
    end_tt = make_actual_start(mesh_device, spec.chunk_tokens)
    trace = output = None

    def forward_and_commit() -> ttnn.Tensor:
        result, new_state = layer.forward(hidden_tt, cache.read(0), start_tt, end_tt)
        cache.commit(0, new_state)
        return result

    def eager_chunk() -> ttnn.Tensor:
        nonlocal eager_output
        if eager_output is not None:
            ttnn.deallocate(eager_output)
        eager_output = forward_and_commit()
        return eager_output

    def replay_chunk() -> ttnn.Tensor:
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        return output

    eager_output = None
    try:
        # Compile the forward and the carry copies outside the capture, then return the carry to zero.
        with ttnn.manage_config("throw_exception_on_fallback", True):
            warm_output = forward_and_commit()
        ttnn.deallocate(warm_output)
        cache.reset()
        ttnn.synchronize_device(mesh_device)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output = forward_and_commit()
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        args = (case, mesh_device, cache)
        bounds = (hidden_tt, start_tt, end_tt)
        traced = [_run_schedule(*args, replay_chunk, *bounds) for _ in range(2)]
        eager = _run_schedule(*args, eager_chunk, *bounds)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
        for tensor in (output, eager_output, hidden_tt, start_tt, end_tt):
            if tensor is not None:
                ttnn.deallocate(tensor)
        cache.deallocate()

    failures, rows = [], []
    for chunk, (chunk_snapshot, reference) in enumerate(zip(traced[0], references, strict=True)):
        chunk_rows, chunk_failures = chunk_gate_rows(chunk, chunk_snapshot, reference.output, reference.state)
        rows.extend(chunk_rows)
        failures.extend(chunk_failures)
    sequence = accuracy_metrics(
        torch.cat([reference.output.bfloat16() for reference in references]),
        torch.cat([chunk_snapshot["output"] for chunk_snapshot in traced[0]]),
    )
    rows.append({"chunk": "all", "tensor": "output", **sequence})
    if sequence["pcc"] < PCC_THRESHOLD:
        failures.append(f"chained output PCC {sequence['pcc']:.6f} < {PCC_THRESHOLD}")

    def mismatches(first, second) -> list[str]:
        return [
            f"chunk {chunk} {name}"
            for chunk, (a, b) in enumerate(zip(first, second, strict=True))
            for name in a
            if not torch.equal(a[name], b[name])
        ]

    repeat_mismatches = mismatches(traced[0], traced[1])
    eager_mismatches = mismatches(traced[0], eager)
    if repeat_mismatches:
        failures.append(f"repeated trace schedule is not bit-identical: {repeat_mismatches}")
    if eager_mismatches:
        failures.append(f"trace replay differs from eager execution: {eager_mismatches}")
    for row in rows:
        print(f"GDN_ACCURACY_ROW {spec.name} " + json.dumps(row, sort_keys=True))
    recurrent_rows = [row for row in rows if row["tensor"] == "recurrent"]
    print(
        "GDN_ACCURACY="
        + json.dumps(
            {
                "case": spec.name,
                "model": model,
                "layout": layout,
                "schedule": schedule,
                "chunk_valid_tokens": list(spec.chunk_valid_tokens),
                "value_heads_per_chip": case.config.num_value_heads // spec.mesh_shape[spec.tensor_parallel_axis],
                "compute_grid": list(compute_grid),
                "min_pcc": min(row["pcc"] for row in rows),
                "weights": weights,
                "inputs": inputs,
                "key_head_slice": spec.key_head_slice,
                "max_output_rel_rmse": max(row["rel_rmse"] for row in rows if row["tensor"] == "output"),
                "max_output_rel_linf": max(row["rel_linf"] for row in rows if row["tensor"] == "output"),
                "max_recurrent_rel_linf": max(row["rel_linf"] for row in recurrent_rows),
                "output_norm_ratio_range": [
                    min(row["norm_ratio"] for row in rows if "norm_ratio" in row),
                    max(row["norm_ratio"] for row in rows if "norm_ratio" in row),
                ],
                "worst_head_state_rel_rmse": max(row["worst_head_rel_rmse"] for row in recurrent_rows),
                "trace_repeat_bit_identical": not repeat_mismatches,
                "trace_equals_eager": not eager_mismatches,
                "passed": not failures,
            },
            sort_keys=True,
        )
    )
    assert not failures, "\n".join(failures)

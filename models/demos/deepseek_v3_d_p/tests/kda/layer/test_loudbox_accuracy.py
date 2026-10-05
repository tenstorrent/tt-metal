# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LoudBox accuracy of the Kimi-K3 KDA layer at the Galaxy per-chip shapes (LB-A and LB-B).

Every schedule runs through the production carry owner (``tt/kimi_k3/kda_state.py``) under one trace: the
chunk's forward and the in-place commit of its carries are captured once at chunk 0's bounds and replayed for
every chunk with new input, ``actual_start`` and ``actual_end`` and no host work in the replay. After each
replay the output and the persisted carries are compared with the chained CPU reference; the whole schedule
then runs a second time from a reset carry and must reproduce the first bit for bit.

Requirements (tt-work artifacts/loudbox-linear-prefill-requirements.md): LB-A 2x4 SP2xTP4 T=1280 and LB-B 8x1
SP8xTP1 T=5120 on one Galaxy TP4 rank's heads, both 640 tokens per SP rank (R1-R6); chained chunks (R7);
ragged tail via ``actual_end`` (R8); PCC >= 0.9995 per chunk and over the sequence (R9); repetition (R10);
trace (R11); synthetic and real layer weights, random and real-text inputs (R13).

CPU references, weight caches and text inputs come from the CPU preparation step
(``python -m models.demos.deepseek_v3_d_p.tests.kda.prepare --case <name>``); this test only loads them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.cases import (
    KDATestCase,
    build_kda_case,
    loudbox_kda_case,
    make_kda_device_case,
)
from models.demos.deepseek_v3_d_p.tests.kda.reference_cache import KDAChunkReference, cpu_references
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    mla_row_permutation,
    reconstruct_convolution_at_sp_rank,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
    to_sp_input,
)
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import accuracy_metrics, make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.timeout(1800)]

_PCC_THRESHOLD = 0.9995
_LAYOUTS = {"LB-A": (2, 4), "LB-B": (8, 1)}
_SEQUENCE_PARALLEL_AXIS = 0
_TENSOR_PARALLEL_AXIS = 1
# (weights, inputs): synthetic and real layer weights on seeded random inputs; real weights on real text.
_MATRIX = (("synthetic", "randn"), ("real", "randn"), ("real", "text"))


def _params() -> list:
    return [
        pytest.param(
            _LAYOUTS[layout],
            fabric_1d_device_params(),
            layout,
            weights,
            inputs,
            schedule,
            id=f"{layout}-{weights}-{inputs}-{schedule}",
        )
        for layout in _LAYOUTS
        for weights, inputs in _MATRIX
        for schedule in ("single", "chained3", "ragged")
    ]


def _snapshot(
    case: KDATestCase, mesh_device: ttnn.MeshDevice, output: ttnn.Tensor, carry, chunk: int
) -> dict[str, torch.Tensor]:
    """Host copy of one chunk's valid output (natural order) and the persisted carries at every SP rank."""
    mesh_shape = tuple(mesh_device.shape)
    sp_size = mesh_shape[_SEQUENCE_PARALLEL_AXIS]
    local_rows = case.spec.chunk_tokens // sp_size
    permutation = mla_row_permutation(chunk * case.spec.chunk_tokens, sp_size, local_rows)
    rotated = reconstruct_sp_tp_tensor(
        output, mesh_device, _SEQUENCE_PARALLEL_AXIS, _TENSOR_PARALLEL_AXIS, tp_dim=2, sp_dim=1
    )
    natural = torch.empty_like(rotated)
    natural[:, permutation, :] = rotated
    local_width = case.config.num_heads // mesh_shape[_TENSOR_PARALLEL_AXIS] * case.config.head_k_dim
    snapshot = {"output": natural[:, : case.spec.chunk_valid_tokens[chunk]].clone()}
    for rank in range(sp_size):
        snapshot[f"recurrent_sp{rank}"] = reconstruct_state_at_sp_rank(
            carry.recurrent, mesh_device, _SEQUENCE_PARALLEL_AXIS, _TENSOR_PARALLEL_AXIS, rank
        ).clone()
        snapshot[f"convolution_sp{rank}"] = reconstruct_convolution_at_sp_rank(
            carry.convolution, mesh_device, _SEQUENCE_PARALLEL_AXIS, _TENSOR_PARALLEL_AXIS, rank, local_width
        ).clone()
    return snapshot


def _expected(reference: KDAChunkReference, name: str) -> torch.Tensor:
    if name == "output":
        return reference.output.bfloat16()
    if name.startswith("recurrent"):
        return reference.state.recurrent
    state = reference.state
    return torch.cat((state.q_convolution, state.k_convolution, state.v_convolution), dim=-1).bfloat16()


def _run_schedule(
    case: KDATestCase,
    mesh_device: ttnn.MeshDevice,
    cache: KdaStateCache,
    layer_idx: int,
    trace: int,
    hidden_tt: ttnn.Tensor,
    start_tt: ttnn.Tensor,
    end_tt: ttnn.Tensor,
    output: ttnn.Tensor,
) -> list[dict[str, torch.Tensor]]:
    """Reset the carry, then replay the captured chunk once per chained chunk with that chunk's bounds."""
    sp_size = tuple(mesh_device.shape)[_SEQUENCE_PARALLEL_AXIS]
    local_rows = case.spec.chunk_tokens // sp_size
    cache.reset()
    snapshots = []
    for chunk, valid in enumerate(case.spec.chunk_valid_tokens):
        start = chunk * case.spec.chunk_tokens
        permutation = mla_row_permutation(start, sp_size, local_rows)
        source = to_sp_input(case.chunk_hidden(chunk)[:, permutation], mesh_device, _SEQUENCE_PARALLEL_AXIS)
        ttnn.copy(source, hidden_tt)
        ttnn.deallocate(source)
        for destination, value in ((start_tt, start), (end_tt, start + valid)):
            source = make_actual_start(mesh_device, value)
            ttnn.copy(source, destination)
            ttnn.deallocate(source)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        snapshots.append(_snapshot(case, mesh_device, output, cache.read(layer_idx), chunk))
    return snapshots


@pytest.mark.parametrize(
    "mesh_device,device_params,layout,weights,inputs,schedule", _params(), indirect=["mesh_device", "device_params"]
)
def test_loudbox_kimi_k3_accuracy(
    mesh_device: ttnn.MeshDevice,
    device_params: dict,
    layout: str,
    weights: str,
    inputs: str,
    schedule: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    spec = loudbox_kda_case(weights, layout, schedule, inputs)
    checkpoint_dir: Path | None = request.getfixturevalue("kimi_k3_checkpoint_dir") if weights == "real" else None
    case = build_kda_case(spec, checkpoint_dir)
    references = cpu_references(case)
    layer, hidden_tt = make_kda_device_case(mesh_device, case)
    layer_idx = case.weights.layer_idx
    cache = KdaStateCache({layer_idx: layer})
    start_tt = make_actual_start(mesh_device, 0)
    end_tt = make_actual_start(mesh_device, spec.chunk_tokens)
    trace = output = None
    try:
        # Compile the forward and the carry copies outside the capture, then return the carry to zero.
        with ttnn.manage_config("throw_exception_on_fallback", True):
            warm_output, warm_state = layer.forward(hidden_tt, cache.read(layer_idx), start_tt, end_tt)
        cache.commit(layer_idx, warm_state)
        ttnn.deallocate(warm_output)
        cache.reset()
        ttnn.synchronize_device(mesh_device)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output, new_state = layer.forward(hidden_tt, cache.read(layer_idx), start_tt, end_tt)
        cache.commit(layer_idx, new_state)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)

        runs = [
            _run_schedule(case, mesh_device, cache, layer_idx, trace, hidden_tt, start_tt, end_tt, output)
            for _ in range(2)
        ]
    finally:
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
        if output is not None:
            ttnn.deallocate(output)
        cache.deallocate()
        for tensor in (hidden_tt, start_tt, end_tt):
            ttnn.deallocate(tensor)

    failures = []
    rows = []
    for chunk, (snapshot, reference) in enumerate(zip(runs[0], references, strict=True)):
        for kind in ("output", "recurrent", "convolution"):
            names = [name for name in snapshot if name.split("_sp")[0] == kind]
            per_rank = {name: accuracy_metrics(_expected(reference, name), snapshot[name]) for name in names}
            worst = min(per_rank, key=lambda name: per_rank[name]["pcc"])
            row = {"chunk": chunk, "tensor": kind, "worst": worst, **per_rank[worst]}
            row["max_abs_any_rank"] = max(metrics["max_abs"] for metrics in per_rank.values())
            rows.append(row)
            if row["pcc"] < _PCC_THRESHOLD:
                failures.append(f"chunk {chunk} {worst} PCC {row['pcc']:.6f} < {_PCC_THRESHOLD}")
    sequence = accuracy_metrics(
        torch.cat([reference.output.bfloat16() for reference in references], dim=1),
        torch.cat([snapshot["output"] for snapshot in runs[0]], dim=1),
    )
    rows.append({"chunk": "all", "tensor": "output", **sequence})
    if sequence["pcc"] < _PCC_THRESHOLD:
        failures.append(f"chained output PCC {sequence['pcc']:.6f} < {_PCC_THRESHOLD}")
    mismatched = [
        f"chunk {chunk} {name}"
        for chunk, (first, second) in enumerate(zip(*runs, strict=True))
        for name in first
        if not torch.equal(first[name], second[name])
    ]
    if mismatched:
        failures.append(f"repeated schedule is not bit-identical: {mismatched}")
    for row in rows:
        print(f"KDA_LOUDBOX_ROW {spec.name} " + json.dumps(row, sort_keys=True))
    if failures:
        # Keep the failing tensors for offline localization (which heads / key channels carry the error).
        dump = tmp_path / f"{spec.name}.pt"
        torch.save(
            {"device": runs[0], "expected": [{name: _expected(r, name) for name in runs[0][0]} for r in references]},
            dump,
        )
        print(f"KDA_LOUDBOX_DUMP={dump}")
        for chunk, (snapshot, reference) in enumerate(zip(runs[0], references, strict=True)):
            error = (snapshot["recurrent_sp0"] - reference.state.recurrent).abs()[0]  # [heads, key, value]
            per_key = error.amax(-1).flatten()
            top = torch.topk(per_key, 8)
            print(
                f"KDA_LOUDBOX_RECURRENT_PEAKS chunk {chunk}: per-head max |err| "
                f"{[round(float(v), 4) for v in error.amax((-1, -2))]}; top (head, key): "
                + ", ".join(
                    f"({int(i) // error.shape[1]},{int(i) % error.shape[1]})={float(v):.3e}" for v, i in zip(*top)
                )
            )
    print(
        "KDA_LOUDBOX_ACCURACY="
        + json.dumps(
            {
                "case": spec.name,
                "layout": layout,
                "weights": weights,
                "inputs": inputs,
                "schedule": schedule,
                "chunk_valid_tokens": list(spec.chunk_valid_tokens),
                "heads_per_chip": case.config.num_heads // tuple(mesh_device.shape)[_TENSOR_PARALLEL_AXIS],
                "pcc_threshold": _PCC_THRESHOLD,
                "bit_identical_repeat": not mismatched,
                "min_pcc": min(row["pcc"] for row in rows),
                "passed": not failures,
            },
            sort_keys=True,
        )
    )
    assert not failures, "\n".join(failures)

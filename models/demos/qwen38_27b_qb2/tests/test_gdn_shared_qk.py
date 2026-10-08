# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Physical shared-normalization diagnostic; model defaults remain unchanged."""

import gc
import hashlib
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.gdn_shared_qk import BATCHES, compare
from models.demos.qwen38_27b_qb2.tests.test_gdn_model_adapter import run_case
from models.demos.qwen38_27b_qb2.tests.test_gdn_step_candidate import check, reference, stimulus, upload
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt.gdn_step import op
from models.demos.qwen38_27b_qb2.tt.gdn_step.shared_qk import prepare
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


def require_identical(left, right):
    a, b = [ttnn.get_device_tensors(tensor) for tensor in (left, right)]
    assert len(a) == len(b) == 4
    for lrank, rrank in zip(a, b):
        assert torch.equal(ttnn.to_torch(lrank), ttnn.to_torch(rrank)), "Shared normalization changed FP32 results"


def long_horizon(mesh, report, path):
    """Changing operands, alternate live scratch and 4096 FP32 state updates."""
    initial = stimulus(12, 720000)[-1]
    hosts, candidate_inputs, baseline_inputs = [], [], []
    for index in range(64):
        rng = torch.Generator().manual_seed(810000 + index)
        q, k = [torch.randn(4, 128, generator=rng).bfloat16().float() for _ in range(2)]
        _, _, v, gates, _ = stimulus(12, 830000 + index)
        expanded = [value.repeat_interleave(3, dim=0) for value in (q, k)]
        values = (*expanded, v, gates)
        hosts.append(values)
        candidate_inputs.append([upload(mesh, value) for value in (q, k, v, gates)])
        baseline_inputs.append([upload(mesh, value) for value in values])
    state = upload(mesh, initial, tiled=True)
    control_state = upload(mesh, initial, tiled=True)
    reset = upload(mesh, initial, tiled=True)
    output = upload(mesh, torch.full((12, 128), float("nan")))
    control_output = upload(mesh, torch.full((12, 128), float("nan")))
    scratch = [tuple(upload(mesh, torch.full((4, 128), float("nan"))) for _ in range(2)) for _ in range(2)]
    persistent = [state, control_state, output, control_output, *(tensor for pair in scratch for tensor in pair)]
    addresses = [tensor.buffer_address() for tensor in persistent]

    def invoke(index):
        q, k, v, gates = candidate_inputs[index]
        normalized = scratch[index % 2]
        prepare(q, k, *normalized)
        op.step(*normalized, v, gates, state, output, value_splits=4, input_buffer_items=2, qk_head_repeat=3)
        op.step(
            *baseline_inputs[index],
            control_state,
            control_output,
            value_splits=4,
            input_buffer_items=2,
            normalize_qk=True,
        )

    invoke(0)  # Compile each descriptor before trace capture.
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        for index in range(64):
            invoke(index)
    except BaseException:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.release_trace(mesh, trace)
        raise
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    expected = initial.clone()
    try:
        for target in (state, control_state):
            ttnn.copy(reset, target)
        for cycle in range(64):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for values in hosts:
                expected, expected_output = reference(expected, *values, normalize_qk=True)
            if cycle % 4 == 3:
                checks = dict(
                    steps=(cycle + 1) * 64,
                    state=check(state, expected),
                    output=check(output, expected_output),
                )
                require_identical(state, control_state)
                require_identical(output, control_output)
                checks["bit_identical_to_fused_normalization"] = True
                report["long_horizon"].append(checks)
                save(path, report)
                print("SHARED_QK_LONG_HORIZON", checks["steps"], flush=True)
    finally:
        ttnn.release_trace(mesh, trace)
    assert addresses == [tensor.buffer_address() for tensor in persistent]
    for inputs, values in zip(candidate_inputs, hosts):
        for i, (tensor, gold) in enumerate(zip(inputs, values)):
            if i < 2:
                gold = gold[::3]
            assert all(torch.equal(ttnn.to_torch(rank), gold) for rank in ttnn.get_device_tensors(tensor))
    report["persistent_addresses_unchanged"] = True
    report["changing_input_cycle_length"] = 64
    report["alternating_scratch_allocations"] = 2
    save(path, report)


@pytest.mark.skipif(os.getenv("QWEN_GDN_SHARED_QK") != "1", reason="explicit allocated-Galaxy diagnostic")
def test_gdn_shared_qk():
    path = Path(os.environ["QWEN_GDN_SHARED_QK_RECEIPT"])
    assert not path.exists(), "Preserve each experimental attempt"
    torch.set_num_threads(8)
    source = Path(__file__).resolve().parents[1]
    files = [
        *op.HERE.glob("*"),
        Path(__file__),
        source / "tests/test_gdn_model_adapter.py",
        source / "tests/gdn_shared_qk.py",
    ]
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        promoted_to_model=False,
        precision="FP32 Q/K normalization and recurrent state; adapter receives existing BF16 model activations",
        scope="Synthetic TP4 adapter timing and changing-input recurrence qualification; no model-eval claim",
        source_sha256={
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
            if p.suffix in (".py", ".cpp", ".hpp", ".h")
        },
        cases=[],
        comparisons=[],
        long_horizon=[],
    )
    save(path, report)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        assert len(report["device_ids"]) == 4
        for batch in BATCHES:
            report.update(state="adapter_comparison", active_batch=batch)
            group = []
            for shared in (False, True, False):
                report["active_shared_qk"] = shared
                save(path, report)
                case = run_case(mesh, batch, shared_qk=shared)
                group.append(case)
                report["cases"].append(case)
                save(path, report)
                gc.collect()
            result = compare(group)
            report["comparisons"].append(result)
            print("SHARED_QK_COMPARISON", result, flush=True)
            save(path, report)
        report["state"] = "long_horizon"
        save(path, report)
        long_horizon(mesh, report, path)
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", passed=False, error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        try:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            finally:
                ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = True
        except BaseException as error:
            report.update(state="failed", passed=False, cleanup_error=str(error)[:2000])
            raise
        finally:
            save(path, report)

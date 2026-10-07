# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate model-layout preparation and in-place state before model integration."""

import gc
import hashlib
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.test_gdn_step_candidate import accuracy, reference
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt.gdn_step.model_adapter import step_from_flat
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


def run_case(mesh, batch):
    rng = torch.Generator().manual_seed(902100 + batch)
    host = [torch.randn(batch, 32, width, generator=rng).bfloat16() for width in (512, 512, 1536)]
    decay = torch.zeros(batch, 32, 12)
    decay[:, 0] = -(0.00001 + 0.00499 * torch.rand(batch, 12, generator=rng))
    beta = torch.zeros(batch, 32, 12).bfloat16()
    beta[:, 0] = (0.2 + 0.6 * torch.rand(batch, 12, generator=rng)).bfloat16()
    initial = torch.randn(batch, 12, 128, 128, generator=rng) * 0.1
    host += [decay, beta]

    def upload(value, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value,
            device=mesh,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    inputs = [upload(value, ttnn.float32 if i == 3 else ttnn.bfloat16) for i, value in enumerate(host)]
    state = upload(initial, ttnn.float32)
    reset = upload(initial, ttnn.float32)
    output = upload(torch.full((batch * 12, 128), float("nan")), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    normalized = []
    for value, scale in ((host[0], 128**-0.5), (host[1], 1.0)):
        value = value[:, 0].float().reshape(batch, 4, 128)
        value = value * torch.rsqrt(value.square().sum(-1, keepdim=True) + 1e-6) * scale
        normalized.append(value.repeat_interleave(3, dim=1).reshape(batch * 12, 128))
    values = host[2][:, 0].float().reshape(batch * 12, 128)
    gates = torch.zeros(batch * 12, 8)
    gates[:, 0] = decay[:, 0].flatten().exp()
    gates[:, 1] = beta[:, 0].flatten().float()
    expected = initial.reshape(batch * 12, 128, 128).clone()
    checks = []

    def verify(actual_output, expected_state, expected_output, *, require_pass=True):
        state_checks = [
            accuracy(ttnn.to_torch(rank).reshape(batch * 12, 128, 128), expected_state)
            for rank in ttnn.get_device_tensors(state)
        ]
        output_checks = [
            accuracy(ttnn.to_torch(rank)[:, 0], expected_output) for rank in ttnn.get_device_tensors(actual_output)
        ]
        assert len(state_checks) == len(output_checks) == 4
        passed = all(row["passed"] for row in state_checks + output_checks)
        if require_pass:
            assert passed, (state_checks, output_checks)
        return dict(passed=passed, state=state_checks, output=output_checks)

    for _ in range(2):
        actual_output = step_from_flat(*inputs, state, output)
        expected, expected_output = reference(expected, *normalized, values, gates)
        checks.append(verify(actual_output, expected, expected_output))
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        traced_output = step_from_flat(*inputs, state, output)
    except BaseException:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.release_trace(mesh, trace)
        raise
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        ttnn.copy(reset, state)
        expected = initial.reshape(batch * 12, 128, 128).clone()
        for _ in range(64):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            expected, expected_output = reference(expected, *normalized, values, gates)
        checks.append(verify(traced_output, expected, expected_output))
        samples = []
        for _ in range(5):
            ttnn.synchronize_device(mesh)
            tick = time.perf_counter()
            for _ in range(100):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter() - tick) * 1e6 / 100)
    finally:
        ttnn.release_trace(mesh, trace)
    for tensor, original in zip(inputs, host):
        assert all(torch.equal(ttnn.to_torch(rank), original) for rank in ttnn.get_device_tensors(tensor))
    result = dict(
        batch=batch,
        passed=True,
        checks=checks,
        traced_call_us=samples,
        median_traced_call_us=statistics.median(samples),
    )
    if os.getenv("QWEN_GDN_NATIVE_CONTROL") == "1":
        # Match the current model boundary, including native head normalization,
        # per-core batch partitioning, concatenation, and recurrent state copy.
        # Caller-provided constants prevent host uploads inside trace capture.
        masks = torch.zeros(1, 1, 32, 96)
        masks[:, :, :16, :16] = 1
        masks[:, :, 16:, 48:64] = 1
        masks[:, :, 16:, 64:80] = 1
        constants = dict(
            eye=upload(torch.eye(32).reshape(1, 1, 32, 32), ttnn.float32),
            tril=upload(torch.ones(32, 32).tril().reshape(1, 1, 32, 32), ttnn.float32),
            ones=upload(torch.ones(1, 1, 32, 32), ttnn.float32),
            masks=upload(masks, ttnn.float32),
        )
        grid = mesh.compute_with_storage_grid_size()
        scan_batch = grid.x * grid.y // 12

        def native():
            outputs, states = [], []
            for start in range(0, batch, scan_batch):
                end = min(start + scan_batch, batch)
                part, new_state = ttnn.transformer.chunk_gated_delta_rule(
                    *(tensor[start:end] for tensor in inputs),
                    initial_state=state[start:end],
                    output_final_state=True,
                    output_head_major=True,
                    chunk_size=32,
                    **constants,
                )
                outputs.append(part)
                states.append(new_state)
            new_state = states[0] if len(states) == 1 else ttnn.concat(states, dim=0)
            ttnn.copy(new_state, state)
            return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=0)

        ttnn.copy(reset, state)
        expected = initial.reshape(batch * 12, 128, 128).clone()
        native_checks = []
        for _ in range(2):
            actual_output = native()
            expected, expected_output = reference(expected, *normalized, values, gates)
            native_checks.append(verify(actual_output, expected, expected_output, require_pass=False))
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            native_output = native()
        except BaseException:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.release_trace(mesh, trace)
            raise
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        try:
            ttnn.copy(reset, state)
            expected = initial.reshape(batch * 12, 128, 128).clone()
            for _ in range(64):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                expected, expected_output = reference(expected, *normalized, values, gates)
            native_checks.append(verify(native_output, expected, expected_output, require_pass=False))
            native_samples = []
            for _ in range(5):
                ttnn.synchronize_device(mesh)
                tick = time.perf_counter()
                for _ in range(100):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                native_samples.append((time.perf_counter() - tick) * 1e6 / 100)
        finally:
            ttnn.release_trace(mesh, trace)
        result["native_control"] = dict(
            accuracy_passed=all(check["passed"] for check in native_checks),
            checks=native_checks,
            traced_call_us=native_samples,
            median_traced_call_us=statistics.median(native_samples),
            speedup=statistics.median(native_samples) / result["median_traced_call_us"],
        )
    return result


@pytest.mark.skipif(os.getenv("QWEN_GDN_MODEL_ADAPTER") != "1", reason="explicit allocated-Galaxy experiment")
def test_gdn_model_adapter():
    path = Path(os.environ["QWEN_GDN_MODEL_ADAPTER_RECEIPT"])
    assert not path.exists(), "Use a new result directory"
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        promoted_to_model=False,
        scope="Model-layout preparation plus single-step GDN; convolution, output gating and projection excluded",
        measurement="Five warm samples of 100 trace replays; preparation and layout costs included",
        source_sha256={
            source.name: hashlib.sha256(source.read_bytes()).hexdigest()
            for source in [
                Path(__file__),
                *Path(step_from_flat.__code__.co_filename).parent.glob("*.py"),
                *Path(step_from_flat.__code__.co_filename).parent.glob("*.cpp"),
            ]
        },
        cases=[],
    )
    save(path, report)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        for batch in (1, 8, 16, 32, 64):
            report.update(state="running", active_batch=batch)
            save(path, report)
            report["cases"].append(run_case(mesh, batch))
            save(path, report)
            gc.collect()
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
        finally:
            try:
                ttnn.close_mesh_device(parent)
            finally:
                save(path, report)

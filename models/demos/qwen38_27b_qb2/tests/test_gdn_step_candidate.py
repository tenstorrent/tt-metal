# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP4 compilation, precision, cache-rebinding, and latency experiment.

The qualified model never imports this candidate. Accuracy success does not
promote it: its separate P1 latency target and full-model gates must also pass.
"""

import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.experiments.gdn_step import op
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


def reference(state, q, k, v, gates):
    decayed = state * gates[:, 0, None, None]
    prediction = torch.einsum("hk,hkv->hv", k, decayed)
    delta = gates[:, 1, None] * (v - prediction)
    updated = decayed + k[:, :, None] * delta[:, None, :]
    return updated, torch.einsum("hk,hkv->hv", q, updated)


def stimulus(heads, seed, *, cancellation=False):
    rng = torch.Generator().manual_seed(seed)
    q, k = [torch.nn.functional.normalize(torch.randn(heads, 128, generator=rng), dim=-1) for _ in range(2)]
    q *= 128**-0.5
    v = torch.randn(heads, 128, generator=rng)
    gates = torch.zeros(heads, 8)
    gates[:, 0] = 0.995 + torch.rand(heads, generator=rng) * 0.00499
    gates[:, 1] = 0.2 + torch.rand(heads, generator=rng) * 0.6
    state = torch.randn(heads, 128, 128, generator=rng) * 0.1
    if cancellation:
        v = torch.einsum("hk,hkv->hv", k, state * gates[:, 0, None, None]) + 1e-6 * v
    return q, k, v, gates, state


def upload(mesh, value, *, tiled=False):
    return ttnn.from_torch(
        value.contiguous(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT if tiled else ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def buffers(mesh, values):
    return [upload(mesh, value, tiled=i == 4) for i, value in enumerate(values)] + [
        upload(mesh, torch.full_like(values[0], float("nan")))
    ]


def accuracy(actual, expected):
    assert actual.shape == expected.shape
    actual, expected = [value.flatten(1).double() for value in (actual, expected)]
    error = actual - expected
    relative_rms = error.square().mean(1).sqrt() / expected.square().mean(1).sqrt().clamp_min(1e-12)
    ac, ec = [value - value.mean(1, keepdim=True) for value in (actual, expected)]
    pcc = (ac * ec).sum(1) / (ac.norm(dim=1) * ec.norm(dim=1)).clamp_min(1e-24)
    finite = bool(torch.isfinite(actual).all())
    return dict(
        passed=finite and bool((pcc >= 0.999).all()) and bool((relative_rms <= 0.005).all()),
        finite=finite,
        min_head_pcc=float(pcc.min()) if finite else None,
        max_head_relative_rms=float(relative_rms.max()) if finite else None,
        max_abs=float(error.abs().max()) if finite else None,
    )


def check(tensor, expected):
    results = [accuracy(ttnn.to_torch(part).float(), expected) for part in ttnn.get_device_tensors(tensor)]
    assert len(results) == 4
    assert all(x["passed"] for x in results), results
    return results


def capture(mesh, calls, *, value_splits=1):
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        for arguments in calls:
            op.step(*arguments, value_splits=value_splits)
    except BaseException:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.release_trace(mesh, trace)
        raise
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    return trace


def short_case(mesh, batch, *, value_splits=1):
    heads = batch * 12
    # Two simultaneously live sets of identically shaped allocations exercise
    # cache-hit address replacement, including the in-place state destination.
    hosts = [stimulus(heads, 20261006 + batch + i, cancellation=bool(i)) for i in range(2)]
    devices = [buffers(mesh, values) for values in hosts]
    expected = [values[-1].clone() for values in hosts]
    checks = []
    cache_entries = []
    initial_entries = mesh.num_program_cache_entries()
    for index in [0, 1, 0, 1]:
        q, k, v, gates, _ = hosts[index]
        expected[index], output = reference(expected[index], q, k, v, gates)
        op.step(*devices[index], value_splits=value_splits)
        ttnn.synchronize_device(mesh)
        cache_entries.append(mesh.num_program_cache_entries())
        checks.append(
            dict(
                allocation=index,
                state=check(devices[index][4], expected[index]),
                output=check(devices[index][5], output),
            )
        )
    # Ensure stepping one set did not overwrite the other set's state.
    check(devices[0][4], expected[0])
    check(devices[1][4], expected[1])
    assert cache_entries[0] > 0, "No cached program was created"
    assert cache_entries[2] == cache_entries[3], "Repeated allocations did not reuse the program cache"
    for device_set, host_set in zip(devices, hosts):
        for tensor, expected_input in zip(device_set[:4], host_set[:4]):
            for rank in ttnn.get_device_tensors(tensor):
                assert torch.equal(ttnn.to_torch(rank), expected_input), "Kernel modified a read-only input"
    trace = capture(mesh, [devices[0]], value_splits=value_splits)
    try:
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
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
    target_us = heads * 128 * 128 * 4 * 2 / (512 * 1e3) + 10
    grid = mesh.compute_with_storage_grid_size()
    return dict(
        batch=batch,
        value_splits=value_splits,
        heads_per_chip=heads,
        active_cores=len(op.work_items(heads, grid.x, grid.y, value_splits)),
        checks=checks,
        traced_call_us=samples,
        median_traced_call_us=statistics.median(samples),
        p1_target_us=target_us,
        p1_latency_target_met=statistics.median(samples) <= target_us,
        program_cache_entries_before=initial_entries,
        program_cache_entries_per_call=cache_entries,
        program_cache_entries_after=mesh.num_program_cache_entries(),
    )


def long_horizon(mesh, report, path, *, value_splits=1):
    heads = 12
    # All steps change inputs; a 64-step cycle avoids timing thousands of H2D
    # uploads. The reference executes every recurrence, including all cycles.
    hosts = [stimulus(heads, 710000 + i)[:4] for i in range(64)]
    inputs = [[upload(mesh, value) for value in values] for values in hosts]
    initial = stimulus(heads, 720000)[-1]
    state = upload(mesh, initial, tiled=True)
    output = upload(mesh, torch.zeros(heads, 128))
    calls = [values + [state, output] for values in inputs]
    op.step(*calls[0], value_splits=value_splits)  # compile outside capture
    trace = capture(mesh, calls, value_splits=value_splits)
    expected = initial.clone()
    reset = upload(mesh, initial, tiled=True)
    ttnn.copy(reset, state)
    ttnn.synchronize_device(mesh)
    try:
        for cycle in range(64):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for values in hosts:
                expected, expected_output = reference(expected, *values)
            if cycle % 4 == 3:
                checkpoint = dict(
                    value_splits=value_splits,
                    steps=(cycle + 1) * 64,
                    state=check(state, expected),
                    output=check(output, expected_output),
                )
                report["long_horizon"].append(checkpoint)
                save(path, report)
                print("GDN_LONG_HORIZON", json.dumps(checkpoint), flush=True)
    finally:
        ttnn.release_trace(mesh, trace)
    # Near-identity decay exposes recurrent-state narrowing without an update
    # masking the lost low mantissa bits.
    q, k, v, gates = [value.clone() for value in hosts[0]]
    gates[:, 0] = 0.99999
    gates[:, 1] = 0
    decay_inputs = [upload(mesh, value) for value in [q, k, v, gates]]
    for _ in range(64):
        op.step(*decay_inputs, state, output, value_splits=value_splits)
        expected, expected_output = reference(expected, q, k, v, gates)
    report["decay_only"].append(
        dict(value_splits=value_splits, steps=64, state=check(state, expected), output=check(output, expected_output))
    )
    save(path, report)


@pytest.mark.skipif(os.getenv("QWEN_GDN_STEP_CANDIDATE") != "1", reason="explicit allocated-Galaxy experiment")
def test_gdn_step_candidate():
    path = Path(os.environ["QWEN_GDN_STEP_RECEIPT"])
    assert not path.exists(), "Use a new result directory"
    value_splits = [int(part) for part in os.getenv("QWEN_GDN_VALUE_SPLITS", "1,2,4").split(",")]
    assert value_splits and len(set(value_splits)) == len(value_splits)
    assert all(part in (1, 2, 4) for part in value_splits)
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        promoted_to_model=False,
        scope="Standalone TP4 recurrence candidate; no convolution/normalization/output gating or model eval",
        precision="FP32 compact vectors, tiled FP32 state, direct FP32 unpack and SFPU arithmetic",
        measurement="Five warm samples of 100 trace replays; includes dispatch; no host readback in timing",
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(op.HERE.glob("*"))
            if p.suffix in {".cpp", ".py"}
        },
        cases=[],
        long_horizon=[],
        decay_only=[],
        value_splits=value_splits,
    )
    save(path, report)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        mesh.enable_program_cache()
        report["device_ids"] = list(mesh.get_device_ids())
        for partitions in value_splits:
            for batch in [1, 8, 16, 32, 64]:
                report.update(state="short_case", active_batch=batch, active_value_splits=partitions)
                save(path, report)
                result = short_case(mesh, batch, value_splits=partitions)
                report["cases"].append(result)
                print("GDN_SHORT_CASE", json.dumps(result), flush=True)
                save(path, report)
                gc.collect()
            report.update(state="long_horizon")
            long_horizon(mesh, report, path, value_splits=partitions)
            gc.collect()
        report.update(
            state="completed",
            passed=True,
            p1_latency_target_met=all(c["p1_latency_target_met"] for c in report["cases"]),
        )
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:2000]))
        raise
    finally:
        try:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            finally:
                ttnn.close_mesh_device(parent)
        except BaseException as error:
            report.update(
                state="failed",
                passed=False,
                cleanup_error=dict(type=type(error).__name__, message=str(error)[:2000]),
            )
            raise
        finally:
            save(path, report)

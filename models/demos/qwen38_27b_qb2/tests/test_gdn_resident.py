# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in physical TP4 screen of register-resident GDN; no model promotion."""

import gc
import hashlib
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings
from models.demos.qwen38_27b_qb2.tests.test_gdn_step_candidate import accuracy, reference, stimulus
from models.demos.qwen38_27b_qb2.tt.gdn_step import op
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


def download(tensor):
    values = [ttnn.to_torch(rank).float() for rank in ttnn.get_device_tensors(tensor)]
    assert len(values) == 4
    return values


def upload(mesh, values, *, tiled=False):
    assert len(values) == 4
    return ttnn.from_torch(
        torch.cat(values, dim=0).contiguous(),
        device=mesh,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT if tiled else ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def invoke(inputs, state, output, resident):
    op.step(*inputs, state, output, value_splits=4, input_buffer_items=2, qk_head_repeat=3, resident_state=resident)


def capture(mesh, inputs, state, output, resident):
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        for values in inputs:
            invoke(values, state, output, resident)
    except BaseException:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.release_trace(mesh, trace)
        raise
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    return trace


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def compare_states(sessions, expected, expected_output):
    actual = [(download(state), download(output)) for state, output in sessions]
    checks = []
    for rank in range(4):
        row = dict(
            rank=rank,
            state_bit_identical=torch.equal(actual[0][0][rank], actual[1][0][rank]),
            output_bit_identical=torch.equal(actual[0][1][rank], actual[1][1][rank]),
            state_dense=accuracy(actual[1][0][rank], expected[rank]),
            output_dense=accuracy(actual[1][1][rank], expected_output[rank]),
            state_sha256=tensor_digest(actual[1][0][rank]),
            output_sha256=tensor_digest(actual[1][1][rank]),
        )
        checks.append(row)
        assert row["state_bit_identical"] and row["output_bit_identical"], row
        assert row["state_dense"]["passed"] and row["output_dense"]["passed"], row
    return checks


def check_inputs(inputs, hosts):
    for device, expected in zip(inputs, hosts):
        for field, tensor in enumerate(device):
            for actual, rank in zip(download(tensor), expected):
                assert torch.equal(actual, rank[field]), "Recurrence modified a read-only input"


def timing_bracket(mesh, inputs, initial, sessions):
    reset = upload(mesh, initial, tiled=True)
    timings, state_hashes, output_hashes = [], [], []
    for resident, label in ((False, "native"), (True, "fused"), (False, "native")):
        state, output = sessions[int(resident)]
        invoke(inputs, state, output, resident)
        trace = capture(mesh, [inputs], state, output, resident)
        try:
            # Capture may execute the call; reset afterwards so every arm
            # has the same initial state and exactly 508 timed/warm steps.
            ttnn.copy(reset, state)
            ttnn.synchronize_device(mesh)
            for _ in range(8):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            samples = []
            for _ in range(5):
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(100):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - start) * 1e6 / 100)
            state_hashes.append([tensor_digest(v) for v in download(state)])
            output_hashes.append([tensor_digest(v) for v in download(output)])
            timings.append(dict(variant=label, traced_call_us=samples))
        finally:
            ttnn.release_trace(mesh, trace)
    assert state_hashes[0] == state_hashes[1] == state_hashes[2], "Timing arms have different recurrent state"
    assert output_hashes[0] == output_hashes[1] == output_hashes[2], "Timing arms have different outputs"
    return dict(
        timings=timings,
        comparison=compare_timings(timings),
        timing_final_states=state_hashes,
        timing_final_outputs=output_hashes,
        identical_steps_per_arm=508,
    )


def run_case(mesh, batch, cycles, report, path):
    heads = batch * 12
    initial = [stimulus(heads, 921000 + batch * 100 + rank)[-1] for rank in range(4)]
    # 64 distinct input sets cycle over 4096 steps in B1 and 64 steps for
    # B16/B32. Independent ranks catch accidental replication/remapping.
    hosts, inputs = [], []
    for step in range(64):
        ranks = []
        for rank in range(4):
            q, k, v, gates, _ = stimulus(heads, 931000 + batch * 10000 + rank * 100 + step)
            q, k = q[: heads // 3], k[: heads // 3]
            if step == 0:
                v = (
                    torch.einsum("hk,hkv->hv", k.repeat_interleave(3, 0), initial[rank] * gates[:, 0, None, None])
                    + 1e-6 * v
                )
            ranks.append((q, k, v, gates))
        hosts.append(ranks)
        inputs.append([upload(mesh, [rank[field] for rank in ranks]) for field in range(4)])
    sessions = [
        (upload(mesh, initial, tiled=True), upload(mesh, [torch.full((heads, 128), float("nan")) for _ in range(4)]))
        for _ in range(2)
    ]
    addresses = [[tensor.buffer_address() for tensor in values] for values in inputs + [list(s) for s in sessions]]
    expected = [state.clone() for state in initial]
    expected_output = [None] * 4
    case = dict(batch=batch, heads_per_rank=heads, cycles=cycles, steps=0, checks=[], passed=False)
    report["cases"].append(case)
    save(path, report)

    # Check the first cancellation-sensitive step before trace construction.
    for rank in range(4):
        q, k, v, gates = hosts[0][rank]
        expected[rank], expected_output[rank] = reference(
            expected[rank], q.repeat_interleave(3, 0), k.repeat_interleave(3, 0), v, gates
        )
    for resident, session in zip((False, True), sessions):
        invoke(inputs[0], *session, resident)
    case["checks"].append(
        dict(steps=1, phase="cancellation", ranks=compare_states(sessions, expected, expected_output))
    )
    save(path, report)

    traces = []
    reset = upload(mesh, initial, tiled=True)
    try:
        for resident, session in zip((False, True), sessions):
            traces.append(capture(mesh, inputs, *session, resident))
        for state, _ in sessions:
            ttnn.copy(reset, state)
        ttnn.synchronize_device(mesh)
        expected = [state.clone() for state in initial]
        for cycle in range(cycles):
            for trace in traces:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for ranks in hosts:
                for rank, (q, k, v, gates) in enumerate(ranks):
                    expected[rank], expected_output[rank] = reference(
                        expected[rank], q.repeat_interleave(3, 0), k.repeat_interleave(3, 0), v, gates
                    )
            case["steps"] = (cycle + 1) * 64
            if cycle == 0 or (cycle + 1) % 4 == 0 or cycle + 1 == cycles:
                case["checks"].append(
                    dict(steps=case["steps"], phase="trace", ranks=compare_states(sessions, expected, expected_output))
                )
                save(path, report)
                print("RESIDENT_GDN_CHECK", batch, case["steps"], "PASS", flush=True)
    finally:
        for trace in traces:
            ttnn.release_trace(mesh, trace)

    # This checks inputs spanning different program-cache bindings, after all
    # trace replays, on all four ranks. No trace-created storage escapes.
    check_inputs(inputs, hosts)
    if batch in (16, 32):
        case.update(timing_bracket(mesh, inputs[0], initial, sessions))
        check_inputs(inputs[:1], hosts[:1])
    assert addresses == [
        [tensor.buffer_address() for tensor in values] for values in inputs + [list(s) for s in sessions]
    ]
    case.update(passed=True, input_and_address_stability=True)
    save(path, report)


@pytest.mark.skipif(os.getenv("QWEN_GDN_RESIDENT") != "1", reason="explicit physical TP4 experiment")
def test_gdn_resident():
    assert not any(
        os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_DISABLE_SFPLOADMACRO", "TT_METAL_SLOW_DISPATCH_MODE")
    )
    path = Path(os.environ["QWEN_GDN_RESIDENT_RECEIPT"])
    assert not path.exists(), "Preserve earlier attempts"
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        promoted_to_serving=False,
        cases=[],
        scope="Physical FP32 recurrence only: B16/B32 64 steps plus B1 4096 steps, four independent ranks",
        comparison="Same shared Q/K, value_splits=4, input_buffer_items=2, unchanged reader/writer and FP32 precision",
        precision="FP32 prepared inputs/state/output; no BF16/BFP4 narrowing",
        performance_scope="Standalone kernel only; projected 48-layer savings are not a full-model measurement",
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(op.HERE.glob("*"))
            if p.suffix in {".py", ".cpp", ".hpp"}
        },
    )
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        mesh.enable_program_cache()
        report["device_ids"] = list(mesh.get_device_ids())
        report["state"] = "running"
        for batch, cycles in ((16, 1), (32, 1), (1, 64)):
            run_case(mesh, batch, cycles, report, path)
            gc.collect()
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        try:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            finally:
                if parent is not None:
                    ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = parent is not None
        finally:
            save(path, report)

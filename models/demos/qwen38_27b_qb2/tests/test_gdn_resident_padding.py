# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Physical TP4 test of unused V rows, preserving 120 recurrence workers."""

import gc
import hashlib
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.gdn_resident_padding import MODES, reader_variant
from models.demos.qwen38_27b_qb2.tests.test_gdn_resident import check_inputs, download, tensor_digest, upload
from models.demos.qwen38_27b_qb2.tests.test_gdn_step_candidate import accuracy, reference, stimulus
from models.demos.qwen38_27b_qb2.tt.gdn_step import op
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


@contextmanager
def reader_mode(mode):
    original = op.kernel_source
    reader = reader_variant(original("reader.cpp"), mode)
    op.kernel_source = lambda name: reader if name == "reader.cpp" else original(name)
    try:
        yield
    finally:
        op.kernel_source = original


def invoke(inputs, state, output):
    op.step(*inputs, state, output, value_splits=4, input_buffer_items=2, qk_head_repeat=3, resident_state=True)


def capture(mesh, inputs, state, output, mode):
    with reader_mode(mode):
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            for values in inputs:
                invoke(values, state, output)
        except BaseException:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.release_trace(mesh, trace)
            raise
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    return trace


def check(sessions, expected, expected_output):
    actual = [(download(state), download(output)) for state, output in sessions]
    rows = []
    for index, mode in enumerate(MODES):
        for rank in range(4):
            state, output = actual[index][0][rank], actual[index][1][rank]
            row = dict(
                mode=mode,
                rank=rank,
                state_exact=torch.equal(state, actual[0][0][rank]),
                output_exact=torch.equal(output, actual[0][1][rank]),
                state_dense=accuracy(state, expected[rank]),
                output_dense=accuracy(output, expected_output[rank]),
                state_sha256=tensor_digest(state),
                output_sha256=tensor_digest(output),
            )
            assert row["state_exact"] and row["output_exact"], row
            assert row["state_dense"]["passed"] and row["output_dense"]["passed"], row
            rows.append(row)
    return rows


def timing_bracket(mesh, inputs, reset, sessions):
    rows = []
    for index, mode, label in (
        (0, "zero", "before"),
        (1, "skip", "skip"),
        (3, "overlap_zero", "overlap_zero"),
        (4, "overlap_skip", "overlap_skip"),
        (0, "zero", "after"),
    ):
        state, output = sessions[index]
        with reader_mode(mode):
            invoke(inputs, state, output)
        trace = capture(mesh, [inputs] * 16, state, output, mode)
        try:
            # Match recurrent histories even if capture executes its programs.
            ttnn.copy(reset, state)
            ttnn.synchronize_device(mesh)
            for _ in range(2):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            samples = []
            for _ in range(5):
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(32):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - start) * 1e6 / (16 * 32))
            rows.append(
                dict(
                    variant=label,
                    call_us=samples,
                    median_us=statistics.median(samples),
                    steps=2592,
                    state=[tensor_digest(v) for v in download(state)],
                    output=[tensor_digest(v) for v in download(output)],
                )
            )
        finally:
            ttnn.release_trace(mesh, trace)
    assert all(r["state"] == rows[0]["state"] and r["output"] == rows[0]["output"] for r in rows)
    control = (rows[0]["median_us"] + rows[-1]["median_us"]) / 2
    drift = abs(rows[0]["median_us"] - rows[-1]["median_us"]) / control
    return dict(
        arms=rows,
        control_us=control,
        comparisons=[
            dict(
                variant=row["variant"],
                saved_us=control - row["median_us"],
                projected_48_layer_saving_ms=(control - row["median_us"]) * 48 / 1000,
            )
            for row in rows[1:-1]
        ],
        control_drift_fraction=drift,
        timing_stable=drift < 0.03,
        projection_is_not_model_measurement=True,
    )


def run_case(mesh, batch, cycles, report, path):
    heads = batch * 12
    initial = [stimulus(heads, 942000 + batch * 100 + rank)[-1] for rank in range(4)]
    hosts, inputs = [], []
    for step in range(64):
        ranks = []
        for rank in range(4):
            q, k, v, gates, _ = stimulus(heads, 951000 + batch * 10000 + rank * 100 + step)
            q, k = q[: heads // 3].contiguous(), k[: heads // 3].contiguous()
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
        for _ in MODES
    ]
    reset = upload(mesh, initial, tiled=True)
    addresses = [tensor.buffer_address() for values in inputs + [list(s) for s in sessions] for tensor in values]
    assert len(set(addresses)) == len(addresses), "Retained allocations must be distinct"
    case = dict(batch=batch, cycles=cycles, steps=0, checks=[], passed=False, addresses=addresses)
    report["cases"].append(case)
    save(path, report)

    expected, expected_output = [], []
    for rank in range(4):
        q, k, v, gates = hosts[0][rank]
        state, output = reference(initial[rank], q.repeat_interleave(3, 0), k.repeat_interleave(3, 0), v, gates)
        expected.append(state)
        expected_output.append(output)
    for mode, session in zip(MODES, sessions):
        with reader_mode(mode):
            invoke(inputs[0], *session)
    case["checks"].append(dict(phase="cancellation", steps=1, ranks=check(sessions, expected, expected_output)))
    save(path, report)

    traces = []
    try:
        for mode, session in zip(MODES, sessions):
            traces.append(capture(mesh, inputs, *session, mode))
        for state, _ in sessions:
            ttnn.copy(reset, state)
        ttnn.synchronize_device(mesh)
        expected = [s.clone() for s in initial]
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
                    dict(phase="trace", steps=case["steps"], ranks=check(sessions, expected, expected_output))
                )
                save(path, report)
                print("GDN_RESIDENT_PADDING", batch, case["steps"], "PASS", flush=True)
    finally:
        for trace in traces:
            ttnn.release_trace(mesh, trace)
    check_inputs(inputs, hosts)
    case["timing"] = timing_bracket(mesh, inputs[1], reset, sessions)
    check_inputs(inputs[:2], hosts[:2])
    assert addresses == [
        tensor.buffer_address() for values in inputs + [list(s) for s in sessions] for tensor in values
    ]
    case.update(passed=True, readonly_inputs=True, stable_addresses=True)
    save(path, report)
    print("GDN_RESIDENT_PADDING_TIMING", batch, case["timing"], flush=True)


@pytest.mark.skipif(os.getenv("QWEN_GDN_RESIDENT_PADDING") != "1", reason="explicit physical TP4 experiment")
def test_gdn_resident_padding():
    assert not any(
        os.getenv(k)
        for k in (
            "TT_METAL_SIMULATOR",
            "TT_METAL_SLOW_DISPATCH_MODE",
            "TT_METAL_KERNEL_PATH",
            "TT_METAL_DISABLE_SFPLOADMACRO",
        )
    )
    path = Path(os.environ["QWEN_GDN_RESIDENT_PADDING_RECEIPT"])
    assert not path.exists(), "Preserve earlier attempts"
    cycles = int(os.getenv("QWEN_GDN_RESIDENT_PADDING_CYCLES", "1"))
    assert cycles in (1, 64), "Bounded 64-step screen or 4096-step qualification only"
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        full_model_qualified=False,
        promoted_to_serving=False,
        cases=[],
        scope="FP32 resident recurrence only; unchanged 120-worker assignment, math, ownership and two-item lookahead",
        reader_sha256={},
    )
    for mode in MODES:
        source = reader_variant(op.kernel_source("reader.cpp"), mode)
        (path.parent / f"reader-{mode}.cpp").write_text(source)
        report["reader_sha256"][mode] = hashlib.sha256(source.encode()).hexdigest()
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        mesh.enable_program_cache()
        report["device_ids"] = list(mesh.get_device_ids())
        assert report["device_ids"] == [0, 4, 12, 8]
        report["state"] = "running"
        for batch in (16, 32):
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

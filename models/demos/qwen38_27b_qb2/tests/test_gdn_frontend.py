# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Physical TP4 comparison of compact convolution/history and GDN preparation."""

import gc
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings
from models.demos.qwen38_27b_qb2.tests.test_decode_conv import first_tp_shard_conv_taps
from models.demos.qwen38_27b_qb2.tests.test_gdn_epilogue import digest, download
from models.demos.qwen38_27b_qb2.tests.test_gdn_flat_prepare import check
from models.demos.qwen38_27b_qb2.tt.decode_conv import make_actual_start, packed_decode_conv
from models.demos.qwen38_27b_qb2.tt.gdn_frontend.op import WIDTHS, convolution
from models.demos.qwen38_27b_qb2.tt.gdn_step.flat_prepare import prepare
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric

CASES = [(b, compact, memory) for b in (16, 32) for compact in (False, True) for memory in ("dram", "l1")]


def native(qkv, history, taps, decay, beta, prepared, actual_start, *, compact_input):
    batch = history.shape[0]
    if compact_input:
        qkv = ttnn.reshape(qkv, [batch, 1, qkv.shape[-1]])
    rows = ttnn.to_layout(qkv[:, :, : sum(WIDTHS)], ttnn.ROW_MAJOR_LAYOUT)
    outputs = packed_decode_conv(rows, history, taps, WIDTHS, actual_start)
    tail = ttnn.concat([history[:, 1:, :], rows], dim=1)
    ttnn.copy(tail, history)
    prepare(*outputs, decay, beta, *prepared)
    return outputs


def assert_history(actual, expected):
    ranks = download(actual)
    assert len(ranks) == 4 and all(torch.equal(value, expected) for value in ranks)
    return [digest(value) for value in ranks]


def run_case(mesh, batch, compact, placement):
    memory = ttnn.DRAM_MEMORY_CONFIG if placement == "dram" else ttnn.L1_MEMORY_CONFIG

    def upload(value, dtype=ttnn.bfloat16, *, row=False, config=None):
        return ttnn.from_torch(
            value.contiguous(),
            device=mesh,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT if row else ttnn.TILE_LAYOUT,
            memory_config=memory if config is None else config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    taps = [upload(value) for value in first_tp_shard_conv_taps(WIDTHS)]
    start = make_actual_start(mesh)
    allocations = []
    report = dict(batch=batch, compact_input=compact, placement=placement, correctness=[], timings=[])
    for index in range(2):
        rng = torch.Generator().manual_seed(810100 + batch + index)
        width = 4160 if compact else sum(WIDTHS)
        host_input = torch.randn(batch, 1, width, generator=rng).bfloat16()
        history_host = torch.randn(batch, 3, sum(WIDTHS), generator=rng).bfloat16()
        qkv = upload(host_input.reshape(1, batch, width) if compact else host_input)
        history, control_history = [upload(history_host, row=True) for _ in range(2)]
        decay = ttnn.exp(upload(-torch.rand(batch, 1, 12, generator=rng), ttnn.float32))
        beta = upload(torch.rand(batch, 1, 12, generator=rng).bfloat16())
        compact_outputs = [upload(torch.full((1, batch, w), float("nan")).bfloat16()) for w in WIDTHS]
        prepared, control_prepared = [
            [
                upload(torch.full(shape, float("nan")), ttnn.float32, row=True, config=ttnn.DRAM_MEMORY_CONFIG)
                for shape in ((batch * 4, 128), (batch * 4, 128), (batch * 12, 128), (batch * 12, 8))
            ]
            for _ in range(2)
        ]
        allocations.append(
            dict(
                qkv=qkv,
                history=history,
                control_history=control_history,
                history_host=history_host,
                host_input=host_input,
                decay=decay,
                beta=beta,
                outputs=compact_outputs,
                prepared=prepared,
                control_prepared=control_prepared,
            )
        )

    def invoke(a, fused):
        if fused:
            convolution(a["qkv"], a["history"], taps, a["outputs"], compact_input=compact)
            prepare(*a["outputs"], a["decay"], a["beta"], *a["prepared"], compact_qkv=True)
            return a["outputs"], a["prepared"], a["history"]
        result = native(
            a["qkv"],
            a["control_history"],
            taps,
            a["decay"],
            a["beta"],
            a["control_prepared"],
            start,
            compact_input=compact,
        )
        return result, a["control_prepared"], a["control_history"]

    def validate(a):
        reference, control_prepared, control_history = invoke(a, False)
        outputs, prepared, history = invoke(a, True)
        expected = [download(t) for t in control_prepared]
        checks = check(prepared, expected)
        conv_checks = []
        for output, control in zip(outputs, reference):
            for value, want in zip(download(output), download(control)):
                assert torch.equal(value[0], want[:, 0]), "Convolution changed BF16 arithmetic"
                conv_checks.append(dict(output_sha256=digest(value[0]), reference_sha256=digest(want[:, 0])))
        assert len(conv_checks) == 12
        a["history_host"] = torch.cat([a["history_host"][:, 1:], a["host_input"][:, :, : sum(WIDTHS)]], dim=1)
        state_hashes = assert_history(history, a["history_host"])
        assert state_hashes == assert_history(control_history, a["history_host"])
        return dict(prepared=checks, convolution=conv_checks, history_sha256=state_hashes)

    # Alternating allocations exercises generic-op cache rebinding; repeated
    # state updates validate chronology and disjoint per-user ownership.
    for index in (0, 1, 0):
        for step in range(4):
            report["correctness"].append(dict(allocation=index, step=step, checks=validate(allocations[index])))
    a = allocations[0]
    original_inputs = [[digest(v) for v in download(t)] for t in (a["qkv"], a["decay"], a["beta"], *taps)]
    original_addresses = [t.buffer_address() for t in (a["history"], *a["outputs"], *a["prepared"])]
    for variant in ("native", "fused", "native"):
        fused = variant == "fused"
        warmup = invoke(a, fused)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            result = invoke(a, fused)
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        try:
            for _ in range(8):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            samples = []
            for _ in range(5):
                ttnn.synchronize_device(mesh)
                tick = time.perf_counter()
                for _ in range(100):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - tick) * 1e6 / 100)
            expected_history = a["host_input"][:, :, : sum(WIDTHS)].repeat(1, 3, 1)
            state_hashes = assert_history(result[2], expected_history)
            report["timings"].append(dict(variant=variant, traced_call_us=samples, history_sha256=state_hashes))
        finally:
            ttnn.release_trace(mesh, trace)
        del warmup, result
    assert original_inputs == [[digest(v) for v in download(t)] for t in (a["qkv"], a["decay"], a["beta"], *taps)]
    assert original_addresses == [t.buffer_address() for t in (a["history"], *a["outputs"], *a["prepared"])]
    # A captured invocation must observe newly copied inputs and live history.
    a["history_host"] = a["host_input"][:, :, : sum(WIDTHS)].repeat(1, 3, 1)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        invoke(a, True)
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        # Capture executes the same saturated input; it does not alter history.
        for key in ("qkv", "decay", "beta"):
            ttnn.copy(allocations[1][key], a[key])
        a["host_input"] = allocations[1]["host_input"]
        reference = invoke(a, False)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        report["changed_input_trace"] = check(a["prepared"], [download(t) for t in reference[1]])
        expected_history = torch.cat([a["history_host"][:, 1:], a["host_input"][:, :, : sum(WIDTHS)]], dim=1)
        assert_history(a["history"], expected_history)
    finally:
        ttnn.release_trace(mesh, trace)
    report.update(passed=True, comparison=compare_timings(report["timings"]))
    return report


@pytest.mark.skipif(os.getenv("QWEN_GDN_FRONTEND") != "1", reason="explicit physical TP4 experiment")
def test_gdn_frontend():
    assert not any(
        os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_DISABLE_SFPLOADMACRO", "TT_METAL_SLOW_DISPATCH_MODE")
    )
    path = Path(os.environ["QWEN_GDN_FRONTEND_RECEIPT"])
    assert not path.exists(), "Preserve each hardware attempt"
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        promoted_to_model=False,
        cases=[],
        scope="Real convolution taps, synthetic activations, convolution/history/preparation only; no model speed claim",
    )
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        for case in CASES:
            report.update(state="running", active_case=case)
            save(path, report)
            report["cases"].append(run_case(mesh, *case))
            print("GDN_FRONTEND", case, report["cases"][-1]["comparison"], flush=True)
            save(path, report)
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

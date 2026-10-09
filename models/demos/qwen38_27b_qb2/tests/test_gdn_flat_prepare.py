# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Physical TP4 screen for direct tiled-input GDN preparation, without weights."""

import gc
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings
from models.demos.qwen38_27b_qb2.tests.test_gdn_epilogue import digest, download
from models.demos.qwen38_27b_qb2.tt.gdn_step.flat_prepare import prepare
from models.demos.qwen38_27b_qb2.tt.gdn_step.shared_qk import prepare as normalize
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric

CASES = [(b, t, p) for b in (32, 16) for t in (1, 32) for p in ("l1", "dram")] + [(1, 32, "dram")]


def native(inputs, outputs, batch):
    def vector(tensor, count):
        row = ttnn.to_layout(tensor[:, :1, :], ttnn.ROW_MAJOR_LAYOUT)
        row = ttnn.reshape(row, [batch, count, 128])
        value = ttnn.typecast(ttnn.to_layout(row, ttnn.TILE_LAYOUT), ttnn.float32)
        value = ttnn.reshape(ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT), [batch * count, 128])
        return ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)

    q, k, v = [vector(value, count) for value, count in zip(inputs[:3], (4, 4, 12))]
    gates = ttnn.concat([inputs[3], ttnn.typecast(inputs[4][:, :1, :], ttnn.float32)], dim=1)
    gates = ttnn.to_layout(ttnn.permute(gates, [0, 2, 1]), ttnn.ROW_MAJOR_LAYOUT)
    gates = ttnn.pad(ttnn.reshape(gates, [batch * 12, 2]), [(0, 0), (0, 6)], 0.0)
    gates = ttnn.to_memory_config(gates, ttnn.DRAM_MEMORY_CONFIG)
    normalize(q, k, *outputs[:2])
    return (q, k), (*outputs[:2], v, gates)


def check(outputs, references):
    rows = []
    for tensor, ranks in zip(outputs, references):
        assert len(ranks) == 4
        for value, expected in zip(download(tensor), ranks):
            rows.append(
                dict(
                    bit_identical=torch.equal(value, expected),
                    finite=bool(torch.isfinite(value).all()),
                    max_abs_error=float((value - expected).abs().max()),
                    output_sha256=digest(value),
                    reference_sha256=digest(expected),
                )
            )
    assert len(rows) == 16 and all(row["bit_identical"] and row["finite"] for row in rows), rows
    return rows


def run_case(mesh, batch, time_rows, placement):
    memory = ttnn.DRAM_MEMORY_CONFIG if placement == "dram" else ttnn.L1_MEMORY_CONFIG

    def upload(value, dtype, *, row=False):
        return ttnn.from_torch(
            value.contiguous(),
            device=mesh,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT if row else ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if row else memory,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    operands, references, keepalive = [], [], []
    for allocation in range(2):
        rng = torch.Generator().manual_seed(610090 + batch * 100 + allocation)
        hosts = [torch.randn(batch, time_rows, width, generator=rng).bfloat16() for width in (512, 512, 1536)]
        for value in hosts[:2]:
            value[0, 0, :128] = 0
            value[0, 0, 128:256] *= 1e-5
        inputs = [upload(value, ttnn.bfloat16) for value in hosts]
        log_decay = upload(-torch.rand(batch, 32, 12, generator=rng), ttnn.float32)
        # Exp remains native and outside both timed preparations.
        decay = ttnn.exp(log_decay[:, :1, :])
        beta = upload(torch.rand(batch, time_rows, 12, generator=rng).bfloat16(), ttnn.bfloat16)
        inputs += [decay, beta]
        shapes = [(batch * 4, 128)] * 2 + [(batch * 12, 128), (batch * 12, 8)]
        outputs = [upload(torch.full(shape, float("nan")), ttnn.float32, row=True) for shape in shapes]
        temporaries, results = native(inputs, outputs, batch)
        expected = [download(tensor) for tensor in results]
        assert all(torch.equal(value, hosts[2][:, 0].reshape(batch * 12, 128).float()) for value in expected[2])
        assert all(torch.count_nonzero(value[:, 2:]).item() == 0 for value in expected[3])
        operands.append((inputs, outputs))
        references.append(expected)
        keepalive.extend((log_decay, *temporaries, *results))
    hashes = [[[digest(value) for value in download(tensor)] for tensor in inputs] for inputs, _ in operands]
    addresses = [[tensor.buffer_address() for tensor in (*inputs, *outputs)] for inputs, outputs in operands]
    report = dict(batch=batch, time_rows=time_rows, placement=placement, correctness=[], timings=[])
    for allocation in (0, 1, 0):
        inputs, outputs = operands[allocation]
        prepare(*inputs, *outputs)
        report["correctness"].append(dict(allocation=allocation, checks=check(outputs, references[allocation])))
    for variant in ("native", "fused", "native"):
        inputs, outputs = operands[0]

        def invoke():
            if variant == "native":
                return native(inputs, outputs, batch)
            prepare(*inputs, *outputs)
            return (), outputs

        warmup = invoke()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            traced = invoke()
        except BaseException:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.release_trace(mesh, trace)
            raise
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        try:
            for _ in range(8):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            check(traced[1], references[0])
            samples = []
            for _ in range(5):
                ttnn.synchronize_device(mesh)
                tick = time.perf_counter()
                for _ in range(100):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - tick) * 1e6 / 100)
            checks = check(traced[1], references[0])
            report["timings"].append(dict(variant=variant, traced_call_us=samples, checks=checks))
        finally:
            ttnn.release_trace(mesh, trace)
        del warmup, traced
    report["comparison"] = compare_timings(report["timings"])
    assert hashes == [[[digest(value) for value in download(t)] for t in inputs] for inputs, _ in operands]
    # Reuse captured addresses with changed inputs; catches stale descriptor/input caching.
    inputs, outputs = operands[0]
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        prepare(*inputs, *outputs)
    except BaseException:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.release_trace(mesh, trace)
        raise
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        for src, dst in zip(operands[1][0], inputs):
            ttnn.copy(src, dst)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        report["changed_input_trace_checks"] = check(outputs, references[1])
    finally:
        ttnn.release_trace(mesh, trace)
    assert addresses == [[t.buffer_address() for t in (*i, *o)] for i, o in operands]
    report.update(passed=True, inputs_unchanged_except_explicit_copy=True, persistent_addresses_unchanged=True)
    return report


@pytest.mark.skipif(os.getenv("QWEN_GDN_FLAT_PREPARE") != "1", reason="explicit physical TP4 experiment")
def test_gdn_flat_prepare():
    assert not any(
        os.getenv(key) for key in ("TT_METAL_SIMULATOR", "TT_METAL_DISABLE_SFPLOADMACRO", "TT_METAL_SLOW_DISPATCH_MODE")
    )
    path = Path(os.environ["QWEN_GDN_FLAT_PREPARE_RECEIPT"])
    assert not path.exists(), "Preserve every hardware attempt"
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        promoted_to_model=False,
        cases=[],
        scope="Synthetic direct preparation only; native exp external; no full-model throughput claim",
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
            print("GDN_FLAT_PREPARE", case, report["cases"][-1]["comparison"], flush=True)
            save(path, report)
            gc.collect()
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", passed=False, error=type(error).__name__, detail=str(error)[:3000])
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

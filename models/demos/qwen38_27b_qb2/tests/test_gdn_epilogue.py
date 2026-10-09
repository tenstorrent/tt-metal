# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Physical TP4 epilogue: native arithmetic, trace replay, rebinding and timing."""

import gc
import hashlib
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import BATCHES, PLACEMENTS, compare_timings
from models.demos.qwen38_27b_qb2.tt.gdn_epilogue.op import epilogue
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


def digest(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def download(tensor):
    ranks = ttnn.get_device_tensors(tensor)
    assert len(ranks) == 4
    return [ttnn.to_torch(rank).float() for rank in ranks]


def check(tensor, references):
    checks = []
    for value, reference in zip(download(tensor), references):
        value, reference = value[:, :1].reshape(-1, 128).double(), reference[:, :1].reshape(-1, 128).double()
        assert torch.isfinite(value).all() and torch.isfinite(reference).all()
        centered = [v - v.mean(-1, keepdim=True) for v in (value, reference)]
        norms = [v.norm(dim=-1) for v in centered]
        pcc = (centered[0] * centered[1]).sum(-1) / (norms[0] * norms[1]).clamp_min(1e-30)
        nonzero = norms[1] > 1e-10
        relative = (value - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-10)
        row = dict(
            min_pcc=float(pcc[nonzero].min()),
            max_relative_rms=float(relative.max()),
            max_abs_error=float((value - reference).abs().max()),
            bit_identical=torch.equal(value, reference),
            output_sha256=digest(value),
            reference_sha256=digest(reference),
        )
        row["passed"] = row["min_pcc"] >= 0.99999 and row["max_relative_rms"] <= 0.001
        checks.append(row)
    assert all(row["passed"] for row in checks), checks
    return checks


def native(inputs, batch, memory):
    raw, gate, weight, _ = inputs
    head_major = ttnn.to_layout(ttnn.reshape(raw, [batch * 12, 1, 128]), ttnn.TILE_LAYOUT, memory_config=memory)
    head_major = ttnn.reshape(head_major, [batch * 12, 32, 128], head_major.padded_shape)
    padded_gate = ttnn.pad(gate, [(0, 0), (0, 31), (0, 0)], 0.0)
    normalized = ttnn.experimental.kda.sigmoid_gated_rms_norm(
        head_major, padded_gate, weight, 12, epsilon=1e-6, output_dtype=ttnn.bfloat16, memory_config=memory
    )
    output = ttnn.mul(normalized, padded_gate, memory_config=memory)
    # Keep all intermediates alive for capture; padded_gate may alias gate.
    return head_major, padded_gate, normalized, output


def run_case(mesh, batch, placement):
    memory = ttnn.DRAM_MEMORY_CONFIG if placement == "dram" else ttnn.L1_MEMORY_CONFIG

    def upload(value, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value.contiguous(),
            device=mesh,
            dtype=dtype,
            layout=layout,
            memory_config=memory,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    inputs, references, normalized_references, keepalive = [], [], [], []
    for allocation in range(2):
        rng = torch.Generator().manual_seed(20261009 + batch * 100 + allocation)
        raw = torch.randn(batch * 12, 128, generator=rng)
        raw[0] = 0
        raw[1] *= 1e-5
        gate = torch.randn(batch, 1, 1536, generator=rng).bfloat16()
        gate.reshape(-1)[:8] = torch.tensor([-30, -10, -1, 0, 1, 10, 30, 0.0001]).bfloat16()
        weight = torch.randn(128, generator=rng).bfloat16()
        values = (
            upload(raw, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
            upload(gate, ttnn.bfloat16),
            upload(weight, ttnn.bfloat16),
            upload(torch.full_like(gate, float("nan")), ttnn.bfloat16),
        )
        outputs = native(values, batch, memory)
        keepalive.extend(outputs)
        references.append(download(outputs[-1]))
        normalized_references.append(download(outputs[-2]))
        inputs.append(values)
    hashes = [[[digest(v) for v in download(t)] for t in values[:3]] for values in inputs]
    addresses = [[t.buffer_address() for t in values] for values in inputs]
    report = dict(batch=batch, placement=placement, input_sha256=hashes, cases=[], timings=[])
    for allocation in (0, 1, 0):
        values = inputs[allocation]
        epilogue(*values, multiply_z=False)
        normalization = check(values[-1], normalized_references[allocation])
        epilogue(*values)
        checks = check(values[-1], references[allocation])
        padded = ttnn.reshape(values[-1], [batch, 32, 1536], values[-1].padded_shape)
        assert all(torch.count_nonzero(rank[:, 1:]).item() == 0 for rank in download(padded))
        report["cases"].append(dict(allocation=allocation, ranks=checks, norm_ranks=normalization, padding_zero=True))
    assert hashes == [[[digest(v) for v in download(t)] for t in values[:3]] for values in inputs]

    for variant in ("native", "fused", "native"):

        def invoke():
            if variant == "native":
                return native(inputs[0], batch, memory)
            epilogue(*inputs[0])
            return (inputs[0][-1],)

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
            check(traced[-1], references[0])
            samples = []
            for _ in range(5):
                ttnn.synchronize_device(mesh)
                tick = time.perf_counter()
                for _ in range(100):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - tick) * 1e6 / 100)
            check(traced[-1], references[0])
            report["timings"].append(dict(variant=variant, traced_call_us=samples))
        finally:
            ttnn.release_trace(mesh, trace)
        del warmup, traced
    report["comparison"] = compare_timings(report["timings"])
    # Capture once, then change values at existing addresses before replay.
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        epilogue(*inputs[0])
    except BaseException:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.release_trace(mesh, trace)
        raise
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        for src, dst in zip(inputs[1][:3], inputs[0][:3]):
            ttnn.copy(src, dst)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        report["changed_input_trace_ranks"] = check(inputs[0][-1], references[1])
        assert addresses == [[t.buffer_address() for t in values] for values in inputs]
        assert hashes[1] == [[digest(v) for v in download(t)] for t in inputs[1][:3]]
        assert hashes[1] == [[digest(v) for v in download(t)] for t in inputs[0][:3]]
    finally:
        ttnn.release_trace(mesh, trace)
    report.update(passed=True, inputs_unchanged_except_explicit_copy=True, persistent_addresses_unchanged=True)
    return report


@pytest.mark.skipif(os.getenv("QWEN_GDN_EPILOGUE") != "1", reason="explicit physical TP4 experiment")
def test_gdn_epilogue():
    assert not any(
        os.getenv(key) for key in ("TT_METAL_SIMULATOR", "TT_METAL_DISABLE_SFPLOADMACRO", "TT_METAL_SLOW_DISPATCH_MODE")
    )
    path = Path(os.environ["QWEN_GDN_EPILOGUE_RECEIPT"])
    assert not path.exists(), "Preserve each experimental attempt"
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        promoted_to_model=False,
        scope="Synthetic TP4 epilogue only; no full-model or context-length throughput claim",
        cases=[],
    )
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        for placement in PLACEMENTS:
            for batch in BATCHES:
                report.update(state="running", active_case=[placement, batch])
                save(path, report)
                report["cases"].append(run_case(mesh, batch, placement))
                print("GDN_EPILOGUE", report["cases"][-1]["comparison"], flush=True)
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

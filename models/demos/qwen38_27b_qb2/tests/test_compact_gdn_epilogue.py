# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact/packed gate and output validation on all four physical TP ranks."""

import gc
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.compact_gdn import CASES
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings
from models.demos.qwen38_27b_qb2.tests.test_gdn_epilogue import digest, download, native
from models.demos.qwen38_27b_qb2.tests.test_gdn_layer_integration import capture
from models.demos.qwen38_27b_qb2.tt.gdn_epilogue.op import epilogue
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


def run_case(mesh, batch, placement, mode):
    compact_gate, compact_output, offset = mode
    memory = ttnn.L1_MEMORY_CONFIG if placement == "l1" else ttnn.DRAM_MEMORY_CONFIG
    options = dict(compact_gate=compact_gate, compact_output=compact_output, gate_offset=offset)

    def upload(values, dtype=ttnn.bfloat16, *, row=False):
        return ttnn.from_torch(
            torch.cat(values, dim=0).contiguous(),
            device=mesh,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT if row else ttnn.TILE_LAYOUT,
            memory_config=memory,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )

    def blank(compact):
        shape = (1 if compact else batch, 32, 1536)
        full = upload([torch.full(shape, float("nan")) for _ in range(4)])
        shape = (1, batch, 1536) if compact else (batch, 1, 1536)
        result = ttnn.reshape(full, shape, full.padded_shape)
        assert result.buffer_address() == full.buffer_address()
        return result

    allocations, references, public_references, norm_references = [], [], [], []
    for allocation in range(3):  # Independent A/B/A-copy owners for replay.
        raws, gates, weights, packed = [], [], [], []
        for rank in range(4):
            rng = torch.Generator().manual_seed(190020 + batch * 100 + rank * 1000 + (allocation % 2))
            raw = torch.randn(batch * 12, 128, generator=rng)
            raw[0] = 0
            raw[1] *= 1e-5
            gate = torch.randn(batch, 1, 1536, generator=rng).bfloat16()
            gate.flatten()[:8] = torch.tensor([-30, -10, -1, 0, 1, 10, 30, 0.0001]).bfloat16()
            raws.append(raw)
            gates.append(gate)
            weights.append(torch.randn(128, generator=rng).bfloat16())
            if compact_gate:
                # Non-gate channels and inactive users are poisoned. A wrong
                # offset/row must fail numerically instead of reading benign zeros.
                width = 4160 if offset else 1536
                value = torch.full((1, 32, width), float("nan"), dtype=torch.bfloat16)
                value[:, :batch, offset : offset + 1536] = gate.reshape(1, batch, 1536)
                packed.append(value)
        raw, public_gate, weight = upload(raws, ttnn.float32, row=True), upload(gates), upload(weights)
        if compact_gate:
            gate_full = upload(packed)
            gate = ttnn.reshape(gate_full, (1, batch, gate_full.shape[-1]), gate_full.padded_shape)
            assert gate.buffer_address() == gate_full.buffer_address()
        else:
            gate = public_gate
        output, control = blank(compact_output), blank(False)
        controls = (raw, public_gate, weight, control)
        native_outputs = native(controls, batch, memory)
        references.append([v[:, :1].reshape(batch, 1536) for v in download(native_outputs[-1])])
        epilogue(*controls)
        public_references.append([v.reshape(batch, 1536) for v in download(control)])
        epilogue(*controls, multiply_z=False)
        norm_references.append([v.reshape(batch, 1536) for v in download(control)])
        allocations.append(dict(values=(raw, gate, weight, output), controls=controls))
        del native_outputs

    def validate(tensor, index, *, norm=False):
        actual = [v.reshape(batch, 1536) for v in download(tensor)]
        expected = norm_references[index] if norm else public_references[index]
        assert all(torch.isfinite(v).all() for v in actual)
        assert all(
            torch.equal(v, w) for v, w in zip(actual, expected)
        ), "Compact path changed public epilogue arithmetic"
        if not norm:
            for value, want in zip(actual, references[index]):
                error = (value - want).double().norm() / want.double().norm().clamp_min(1e-20)
                assert error <= 0.001, float(error)
        shape = (1 if compact_output else batch, 32, 1536)
        full = ttnn.reshape(tensor, shape, tensor.padded_shape)
        for value in download(full):
            padding = value[:, batch:] if compact_output else value[:, 1:]
            assert torch.count_nonzero(padding).item() == 0, "Output padding retained sentinel or another user"
        return [digest(v) for v in actual]

    snapshots = [[[digest(v) for v in download(t)] for t in a["values"][:3]] for a in allocations]
    addresses = [[t.buffer_address() for t in a["values"]] for a in allocations]
    checks = []
    for index in (0, 1, 0):
        values = allocations[index]["values"]
        epilogue(*values, **options, multiply_z=False)
        norm = validate(values[-1], index, norm=True)
        epilogue(*values, **options)
        checks.append(dict(allocation=index, norm_sha256=norm, output_sha256=validate(values[-1], index)))
    assert snapshots == [[[digest(v) for v in download(t)] for t in a["values"][:3]] for a in allocations]
    timings = []
    for variant in ("native", "fused", "native"):
        a = allocations[0]

        def invoke():
            if variant == "native":
                epilogue(*a["controls"])
                return a["controls"][-1]
            epilogue(*a["values"], **options)
            return a["values"][-1]

        invoke()
        trace, result = capture(mesh, invoke)
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
            assert all(torch.equal(v.reshape(batch, 1536), w) for v, w in zip(download(result), public_references[0]))
            timings.append(dict(variant=variant, traced_call_us=samples))
        finally:
            ttnn.release_trace(mesh, trace)

    def replay_call():
        epilogue(*allocations[0]["values"], **options)
        return allocations[0]["values"][-1]

    trace, result = capture(mesh, replay_call)
    replay_hashes = []
    try:
        for index in (2, 1, 2):
            for src, dst in zip(allocations[index]["values"][:3], allocations[0]["values"][:3]):
                ttnn.copy(src, dst)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            replay_hashes.append(validate(result, index))
        assert replay_hashes[0] == replay_hashes[2] and replay_hashes[0] != replay_hashes[1]
    finally:
        ttnn.release_trace(mesh, trace)
    assert addresses == [[t.buffer_address() for t in a["values"]] for a in allocations]
    assert snapshots == [[[digest(v) for v in download(t)] for t in a["values"][:3]] for a in allocations]
    return dict(
        batch=batch,
        placement=placement,
        mode=list(mode),
        checks=checks,
        passed=True,
        changed_input_trace=True,
        input_and_address_stability=True,
        timings=timings,
        comparison=compare_timings(timings),
        kernel_only=True,
    )


@pytest.mark.skipif(os.getenv("QWEN_COMPACT_EPILOGUE") != "1", reason="explicit physical TP4 experiment")
def test_compact_gdn_epilogue():
    assert not any(
        os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_DISABLE_SFPLOADMACRO", "TT_METAL_SLOW_DISPATCH_MODE")
    )
    path = Path(os.environ["QWEN_COMPACT_EPILOGUE_RECEIPT"])
    assert not path.exists()
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        cases=[],
        promoted_to_serving=False,
        scope="26 TP4 layout cases; kernel timing only, not complete GDN or full-model gain",
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

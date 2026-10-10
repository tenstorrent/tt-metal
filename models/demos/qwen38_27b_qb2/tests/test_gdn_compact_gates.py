# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact native-math gate packing screen and real-weight recurrent comparison."""

import gc
import os
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.gdn_compact_gates import CASES, validate_report
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings
from models.demos.qwen38_27b_qb2.tests.test_gdn_epilogue import digest, download
from models.demos.qwen38_27b_qb2.tests.test_gdn_layer_integration import capture
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder
from models.demos.qwen38_27b_qb2.tt.gdn_step.flat_prepare import prepare
from models.demos.qwen38_27b_qb2.tt.gdn_step.gates import from_packed
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


def snapshots(tensors):
    result = []
    for tensor in tensors:
        ranks = download(tensor)
        assert len(ranks) == 4 and all(torch.isfinite(v).all() for v in ranks)
        result.append([digest(v) for v in ranks])
    return result


def run_case(mesh, batch, placement):
    memory = ttnn.L1_MEMORY_CONFIG if placement == "l1" else ttnn.DRAM_MEMORY_CONFIG

    def upload(ranks, *, fp32=False, row=False):
        return ttnn.from_torch(
            torch.cat(ranks, dim=0).contiguous(),
            device=mesh,
            dtype=ttnn.float32 if fp32 else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if row else ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if row else memory,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )

    slots = []
    shapes = ((batch * 4, 128), (batch * 4, 128), (batch * 12, 128), (batch * 12, 8))
    for allocation in (0, 1, 0):
        operands = [[] for _ in range(6)]
        for rank in range(4):
            rng = torch.Generator().manual_seed(187420 + batch * 100 + rank * 10000 + allocation)
            packed = torch.randn(1, batch, 4160, generator=rng).bfloat16()
            # Extreme gates and distinct users/ranks expose dropped rows and offsets.
            packed[0, 0, 4096:4104] = torch.tensor([-30, -10, -1, 0, 1, 10, 30, 0.0001]).bfloat16()
            packed[0, 0, 4128:4136] = torch.tensor([-40, -20, -1, 0, 1, 19.5, 20.5, 40]).bfloat16()
            vals = [
                packed,
                *[torch.randn(1, batch, w, generator=rng).bfloat16() for w in (512, 512, 1536)],
                -torch.rand(1, 1, 12, generator=rng) * 2 - 0.1,
                torch.randn(1, 1, 12, generator=rng),
            ]
            for target, value in zip(operands, vals):
                target.append(value)
        inputs = [upload(values, fp32=i >= 4) for i, values in enumerate(operands)]
        outputs = [
            [upload([torch.zeros(shape) for _ in range(4)], fp32=True, row=True) for shape in shapes] for _ in range(2)
        ]
        slots.append((inputs, outputs))
    initial = [snapshots(i) for i, _ in slots]
    addresses = [[t.buffer_address() for t in (*i, *o[0], *o[1])] for i, o in slots]

    def invoke(slot, compact):
        inputs, outputs = slot
        packed, q, k, v, a, bias = inputs
        g, beta = from_packed(packed, a, bias, compact=compact)
        decay = ttnn.exp(g)
        result = outputs[int(compact)]
        prepare(q, k, v, decay, beta, *result, compact_qkv=True, compact_gates=compact)
        return result

    checks = []
    for index in (0, 1, 0):
        control = snapshots(invoke(slots[index], False))
        candidate = snapshots(invoke(slots[index], True))
        assert candidate == control, f"Compact gate arithmetic or delivery differs at B{batch}/{placement}"
        checks.append(dict(allocation=index, prepared_sha256=candidate, bit_identical=True, finite=True))
    timings = []
    for variant in ("native", "fused", "native"):
        compact = variant == "fused"
        invoke(slots[0], compact)
        trace, result = capture(mesh, lambda: invoke(slots[0], compact))
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
            assert snapshots(result) == checks[0]["prepared_sha256"]
            timings.append(dict(variant=variant, traced_call_us=samples))
        finally:
            ttnn.release_trace(mesh, trace)
    trace, result = capture(mesh, lambda: invoke(slots[0], True))
    try:
        for index in (1, 2):
            for src, dst in zip(slots[index][0], slots[0][0]):
                ttnn.copy(src, dst)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            assert snapshots(result) == checks[1 if index == 1 else 0]["prepared_sha256"]
    finally:
        ttnn.release_trace(mesh, trace)
    assert initial == [snapshots(i) for i, _ in slots]
    assert addresses == [[t.buffer_address() for t in (*i, *o[0], *o[1])] for i, o in slots]
    return dict(
        batch=batch,
        placement=placement,
        checks=checks,
        timings=timings,
        comparison=compare_timings(timings),
        input_immutability=True,
        stable_addresses=True,
        changed_input_trace=True,
    )


def run_layer(layer, mesh, batch):
    rng = torch.Generator().manual_seed(531700 + batch)
    sources = [
        ttnn.from_torch(
            torch.randn(1, 1, batch, 5120, generator=rng).bfloat16(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for _ in range(4)
    ]
    slots, traces = [], []
    try:
        for compact in (False, True):
            state = layer.allocate_state(batch_size=batch)
            slots.append(
                dict(
                    compact=compact,
                    state=state,
                    x=ttnn.clone(sources[0]),
                    zeros=[ttnn.zeros_like(t) for t in (state.recurrent, state.conv)],
                )
            )
        for slot in slots:
            layer.policy["compact_gdn_gates"] = slot["compact"]
            layer._delta(slot["x"], slot["state"], decode=True)
            for zero, dest in zip(slot["zeros"], (slot["state"].recurrent, slot["state"].conv)):
                ttnn.copy(zero, dest)
        for slot in slots:
            layer.policy["compact_gdn_gates"] = slot["compact"]
            trace, output = capture(mesh, lambda: layer._delta(slot["x"], slot["state"], decode=True))
            traces.append(trace)
            slot["output"] = output
        for slot in slots:
            for zero, dest in zip(slot["zeros"], (slot["state"].recurrent, slot["state"].conv)):
                ttnn.copy(zero, dest)
        checks = []
        for step in range(64):
            source = sources[(step * 7 + step // 3) % 4]
            observed = []
            for slot, trace in zip(slots, traces):
                ttnn.copy(source, slot["x"])
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                if step + 1 in (1, 2, 4, 8, 16, 32, 64):
                    state = slot["state"]
                    observed.append(snapshots([state.recurrent, state.conv, slot["output"]]))
            if observed:
                assert observed[0] == observed[1], f"Real-weight compact gates differ at B{batch}, step {step+1}"
                checks.append(
                    dict(step=step + 1, state_history_output_sha256=observed[0], bit_identical=True, finite=True)
                )
        return dict(batch=batch, checkpoints=checks, real_weights=True, persistent_sessions_precede_trace_capture=True)
    finally:
        for trace in traces:
            ttnn.release_trace(mesh, trace)
        layer.policy["compact_gdn_gates"] = False


@pytest.mark.skipif(os.getenv("QWEN_GDN_COMPACT_GATES") != "1", reason="explicit allocated Galaxy experiment")
def test_gdn_compact_gates():
    assert not any(
        os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_SLOW_DISPATCH_MODE", "TT_METAL_DISABLE_SFPLOADMACRO")
    )
    path = Path(os.environ["QWEN_GDN_COMPACT_GATES_RECEIPT"])
    assert not path.exists()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint_path()
    precision = load_precision(root / "config/precision_single_step_compact_gdn_bfp8_all.json")
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        cases=[],
        layers=[],
        source_sha256=model_source_hashes(root),
        precision=precision,
        checkpoint=str(checkpoint),
        scope="Exact compact gate/preparation boundary plus real-weight 64-step recurrence; not full-model or GPQA qualification",
    )
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        for case in CASES:
            report.update(state="gate_preparation", active_case=case)
            save(path, report)
            report["cases"].append(run_case(mesh, *case))
            save(path, report)
            gc.collect()
        layer = Qwen38TPDecoder.from_state_dict(
            Checkpoint(checkpoint).layer(0),
            hf_config=AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config,
            layer_idx=0,
            mesh_device=mesh,
            policy={**decoder_policy(precision, 0), "ring": False, "compact_decode_residual": True},
        )
        setup = layer.allocate_state(batch_size=32)
        del setup
        for batch in (16, 32):
            report.update(state="real_weight", active_batch=batch)
            save(path, report)
            report["layers"].append(run_layer(layer, mesh, batch))
            save(path, report)
            gc.collect()
        report.update(state="completed")
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
            if report["state"] == "completed":
                report.update(validation=validate_report(report), passed=True)
        finally:
            save(path, report)

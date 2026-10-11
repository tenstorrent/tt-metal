# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded native smoke for fused recurrence/epilogue, not model qualification."""

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.test_gdn_resident import download, tensor_digest, upload
from models.demos.qwen38_27b_qb2.tests.test_gdn_step_candidate import stimulus
from models.demos.qwen38_27b_qb2.tt.gdn_epilogue.op import epilogue
from models.demos.qwen38_27b_qb2.tt.gdn_fused.op import fused_step
from models.demos.qwen38_27b_qb2.tt.gdn_step.op import step
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


@pytest.mark.skipif(os.getenv("QWEN_GDN_FUSED_SMOKE") != "1", reason="explicit physical fused-GDN experiment")
def test_gdn_fused_smoke():
    assert not any(os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_SLOW_DISPATCH_MODE", "TT_METAL_KERNEL_PATH"))
    path = Path(os.environ["QWEN_GDN_FUSED_RECEIPT"])
    assert not path.exists()
    torch.set_num_threads(8)
    report = dict(state="opening", passed=False, cleanup_completed=False, cases=[], full_model_qualified=False)
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=0)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        mesh.enable_program_cache()
        report["device_ids"] = list(mesh.get_device_ids())
        assert report["device_ids"] == [0, 4, 12, 8]

        def tiled(values):
            return ttnn.from_torch(
                torch.cat(values, dim=0).contiguous(),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )

        def hashes(tensor):
            return [tensor_digest(v) for v in download(tensor)]

        for batch in (16, 32):
            # Two independent allocations exercise cached-descriptor rebinding;
            # a third instance omits diagnostic raw DRAM writes.
            for instance, capture_raw in enumerate((True, True, False)):
                hosts = [stimulus(batch * 12, 82400 + batch * 100 + rank + instance * 1000) for rank in range(4)]
                host_inputs = [
                    (q[: batch * 4].contiguous(), k[: batch * 4].contiguous(), v, gates) for q, k, v, gates, _ in hosts
                ]
                inputs = [upload(mesh, [row[i] for row in host_inputs]) for i in range(4)]
                state = [upload(mesh, [h[-1] for h in hosts], tiled=True) for _ in range(2)]
                raw = [upload(mesh, [torch.full((batch * 12, 128), float("nan")) for _ in range(4)]) for _ in range(2)]
                packed, weights = [], []
                for rank in range(4):
                    rng = torch.Generator().manual_seed(58291 + rank + instance * 100)
                    value = torch.full((1, 32, 4160), float("nan"), dtype=torch.bfloat16)
                    value[:, :batch, 2560:4096] = torch.randn(1, batch, 1536, generator=rng).bfloat16()
                    packed.append(value)
                    weights.append(torch.randn(128, generator=rng).bfloat16())
                gate_full, weight = tiled(packed), tiled(weights)
                gate = ttnn.reshape(gate_full, (1, batch, 4160), gate_full.padded_shape)
                output_full = [tiled([torch.full((1, 32, 1536), float("nan")) for _ in range(4)]) for _ in range(2)]
                output = [ttnn.reshape(t, (1, batch, 1536), t.padded_shape) for t in output_full]
                readonly = [*inputs, gate, weight]
                before = [hashes(t) for t in readonly]
                raw_poison = hashes(raw[1])
                case = dict(batch=batch, instance=instance, capture_raw=capture_raw, steps=[], passed=False)
                report["cases"].append(case)
                report["state"] = "running"
                save(path, report)
                for repeat in range(4):
                    step(
                        *inputs,
                        state[0],
                        raw[0],
                        value_splits=4,
                        input_buffer_items=2,
                        qk_head_repeat=3,
                        resident_state=True,
                    )
                    epilogue(
                        raw[0],
                        gate,
                        weight,
                        output[0],
                        compact_gate=True,
                        compact_output=True,
                        gate_offset=2560,
                        input_padding="skip",
                    )
                    fused_step(*inputs, state[1], raw[1], gate, weight, output[1], capture_raw=capture_raw)
                    ttnn.synchronize_device(mesh)
                    checks = dict(
                        state=hashes(state[0]) == hashes(state[1]), output=hashes(output[0]) == hashes(output[1])
                    )
                    checks["raw"] = hashes(raw[1]) == (hashes(raw[0]) if capture_raw else raw_poison)
                    assert all(checks.values()), checks
                    assert all(torch.isfinite(v).all() for v in download(output[1]))
                    padded = ttnn.reshape(output[1], (1, 32, 1536), output[1].padded_shape)
                    assert all(torch.count_nonzero(v[:, batch:]).item() == 0 for v in download(padded))
                    case["steps"].append(dict(repeat=repeat, **checks))
                    save(path, report)
                assert before == [hashes(t) for t in readonly]
                case["passed"] = True
                save(path, report)
                for tensor in (*inputs, *state, *raw, gate, weight, *output):
                    ttnn.deallocate(tensor)
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
                if parent is not None:
                    ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = parent is not None
        finally:
            save(path, report)

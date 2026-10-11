# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded phase attribution of current resident recurrence and packed-L1 epilogue."""

import hashlib
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.gdn_phase_profile import instrument
from models.demos.qwen38_27b_qb2.tests.test_gdn_resident import download, tensor_digest, upload
from models.demos.qwen38_27b_qb2.tests.test_gdn_step_candidate import accuracy, reference, stimulus
from models.demos.qwen38_27b_qb2.tt.gdn_epilogue import op as epilogue
from models.demos.qwen38_27b_qb2.tt.gdn_step import op as recurrence
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


@pytest.mark.skipif(os.getenv("QWEN_GDN_PIPELINE_PROFILE") != "1", reason="explicit allocated-Galaxy diagnostic")
def test_gdn_pipeline_profile():
    import tracy

    path = Path(os.environ["QWEN_GDN_PHASE_RECEIPT"])
    assert not path.exists(), "Preserve prior diagnostic attempts"
    torch.set_num_threads(8)
    original_recurrence, original_epilogue = recurrence.kernel_source, epilogue.source
    sources = {name: original_recurrence(name) for name in ("reader.cpp", "writer.cpp", "compute_resident.cpp")}
    sources.update(
        {"epilogue/" + name: original_epilogue(name) for name in ("reader.cpp", "writer.cpp", "compute.cpp")}
    )
    annotated = {name: instrument(name, text) for name, text in sources.items()}
    for name, text in annotated.items():
        (path.parent / ("instrumented-" + name.replace("/", "-"))).write_text(text)
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        scope="Synthetic current resident recurrence plus packed-L1 epilogue; no full-model throughput claim",
        source_sha256={name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()},
        instrumented_sha256={name: hashlib.sha256(text.encode()).hexdigest() for name, text in annotated.items()},
        counter_mask=int(os.getenv("TT_METAL_PROFILE_PERF_COUNTERS", "0")),
        cases=[],
        max_kernel_calls=48,
        kernel_calls=0,
    )
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=0)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())

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
            return [tensor_digest(value) for value in download(tensor)]

        for batch in (16, 32):
            hosts = [stimulus(batch * 12, 20261011 + batch * 100 + rank) for rank in range(4)]
            host_inputs = [
                (q[: batch * 4].contiguous(), k[: batch * 4].contiguous(), v, gates) for q, k, v, gates, _ in hosts
            ]
            packed, weights = [], []
            for rank in range(4):
                rng = torch.Generator().manual_seed(81200 + batch * 10 + rank)
                value = torch.full((1, 32, 4160), float("nan"), dtype=torch.bfloat16)
                value[:, :batch, 2560:4096] = torch.randn(1, batch, 1536, generator=rng).bfloat16()
                packed.append(value)
                weights.append(torch.randn(128, generator=rng).bfloat16())
            padding_controls = []
            for padding in ("zero", "skip"):
                controls = []
                for profiled in (False, True):
                    recurrence.kernel_source = (lambda name: annotated[name]) if profiled else original_recurrence
                    epilogue.source = (lambda name: annotated["epilogue/" + name]) if profiled else original_epilogue
                    inputs = [upload(mesh, [row[field] for row in host_inputs]) for field in range(4)]
                    state = upload(mesh, [row[-1] for row in hosts], tiled=True)
                    raw = upload(mesh, [torch.full((batch * 12, 128), float("nan")) for _ in range(4)])
                    gate_full, weight = tiled(packed), tiled(weights)
                    gate = ttnn.reshape(gate_full, (1, batch, 4160), gate_full.padded_shape)
                    out_full = tiled([torch.full((1, 32, 1536), float("nan")) for _ in range(4)])
                    output = ttnn.reshape(out_full, (1, batch, 1536), out_full.padded_shape)
                    readonly = [*inputs, gate, weight]
                    before = [hashes(tensor) for tensor in readonly]
                    expected = [row[-1].clone() for row in hosts]
                    expected_output = [None] * 4
                    for step in range(3):
                        label = f"GDN_PIPELINE_B{batch}_{padding.upper()}_ZONES{int(profiled)}_STEP{step}"
                        tracy.signpost(label + "_BEGIN")
                        recurrence.step(
                            *inputs,
                            state,
                            raw,
                            value_splits=4,
                            input_buffer_items=2,
                            qk_head_repeat=3,
                            resident_state=True,
                        )
                        epilogue.epilogue(
                            raw,
                            gate,
                            weight,
                            output,
                            compact_gate=True,
                            compact_output=True,
                            gate_offset=2560,
                            input_padding=padding,
                        )
                        ttnn.synchronize_device(mesh)
                        tracy.signpost(label + "_END")
                        report["kernel_calls"] += 2
                        for rank, (q, k, v, gates) in enumerate(host_inputs):
                            expected[rank], expected_output[rank] = reference(
                                expected[rank], q.repeat_interleave(3, 0), k.repeat_interleave(3, 0), v, gates
                            )
                        ttnn.ReadDeviceProfiler(mesh)
                    checks = []
                    for rank, (actual_state, actual_raw, actual_out) in enumerate(
                        zip(download(state), download(raw), download(output))
                    ):
                        row = dict(
                            rank=rank,
                            state=accuracy(actual_state, expected[rank]),
                            raw_output=accuracy(actual_raw, expected_output[rank]),
                            finite_output=bool(torch.isfinite(actual_out).all()),
                        )
                        assert row["state"]["passed"] and row["raw_output"]["passed"] and row["finite_output"], row
                        checks.append(row)
                    assert before == [hashes(tensor) for tensor in readonly], "Diagnostic changed read-only inputs"
                    padded = ttnn.reshape(output, (1, 32, 1536), output.padded_shape)
                    assert all(torch.count_nonzero(value[:, batch:]).item() == 0 for value in download(padded))
                    result = dict(state=hashes(state), raw_output=hashes(raw), output=hashes(output))
                    controls.append(result)
                    if profiled:
                        assert controls[0] == result, "Timing zones changed state or output"
                    report["cases"].append(
                        dict(batch=batch, padding=padding, profiled=profiled, checks=checks, hashes=result)
                    )
                    save(path, report)
                    for tensor in (*inputs, state, raw, gate, weight, output):
                        ttnn.deallocate(tensor)
                    ttnn.ReadDeviceProfiler(mesh)
                padding_controls.append(controls)
            assert padding_controls[0] == padding_controls[1], "Padding variants changed live values"
        assert report["kernel_calls"] == 48
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", passed=False, error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        recurrence.kernel_source, epilogue.source = original_recurrence, original_epilogue
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

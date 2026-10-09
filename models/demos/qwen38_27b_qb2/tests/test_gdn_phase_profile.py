# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded GDN wait/transfer/math attribution with one versus two input buffers."""

import hashlib
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.gdn_phase_profile import instrument
from models.demos.qwen38_27b_qb2.tests.test_gdn_step_candidate import check, reference, stimulus, upload
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt.gdn_step import op
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


@pytest.mark.skipif(os.getenv("QWEN_GDN_PHASE_PROFILE") != "1", reason="explicit allocated-Galaxy diagnostic")
def test_gdn_phase_profile():
    import tracy

    path = Path(os.environ["QWEN_GDN_PHASE_RECEIPT"])
    assert not path.exists(), "Preserve every profiler attempt"
    torch.set_num_threads(8)
    original = op.kernel_source
    sources = {name: original(name) for name in ("reader.cpp", "writer.cpp", "compute.cpp")}
    annotated = {name: instrument(name, text) for name, text in sources.items()}
    for name, text in annotated.items():
        (path.parent / ("instrumented-" + name)).write_text(text)
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        scope="Synthetic shared-Q/K recurrence only; diagnostic instrumentation overhead, not serving performance",
        source_sha256={name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()},
        instrumented_sha256={name: hashlib.sha256(text.encode()).hexdigest() for name, text in annotated.items()},
        cases=[],
        max_kernel_calls=24,
        kernel_calls=0,
    )
    save(path, report)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=0)
    mesh = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        for batch in (32, 16):
            values = list(stimulus(batch * 12, 20261009 + batch))
            q, k = [value[: batch * 4].contiguous() for value in values[:2]]
            reference_values = [q.repeat_interleave(3, 0), k.repeat_interleave(3, 0), *values[2:]]
            for buffers in (1, 2):
                controls = []
                for profiled in (False, True):
                    op.kernel_source = (lambda name: annotated[name]) if profiled else original
                    inputs = [upload(mesh, value, tiled=index == 4) for index, value in enumerate([q, k, *values[2:]])]
                    output = upload(mesh, torch.full_like(values[0], float("nan")))
                    expected = values[4].clone()
                    for step in range(3):
                        label = f"GDN_PHASE_B{batch}_BUFFERS{buffers}_ZONES{int(profiled)}_STEP{step}"
                        tracy.signpost(label + "_BEGIN")
                        op.step(*inputs, output, value_splits=4, input_buffer_items=buffers, qk_head_repeat=3)
                        ttnn.synchronize_device(mesh)
                        tracy.signpost(label + "_END")
                        report["kernel_calls"] += 1
                        expected, expected_output = reference(expected, *reference_values[:4])
                        ttnn.ReadDeviceProfiler(mesh)
                    checks = dict(state=check(inputs[4], expected), output=check(output, expected_output))

                    def digest(tensor):
                        return [
                            hashlib.sha256(
                                ttnn.to_torch(rank).contiguous().view(torch.uint8).numpy().tobytes()
                            ).hexdigest()
                            for rank in ttnn.get_device_tensors(tensor)
                        ]

                    hashes = dict(state=digest(inputs[4]), output=digest(output))
                    controls.append(hashes)
                    if profiled:
                        assert controls[0] == controls[1], "Timing zones changed recurrence output/state"
                    report["cases"].append(
                        dict(batch=batch, buffers=buffers, profiled=profiled, checks=checks, hashes=hashes)
                    )
                    save(path, report)
                    for tensor in (*inputs, output):
                        ttnn.deallocate(tensor)
                    ttnn.ReadDeviceProfiler(mesh)
        assert report["kernel_calls"] == 24
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", passed=False, error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        op.kernel_source = original
        try:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            finally:
                ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = True
        finally:
            save(path, report)

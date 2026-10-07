# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-weight P0 trace: GDN/GQA at 8K, 128K and near-256K.

The eager diagnostic window attributes device operations to stages. Its host
duration includes profiling and Python dispatch, so it is not traced-model TPOT.
"""

import functools
import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.layer_profile_report import PROFILE_CASES
from models.demos.qwen38_27b_qb2.tests.test_galaxy_perf_sweep import drop_request_buffers, run_batch
from models.demos.qwen38_27b_qb2.tt.generator import build_generator, configure_fabric


def mark_stage(method, label):
    import tracy

    @functools.wraps(method)
    def marked(*args, **kwargs):
        tracy.signpost(f"{label}_BEGIN")
        try:
            return method(*args, **kwargs)
        finally:
            tracy.signpost(f"{label}_END")

    return marked


@pytest.mark.skipif(os.getenv("QWEN_GALAXY_LAYER_PROFILE") != "1", reason="explicit allocated-Galaxy profile")
def test_galaxy_layer_profile():
    import tracy

    output = Path(os.environ["QWEN_PROFILE_RECEIPT"])
    assert not output.exists(), "Use a new receipt path"
    torch.set_num_threads(8)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    gen = None
    report = dict(
        passed=False,
        state="loading",
        layers=[0, 3],
        cases=[dict(input_tokens=length, batch=batch) for length, batch in PROFILE_CASES],
        optimization_priority="128K/256K aggregate throughput; short-context regressions may be accepted",
        topology="linear",
        scope="Reduced two-layer diagnostic, not full-model throughput or accuracy qualification",
        measurement="Eager warm device operations with Tracy stage signposts; no host-time TPOT claim",
        cells=[],
    )
    originals = []
    try:
        source = Path(__file__).resolve().parents[1]
        report["source_sha256"] = {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((source / "tt").glob("*.py"))
        }
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        gen = build_generator(source, mesh, layer_indices=[0, 3], topology=ttnn.Topology.Linear)
        assert [layer.kind for layer in gen.model.layers] == ["linear_attention", "full_attention"]
        report["device_ids"] = list(mesh.get_device_ids())
        report["precision"] = gen.model.precision
        ttnn.ReadDeviceProfiler(mesh)
        base_ids = gen.tokenizer.encode(
            "Explain how scientific experiments test a hypothesis. ", add_special_tokens=False
        )
        report["state"] = "profiling"
        for length, batch in PROFILE_CASES:
            # Restore methods before the next geometry's warmup: only the
            # diagnostic window should have stage signposts.
            for layer, name, method in originals:
                setattr(layer, name, method)
            originals.clear()
            drop_request_buffers(gen)
            gen._ensure_cache(batch, length + 127)
            gen._ensure_history(127)
            ids = (base_ids * (length // len(base_ids) + 1))[:length]
            prompts = torch.tensor([ids] * batch, dtype=torch.int64)
            warmup = run_batch([gen], prompts, 4)
            ttnn.ReadDeviceProfiler(mesh)
            repeat = run_batch([gen], prompts, 4)
            assert warmup["output_sha256_per_replica"] == repeat["output_sha256_per_replica"]
            assert repeat["trace_captures"] == 0
            ttnn.ReadDeviceProfiler(mesh)
            # Trace scratch reservations must be released before eager ops.
            gen._release_traces()
            for index, layer in zip(report["layers"], gen.model.layers):
                for name in ("decode_forward", "_norm", "_delta", "_full_decode", "_finish"):
                    method = getattr(layer, name)
                    originals.append((layer, name, method))
                    setattr(layer, name, mark_stage(method, f"P0_S{length}_B{batch}_L{index}_{name}"))
            label = f"P0_S{length}_B{batch}_MODEL"
            tracy.signpost(f"{label}_BEGIN")
            started = time.perf_counter()
            diagnostic_logits = gen._model_step()
            ttnn.synchronize_device(mesh)
            diagnostic_s = time.perf_counter() - started
            tracy.signpost(f"{label}_END")
            ttnn.ReadDeviceProfiler(mesh)
            # Verify the diagnostic produced finite logits without including
            # full logits readback in the marked device-operation window.
            logits = gen._host_logits(diagnostic_logits)
            assert torch.isfinite(logits).all()
            del diagnostic_logits, logits
            report["cells"].append(
                dict(
                    input_tokens=length,
                    batch=batch,
                    state="completed",
                    diagnostic_host_s=diagnostic_s,
                    warm_repeat=repeat,
                )
            )
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"GALAXY_LAYER_PROFILE_COMPLETE input_tokens={length} batch={batch}", flush=True)
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:1000]))
        raise
    finally:
        try:
            output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            try:
                if gen is not None:
                    for layer, name, method in originals:
                        setattr(layer, name, method)
                    gen.close()
            finally:
                ttnn.close_mesh_device(parent)

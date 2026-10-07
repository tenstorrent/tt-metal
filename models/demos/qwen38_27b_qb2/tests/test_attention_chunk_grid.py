# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure accuracy and latency across SDPA chunk/core budgets at fixed inputs."""

import gc
import hashlib
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.attention_tuning import geometry
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import run_case, save
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


@pytest.mark.skipif(os.getenv("QWEN_ATTENTION_CHUNK_GRID") != "1", reason="explicit allocated-Galaxy diagnostic")
def test_attention_chunk_grid():
    path = Path(os.environ["QWEN_ATTENTION_CHUNK_GRID_RECEIPT"])
    assert not path.exists(), "Use a new result directory"
    cases = ((8192, 1), (8192, 16), (131072, 8), (262016, 4))
    modes = (
        ("hifi4_fp32_full_tile_accurate_exp",)
        if os.getenv("QWEN_ATTENTION_FULL_TILE") == "1"
        else ("native", "hifi4_fp32")
    )
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        diagnostic_complete=False,
        promoted_to_model=False,
        scope="Synthetic attention chunk/core allocation; no full-model qualification",
        reference="FP32 causal attention on the same quantized KV inputs, seed and page table for every variant",
        acceptance="Each geometry must have at least one variant passing every user's unchanged PCC/RMS gates",
        source_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("test_attention_chunk_grid.py", "test_long_context_attention.py", "attention_tuning.py")
        },
        cases=[],
    )
    save(path, report)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        for length, batch in cases:
            for precision in modes:
                for cores in (16, 32):
                    case = geometry(length, batch)
                    report["cases"].append(case)
                    run_case(
                        mesh,
                        case,
                        report,
                        path,
                        chunks=(128, 256, 512),
                        precision=precision,
                        max_cores_per_head_batch=cores,
                        require_native_accuracy=False,
                    )
                    ttnn.synchronize_device(mesh)
                    gc.collect()
        report["qualified_geometries"] = [
            [length, batch]
            for length, batch in cases
            if any(
                case.get("passing_chunks")
                for case in report["cases"]
                if (case["input_tokens"], case["batch"]) == (length, batch)
            )
        ]
        report.update(
            state="completed", diagnostic_complete=True, passed=len(report["qualified_geometries"]) == len(cases)
        )
        save(path, report)
        assert report["passed"], "Some geometries have no numerically passing chunk/core configuration"
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:2000]))
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
        finally:
            try:
                ttnn.close_mesh_device(parent)
            finally:
                save(path, report)

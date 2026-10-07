# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare native and accurate SDPA math without changing deployed model policy."""

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


@pytest.mark.skipif(os.getenv("QWEN_ATTENTION_PRECISION") != "1", reason="explicit allocated-Galaxy diagnostic")
def test_attention_precision():
    path = Path(os.environ["QWEN_ATTENTION_PRECISION_RECEIPT"])
    assert not path.exists(), "Use a new result directory"
    torch.set_num_threads(8)
    modes = ("native", "hifi4_fp32", "hifi4_fp32_accurate_exp")
    cases = ((8192, 1), (8192, 16), (131072, 8), (262016, 4))
    report = dict(
        state="opening",
        passed=False,
        diagnostic_complete=False,
        promoted_to_model=False,
        scope="Synthetic attention math diagnosis; no model accuracy or deployment qualification",
        reference="Full causal FP32 attention on quantized device KV, same input seed across math modes",
        acceptance="Every case at HiFi4/FP32/accurate-exp must pass the unchanged per-user PCC and relative-RMS limits",
        source_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("test_attention_precision.py", "test_long_context_attention.py", "attention_tuning.py")
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
            for mode in modes:
                case = geometry(length, batch)
                case["precision_mode"] = mode
                report["cases"].append(case)
                run_case(mesh, case, report, path, chunks=(), precision=mode, require_native_accuracy=False)
                ttnn.synchronize_device(mesh)
                gc.collect()
        report["diagnostic_complete"] = True
        report["passed_modes"] = [
            mode
            for mode in modes
            if all(case.get("passed") is True for case in report["cases"] if case["precision_mode"] == mode)
        ]
        report["passed"] = "hifi4_fp32_accurate_exp" in report["passed_modes"]
        report["state"] = "completed"
        save(path, report)
        assert report["passed"], "Accurate attention configuration failed the unchanged numerical reference"
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

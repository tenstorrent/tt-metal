# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Numerical and trace checks of the exact decoder helper, including device padding."""

import gc
import hashlib
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.attention_tuning import geometry
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import run_case, save
from models.demos.qwen38_27b_qb2.tt.decode_attention import paged_decode
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


@pytest.mark.skipif(os.getenv("QWEN_MODEL_ATTENTION") != "1", reason="explicit allocated-Galaxy test")
def test_model_decode_attention():
    path = Path(os.environ["QWEN_MODEL_ATTENTION_RECEIPT"])
    assert not path.exists(), "Use a new result directory"
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        scope="Actual decoder attention helper, including device Q padding and output slicing; no full-model eval",
        source_sha256={
            str(source.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
            for source in (Path(__file__), Path(run_case.__code__.co_filename), Path(paged_decode.__code__.co_filename))
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
        for length, batch, capacity in (
            (1, 1, 32),
            (127, 1, 128),
            (257, 1, 288),
            (1024, 16, 1152),
            (8192, 1, 8704),
            (8192, 16, 8704),
            (131072, 8, 131584),
            (262016, 4, 262144),
        ):
            case = geometry(length, batch)
            chunk = 256
            while capacity % chunk:
                chunk //= 2
            case.update(
                aligned_capacity=capacity,
                native_chunk=chunk,
                extra_pool_tokens=(capacity - case["native_capacity"]) * batch,
                pool_tokens=capacity * batch,
            )
            report["cases"].append(case)
            run_case(mesh, case, report, path, chunks=(), precision="model_accurate")
            gc.collect()
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
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

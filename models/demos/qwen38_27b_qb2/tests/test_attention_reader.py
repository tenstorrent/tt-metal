# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure the real paged-attention op with an isolated KV reader override."""

import gc
import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.attention_placement import placement, useful_kv_bytes
from models.demos.qwen38_27b_qb2.tests.attention_reader import CASES, compilation_evidence, verify_overlay
from models.demos.qwen38_27b_qb2.tests.attention_tuning import geometry
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import run_case, save
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


@pytest.mark.skipif(os.getenv("QWEN_ATTENTION_READER") != "1", reason="explicit allocated-Galaxy reader diagnostic")
def test_attention_reader():
    path = Path(os.environ["QWEN_ATTENTION_READER_RECEIPT"])
    assert not path.exists(), "Use a new result directory"
    manifest = json.loads(Path(os.environ["QWEN_ATTENTION_READER_MANIFEST"]).read_text())
    verify_overlay(manifest, os.environ["TT_METAL_KERNEL_PATH"], Path.cwd())
    cache = Path(os.environ["TT_METAL_CACHE"])
    report = dict(
        state="opening",
        passed=False,
        variant=manifest["variant"],
        overlay=manifest,
        cases=[],
        precision_change=False,
        promoted_to_model=False,
        scope="Real paged-SDPA op on synthetic inputs; no full-model speedup or reference-eval claim",
        measurement="Five samples of 100 trace replays; excludes setup and host readback",
    )
    save(path, report)
    torch.set_num_threads(8)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        grid = mesh.compute_with_storage_grid_size()
        report["worker_grid"] = [grid.x, grid.y]
        report["device_ids"] = list(mesh.get_device_ids())
        for length, batch in CASES:
            case = geometry(length, batch)
            core_info = placement("native", batch, (grid.x, grid.y))
            native_threshold = ((512 // core_info["active_cores"]) * 1152) // 2048
            case.update(
                native_chunk=256,
                native_barrier_threshold=native_threshold,
                kv_barrier_threshold=native_threshold
                if manifest["variant"] == "native"
                else int(manifest["variant"][2:]),
                active_cores=core_info["active_cores"],
            )
            report["cases"].append(case)
            run_case(mesh, case, report, path, chunks=(), precision="hifi4_fp32_full_tile_accurate_exp")
            call_us = case["candidates"][0]["median_traced_call_us"]
            case["useful_kv_gb_s"] = useful_kv_bytes(case["positions"]) / (call_us * 1000)
            report["compilation_evidence"] = compilation_evidence(cache, manifest)
            save(path, report)
            print("ATTENTION_READER_CASE", manifest["variant"], length, batch, call_us, flush=True)
            ttnn.synchronize_device(mesh)
            gc.collect()
        verify_overlay(manifest, os.environ["TT_METAL_KERNEL_PATH"], Path.cwd())
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
                report["cleanup_completed"] = True
            finally:
                save(path, report)

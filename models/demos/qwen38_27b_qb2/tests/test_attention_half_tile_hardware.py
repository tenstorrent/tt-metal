# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Physical TP4 full/partial/full-query timing with production instructions."""

import gc
import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.attention_half_tile import (
    HARDWARE_CASES,
    compare_hardware,
    compilation_evidence,
    verify_overlay,
)
from models.demos.qwen38_27b_qb2.tests.attention_placement import useful_kv_bytes
from models.demos.qwen38_27b_qb2.tests.attention_tuning import geometry
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import run_case, save
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


@pytest.mark.skipif(os.getenv("QWEN_HALF_TILE_HARDWARE") != "1", reason="explicit allocated-Galaxy diagnostic")
def test_attention_half_tile_hardware():
    path = Path(os.environ["QWEN_HALF_TILE_RECEIPT"])
    assert not path.exists(), "Preserve attempts"
    for name in ("TT_METAL_SIMULATOR", "TT_METAL_DISABLE_SFPLOADMACRO", "TT_METAL_DEVICE_PROFILER"):
        assert os.getenv(name, "") in ("", "0"), f"Unexpected diagnostic mode: {name}"
    manifest = json.loads(Path(os.environ["QWEN_HALF_TILE_MANIFEST"]).read_text())
    assert manifest["variant"] == "accurate_partial"
    verify_overlay(manifest, os.environ["TT_METAL_KERNEL_PATH"], Path.cwd())
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        cases=[],
        comparisons=[],
        overlay=manifest,
        precision_change=False,
        promoted_to_model=False,
        production_instruction_path=True,
        scope="Synthetic TP4 attention kernel; excludes model query padding/slicing and full-model/eval qualification",
        controls="Unchanged full-query arithmetic in the same overlay brackets each partial-query case",
        measurement="Five samples of 100 trace replays plus repeated measurement per variant; readback excluded",
    )
    save(path, report)
    torch.set_num_threads(8)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = mesh = None
    try:
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        for length, batch in HARDWARE_CASES:
            triple = []
            for variant in ("full_before", "partial", "full_after"):
                case = geometry(length, batch)
                case.update(native_chunk=256, variant=variant)
                triple.append(case)
                report["cases"].append(case)
                run_case(
                    mesh,
                    case,
                    report,
                    path,
                    chunks=(),
                    require_native_accuracy=variant != "partial",
                    precision="hifi4_fp32_accurate_exp"
                    if variant == "partial"
                    else "hifi4_fp32_full_tile_accurate_exp",
                )
                report["compilation_evidence"] = compilation_evidence(Path(os.environ["TT_METAL_CACHE"]), manifest)
                ttnn.synchronize_device(mesh)
                gc.collect()
                save(path, report)
            comparison = compare_hardware(triple)
            comparison["full_query_useful_kv_gb_s"] = useful_kv_bytes(triple[0]["positions"]) / (
                comparison["full_query_us"] * 1000
            )
            comparison["partial_query_useful_kv_gb_s"] = useful_kv_bytes(triple[1]["positions"]) / (
                comparison["partial_query_us"] * 1000
            )
            report["comparisons"].append(comparison)
            save(path, report)
            print("HALF_TILE_HARDWARE", json.dumps(comparison), flush=True)
        verify_overlay(manifest, os.environ["TT_METAL_KERNEL_PATH"], Path.cwd())
        report.update(
            state="completed",
            passed=True,
            all_candidates_qualified=all(c["timing_comparison_qualified"] for c in report["comparisons"]),
        )
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
        finally:
            try:
                if parent is not None:
                    ttnn.close_mesh_device(parent)
                    report["cleanup_completed"] = True
            finally:
                save(path, report)

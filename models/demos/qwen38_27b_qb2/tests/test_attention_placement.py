# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-precision placement experiments at the user's long-context operating points."""

import gc
import hashlib
import os
import statistics
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.attention_placement import LAYOUTS, useful_kv_bytes
from models.demos.qwen38_27b_qb2.tests.attention_tuning import geometry
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import run_case, save
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


@pytest.mark.skipif(os.getenv("QWEN_ATTENTION_PLACEMENT") != "1", reason="explicit allocated-Galaxy placement sweep")
def test_attention_placement():
    path = Path(os.environ["QWEN_ATTENTION_PLACEMENT_RECEIPT"])
    assert not path.exists(), "Use a new receipt directory"
    source = Path(__file__).resolve().parent
    native = Path(os.environ["TT_METAL_HOME"])
    files = [
        source / name
        for name in (
            "test_attention_placement.py",
            "test_long_context_attention.py",
            "attention_placement.py",
            "attention_tuning.py",
        )
    ]
    files.extend(
        native / relative
        for relative in (
            "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp",
            "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp",
        )
    )
    report = dict(
        state="opening",
        passed=False,
        cases=[],
        comparisons=[],
        promoted_to_model=False,
        scope="Synthetic TP4 attention placement and common-output-layout timing; no full-model throughput or eval claim",
        precision=dict(q="bfloat16", kv="bfloat8_b", fidelity="HiFi4", fp32_accumulation=True, approximate_exp=False),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        measurement="Five samples of 100 trace replays; includes sharded-output conversion to DRAM, excludes setup/readback",
        caveat="KV remains interleaved across banks; FlashMLA reference locations do not make reads bank-local",
    )
    save(path, report)
    torch.set_num_threads(8)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        grid = mesh.compute_with_storage_grid_size()
        report["worker_grid"] = [grid.x, grid.y]
        save(path, report)
        for length, batch in ((32768, 16), (131072, 8), (262016, 4), (32768, 32), (131072, 16), (262016, 8)):
            rows = []
            for index, name in enumerate((*LAYOUTS, "native")):
                case = geometry(length, batch)
                case.update(native_chunk=256, measurement_index=index, placement_name=name)
                report["cases"].append(case)
                run_case(
                    mesh,
                    case,
                    report,
                    path,
                    chunks=(),
                    precision="hifi4_fp32_full_tile_accurate_exp",
                    core_placement=name,
                    require_native_accuracy=name == "native",
                )
                candidate = case["candidates"][0]
                microseconds = candidate["median_traced_call_us"]
                row = dict(
                    name=name,
                    traced_call_us=microseconds,
                    passed=case["passed"],
                    selection=case["selection"],
                    useful_kv_bytes=useful_kv_bytes(case["positions"]),
                    useful_kv_gb_s=useful_kv_bytes(case["positions"]) / (microseconds * 1000),
                )
                rows.append(row)
                case["bandwidth_summary"] = row
                save(path, report)
                ttnn.synchronize_device(mesh)
                gc.collect()
            drift = abs(rows[-1]["traced_call_us"] / rows[0]["traced_call_us"] - 1)
            valid = [row for row in rows[:-1] if row["passed"] and row["selection"]["timing_comparison_qualified"]]
            fastest = min(valid, key=lambda row: row["traced_call_us"])
            matched = []
            for left, right in (("row_major_64", "flash_mla_64"), ("row_major_80", "outer_columns_80")):
                a, b = (next(row for row in rows if row["name"] == name) for name in (left, right))
                matched.append(
                    dict(
                        control=left,
                        candidate=right,
                        both_numerically_passed=a["passed"] and b["passed"],
                        control_over_candidate=a["traced_call_us"] / b["traced_call_us"],
                    )
                )
            comparison = dict(
                input_tokens=length,
                batch=batch,
                baseline_repeat_drift_fraction=drift,
                timing_comparison_qualified=drift <= 0.03,
                fastest_passing_layout=fastest["name"],
                baseline_over_fastest=statistics.mean((rows[0]["traced_call_us"], rows[-1]["traced_call_us"]))
                / fastest["traced_call_us"],
                matched_placement_comparisons=matched,
            )
            report["comparisons"].append(comparison)
            save(path, report)
            print("ATTENTION_PLACEMENT_COMPARISON", comparison, flush=True)
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

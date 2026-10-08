# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context placement/chunk diagnostic with unchanged model precision."""

import gc
import hashlib
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.attention_placement_long import CASES, VARIANTS, comparison
from models.demos.qwen38_27b_qb2.tests.attention_tuning import geometry
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import run_case, save
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


@pytest.mark.skipif(os.getenv("QWEN_ATTENTION_PLACEMENT_LONG") != "1", reason="explicit Galaxy diagnostic")
def test_attention_placement_long():
    path = Path(os.environ["QWEN_ATTENTION_PLACEMENT_RECEIPT"])
    assert not path.exists(), "Use a new receipt directory"
    source = Path(__file__).resolve().parent
    native = Path(os.environ["TT_METAL_HOME"])
    files = [
        source / name
        for name in (
            "test_attention_placement_long.py",
            "attention_placement_long.py",
            "test_long_context_attention.py",
            "attention_placement.py",
            "attention_tuning.py",
        )
    ]
    files.extend((native / "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device").rglob("*.cpp"))
    report = dict(
        state="opening",
        passed=False,
        cases=[],
        comparisons=[],
        promoted_to_model=False,
        cleanup_completed=False,
        scope="Synthetic TP4 placement/chunk diagnostic; no full-model or reference-eval qualification",
        precision=dict(q="bfloat16", kv="bfloat8_b", fidelity="HiFi4", fp32_accumulation=True, approximate_exp=False),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        measurement="Five samples of 100 trace replays and repeated control for each variant; includes output conversion",
        hypotheses=[
            "Separate location changes from the number of workers/partial reductions",
            "512-token chunks may reduce accumulated rounding error and reader overhead at high batch/context",
            "Explicit full-grid control accounts for reducer placement and output-sharding cost",
        ],
        caveat="KV stays DRAM-interleaved; outside-column placement does not make reads bank-local",
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
        assert (grid.x, grid.y) == (12, 10), "Experiment requires measured 12x10 worker grid"
        save(path, report)
        for length, batch in CASES:
            rows = []
            for index, (name, chunk) in enumerate(VARIANTS):
                case = geometry(length, batch)
                case.update(native_chunk=chunk, measurement_index=index, placement_name=name)
                report["cases"].append(case)
                rows.append(case)
                run_case(
                    mesh,
                    case,
                    report,
                    path,
                    chunks=(),
                    precision="hifi4_fp32_full_tile_accurate_exp",
                    core_placement=name,
                    require_native_accuracy=name == "native" and chunk == 256,
                )
                ttnn.synchronize_device(mesh)
                gc.collect()
            result = comparison(rows)
            report["comparisons"].append(result)
            save(path, report)
            print("LONG_PLACEMENT_COMPARISON", result, flush=True)
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

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded native counters for real-weight projections and exact model SDPA."""

import gc
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.attention_tuning import accuracy, geometry, reference
from models.demos.qwen38_27b_qb2.tests.large_operator_profile import selected_labels
from models.demos.qwen38_27b_qb2.tests.projection_sweep import ROLES
from models.demos.qwen38_27b_qb2.tests.test_gdn_epilogue import digest, download
from models.demos.qwen38_27b_qb2.tests.test_projection_sweep import metrics, upload
from models.demos.qwen38_27b_qb2.tt.decode_attention import paged_decode
from models.demos.qwen38_27b_qb2.tt.decoder import Qwen38Decoder
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


def sampled(mesh, label, invoke, check, report, path):
    import tracy

    # Drain device buffers into host memory while retaining final metadata
    # processing. Do not enable MID_RUN_DUMP: that skips counter enrichment.
    # Setup, warmup and readback can otherwise fill BRISC's instruction-counter
    # buffer even when only one operator boundary is selected.
    ttnn.ReadDeviceProfiler(mesh)
    warm = invoke()
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    del warm
    case = dict(label=label, samples=[], passed=False)
    report["cases"].append(case)
    for repeat in range(3):
        tracy.signpost(header=f"LARGE_OP_{label}_{repeat}_BEGIN")
        result = invoke()
        ttnn.synchronize_device(mesh)
        tracy.signpost(header=f"LARGE_OP_{label}_{repeat}_END")
        ttnn.ReadDeviceProfiler(mesh)
        # Readback conversion is deliberately outside the profiled boundary.
        values = download(ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG))
        checks = check(values)
        assert len(values) == 4 and len(checks) == 4 and all(c["passed"] for c in checks), checks
        case["samples"].append(dict(repeat=repeat, hashes=[digest(v) for v in values], accuracy=checks))
        save(path, report)
        del result, values
    assert all(s["hashes"] == case["samples"][0]["hashes"] for s in case["samples"])
    case["passed"] = True
    save(path, report)


def projections(mesh, report, path, checkpoint, precision):
    layer = Qwen38TPDecoder.from_state_dict(
        Checkpoint(checkpoint).layer(0),
        hf_config=AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config,
        layer_idx=0,
        mesh_device=mesh,
        policy={**decoder_policy(precision, 0), "ring": False, "compact_decode_residual": True},
    )
    for role in ("output", "down"):
        labels = selected_labels(report["boundary_selection"])
        if not any(label.startswith(role.upper() + "_") for label in labels):
            continue
        spec = ROLES[role]
        canonical = download(layer.weights[spec["name"] + ".weight"])
        rng = torch.Generator().manual_seed(92716 + spec["k"])
        host = torch.randn(16, spec["k"] * 4, generator=rng).bfloat16()
        x = upload(mesh, host, role=role, input_layout="compact_l1")
        gold = [a.float() @ w.float() for a, w in zip(host.chunk(4, dim=-1), canonical)]
        for boundary in ("LOCAL", "COMPLETE"):
            if f"{role.upper()}_{boundary}" not in labels:
                continue

            def invoke():
                if boundary == "LOCAL":
                    return Qwen38Decoder._linear(layer, x, spec["name"], keep_sharded=True)
                return layer._linear(x, spec["name"], keep_sharded=True)

            def check(values):
                expected = gold if boundary == "LOCAL" else [sum(gold)] * 4
                checks = [metrics(v[..., : spec["n"]], g) for v, g in zip(values, expected)]
                for result in checks:
                    result["passed"] = result["pcc"] >= 0.999 and result["relative_rms"] <= 0.015
                return checks

            sampled(mesh, f"{role.upper()}_{boundary}", invoke, check, report, path)
        del x, host, gold, canonical
        gc.collect()
    del layer
    gc.collect()


def attention(mesh, report, path):
    case = geometry(32768, 16)
    capacity = case["aligned_capacity"]
    pages = capacity // 32
    rng = torch.Generator().manual_seed(20261006 + 32768 + 16)
    table = torch.randperm(16 * pages, generator=rng, dtype=torch.int32).reshape(16, pages)
    query = torch.randn((1, 16, 6, 256), generator=rng).bfloat16()

    def upload_replicated(value, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value,
            dtype=dtype,
            layout=layout,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    q = upload_replicated(query, ttnn.bfloat16)
    tt_table = upload_replicated(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    positions = upload_replicated(torch.tensor(case["positions"], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    key = upload_replicated(torch.randn((16 * pages, 1, 32, 256), generator=rng).bfloat16(), ttnn.bfloat8_b)
    host_value = torch.randn((16 * pages, 1, 32, 256), generator=rng).bfloat16()
    for user, position in enumerate(case["positions"]):
        for page in range((position + 1) // 32, pages):
            host_value[int(table[user, page]), 0, max(0, position + 1 - page * 32) :, :] = 32
    value = upload_replicated(host_value, ttnn.bfloat8_b)
    del host_value
    key_quantized = ttnn.to_torch(ttnn.get_device_tensors(key)[0])
    value_quantized = ttnn.to_torch(ttnn.get_device_tensors(value)[0])
    expected = reference(query, key_quantized, value_quantized, table, case["positions"])
    del key_quantized, value_quantized
    report["attention_geometry"] = case
    sampled(
        mesh,
        "SDPA_COMPLETE",
        lambda: paged_decode(
            q,
            key,
            value,
            positions=positions,
            page_table=tt_table,
            policy={"decode_attention": "accurate_full_tile"},
        ),
        lambda values: [accuracy(v, expected) for v in values],
        report,
        path,
    )


@pytest.mark.skipif(os.getenv("QWEN_LARGE_OPERATOR_PROFILE") != "1", reason="explicit physical profiling")
def test_large_operator_profile():
    assert not any(os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_SLOW_DISPATCH_MODE", "TT_METAL_KERNEL_PATH"))
    assert not os.getenv("TT_METAL_PROFILER_MID_RUN_DUMP")
    path = Path(os.environ["QWEN_LARGE_OPERATOR_RECEIPT"])
    assert not path.exists()
    torch.set_num_threads(8)
    source = Path(__file__).resolve().parents[1]
    precision = load_precision(source / "config/precision_single_step_compact_gdn_resident_gates_padding_bfp8_all.json")
    checkpoint = checkpoint_path()
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        cases=[],
        source_sha256=model_source_hashes(source),
        precision=precision,
        counter_mask=int(os.getenv("TT_METAL_PROFILE_PERF_COUNTERS", "0")),
        boundary_selection=os.getenv("QWEN_LARGE_OPERATOR_BOUNDARY", "all"),
        profiler_drained_outside_samples=True,
        batch=16,
        context=32768,
        scope="Real layer0 output/down weights and synthetic model SDPA; three eager profiled calls per boundary",
        full_model_qualified=False,
        promoted_to_serving=False,
    )
    labels = selected_labels(report["boundary_selection"])
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=0)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        mesh.enable_program_cache()
        report["device_ids"] = list(mesh.get_device_ids())
        assert report["device_ids"] == [0, 4, 12, 8]
        report["state"] = "running"
        if any(label != "SDPA_COMPLETE" for label in labels):
            projections(mesh, report, path, checkpoint, precision)
        if "SDPA_COMPLETE" in labels:
            attention(mesh, report, path)
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
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

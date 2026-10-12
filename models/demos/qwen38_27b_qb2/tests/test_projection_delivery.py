# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test direct native matmul-to-collective delivery at the complete boundary."""

import copy
import gc
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig
from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.projection_sweep import ROLES, compare
from models.demos.qwen38_27b_qb2.tests.test_gdn_epilogue import digest, download
from models.demos.qwen38_27b_qb2.tests.test_projection_sweep import measure, upload
from models.demos.qwen38_27b_qb2.tt.decoder import Qwen38Decoder
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


def run_case(layer, mesh, batch, role, report, path, *, input_layout="compact_l1"):
    spec = ROLES[role]
    canonical = download(layer.weights[spec["name"] + ".weight"])
    rng = torch.Generator().manual_seed(20261011 + batch + spec["k"])
    hosts = [torch.randn(batch, spec["k"] * 4, generator=rng).bfloat16() for _ in range(2)]
    inputs = [upload(mesh, value, role=role, input_layout=input_layout) for value in (*hosts, hosts[0])]
    keep_sharded = input_layout == "compact_l1"
    assert len({t.buffer_address() for t in inputs}) == 3
    golds = [[a.float() @ w.float() for a, w in zip(host.chunk(4, dim=-1), canonical)] for host in hosts]
    baseline = []
    for x in inputs[:2]:
        value = layer._linear(x, spec["name"], keep_sharded=keep_sharded)
        baseline.append([v.reshape(batch, spec["n"]) for v in download(value)])
        del value
    readonly = [[digest(v) for v in download(x)] for x in inputs[1:]]
    config = {key: layer.policy[role + "_" + key] for key in ("readers", "cores", "block")}
    weight = layer.dram_weights[spec["name"] + ".weight"]
    weight_address = weight.buffer_address()
    group = []
    for label, delivery in (
        ("before", "reshard"),
        ("native_input", "native_input"),
        ("matmul_output", "matmul_output"),
        ("after", "reshard"),
    ):
        candidate = copy.copy(layer)
        candidate.policy = dict(layer.policy, projection_delivery=delivery)
        effective = candidate._projection_delivery(inputs[0], spec["name"], keep_sharded)
        assert effective == (delivery if keep_sharded else "reshard")
        geometry = None
        if keep_sharded:
            partial = Qwen38Decoder._linear(candidate, inputs[0], spec["name"], keep_sharded=True)
            shard = partial.memory_config().shard_spec
            geometry = dict(
                shape=list(partial.shape),
                shard_shape=list(shard.shape),
                cores=shard.grid.num_cores(),
                grid=str(shard.grid),
            )
            assert geometry["cores"] == (40 if delivery == "matmul_output" else 8), geometry
            assert geometry["shard_shape"] == [32, 128 if delivery == "matmul_output" else 640], geometry
            del partial
        report.update(active_batch=batch, active_role=role, active_delivery=delivery)
        save(path, report)
        row = measure(
            candidate,
            role,
            config,
            weight,
            inputs,
            golds,
            baseline,
            label=label,
            input_layout=input_layout,
            trace_calls=16,
        )
        row.update(batch=batch, role=role, delivery=delivery, effective_delivery=effective, partial_layout=geometry)
        row["exact_baseline"] = all(m["equal"] for c in row["accuracy"] for m in c["baseline"])
        report["cases"].append(row)
        group.append(row)
        save(path, report)
        assert row["accuracy_passed"] and row["changed_input_trace_passed"] and row["exact_baseline"], row
        assert weight.buffer_address() == weight_address
        assert readonly == [[digest(v) for v in download(x)] for x in inputs[1:]]
        gc.collect()
    assert all(row["output_sha256"] == group[0]["output_sha256"] for row in group)
    for row in group[1:-1]:
        comparison = dict(
            batch=batch,
            role=role,
            input_layout=input_layout,
            delivery=row["delivery"],
            effective_delivery=row["effective_delivery"],
            **compare(group[0], row, group[-1], role=role),
        )
        report["comparisons"].append(comparison)
        print("PROJECTION_DELIVERY", comparison, flush=True)
    save(path, report)


@pytest.mark.skipif(os.getenv("QWEN_PROJECTION_DELIVERY") != "1", reason="explicit physical TP4 experiment")
def test_projection_delivery():
    assert not any(
        os.getenv(k)
        for k in (
            "TT_METAL_SIMULATOR",
            "TT_METAL_KERNEL_PATH",
            "TT_METAL_SLOW_DISPATCH_MODE",
            "TT_METAL_DISABLE_SFPLOADMACRO",
        )
    )
    assert TRACE_ALLOC_TRACKING, "Enable trace allocation tracking at process startup"
    path = Path(os.environ["QWEN_PROJECTION_DELIVERY_RECEIPT"])
    assert not path.exists(), "Preserve earlier attempts"
    torch.set_num_threads(8)
    source = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint_path()
    precision = load_precision(source / "config/precision_single_step_compact_gdn_resident_gates_padding_bfp8_all.json")
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        cases=[],
        comparisons=[],
        source_sha256=model_source_hashes(source),
        checkpoint=str(checkpoint),
        precision=precision,
        trace_allocation_tracking=True,
        scope="Real layer-0 BFP8 weights; B16/B32 projection, layout and all-reduce, exact baseline and changed-input trace checks",
        full_model_qualified=False,
        promoted_to_serving=False,
    )
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        mesh.enable_program_cache()
        report["device_ids"] = list(mesh.get_device_ids())
        assert report["device_ids"] == [0, 4, 12, 8]
        layer = Qwen38TPDecoder.from_state_dict(
            Checkpoint(checkpoint).layer(0),
            hf_config=AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config,
            layer_idx=0,
            mesh_device=mesh,
            policy={**decoder_policy(precision, 0), "ring": False, "compact_decode_residual": True},
        )
        report["state"] = "running"
        for batch in (16, 32):
            for role in ("output", "down"):
                run_case(layer, mesh, batch, role, report, path)
                gc.collect()
            run_case(layer, mesh, batch, "output", report, path, input_layout="public_dram")
            gc.collect()
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

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight TP4 output/down projection tuning, including layouts and CCL."""

import copy
import gc
import os
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.projection_sweep import (
    BATCHES,
    BOUNDARIES,
    ROLES,
    candidates,
    compare,
    geometry,
    input_contract,
)
from models.demos.qwen38_27b_qb2.tests.test_gdn_epilogue import digest, download
from models.demos.qwen38_27b_qb2.tests.test_gdn_layer_integration import capture
from models.demos.qwen38_27b_qb2.tt.decoder import Qwen38Decoder
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


def metrics(actual, expected):
    width = expected.shape[-1]
    actual, expected = actual.float().reshape(-1, width), expected.float().reshape(-1, width)
    assert actual.shape == expected.shape and torch.isfinite(actual).all() and torch.isfinite(expected).all()
    rms = ((actual - expected).square().mean(-1) / expected.square().mean(-1).clamp_min(1e-20)).sqrt().max().item()
    left, right = actual - actual.mean(-1, keepdim=True), expected - expected.mean(-1, keepdim=True)
    pcc = ((left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-20)).min().item()
    return dict(relative_rms=rms, pcc=pcc, equal=torch.equal(actual, expected))


def upload(mesh, values, *, role, input_layout):
    # Different rank/user inputs catch accidental replication and all-reduce omissions.
    contract = input_contract(values.shape[0], role, input_layout)
    shape = [*contract["shape"][:-1], values.shape[-1]]
    memory = ttnn.L1_MEMORY_CONFIG if contract["memory"] == "l1" else ttnn.DRAM_MEMORY_CONFIG
    result = ttnn.from_torch(
        values.reshape(shape),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )
    assert list(result.shape) == contract["shape"] and list(result.padded_shape) == contract["padded_shape"]
    assert result.memory_config() == memory
    return result


def weight_for(layer, role, config, canonical):
    """Repack the exact dequantized baseline tiles; verify live columns afterwards."""
    spec = ROLES[role]
    dimensions = geometry(role, config)
    pieces = [torch.nn.functional.pad(w, (0, dimensions["padded_n"] - spec["n"])) for w in canonical]
    source = ttnn.from_torch(
        torch.cat(pieces, dim=1),
        device=layer.device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(layer.device, dim=1),
    )
    banks = layer.device.dram_grid_size().x
    assert banks == 8
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, [spec["k"], dimensions["weight_shard_width"]], ttnn.ShardOrientation.ROW_MAJOR),
    )
    result = ttnn.to_memory_config(source, memory)
    restored = download(ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG))
    assert len(restored) == 4
    for value, expected in zip(restored, canonical):
        assert torch.equal(value[:, : spec["n"]].float(), expected.float()), "Weight repacking changed BFP8 values"
        assert torch.count_nonzero(value[:, spec["n"] :]) == 0
    return result


def measure(layer, role, config, weight, inputs, golds, baseline, *, label, input_layout):
    spec = ROLES[role]
    candidate = copy.copy(layer)
    candidate.policy = dict(layer.policy, **{role + "_" + key: value for key, value in config.items()})
    candidate.dram_weights = dict(layer.dram_weights, **{spec["name"] + ".weight": weight})
    contract = input_contract(
        inputs[0].shape[-2] if input_layout == "compact_l1" else inputs[0].shape[0], role, input_layout
    )
    keep_sharded = contract["keep_sharded"]

    def invoke(x):
        return candidate._linear(x, spec["name"], keep_sharded=keep_sharded)

    # Include the declared compact/public GDN or compact MLP projection boundary,
    # including native TP all-reduce. Setup, weight repacking and readbacks are excluded.
    warmup = invoke(inputs[0])
    ttnn.synchronize_device(layer.device)
    trace, output = capture(layer.device, lambda: invoke(inputs[0]))
    checks, hashes, initial_hashes = [], [], None
    try:
        for index in (0, 1, 0):
            if index == 1:
                ttnn.copy(inputs[1], inputs[0])
            else:
                ttnn.copy(inputs[2], inputs[0])
            # Independently evaluate each local quantized GEMM before CCL.
            local = Qwen38Decoder._linear(candidate, inputs[0], spec["name"], keep_sharded=True)
            partials = download(ttnn.to_memory_config(local, ttnn.DRAM_MEMORY_CONFIG))
            local_checks = [metrics(v[..., : spec["n"]], g) for v, g in zip(partials, golds[index])]
            del partials, local
            ttnn.execute_trace(layer.device, trace, cq_id=0, blocking=True)
            outputs = [v.reshape(-1, spec["n"]) for v in download(output)]
            dense_sum = sum(golds[index])
            reduced = [metrics(v, dense_sum) for v in outputs]
            base_checks = [metrics(v, b) for v, b in zip(outputs, baseline[index])] if baseline else []
            passed = (
                len(outputs) == 4
                and all(m["pcc"] >= 0.999 and m["relative_rms"] <= 0.015 for m in local_checks + reduced)
                and all(m["pcc"] >= 0.99999 and m["relative_rms"] <= 0.003 for m in base_checks)
            )
            checks.append(dict(input=index, local=local_checks, reduced=reduced, baseline=base_checks, passed=passed))
            if index == 0:
                hashes = [digest(v) for v in outputs]
                if initial_hashes is None:
                    initial_hashes = hashes
                else:
                    assert hashes == initial_hashes, "A/B/A replay did not restore the first result"
            else:
                assert hashes != [digest(v) for v in outputs], "Changed trace input was ignored"
        for _ in range(8):
            ttnn.execute_trace(layer.device, trace, cq_id=0, blocking=True)
        samples = []
        for _ in range(5):
            ttnn.synchronize_device(layer.device)
            tick = time.perf_counter()
            for _ in range(100):
                ttnn.execute_trace(layer.device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(layer.device)
            samples.append((time.perf_counter() - tick) * 1e6 / 100)
        assert hashes == [digest(v.reshape(-1, spec["n"])) for v in download(output)]
        row = dict(
            label=label,
            input_layout=input_layout,
            input_contract=contract,
            config=config,
            geometry=geometry(role, config),
            samples_us=samples,
            output_sha256=hashes,
            accuracy=checks,
            accuracy_passed=all(c["passed"] for c in checks),
            changed_input_trace_passed=all(c["passed"] for c in checks),
            full_model_measured=False,
        )
    finally:
        ttnn.release_trace(layer.device, trace)
    del warmup, output
    return row


@pytest.mark.skipif(os.getenv("QWEN_PROJECTION_SWEEP") != "1", reason="explicit allocated Galaxy projection experiment")
def test_projection_sweep():
    assert not any(
        os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_SLOW_DISPATCH_MODE", "TT_METAL_DISABLE_SFPLOADMACRO")
    )
    path = Path(os.environ["QWEN_PROJECTION_RECEIPT"])
    assert not path.exists(), "Preserve each hardware attempt"
    torch.set_num_threads(8)
    source = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint_path()
    precision = load_precision(source / "config/precision_single_step_shared_qk_bfp8_all.json")
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        cases=[],
        comparisons=[],
        source_sha256=model_source_hashes(source),
        checkpoint=str(checkpoint),
        precision=precision,
        scope="Actual layer-0 BFP8 weights, rank-distinct inputs; separate compact-L1/public-DRAM projection+layout+CCL boundaries, not full-model performance",
        promoted_to_serving=False,
    )
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        assert len(report["device_ids"]) == 4
        layer = Qwen38TPDecoder.from_state_dict(
            Checkpoint(checkpoint).layer(0),
            hf_config=AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config,
            layer_idx=0,
            mesh_device=mesh,
            policy={**decoder_policy(precision, 0), "ring": False, "compact_decode_residual": True},
        )
        for batch in BATCHES:
            for role, input_layout in BOUNDARIES:
                spec = ROLES[role]
                contract = input_contract(batch, role, input_layout)
                report.update(state="running", active_batch=batch, active_role=role, active_input_layout=input_layout)
                canonical = download(layer.weights[spec["name"] + ".weight"])
                assert len(canonical) == 4
                rng = torch.Generator().manual_seed(92700 + batch + spec["k"])
                hosts = [torch.randn(batch, spec["k"] * 4, generator=rng).bfloat16() for _ in range(2)]
                inputs = [upload(mesh, h, role=role, input_layout=input_layout) for h in (*hosts, hosts[0])]
                golds = [[a.float() @ w.float() for a, w in zip(h.chunk(4, dim=-1), canonical)] for h in hosts]
                baseline = []
                for index in (0, 1):
                    value = layer._linear(inputs[index], spec["name"], keep_sharded=contract["keep_sharded"])
                    baseline.append([v.reshape(batch, spec["n"]) for v in download(value)])
                    del value
                weights = {2: layer.dram_weights[spec["name"] + ".weight"]}
                configs = candidates(role)
                group = []
                for index, config in enumerate([configs[0], *configs[1:], configs[0]]):
                    label = "before" if index == 0 else "after" if index == len(configs) else f"candidate-{index}"
                    report["active_config"] = config
                    save(path, report)
                    if config["readers"] not in weights:
                        weights[config["readers"]] = weight_for(layer, role, config, canonical)
                    row = measure(
                        layer,
                        role,
                        config,
                        weights[config["readers"]],
                        inputs,
                        golds,
                        baseline,
                        label=label,
                        input_layout=input_layout,
                    )
                    row.update(batch=batch, role=role)
                    group.append(row)
                    report["cases"].append(row)
                    save(path, report)
                    gc.collect()
                assert group[0]["accuracy_passed"] and group[-1]["accuracy_passed"], "Baseline reference failed"
                for row in group[1:-1]:
                    result = compare(group[0], row, group[-1], role=role)
                    report["comparisons"].append(
                        dict(batch=batch, role=role, input_layout=input_layout, config=row["config"], **result)
                    )
                del weights, inputs, baseline, canonical
                gc.collect()
                save(path, report)
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

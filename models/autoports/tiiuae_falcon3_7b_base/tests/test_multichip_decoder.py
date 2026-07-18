# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tests.test_functional_decoder import (
    REAL_WEIGHT_PCC,
    _assert_pcc,
    _config,
    _hf_decode,
    _hf_layer,
    _hf_prefill,
    _real_layer_state_dict,
)
from models.autoports.tiiuae_falcon3_7b_base.tests.test_optimized_decoder import (
    _recorded_layer14_inputs,
    _recorded_layer14_seq31_inputs,
    _release_model,
)
from models.autoports.tiiuae_falcon3_7b_base.tt.functional_decoder import IR_REPRESENTATIVE_LAYER
from models.autoports.tiiuae_falcon3_7b_base.tt.multichip_decoder import (
    TARGET_MESH_SHAPE,
    TENSOR_PARALLEL_SIZE,
    MultichipDecoder,
)
from models.autoports.tiiuae_falcon3_7b_base.tt.optimized_decoder import OptimizedDecoder
from models.common.utility_functions import comp_pcc

RESULTS_DIR = Path(__file__).parents[1] / "doc" / "multichip_decoder" / "results"
REPO_ROOT = Path(__file__).parents[4]
IMPLEMENTATION_PATH = Path(__file__).parents[1] / "tt" / "multichip_decoder.py"
OPTIMIZED_BATCH1_RESULT = (
    Path(__file__).parents[1] / "doc" / "optimized_decoder" / "results" / "final" / "final_batch1.json"
)
OPTIMIZED_BATCH32_RESULT = OPTIMIZED_BATCH1_RESULT.with_name("final_batch32.json")


def _write_result_artifact(filename: str, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("generated_at_utc", datetime.now(timezone.utc).isoformat())
    payload.setdefault(
        "repo_head",
        subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True).strip(),
    )
    payload.setdefault("implementation_sha256", hashlib.sha256(IMPLEMENTATION_PATH.read_bytes()).hexdigest())
    payload.setdefault("test_sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    payload.setdefault("hardware", "4x Blackhole p300c, mesh 1x4, FABRIC_1D_RING")
    output_dir = Path(os.getenv("FALCON3_MULTICHIP_RESULTS_DIR", RESULTS_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _mesh_input(hidden_states: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        hidden_states.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _mesh_int32(host: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        host.to(torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _first_rank(tensor: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _assert_replicated(tensor: ttnn.Tensor) -> None:
    ranks = ttnn.get_device_tensors(tensor)
    assert len(ranks) == TENSOR_PARALLEL_SIZE
    first = ttnn.to_torch(ranks[0])
    for rank in ranks[1:]:
        assert torch.equal(first, ttnn.to_torch(rank))


def _paged_cache_to_torch(cache: ttnn.Tensor, page_table: torch.Tensor, logical_length: int) -> torch.Tensor:
    """Rebuild `[batch,global_kv_heads,seq,head_dim]` from rank-local physical pages."""
    rank_caches = [ttnn.to_torch(rank) for rank in ttnn.get_device_tensors(cache)]
    users = []
    block_size = int(rank_caches[0].shape[2])
    for user_id in range(page_table.shape[0]):
        rank_users = []
        for rank_cache in rank_caches:
            logical_pages = [rank_cache[int(page_id)] for page_id in page_table[user_id]]
            rank_users.append(torch.cat(logical_pages, dim=1)[:, :logical_length, :])
        users.append(torch.cat(rank_users, dim=0))
    assert block_size == 32
    return torch.stack(users)


def _contiguous_cache_to_torch(cache: ttnn.Tensor, logical_length: int) -> torch.Tensor:
    return torch.cat(
        [ttnn.to_torch(rank)[:, :, :logical_length, :] for rank in ttnn.get_device_tensors(cache)],
        dim=1,
    )


@torch.no_grad()
def _hf_key_value_samples(layer, config, hidden_states: torch.Tensor, positions: list[int]):
    """Reference token-local K/V projections and RoPE without full-sequence HF attention."""
    samples = hidden_states[:, positions, :]
    normed = layer.input_layernorm(samples)
    batch, sample_count, _ = samples.shape
    key = (
        layer.self_attn.k_proj(normed)
        .view(batch, sample_count, config.num_key_value_heads, config.head_dim)
        .transpose(1, 2)
    )
    value = (
        layer.self_attn.v_proj(normed)
        .view(batch, sample_count, config.num_key_value_heads, config.head_dim)
        .transpose(1, 2)
    )
    position_ids = torch.tensor(positions, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    cos, sin = LlamaRotaryEmbedding(config)(samples, position_ids)
    key, _ = apply_rotary_pos_emb(key, key, cos, sin)
    return key, value


def _single_device_input(hidden_states: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        hidden_states.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _single_device_position(batch: int, position: int, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.full((batch,), position, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _release_tensors(*tensors) -> None:
    for tensor in tensors:
        if tensor is not None:
            tensor.deallocate(True)


def _trace_mesh_callable(mesh_device, function, *, samples: int = 5, iterations: int = 100):
    warm_output = function()
    ttnn.synchronize_device(mesh_device)
    warm_output.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = function()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for _ in range(10):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        timings = []
        for _ in range(samples):
            start = time.perf_counter()
            for _ in range(iterations):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            timings.append((time.perf_counter() - start) * 1000.0 / iterations)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    return trace_output, timings


def test_multichip_runtime_is_owned_and_host_fallback_free():
    assert MultichipDecoder.single_chip_baseline_cls is OptimizedDecoder
    assert TARGET_MESH_SHAPE == (1, 4)
    hot_methods = (
        MultichipDecoder.prefill_forward,
        MultichipDecoder.decode_forward,
        MultichipDecoder._prefill_attention,
        MultichipDecoder._prefill_linear_chunked,
        MultichipDecoder._decode_attention,
        MultichipDecoder._decode_qkv,
        MultichipDecoder._prefill_mlp,
        MultichipDecoder._prefill_mlp_chunk,
        MultichipDecoder._decode_mlp,
        MultichipDecoder._all_reduce_partial,
    )
    forbidden = ("torch", "from_torch", "to_torch", "OptimizedDecoder.")
    for method in hot_methods:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} contains runtime fallback token {token!r}"


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_TOPOLOGY") != "1",
    reason="manual replicated-vs-sharded residual topology gate",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)
@pytest.mark.timeout(1800)
def test_replicated_vs_deferred_gather_residual_topology(mesh_device):
    """Measure coherent decoder boundaries, including the following RMSNorm consumer."""
    torch.manual_seed(7)
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=128,
    )
    partial_host = torch.randn(1, 1, 1, config.hidden_size, dtype=torch.bfloat16)
    residual_host = torch.randn_like(partial_host)
    partial = ttnn.from_torch(
        partial_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    residual_replicated = ttnn.from_torch(
        residual_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    residual_sharded = ttnn.from_torch(
        residual_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )

    def replicated_boundary():
        reduced = ttnn.all_reduce(
            partial,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        added = ttnn.add(residual_replicated, reduced, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        reduced.deallocate(True)
        output = ttnn.rms_norm(
            added,
            epsilon=model.rms_norm_eps,
            weight=model.input_norm_weight,
            compute_kernel_config=model.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        added.deallocate(True)
        return output

    def deferred_gather_boundary():
        scattered = ttnn.reduce_scatter(
            partial,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        added_shard = ttnn.add(residual_sharded, scattered, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scattered.deallocate(True)
        gathered = ttnn.all_gather(
            added_shard,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        added_shard.deallocate(True)
        output = ttnn.rms_norm(
            gathered,
            epsilon=model.rms_norm_eps,
            weight=model.input_norm_weight,
            compute_kernel_config=model.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gathered.deallocate(True)
        return output

    replicated_output = replicated_boundary()
    deferred_output = deferred_gather_boundary()
    topology_pcc = comp_pcc(_first_rank(replicated_output), _first_rank(deferred_output))[1]
    assert topology_pcc >= 0.9999
    _assert_replicated(replicated_output)
    _assert_replicated(deferred_output)
    _release_tensors(replicated_output, deferred_output)

    replicated_trace_output, replicated_samples = _trace_mesh_callable(mesh_device, replicated_boundary)
    deferred_trace_output, deferred_samples = _trace_mesh_callable(mesh_device, deferred_gather_boundary)
    replicated_ms = sorted(replicated_samples)[len(replicated_samples) // 2]
    deferred_ms = sorted(deferred_samples)[len(deferred_samples) // 2]
    _write_result_artifact(
        "residual_topology_probe.json",
        {
            "shape": [1, 1, 1, config.hidden_size],
            "dtype": "bfloat16",
            "mesh": list(TARGET_MESH_SHAPE),
            "iterations_per_sample": 100,
            "replicated_all_reduce_add_rmsnorm_ms": replicated_ms,
            "replicated_samples": replicated_samples,
            "reduce_scatter_add_all_gather_rmsnorm_ms": deferred_ms,
            "deferred_gather_samples": deferred_samples,
            "topology_pcc": topology_pcc,
            "selected": "replicated_all_reduce" if replicated_ms <= deferred_ms else "deferred_gather",
        },
    )
    _release_tensors(
        replicated_trace_output,
        deferred_trace_output,
        partial,
        residual_replicated,
        residual_sharded,
    )
    _release_model(model)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_FRACTURED_BOUNDARY") != "1",
    reason="manual safe fractured-residual through distributed norm and next-QKV gate",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)
@pytest.mark.timeout(1800)
def test_safe_fractured_residual_through_distributed_norm_and_qkv(mesh_device):
    """Measure a complete safe RS -> distributed RMSNorm -> next-QKV boundary."""
    torch.manual_seed(20260718)
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=128,
    )
    rows = 32
    hidden = config.hidden_size
    partial_host = torch.randn(1, 1, rows, hidden, dtype=torch.bfloat16) * 0.01
    residual_host = torch.randn_like(partial_host)
    gamma_host = state_dict[f"model.layers.{IR_REPRESENTATIVE_LAYER}.input_layernorm.weight"]
    partial = ttnn.from_torch(
        partial_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    residual_replicated = ttnn.from_torch(
        residual_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    residual_fractured = ttnn.from_torch(
        residual_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    gamma_fractured = ttnn.from_torch(
        gamma_host.reshape(1, 1, hidden // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )

    def next_qkv(normed):
        qkv_input = model._move_owned(normed, model.qkv_input_memory_config)
        output = ttnn.matmul(
            qkv_input,
            model.qkv_decode_weight,
            dtype=ttnn.bfloat16,
            program_config=model.qkv_decode_program_config,
            compute_kernel_config=model.attention_compute_config,
            memory_config=model.qkv_output_memory_config,
        )
        qkv_input.deallocate(True)
        return output

    def replicated_boundary():
        reduced = ttnn.all_reduce(
            partial,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        added = ttnn.add(residual_replicated, reduced, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        reduced.deallocate(True)
        normed = ttnn.rms_norm(
            added,
            epsilon=model.rms_norm_eps,
            weight=model.input_norm_weight,
            compute_kernel_config=model.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        added.deallocate(True)
        return next_qkv(normed)

    def fractured_boundary():
        scattered = ttnn.reduce_scatter(
            partial,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        local_added = ttnn.add(residual_fractured, scattered, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scattered.deallocate(True)
        stats = ttnn.rms_norm_pre_all_gather(
            local_added,
            compute_kernel_config=model.norm_compute_config,
            dtype=ttnn.bfloat16,
        )
        stats = ttnn.reshape(stats, ttnn.Shape((1, 1, rows, 32)))
        gathered_stats = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        stats.deallocate(True)
        local_normed = ttnn.rms_norm_post_all_gather(
            local_added,
            gathered_stats,
            epsilon=model.rms_norm_eps,
            weight=gamma_fractured,
            compute_kernel_config=model.norm_compute_config,
        )
        local_added.deallocate(True)
        gathered_stats.deallocate(True)
        gathered_normed = ttnn.all_gather(
            local_normed,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        local_normed.deallocate(True)
        return next_qkv(gathered_normed)

    replicated_output = replicated_boundary()
    fractured_output = fractured_boundary()
    replicated_host = ttnn.to_torch(
        replicated_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
    )
    fractured_host = ttnn.to_torch(
        fractured_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
    )
    pcc = comp_pcc(replicated_host.float(), fractured_host.float())[1]
    assert pcc >= 0.999
    _release_tensors(replicated_output, fractured_output)

    replicated_trace_output, replicated_samples = _trace_mesh_callable(
        mesh_device, replicated_boundary, samples=5, iterations=100
    )
    fractured_trace_output, fractured_samples = _trace_mesh_callable(
        mesh_device, fractured_boundary, samples=5, iterations=100
    )
    replicated_ms = sorted(replicated_samples)[len(replicated_samples) // 2]
    fractured_ms = sorted(fractured_samples)[len(fractured_samples) // 2]
    _write_result_artifact(
        "fractured_boundary.json",
        {
            "boundary": "row-parallel partial -> residual add -> distributed RMSNorm -> next QKV",
            "mesh": list(TARGET_MESH_SHAPE),
            "shape": [1, 1, rows, hidden],
            "weights": "real layer-14 RMSNorm and QKV",
            "ccl_dtype": "bfloat16",
            "num_links": 2,
            "effective_topology": "Ring",
            "replicated_all_reduce_add_rmsnorm_qkv_ms": replicated_ms,
            "replicated_samples": replicated_samples,
            "fractured_reduce_scatter_add_distributed_rmsnorm_all_gather_qkv_ms": fractured_ms,
            "fractured_samples": fractured_samples,
            "fractured_over_replicated_ratio": fractured_ms / replicated_ms,
            "output_pcc": pcc,
            "selected": "fractured" if fractured_ms < replicated_ms else "replicated",
            "fused_mm_rs_rejected_source": "models/demos/gpt_oss/tt/attention/operations.py:is_shape_fused_mm_rs_supported",
            "fused_mm_rs_issue": "#46181",
            "fused_mm_rs_blocker": "Blackhole race reads matmul blocks before they are fully written; repository gate returns false",
        },
    )
    _release_tensors(
        replicated_trace_output,
        fractured_trace_output,
        partial,
        residual_replicated,
        residual_fractured,
        gamma_fractured,
    )
    _release_model(model)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_HETEROGENEOUS_POSITIONS") != "1",
    reason="manual per-user device-position RoPE and trace gate",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)
@pytest.mark.timeout(1800)
def test_decode_uses_heterogeneous_device_positions_per_user(mesh_device):
    """A batched decode must match independent batch-one calls at distinct positions."""
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    _, decode_31, decode_32 = _recorded_layer14_seq31_inputs(1)
    positions = (17, 31)
    batch_hidden = torch.cat((decode_31, decode_32), dim=0)
    batch_model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=2,
        max_cache_len=128,
    )
    single_model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=128,
    )

    batch_key, batch_value = batch_model.allocate_kv_cache()
    batch_input = _mesh_input(batch_hidden, mesh_device)
    batch_position = _mesh_int32(torch.tensor(positions, dtype=torch.int32), mesh_device)
    batch_output = batch_model.decode_forward(
        batch_input,
        key_cache=batch_key,
        value_cache=batch_value,
        cache_position=batch_position,
        position_index=max(positions),
    )
    _assert_replicated(batch_output)
    actual = _first_rank(batch_output).squeeze(0)

    user_pcc = {}
    key_pcc = {}
    value_pcc = {}
    single_tensors = []
    for user, (position, hidden) in enumerate(zip(positions, (decode_31, decode_32))):
        single_key, single_value = single_model.allocate_kv_cache()
        single_input = _mesh_input(hidden, mesh_device)
        single_position = _mesh_int32(torch.tensor([position], dtype=torch.int32), mesh_device)
        single_output = single_model.decode_forward(
            single_input,
            key_cache=single_key,
            value_cache=single_value,
            cache_position=single_position,
            position_index=position,
        )
        expected = _first_rank(single_output).squeeze(0)[0]
        passed, pcc = comp_pcc(expected.float(), actual[user].float(), pcc=0.999)
        assert passed, f"heterogeneous user {user} position {position}: PCC={pcc}"
        user_pcc[str(position)] = pcc

        batch_key_rank0 = ttnn.to_torch(ttnn.get_device_tensors(batch_key)[0])[user, :, position, :]
        batch_value_rank0 = ttnn.to_torch(ttnn.get_device_tensors(batch_value)[0])[user, :, position, :]
        single_key_rank0 = ttnn.to_torch(ttnn.get_device_tensors(single_key)[0])[0, :, position, :]
        single_value_rank0 = ttnn.to_torch(ttnn.get_device_tensors(single_value)[0])[0, :, position, :]
        key_passed, key_pcc_value = comp_pcc(single_key_rank0.float(), batch_key_rank0.float(), pcc=0.999)
        value_passed, value_pcc_value = comp_pcc(single_value_rank0.float(), batch_value_rank0.float(), pcc=0.999)
        assert key_passed and value_passed
        key_pcc[str(position)] = key_pcc_value
        value_pcc[str(position)] = value_pcc_value
        single_tensors.extend((single_key, single_value, single_input, single_position, single_output))

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = batch_model.decode_forward(
        batch_input,
        key_cache=batch_key,
        value_cache=batch_value,
        cache_position=batch_position,
        position_index=max(positions),
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        first = _first_rank(trace_output)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        second = _first_rank(trace_output)
        assert torch.equal(first, second)
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    _write_result_artifact(
        "heterogeneous_positions.json",
        {
            "weights": "real_layer_14",
            "batch": 2,
            "positions": list(positions),
            "output_pcc_vs_independent_batch1": user_pcc,
            "rank0_key_cache_pcc_vs_independent_batch1": key_pcc,
            "rank0_value_cache_pcc_vs_independent_batch1": value_pcc,
            "rank_outputs_bitwise_equal": True,
            "trace_replays_bitwise_deterministic": True,
        },
    )
    _release_tensors(
        batch_key,
        batch_value,
        batch_input,
        batch_position,
        batch_output,
        trace_output,
        *single_tensors,
    )
    _release_model(batch_model)
    _release_model(single_model)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_LONG_PREFILL") != "1",
    reason="manual non-aligned prefill above the internal MLP chunk boundary",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)
@pytest.mark.timeout(1800)
def test_real_layer_paged_prefill_1025_matches_hf(mesh_device):
    """Exercise the 1,024-row chunk boundary with a non-aligned logical prompt."""
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    base, _, _ = _recorded_layer14_seq31_inputs(1)
    seq_len = 1025
    prefill = base.repeat(1, math.ceil(seq_len / base.shape[1]), 1)[:, :seq_len, :]

    hf_start = time.perf_counter()
    hf_cache = DynamicCache(config=config)
    expected = _hf_prefill(hf_layer, config, prefill, cache=hf_cache)
    hf_seconds = time.perf_counter() - hf_start

    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=seq_len,
    )
    pages = math.ceil(seq_len / model.page_block_size)
    page_table_host = torch.roll(torch.arange(pages, dtype=torch.int32), shifts=3).unsqueeze(0)
    page_table = _mesh_int32(page_table_host, mesh_device)
    key_cache, value_cache = model.allocate_kv_cache(paged=True)
    tt_prefill = _mesh_input(prefill, mesh_device)
    tt_start = time.perf_counter()
    output = model.prefill_forward(
        tt_prefill,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    tt_seconds = time.perf_counter() - tt_start
    _assert_replicated(output)
    actual = _first_rank(output).squeeze(0)
    output_pcc = _assert_pcc("multichip real paged prefill seq=1025", expected, actual, REAL_WEIGHT_PCC)
    hf_layer_cache = hf_cache.layers[IR_REPRESENTATIVE_LAYER]
    actual_key = _paged_cache_to_torch(key_cache, page_table_host, seq_len)
    actual_value = _paged_cache_to_torch(value_cache, page_table_host, seq_len)
    key_pcc = _assert_pcc("multichip paged key cache seq=1025", hf_layer_cache.keys, actual_key, REAL_WEIGHT_PCC)
    value_pcc = _assert_pcc(
        "multichip paged value cache seq=1025", hf_layer_cache.values, actual_value, REAL_WEIGHT_PCC
    )
    assert tuple(output.shape) == (1, 1, seq_len, config.hidden_size)
    assert torch.isfinite(actual).all()
    _write_result_artifact(
        "long_prefill_1025.json",
        {
            "weights": "real_layer_14",
            "mesh": list(TARGET_MESH_SHAPE),
            "batch": 1,
            "logical_sequence_length": seq_len,
            "internal_mlp_chunk_rows": 1024,
            "internal_mlp_chunks": 2,
            "page_block_size": model.page_block_size,
            "page_table": "cyclic_permutation_shift_3",
            "output_pcc_vs_hf": output_pcc,
            "key_cache_pcc_vs_hf": key_pcc,
            "value_cache_pcc_vs_hf": value_pcc,
            "finite_replicated_output": True,
            "hf_reference_seconds": hf_seconds,
            "first_multichip_execution_seconds": tt_seconds,
        },
    )
    _release_tensors(tt_prefill, output, key_cache, value_cache, page_table)
    _release_model(model)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_MAX_CONTEXT") != "1",
    reason="manual advertised-context full-prefill and final-position gate",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)
@pytest.mark.timeout(1800)
def test_batch1_advertised_context_paged_cache_and_last_position(mesh_device):
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    base, _, _ = _recorded_layer14_seq31_inputs(1)
    _, decode_hidden = _recorded_layer14_inputs(1)
    max_cache_len = config.max_position_embeddings
    prefill = base.repeat(1, math.ceil(max_cache_len / base.shape[1]), 1)[:, :max_cache_len, :]
    sample_positions = [0, 31, 1023, 1024, 16383, max_cache_len - 1]
    expected_key, expected_value = _hf_key_value_samples(hf_layer, config, prefill, sample_positions)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=1,
    )
    assert model.max_cache_len == config.max_position_embeddings == 32768
    key_cache, value_cache = model.allocate_kv_cache(paged=True)
    pages = max_cache_len // model.page_block_size
    page_table_host = torch.roll(torch.arange(pages, dtype=torch.int32), shifts=1).unsqueeze(0)
    page_table = _mesh_int32(page_table_host, mesh_device)
    tt_prefill = _mesh_input(prefill, mesh_device)
    prefill_start = time.perf_counter()
    prefill_output = model.prefill_forward(
        tt_prefill,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_seconds = time.perf_counter() - prefill_start
    assert tuple(prefill_output.shape) == (1, 1, max_cache_len, config.hidden_size)
    first_output = ttnn.slice(prefill_output, [0, 0, 0, 0], [1, 1, 1, config.hidden_size])
    last_output = ttnn.slice(
        prefill_output,
        [0, 0, max_cache_len - 1, 0],
        [1, 1, max_cache_len, config.hidden_size],
    )
    _assert_replicated(first_output)
    _assert_replicated(last_output)
    assert torch.isfinite(_first_rank(first_output)).all()
    assert torch.isfinite(_first_rank(last_output)).all()

    actual_key = _paged_cache_to_torch(key_cache, page_table_host, max_cache_len)[:, :, sample_positions, :]
    actual_value = _paged_cache_to_torch(value_cache, page_table_host, max_cache_len)[:, :, sample_positions, :]
    sampled_key_pcc = _assert_pcc("multichip max-context sampled key cache", expected_key, actual_key, REAL_WEIGHT_PCC)
    sampled_value_pcc = _assert_pcc(
        "multichip max-context sampled value cache", expected_value, actual_value, REAL_WEIGHT_PCC
    )
    tt_hidden = _mesh_input(decode_hidden, mesh_device)
    cache_position = _mesh_int32(torch.tensor([max_cache_len - 1], dtype=torch.int32), mesh_device)
    output = model.decode_forward(
        tt_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=max_cache_len - 1,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_replicated(output)
    output_host = _first_rank(output)
    assert tuple(output_host.shape) == (1, 1, 1, config.hidden_size)
    assert torch.isfinite(output_host).all()
    local_cache_shape = list(ttnn.get_device_tensors(key_cache)[0].shape)
    assert local_cache_shape == [pages, 1, model.page_block_size, model.head_dim]
    _write_result_artifact(
        "max_context_batch1.json",
        {
            "max_cache_len": max_cache_len,
            "batch": 1,
            "page_block_size": model.page_block_size,
            "physical_pages_per_device": pages,
            "local_kv_heads_per_device": model.local_num_kv_heads,
            "local_cache_shape_each_k_or_v": local_cache_shape,
            "full_prefill_executed": True,
            "full_prefill_logical_sequence_length": max_cache_len,
            "full_prefill_internal_mlp_chunks": max_cache_len // 1024,
            "full_prefill_seconds": prefill_seconds,
            "sampled_reference_positions": sample_positions,
            "sampled_key_cache_pcc_vs_hf": sampled_key_pcc,
            "sampled_value_cache_pcc_vs_hf": sampled_value_pcc,
            "first_and_last_prefill_outputs_finite_replicated": True,
            "last_position_executed": max_cache_len - 1,
            "page_table": "cyclic_permutation",
            "finite_replicated_output": True,
        },
    )
    _release_tensors(
        key_cache,
        value_cache,
        page_table,
        tt_prefill,
        prefill_output,
        first_output,
        last_output,
        tt_hidden,
        cache_position,
        output,
    )
    _release_model(model)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)
@pytest.mark.timeout(1800)
def test_real_layer_paged_non_aligned_prefill_decode_cache_and_trace(mesh_device):
    """Validate the selected dense layer, local cache ownership, page mapping, stacking, and trace."""
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=128,
    )
    assert model.precision_policy_name == "all_bfp4_lofi"
    assert (model.local_num_heads, model.local_num_kv_heads) == (3, 1)
    assert tuple(model.qkv_weight.shape) == (3072, 1280)
    assert tuple(model.qkv_decode_weight.shape) == (3072, 1280)
    assert model.qkv_decode_padding == 0
    assert tuple(model.o_weight.shape) == (768, 3072)
    assert tuple(model.gate_weight.shape) == (3072, 5760)
    assert tuple(model.gate_decode_weight.shape) == (3072, 6144)
    assert tuple(model.down_decode_weight.shape) == (6144, 3072)
    assert model.mlp_decode_padding == 384
    assert model.dram_banks == 8
    assert (model.qkv_grid.x, model.qkv_grid.y) == (8, 1)
    assert (model.o_grid.x, model.o_grid.y) == (8, 1)
    assert (model.gate_up_grid.x, model.gate_up_grid.y) == (8, 3)
    assert (model.down_grid.x, model.down_grid.y) == (8, 1)
    assert tuple(model.down_weight.shape) == (5760, 3072)
    assert model.visible_device_count == TENSOR_PARALLEL_SIZE
    assert model.detected_num_links == 2
    assert model.num_links == 2
    assert model.topology == ttnn.Topology.Ring

    prefill, decode_31, decode_32 = _recorded_layer14_seq31_inputs(1)
    hf_cache = DynamicCache(config=config)
    expected_prefill = _hf_prefill(hf_layer, config, prefill, cache=hf_cache)
    page_table_host = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
    page_table = _mesh_int32(page_table_host, mesh_device)
    key_cache, value_cache = model.allocate_kv_cache(paged=True)
    tt_prefill = _mesh_input(prefill, mesh_device)
    tt_prefill_output = model.prefill_forward(
        tt_prefill,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    _assert_replicated(tt_prefill_output)
    assert tuple(tt_prefill_output.shape) == (1, 1, 31, config.hidden_size)
    actual_prefill = _first_rank(tt_prefill_output).squeeze(0)
    prefill_pcc = comp_pcc(expected_prefill, actual_prefill)[1]
    _assert_pcc("multichip real prefill seq=31", expected_prefill, actual_prefill, REAL_WEIGHT_PCC)

    hf_layer_cache = hf_cache.layers[IR_REPRESENTATIVE_LAYER]
    actual_key = _paged_cache_to_torch(key_cache, page_table_host, 31)
    actual_value = _paged_cache_to_torch(value_cache, page_table_host, 31)
    prefill_key_pcc = comp_pcc(hf_layer_cache.keys, actual_key)[1]
    prefill_value_pcc = comp_pcc(hf_layer_cache.values, actual_value)[1]
    _assert_pcc("multichip paged key cache seq=31", hf_layer_cache.keys, actual_key, REAL_WEIGHT_PCC)
    _assert_pcc("multichip paged value cache seq=31", hf_layer_cache.values, actual_value, REAL_WEIGHT_PCC)

    outputs = []
    decode_pccs = {}
    cache_pccs = {}
    tt_decode_inputs = []
    tt_positions = []
    for position, decode_hidden in ((31, decode_31), (32, decode_32)):
        expected = _hf_decode(hf_layer, config, decode_hidden, hf_cache, position)
        tt_hidden = _mesh_input(decode_hidden, mesh_device)
        tt_position = _mesh_int32(torch.tensor([position], dtype=torch.int32), mesh_device)
        tt_output = model.decode_forward(
            tt_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=tt_position,
            position_index=position,
            page_table=page_table,
        )
        _assert_replicated(tt_output)
        actual = _first_rank(tt_output).squeeze(0)
        decode_pccs[str(position)] = comp_pcc(expected, actual)[1]
        _assert_pcc(f"multichip real decode position={position}", expected, actual, REAL_WEIGHT_PCC)
        outputs.append(tt_output)
        tt_decode_inputs.append(tt_hidden)
        tt_positions.append(tt_position)

        logical_length = position + 1
        actual_key = _paged_cache_to_torch(key_cache, page_table_host, logical_length)
        actual_value = _paged_cache_to_torch(value_cache, page_table_host, logical_length)
        cache_pccs[str(position)] = {
            "key": comp_pcc(hf_cache.layers[IR_REPRESENTATIVE_LAYER].keys, actual_key)[1],
            "value": comp_pcc(hf_cache.layers[IR_REPRESENTATIVE_LAYER].values, actual_value)[1],
        }
        _assert_pcc(
            f"multichip paged key cache position={position}",
            hf_cache.layers[IR_REPRESENTATIVE_LAYER].keys,
            actual_key,
            REAL_WEIGHT_PCC,
        )
        _assert_pcc(
            f"multichip paged value cache position={position}",
            hf_cache.layers[IR_REPRESENTATIVE_LAYER].values,
            actual_value,
            REAL_WEIGHT_PCC,
        )

    # A replicated DRAM output is directly stackable as the next layer's input.
    stacked = model.decode_forward(
        outputs[-1],
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=tt_positions[-1],
        position_index=32,
        page_table=page_table,
    )
    assert tuple(stacked.shape) == (1, 1, 1, config.hidden_size)
    _assert_replicated(stacked)
    stacked.deallocate(True)

    # Warm the exact fixed-position path, then prove mesh trace capture/replay.
    warm = model.decode_forward(
        tt_decode_inputs[-1],
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=tt_positions[-1],
        position_index=32,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    entries_before = mesh_device.num_program_cache_entries()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        tt_decode_inputs[-1],
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=tt_positions[-1],
        position_index=32,
        page_table=page_table,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        first = _first_rank(trace_output)
        for _ in range(8):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        second = _first_rank(trace_output)
        assert torch.equal(first, second), "mesh trace replay is not bitwise deterministic"
        assert mesh_device.num_program_cache_entries() == entries_before
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    _write_result_artifact(
        "paged_non_aligned_correctness.json",
        {
            "weights": "real_layer_14",
            "mesh": list(TARGET_MESH_SHAPE),
            "batch": 1,
            "prefill_sequence_length": 31,
            "page_table": [2, 0, 3, 1],
            "prefill_pcc_vs_hf": prefill_pcc,
            "prefill_key_cache_pcc_vs_hf": prefill_key_pcc,
            "prefill_value_cache_pcc_vs_hf": prefill_value_pcc,
            "decode_pcc_vs_hf": decode_pccs,
            "decode_cache_pcc_vs_hf": cache_pccs,
            "rank_outputs_bitwise_equal": True,
            "trace_replays": 8,
            "trace_replay_bitwise_deterministic": True,
            "trace_program_cache_entries_stable": True,
            "stacked_decoder_layout": [1, 1, 1, config.hidden_size],
        },
    )

    _release_tensors(
        tt_prefill,
        tt_prefill_output,
        page_table,
        key_cache,
        value_cache,
        trace_output,
        *outputs,
        *tt_decode_inputs,
        *tt_positions,
    )
    _release_model(model)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_BASELINE") != "1",
    reason="manual direct optimized-baseline comparison owns and reopens devices",
)
@pytest.mark.timeout(1800)
def test_multichip_directly_matches_single_chip_optimized_baseline():
    """Run identical real layer-14 inputs first on one chip, then on the 1x4 ring."""
    if ttnn.get_num_devices() < TENSOR_PARALLEL_SIZE:
        pytest.skip("four Blackhole devices are required")
    config = _config()
    batch = 32
    seq_len = 17
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    prefill, decode_hidden = _recorded_layer14_inputs(batch)

    baseline_device = ttnn.open_device(device_id=0, trace_region_size=100_000_000)
    try:
        baseline = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=baseline_device,
            batch=batch,
            max_cache_len=128,
        )
        baseline_input = _single_device_input(prefill, baseline_device)
        baseline_key, baseline_value = baseline.allocate_kv_cache()
        baseline_prefill_tt = baseline.prefill_forward(
            baseline_input,
            key_cache=baseline_key,
            value_cache=baseline_value,
        )
        baseline_prefill = ttnn.to_torch(baseline_prefill_tt).squeeze(0)
        baseline_decode_input = _single_device_input(decode_hidden, baseline_device)
        baseline_position = _single_device_position(batch, seq_len, baseline_device)
        baseline_decode_tt = baseline.decode_forward(
            baseline_decode_input,
            key_cache=baseline_key,
            value_cache=baseline_value,
            cache_position=baseline_position,
            position_index=seq_len,
        )
        baseline_decode = ttnn.to_torch(baseline_decode_tt).squeeze(0)
        baseline_key_host = ttnn.to_torch(baseline_key)[:, :, : seq_len + 1, :]
        baseline_value_host = ttnn.to_torch(baseline_value)[:, :, : seq_len + 1, :]
        _release_tensors(
            baseline_input,
            baseline_key,
            baseline_value,
            baseline_prefill_tt,
            baseline_decode_input,
            baseline_position,
            baseline_decode_tt,
        )
        _release_model(baseline)
    finally:
        ttnn.close_device(baseline_device)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = None
    try:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*TARGET_MESH_SHAPE),
            trace_region_size=100_000_000,
        )
        model = MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=128,
        )
        tt_prefill = _mesh_input(prefill, mesh_device)
        key_cache, value_cache = model.allocate_kv_cache()
        tt_prefill_output = model.prefill_forward(
            tt_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        actual_prefill = _first_rank(tt_prefill_output).squeeze(0)
        tt_decode = _mesh_input(decode_hidden, mesh_device)
        tt_position = _mesh_int32(torch.full((batch,), seq_len, dtype=torch.int32), mesh_device)
        tt_decode_output = model.decode_forward(
            tt_decode,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=tt_position,
            position_index=seq_len,
        )
        actual_decode = _first_rank(tt_decode_output).squeeze(0)
        prefill_pcc = comp_pcc(baseline_prefill, actual_prefill)[1]
        decode_pcc = comp_pcc(baseline_decode, actual_decode)[1]
        key_pcc = comp_pcc(
            baseline_key_host,
            _contiguous_cache_to_torch(key_cache, seq_len + 1),
        )[1]
        value_pcc = comp_pcc(
            baseline_value_host,
            _contiguous_cache_to_torch(value_cache, seq_len + 1),
        )[1]
        assert prefill_pcc >= REAL_WEIGHT_PCC
        assert decode_pcc >= REAL_WEIGHT_PCC
        assert key_pcc >= REAL_WEIGHT_PCC
        assert value_pcc >= REAL_WEIGHT_PCC
        _write_result_artifact(
            "direct_optimized_baseline_pcc.json",
            {
                "batch": batch,
                "sequence_length": seq_len,
                "weights": "real_layer_14",
                "single_chip_baseline": "OptimizedDecoder defaults",
                "multichip": "MultichipDecoder TP=4 defaults",
                "prefill_pcc": prefill_pcc,
                "decode_pcc": decode_pcc,
                "key_cache_pcc": key_pcc,
                "value_cache_pcc": value_pcc,
            },
        )
        _release_tensors(
            tt_prefill,
            key_cache,
            value_cache,
            tt_prefill_output,
            tt_decode,
            tt_position,
            tt_decode_output,
        )
        _release_model(model)
    finally:
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        gc.collect()


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_PERF") != "1",
    reason="manual warmed mesh performance gate",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)
@pytest.mark.timeout(1800)
def test_warmed_multichip_trace_performance(mesh_device):
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    batch = int(os.getenv("FALCON3_MULTICHIP_PERF_BATCH", "1"))
    if batch not in (1, 32):
        raise ValueError("performance workload batch must be 1 or 32")
    prefill, decode_hidden = _recorded_layer14_inputs(batch)
    legacy_mlp_target_cores = os.getenv("FALCON3_MULTICHIP_MLP_CORES")
    gate_up_target_cores = int(os.getenv("FALCON3_MULTICHIP_GATE_UP_CORES", legacy_mlp_target_cores or "24"))
    down_target_cores = int(os.getenv("FALCON3_MULTICHIP_DOWN_CORES", legacy_mlp_target_cores or "8"))
    qkv_target_cores = int(os.getenv("FALCON3_MULTICHIP_QKV_CORES", "8"))
    o_target_cores = int(os.getenv("FALCON3_MULTICHIP_O_CORES", "8"))
    ccl_dtype_name = os.getenv("FALCON3_MULTICHIP_CCL_DTYPE", "bf16")
    ccl_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[ccl_dtype_name]
    num_links = int(os.getenv("FALCON3_MULTICHIP_NUM_LINKS", "2"))
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=128,
        qkv_target_cores=qkv_target_cores,
        gate_up_target_cores=gate_up_target_cores,
        down_target_cores=down_target_cores,
        o_target_cores=o_target_cores,
        ccl_dtype=ccl_dtype,
        num_links=num_links,
    )
    tt_prefill = _mesh_input(prefill, mesh_device)
    key_cache, value_cache = model.allocate_kv_cache()
    prefill_warm = model.prefill_forward(tt_prefill, key_cache=key_cache, value_cache=value_cache)
    ttnn.synchronize_device(mesh_device)
    prefill_warm.deallocate(True)
    prefill_samples = []
    for _ in range(5):
        start = time.perf_counter()
        prefill_output = model.prefill_forward(tt_prefill, key_cache=key_cache, value_cache=value_cache)
        ttnn.synchronize_device(mesh_device)
        prefill_samples.append((time.perf_counter() - start) * 1000.0)
        prefill_output.deallocate(True)
    multichip_prefill_ms = sorted(prefill_samples)[len(prefill_samples) // 2]

    tt_hidden = _mesh_input(decode_hidden, mesh_device)
    cache_position = _mesh_int32(torch.full((batch,), 17, dtype=torch.int32), mesh_device)
    warm = model.decode_forward(
        tt_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=17,
    )
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        tt_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=17,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for _ in range(10):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(100):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter() - start) * 10.0)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    multichip_ms = sorted(samples)[len(samples) // 2]
    baseline_path = OPTIMIZED_BATCH1_RESULT if batch == 1 else OPTIMIZED_BATCH32_RESULT
    baseline_payload = json.loads(baseline_path.read_text())
    baseline_result = baseline_payload["results"]["optimized_selected_dram_all_bfp4_48c"]
    baseline_ms = baseline_result["decode_ms"]
    decode_speedup = baseline_ms / multichip_ms
    decode_efficiency = decode_speedup / TENSOR_PARALLEL_SIZE
    baseline_prefill_ms = baseline_result.get("prefill_ms")
    prefill_speedup = None if baseline_prefill_ms is None else baseline_prefill_ms / multichip_prefill_ms
    prefill_efficiency = None if prefill_speedup is None else prefill_speedup / TENSOR_PARALLEL_SIZE
    _write_result_artifact(
        os.getenv("FALCON3_MULTICHIP_PERF_FILENAME", f"final_batch{batch}.json"),
        {
            "batch": batch,
            "sequence_length": 17,
            "weights": "real_layer_14",
            "qkv_target_cores": qkv_target_cores,
            "gate_up_target_cores": gate_up_target_cores,
            "down_target_cores": down_target_cores,
            "o_target_cores": o_target_cores,
            "physical_decode_geometry": {
                "qkv_grid": [model.qkv_grid.x, model.qkv_grid.y],
                "qkv_grid_cores": model.qkv_grid.x * model.qkv_grid.y,
                "qkv_local_width": model.local_qkv_size,
                "qkv_padded_width": model.local_qkv_decode_size,
                "qkv_padding": model.qkv_decode_padding,
                "o_grid": [model.o_grid.x, model.o_grid.y],
                "o_grid_cores": model.o_grid.x * model.o_grid.y,
                "o_local_input_width": model.local_hidden_size,
                "gate_up_grid": [model.gate_up_grid.x, model.gate_up_grid.y],
                "gate_up_grid_cores": model.gate_up_grid.x * model.gate_up_grid.y,
                "mlp_local_width": model.local_intermediate_size,
                "mlp_padded_width": model.local_intermediate_decode_size,
                "mlp_padding": model.mlp_decode_padding,
                "down_grid": [model.down_grid.x, model.down_grid.y],
                "down_grid_cores": model.down_grid.x * model.down_grid.y,
            },
            "ccl_dtype": ccl_dtype_name,
            "num_links": num_links,
            "effective_ccl_topology": "Ring",
            "prefill_samples": prefill_samples,
            "multichip_prefill_ms": multichip_prefill_ms,
            "single_chip_prefill_ms": baseline_prefill_ms,
            "prefill_speedup": prefill_speedup,
            "prefill_parallel_efficiency": prefill_efficiency,
            "iterations_per_sample": 100,
            "samples": samples,
            "single_chip_baseline_ms": baseline_ms,
            "single_chip_provenance": str(baseline_path.relative_to(Path(__file__).parents[4])),
            "multichip_trace_ms": multichip_ms,
            "speedup": decode_speedup,
            "parallel_efficiency": decode_efficiency,
        },
    )
    _release_tensors(tt_prefill, tt_hidden, cache_position, key_cache, value_cache, trace_output)
    _release_model(model)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_PROFILE") != "1",
    reason="manual Tracy profile gate",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)
@pytest.mark.timeout(1800)
def test_multichip_profile_signposts(mesh_device):
    from tracy import signpost

    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    prefill, decode_hidden = _recorded_layer14_inputs(1)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=128,
    )
    key_cache, value_cache = model.allocate_kv_cache()
    tt_prefill = _mesh_input(prefill, mesh_device)
    prefill_warm = model.prefill_forward(tt_prefill, key_cache=key_cache, value_cache=value_cache)
    ttnn.synchronize_device(mesh_device)
    prefill_warm.deallocate(True)
    ttnn.ReadDeviceProfiler(mesh_device)
    signpost(header="MULTICHIP_PREFILL")
    prefill_output = model.prefill_forward(tt_prefill, key_cache=key_cache, value_cache=value_cache)
    ttnn.synchronize_device(mesh_device)
    signpost(header="MULTICHIP_PREFILL_END")
    ttnn.ReadDeviceProfiler(mesh_device)

    tt_decode = _mesh_input(decode_hidden, mesh_device)
    cache_position = _mesh_int32(torch.tensor([17], dtype=torch.int32), mesh_device)
    warm = model.decode_forward(
        tt_decode,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=17,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        tt_decode,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=17,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    signpost(header="MULTICHIP_DECODE")
    for _ in range(3):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    signpost(header="MULTICHIP_DECODE_END")
    ttnn.ReadDeviceProfiler(mesh_device)
    ttnn.release_trace(mesh_device, trace_id)
    _release_tensors(
        key_cache,
        value_cache,
        tt_prefill,
        prefill_output,
        tt_decode,
        cache_position,
        warm,
        trace_output,
    )
    _release_model(model)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import random

from loguru import logger
import pytest
import torch
import ttnn

from tests.nightly.tg.ccl.moe.test_moe_compute_6U import (
    create_sharded_memory_config,
    create_torch_w0,
    create_torch_w1,
    create_torch_w2,
    gen_expert_mapping,
)
from ttnn.experimental.moe_compute_utils import (
    determine_compute_matmul_cores,
    get_w0_w1_memory_config,
    get_w2_memory_config,
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w2_tensor_for_moe_compute,
)


MESH_GRAPH_DESC_QUAD = "tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto"


def _is_quad_mesh_graph_descriptor_set() -> bool:
    return os.environ.get("TT_MESH_GRAPH_DESC_PATH", "").endswith(MESH_GRAPH_DESC_QUAD)


def _torch_dtype(tt_dtype):
    if tt_dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        return torch.bfloat16
    if tt_dtype == ttnn.float32:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {tt_dtype}")


def _sha256(tensor: torch.Tensor) -> str:
    host = tensor.detach().contiguous().cpu()
    return hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()


def _hash_mesh_tensor(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice, *, dim: int = 0) -> str:
    host = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
    return _sha256(host)


def _hash_combine_tensor(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> str:
    return _hash_mesh_tensor(tensor, mesh_device, dim=1)


def _make_sparse_inputs(
    *,
    tokens_per_device: int,
    hidden_size: int,
    experts: int,
    selected_experts_k: int,
    mesh_shape: tuple[int, int],
    cluster_axis: int,
    expert_mapping: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis]
    total_tokens = tokens_per_device * num_dispatch_devices

    original_tokens = torch.rand(num_dispatch_devices, tokens_per_device, hidden_size, dtype=dtype) - 0.5
    expert_indices = torch.zeros(num_dispatch_devices, tokens_per_device, selected_experts_k, dtype=torch.uint16)
    for src_device in range(num_dispatch_devices):
        for token_idx in range(tokens_per_device):
            expert_indices[src_device, token_idx, :] = torch.randperm(experts)[:selected_experts_k].to(torch.uint16)

    expert_scores = torch.rand(num_dispatch_devices, tokens_per_device, selected_experts_k, dtype=dtype) + 1e-5
    expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)

    sparse_buffer = torch.rand(num_devices, total_tokens, hidden_size, dtype=dtype)
    for src_device in range(num_dispatch_devices):
        for token_idx in range(tokens_per_device):
            sparse_token_idx = src_device * tokens_per_device + token_idx
            token = original_tokens[src_device, token_idx, :]
            for k in range(selected_experts_k):
                expert_id = int(expert_indices[src_device, token_idx, k].item())
                target_device = int(expert_mapping[src_device, expert_id].item())
                sparse_buffer[target_device, sparse_token_idx, :] = token

    return sparse_buffer, expert_indices, expert_scores


@pytest.mark.skipif(
    not _is_quad_mesh_graph_descriptor_set(),
    reason=f"Requires QUAD mesh graph descriptor ending in {MESH_GRAPH_DESC_QUAD}",
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((16, 8), (16, 8))], indirect=["mesh_device"])
def test_moe_compute_quad_compute_output_determinism(mesh_device, mesh_shape, device_params):
    """Repro for DeepSeek QUAD MoE nondeterminism: same inputs should produce same raw compute output."""

    torch.manual_seed(2003)
    random.seed(2003)

    cluster_axis = 0
    num_layers = 1
    tokens_per_device = 8
    selected_experts_k = 8
    hidden_size = 7168
    matmul_n = 2048
    output_height_shard_dim = 4
    output_width_shard_dim = 4
    dtype = ttnn.bfloat16
    experts = 256

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis]
    num_replicated_devices = num_devices // num_dispatch_devices
    total_tokens = tokens_per_device * num_dispatch_devices
    experts_per_cluster = experts // num_replicated_devices
    experts_per_device = experts // num_devices

    expert_mapping = gen_expert_mapping(
        num_devices, num_replicated_devices, cluster_axis, experts, experts_per_cluster, experts_per_device
    )
    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    sparse_buffer, expert_indices, expert_scores = _make_sparse_inputs(
        tokens_per_device=tokens_per_device,
        hidden_size=hidden_size,
        experts=experts,
        selected_experts_k=selected_experts_k,
        mesh_shape=mesh_shape,
        cluster_axis=cluster_axis,
        expert_mapping=expert_mapping,
        dtype=_torch_dtype(dtype),
    )

    tilize_drain_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(6, 9), ttnn.CoreCoord(6, 9))})
    expert_indices_mem_config = create_sharded_memory_config(
        tilize_drain_core, [total_tokens, selected_experts_k], ttnn.uint16
    )
    expert_scores_mem_config = create_sharded_memory_config(tilize_drain_core, [total_tokens, selected_experts_k], dtype)

    tt_sparse_buffer = ttnn.from_torch(
        sparse_buffer,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    expert_indices_flat = expert_indices.reshape(total_tokens, selected_experts_k)
    expert_indices_replicated = expert_indices_flat.unsqueeze(0).repeat(num_devices, 1, 1)
    tt_expert_indices = ttnn.from_torch(
        expert_indices_replicated,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=expert_indices_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    expert_scores_flat = expert_scores.reshape(total_tokens, selected_experts_k)
    expert_scores_replicated = expert_scores_flat.unsqueeze(0).repeat(num_devices, 1, 1)
    tt_expert_scores = ttnn.from_torch(
        expert_scores_replicated,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=expert_scores_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    ring2cores, dram_core_range_set = determine_compute_matmul_cores(mesh_device)
    torch_w0 = create_torch_w0(num_layers, experts_per_device, hidden_size, matmul_n)
    torch_w1 = create_torch_w1(num_layers, experts_per_device, hidden_size, matmul_n)
    torch_w2 = create_torch_w2(num_layers, experts_per_device, matmul_n, hidden_size)

    tt_w0_w1 = ttnn.from_torch(
        prepare_w0_w1_tensor_for_moe_compute(
            torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, matmul_n, ring2cores
        ),
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=get_w0_w1_memory_config(num_layers, experts_per_device, hidden_size, dram_core_range_set),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_w2 = ttnn.from_torch(
        prepare_w2_tensor_for_moe_compute(torch_w2, num_layers, experts_per_device, matmul_n, hidden_size, ring2cores),
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=get_w2_memory_config(num_layers, experts_per_device, matmul_n, dram_core_range_set),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    combine_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(c, c) for c in ttnn.experimental.get_moe_combine_cores(mesh_device)]
    )
    combine_barrier_semaphore = ttnn.create_global_semaphore(mesh_device, combine_core_range_set, 0)
    mux_core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(4, 7))})

    def run_once(iteration: int) -> tuple[str, str]:
        optional_output_tensor = ttnn.from_torch(
            torch.zeros([selected_experts_k, total_tokens, hidden_size], dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )

        outputs = ttnn.experimental.moe_compute(
            tt_sparse_buffer,
            tt_expert_indices,
            tt_expert_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=0,
            output_height_shard_dim=output_height_shard_dim,
            output_width_shard_dim=output_width_shard_dim,
            cluster_axis=cluster_axis,
            mux_core_range_set=mux_core_range_set,
            optional_output_tensor=optional_output_tensor,
            optional_cross_device_semaphore=combine_barrier_semaphore,
        )
        ttnn.synchronize_device(mesh_device)

        matmul_hash = _hash_mesh_tensor(outputs[4], mesh_device, dim=0)
        logger.info(f"iteration={iteration} matmul_output_sha256={matmul_hash}")
        combine_hash = _hash_combine_tensor(outputs[5], mesh_device)
        logger.info(f"iteration={iteration} matmul_output_sha256={matmul_hash} combine_output_sha256={combine_hash}")

        ttnn.deallocate(outputs[0])
        ttnn.deallocate(outputs[1])
        ttnn.deallocate(outputs[2])
        ttnn.deallocate(outputs[4])
        ttnn.deallocate(outputs[5])
        return matmul_hash, combine_hash

    first_matmul_hash, first_combine_hash = run_once(0)
    second_matmul_hash, second_combine_hash = run_once(1)

    assert first_matmul_hash == second_matmul_hash and first_combine_hash == second_combine_hash, (
        "ttnn.experimental.moe_compute is nondeterministic for fixed inputs: "
        f"matmul iter0={first_matmul_hash}, iter1={second_matmul_hash}; "
        f"combine iter0={first_combine_hash}, iter1={second_combine_hash}."
    )

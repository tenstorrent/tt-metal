# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TG (Single Galaxy) Dispatch Test - 4x8 Mesh

Tests all_to_all_dispatch_metadata operation on single galaxy (32 devices, 4x8 mesh).

This test validates dispatch operations with:
- 64 experts (2 per device) - same per-device workload as quad
- 4x8 mesh configuration
- Both correctness and performance modes

If this test fails, dispatch is likely broken on quad as well.
"""

import random

import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.t3000.ccl.test_all_to_all_dispatch import get_mesh_mapper, tt_to_torch_dtype
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import gen_expert_mapping


def gen_tensors_for_metadata_op(
    batch,
    experts,
    selected_experts_k,
    hidden_size,
    seq_len,
    mesh_shape,
    devices,
    cluster_axis=1,
    scheme="random",
    dtype=torch.bfloat16,
):
    """Generate tensors for the all_to_all_dispatch_metadata operation."""
    num_dispatch_devices = mesh_shape[cluster_axis] if cluster_axis is not None else devices
    num_replicated_devices = devices // num_dispatch_devices
    experts_per_cluster = experts // num_replicated_devices
    experts_per_device = experts // devices

    # Generate input tokens
    input_tokens = torch.rand(batch, 1, seq_len, hidden_size, dtype=dtype) - 0.5

    # Generate expert indices (ensure no repeats within a token)
    expert_indices = torch.zeros(batch, 1, seq_len, selected_experts_k, dtype=torch.int16)
    for b in range(batch):
        for s in range(seq_len):
            # Randomly sample selected_experts_k unique experts
            selected = torch.randperm(experts)[:selected_experts_k].to(torch.int16)
            expert_indices[b, 0, s, :] = selected

    # Generate new format expert mapping
    expert_mapping_new = gen_expert_mapping(
        devices, num_replicated_devices, cluster_axis, experts, experts_per_cluster, experts_per_device
    )

    # Generate golden output and metadata tensors using the new expert mapping
    total_tokens = batch * seq_len

    # Metadata golden: expert_indices replicated across all devices
    # Shape: [batch, 1, seq_len, k] -> [1, total_tokens, k] -> [devices, total_tokens, k]
    expert_indices_flat = expert_indices.reshape(1, total_tokens, selected_experts_k)
    metadata_tensor = expert_indices_flat.repeat(devices, 1, 1)

    # Scores golden: same as metadata, normalized expert scores replicated across all devices
    expert_scores = torch.rand(expert_indices.shape, dtype=torch.float32).to(dtype)
    expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)
    scores_reshaped = expert_scores.permute(1, 0, 2, 3)  # [1, batch, seq_len, k]
    scores_golden = scores_reshaped.repeat(devices, 1, 1, 1)  # [devices, batch, seq_len, k]
    scores_tensor = scores_golden.reshape(devices, total_tokens, selected_experts_k)

    # Output tensor golden: route tokens to devices based on expert ownership
    sparse_output_token_tensor = torch.rand(devices, total_tokens, hidden_size, dtype=dtype)
    for b in range(batch):
        for s in range(seq_len):
            t = b * seq_len + s  # Token index in total_tokens
            for k in range(selected_experts_k):
                expert_id = expert_indices[b, 0, s, k].item()
                # Look up which device owns this expert (use device 0's view of the mapping)
                target_device = expert_mapping_new[0, expert_id].item()
                # Copy token to that device's output
                sparse_output_token_tensor[target_device, t, :] = input_tokens[b, 0, s, :]

    return (
        input_tokens,
        expert_indices,
        expert_scores,
        expert_mapping_new,
        sparse_output_token_tensor,
        metadata_tensor,
        scores_tensor,
    )


def run_all_to_all_dispatch_metadata_test(
    mesh_device,
    mesh_shape,
    batch,
    experts,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    trace_mode,
    num_links=4,
    scheme="random_sequential_experts",
    dtype=ttnn.bfloat16,
    cluster_axis=0,
    shard_dim=0,
):
    """Run dispatch test on TG mesh."""
    torch.manual_seed(2005)
    random.seed(2005)
    mesh_device.enable_program_cache()
    devices = mesh_shape[0] * mesh_shape[1]

    expert_indices_tensors = []
    expert_scores_tensors = []
    expert_mapping_new_tensors = []
    input_tensors = []

    torch_expert_mappings_new = []
    torch_expert_scores_list = []

    output_tensor_goldens_list = []
    output_metadata_goldens_list = []
    output_scores_goldens_list = []
    mesh_mapper = get_mesh_mapper(mesh_device, mesh_shape, cluster_axis, shard_dim)

    if cluster_axis == 1:
        shard_dims = (None, shard_dim)
    else:
        shard_dims = (shard_dim, None)

    total_tokens = batch * seq_len
    num_dispatch_devices = mesh_shape[cluster_axis]
    tokens_per_device = batch // num_dispatch_devices

    num_cores_y = min(8, tokens_per_device)
    num_cores_x = (tokens_per_device + num_cores_y - 1) // num_cores_y
    input_indices_scores_core_range = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
    )
    input_indices_shard_spec = ttnn.ShardSpec(
        input_indices_scores_core_range,
        [1, seq_len * select_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_indices_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        input_indices_shard_spec,
    )

    for iter in range(num_iters):
        (
            input_tokens,
            expert_indices,
            expert_scores,
            expert_mapping_new,
            sparse_output_token_tensor,
            metadata_tensor,
            scores_tensor,
        ) = gen_tensors_for_metadata_op(
            batch,
            experts,
            select_experts_k,
            hidden_size,
            seq_len,
            mesh_shape,
            devices,
            cluster_axis=cluster_axis,
            scheme=scheme,
            dtype=tt_to_torch_dtype(dtype),
        )

        output_tensor_goldens_list.append(sparse_output_token_tensor)
        output_metadata_goldens_list.append(metadata_tensor)
        output_scores_goldens_list.append(scores_tensor)
        torch_expert_mappings_new.append(expert_mapping_new)
        torch_expert_scores_list.append(expert_scores)

        tt_input = ttnn.from_torch(
            input_tokens,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        tt_expert_indices = ttnn.from_torch(
            expert_indices,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_indices_sharded_mem_config,
            mesh_mapper=mesh_mapper,
        )

        tt_expert_scores = ttnn.from_torch(
            expert_scores,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=input_indices_sharded_mem_config,
            mesh_mapper=mesh_mapper,
        )

        tt_expert_mapping_new = ttnn.from_torch(
            expert_mapping_new,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )

        input_tensors.append(tt_input)
        expert_indices_tensors.append(tt_expert_indices)
        expert_scores_tensors.append(tt_expert_scores)
        expert_mapping_new_tensors.append(tt_expert_mapping_new)

    tt_out_tensor_list = []

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []
        tt_metadata_list = []
        tt_scores_out_list = []

        for i in range(n_iters):
            output_tensor, indices_tensor, scores_tensor = ttnn.experimental.all_to_all_dispatch_metadata(
                input_tensors[i],
                expert_indices_tensors[i],
                expert_scores_tensors[i],
                expert_mapping_new_tensors[i],
                cluster_axis=cluster_axis,
                num_links=num_links,
                drain_sync_tilizer_core=(0, 0),
                worker_mode=ttnn.WorkerMode.DIRECT,
                dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
            )

            if not trace_mode:
                ttnn.synchronize_device(mesh_device)
            if store_all_results:
                tt_output_list.append(output_tensor)
                tt_metadata_list.append(indices_tensor)
                tt_scores_out_list.append(scores_tensor)

        if store_all_results:
            return tt_output_list, tt_metadata_list, tt_scores_out_list
        else:
            return [output_tensor], [indices_tensor], [scores_tensor]

    if trace_mode:
        logger.info("Compiling model")
        run_op(1, store_all_results=True)
        ttnn.synchronize_device(mesh_device)

        logger.info("Capturing Trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_out_tensor_list, tt_metadata_list, tt_scores_out_list = run_op(num_iters, store_all_results=True)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        logger.info("Executing Trace")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)
    else:
        tt_out_tensor_list, tt_metadata_list, tt_scores_out_list = run_op(num_iters, store_all_results=True)

    # Validation
    passed = True
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_torch_tensor = ttnn.to_torch(
            tt_out_tensor_list[tensor_index],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim),
        )

        tt_metadata_tensor = ttnn.to_torch(
            tt_metadata_list[tensor_index],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim),
        )

        tt_scores_out_tensor = ttnn.to_torch(
            tt_scores_out_list[tensor_index],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim),
        )

        devices = tt_metadata_tensor.shape[0]
        total_tokens_out = tt_metadata_tensor.shape[1]
        selected_experts_k = tt_metadata_tensor.shape[2]

        # Verify metadata
        metadata_all_close = torch.allclose(
            tt_metadata_tensor, output_metadata_goldens_list[tensor_index].to(torch.uint16)
        )
        if not metadata_all_close:
            logger.warning(f"FAILED metadata validation at iteration {tensor_index}")
            passed = False

        # Verify scores
        scores_all_close = torch.allclose(
            tt_scores_out_tensor, output_scores_goldens_list[tensor_index], rtol=1e-2, atol=1e-2
        )
        if not scores_all_close:
            logger.warning(f"FAILED scores validation at iteration {tensor_index}")
            passed = False

        # Verify output tokens
        if tensor_index == 0:
            logger.info(f"Output tensor shape: {tt_torch_tensor.shape}")
            logger.info(f"Golden tensor shape: {output_tensor_goldens_list[tensor_index].shape}")
            logger.info(f"Metadata tensor shape: {tt_metadata_tensor.shape}")
            logger.info(
                f"devices={devices}, total_tokens_out={total_tokens_out}, selected_experts_k={selected_experts_k}"
            )

        tokens_per_src_device = total_tokens_out // devices
        for t in range(total_tokens_out):
            src_device = t // tokens_per_src_device
            for k in range(selected_experts_k):
                expert_id = tt_metadata_tensor[src_device, t, k]
                target_device = torch_expert_mappings_new[tensor_index][src_device, expert_id].item()
                is_all_equal = torch.equal(
                    tt_torch_tensor[target_device, t, :],
                    output_tensor_goldens_list[tensor_index][target_device, t, :],
                )
                if not is_all_equal:
                    if tensor_index == 0 and t == 0:
                        logger.warning(
                            f"FAILED at iteration {tensor_index}, token {t}, expert_id {expert_id}, "
                            f"src_device {src_device}, target_device {target_device}"
                        )
                        failed_indices = torch.where(
                            tt_torch_tensor[target_device, t, :]
                            != output_tensor_goldens_list[tensor_index][target_device, t, :]
                        )
                        logger.warning(f"First 10 failing indices: {failed_indices[0][:10]}")
                        logger.warning(
                            f"TT output (first 10): {tt_torch_tensor[target_device, t, failed_indices[0][:10]]}"
                        )
                        logger.warning(
                            f"Golden (first 10): {output_tensor_goldens_list[tensor_index][target_device, t, failed_indices[0][:10]]}"
                        )
                    passed = False
                    break
            if not passed:
                break

    logger.info(f"Device has {mesh_device.num_program_cache_entries()} program cache entries")
    assert passed, "TG Dispatch test failed!"
    logger.info("✓ TG Dispatch test passed!")


@pytest.mark.requires_device("TG")  # Only run on single galaxy
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device, cluster_axis",
    [
        pytest.param((4, 8), (4, 8), 0, id="4x8_tg"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("experts_per_device", [2])
def test_correctness(mesh_device, mesh_shape, cluster_axis, experts_per_device):
    """Correctness test for TG dispatch."""
    batches_per_device = 32
    num_devices = mesh_shape[0] * mesh_shape[1]  # Total devices = 32 for 4x8
    experts = experts_per_device * num_devices  # 2 * 32 = 64 experts
    select_experts_k = 8
    hidden_size = 7168
    seq_len = 1
    num_iters = 10
    num_links = 4
    dtype = ttnn.bfloat16
    scheme = "random_sequential_experts"

    dispatch_devices = mesh_shape[cluster_axis]
    batch = batches_per_device * dispatch_devices
    trace_mode = True

    run_all_to_all_dispatch_metadata_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        trace_mode,
        num_links=num_links,
        scheme=scheme,
        dtype=dtype,
        cluster_axis=cluster_axis,
    )


@pytest.mark.requires_device("TG")
@pytest.mark.skip(reason="Performance test - enable manually for perf validation")
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device, cluster_axis",
    [
        pytest.param((4, 8), (4, 8), 0, id="4x8_tg"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("experts_per_device", [2])
@pytest.mark.parametrize("select_experts_k", [8, 4, 2])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("num_iters", [40])
def test_decode_perf(
    mesh_device,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    num_iters,
):
    """Performance test for TG dispatch."""
    num_devices = mesh_shape[0] * mesh_shape[1]  # Total devices = 32 for 4x8
    experts = experts_per_device * num_devices  # 2 * 32 = 64 experts
    seq_len = 1
    num_links = 4
    dtype = ttnn.bfloat16
    scheme = "random_sequential_experts"

    dispatch_devices = mesh_shape[cluster_axis]
    batch = batches_per_device * dispatch_devices
    trace_mode = True

    run_all_to_all_dispatch_metadata_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        trace_mode,
        num_links=num_links,
        scheme=scheme,
        dtype=dtype,
        cluster_axis=cluster_axis,
    )

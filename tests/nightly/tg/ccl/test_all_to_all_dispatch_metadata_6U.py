# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import random
from loguru import logger
import torch
import ttnn

from tests.nightly.t3000.ccl.test_all_to_all_dispatch import (
    get_mesh_mapper,
    gen_tensors,
    tt_to_torch_dtype,
)


# Mesh graph descriptor paths for different mesh configurations
MESH_GRAPH_DESC_16x1 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_16x1_torus_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_1x16 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x16_torus_graph_descriptor.textproto"
)


def is_mesh_graph_descriptor_set(expected_path):
    """Check if TT_MESH_GRAPH_DESC_PATH is set to the expected path."""
    return os.environ.get("TT_MESH_GRAPH_DESC_PATH") == expected_path


def create_fabric_router_config(*, max_payload_size=None):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn.FabricRouterConfig()
    if max_payload_size is not None:
        config.max_packet_payload_size_bytes = max_payload_size
    return config


from models.perf.benchmarking_utils import BenchmarkProfiler

from tracy import signpost


def gen_expert_mapping_new_format_from_old(expert_mapping_old, mesh_shape):
    """
    Convert old format expert mapping to new format.

    Old format: [1, 1, experts, devices] - one-hot encoding where expert_mapping_old[0, 0, e, d] = 1
                means expert e is on device d.
    New format: [devices, experts] - direct device ID lookup where expert_mapping_new[src_device, e] = d
                means expert e is on device d (from the perspective of src_device).

    For now, all devices see the same mapping (no replicated experts), so we just replicate
    the same row for each source device.

    In the future, this can be extended to support replicated experts where each device
    sees the "optimal" device (e.g., shortest distance) for each expert.
    """
    devices = mesh_shape[0] * mesh_shape[1]
    experts = expert_mapping_old.shape[2]

    # Extract device assignment from one-hot encoding
    # expert_mapping_old has shape [1, 1, experts, devices]
    # For each expert, find which device has the 1
    expert_mapping_new = torch.zeros(1, experts, dtype=torch.uint16)
    for e in range(experts):
        device_assignment = expert_mapping_old[0, 0, e, :]
        device_id = torch.where(device_assignment == 1)[0].item()
        expert_mapping_new[0, e] = device_id

    # Replicate across all devices (same mapping for now)
    expert_mapping_new = expert_mapping_new.repeat(devices, 1)
    return expert_mapping_new


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
    """
    Generate tensors for the all_to_all_dispatch_metadata operation.

    This function generates tensors with shapes matching the new operation format:
    - Output: [devices, total_tokens, hidden_size] where total_tokens = batch * seq_len
    - Metadata (indices): [devices, total_tokens, selected_experts_k]
    - Scores: [devices, total_tokens, selected_experts_k]

    Returns:
        input_tokens: [batch, 1, seq_len, hidden_size] - input tokens per device
        expert_indices: [batch, 1, seq_len, selected_experts_k] - expert indices per device
        expert_scores: [batch, 1, seq_len, selected_experts_k] - expert scores per device
        expert_mapping_old: [1, 1, experts, devices] - old format expert to device mapping (one-hot)
        expert_mapping_new: [devices, experts] - new format expert to device mapping (direct device ID)
        sparse_output_token_tensor: [devices, total_tokens, hidden_size] - golden output tokens
        metadata_tensor: [devices, total_tokens, selected_experts_k] - golden indices (all-gathered)
        scores_tensor: [devices, total_tokens, selected_experts_k] - golden scores (all-gathered)
    """
    # Use original gen_tensors to get base tensors
    input_tokens, expert_indices, expert_mapping_old, sparse_output_orig, metadata_orig = gen_tensors(
        batch, experts, selected_experts_k, hidden_size, seq_len, mesh_shape, devices, scheme=scheme, dtype=dtype
    )

    # Convert old format expert mapping to new format: [devices, experts] with device IDs
    # This ensures the new format matches the scheme (random vs sequential)
    expert_mapping_new = gen_expert_mapping_new_format_from_old(expert_mapping_old, mesh_shape)

    total_tokens = batch * seq_len

    # Reshape sparse output from [devices, batch, seq_len, hidden_size] to [devices, total_tokens, hidden_size]
    sparse_output_token_tensor = sparse_output_orig.reshape(devices, total_tokens, hidden_size)

    # Reshape metadata from [devices, batch, seq_len, selected_experts_k] to [devices, total_tokens, selected_experts_k]
    metadata_tensor = metadata_orig.reshape(devices, total_tokens, selected_experts_k)

    # Generate expert scores (same shape as expert_indices)
    # Shape: [batch, 1, seq_len, selected_experts_k]
    expert_scores = torch.rand(expert_indices.shape, dtype=torch.float32).to(dtype)
    # Normalize scores so they sum to 1 per token (softmax-like)
    expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)

    # Create scores golden tensor (all-gathered scores, same structure as metadata)
    # First reshape expert_scores from [batch, 1, seq_len, k] to [1, batch, seq_len, k]
    scores_reshaped = expert_scores.permute(1, 0, 2, 3)  # [1, batch, seq_len, k]
    # Replicate across devices (same as metadata golden)
    scores_golden = scores_reshaped.repeat(devices, 1, 1, 1)  # [devices, batch, seq_len, k]
    # Reshape to [devices, total_tokens, selected_experts_k]
    scores_tensor = scores_golden.reshape(devices, total_tokens, selected_experts_k)

    return (
        input_tokens,
        expert_indices,
        expert_scores,
        expert_mapping_old,
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
    warmup_iters,
    trace_mode,
    num_links=4,
    scheme="random_sequential_experts",
    dtype=ttnn.bfloat16,
    profiler=BenchmarkProfiler(),
    cluster_axis=1,
    shard_dim=0,
    worker_mode=ttnn.WorkerMode.DIRECT,
    dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
    use_persistent_mode=False,
):
    torch.manual_seed(2005)
    random.seed(2005)
    mesh_device.enable_program_cache()
    devices = mesh_shape[0] * mesh_shape[1]

    expert_indices_tensors = []
    expert_scores_tensors = []
    expert_mapping_tensors = []
    expert_mapping_new_tensors = []  # New format mapping tensors
    input_tensors = []

    torch_expert_mappings = []
    torch_expert_mappings_new = []  # New format mappings
    torch_expert_scores_list = []

    output_tensor_goldens_list = []
    output_metadata_goldens_list = []
    output_scores_goldens_list = []
    mesh_mapper = get_mesh_mapper(mesh_device, mesh_shape, cluster_axis, shard_dim)

    # Compute dims tuple for ShardTensor2dMesh based on cluster_axis
    # cluster_axis=1: shard along mesh axis 1 (columns) -> dims=(None, shard_dim)
    # cluster_axis=0: shard along mesh axis 0 (rows) -> dims=(shard_dim, None)
    if cluster_axis == 1:
        shard_dims = (None, shard_dim)
    else:  # cluster_axis == 0
        shard_dims = (shard_dim, None)

    total_tokens = batch * seq_len

    # Compute tokens per device for height sharding the input indices/scores
    # After mesh sharding, each device gets batch/devices tokens
    tokens_per_device = batch // devices

    # Create height sharded memory config for input indices and scores
    # 1 row per core, with tokens_per_device cores total
    # Arrange cores in a grid - use 8 rows (Y) and ceil(tokens_per_device/8) columns (X)
    num_cores_y = min(8, tokens_per_device)
    num_cores_x = (tokens_per_device + num_cores_y - 1) // num_cores_y
    input_indices_scores_core_range = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
    )
    # Shard shape: [1 row, seq_len * select_experts_k columns]
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
    logger.info(
        f"Input indices/scores height sharded: {tokens_per_device} tokens across {num_cores_x}x{num_cores_y} cores, "
        f"shard shape [1, {seq_len * select_experts_k}]"
    )

    for iter in range(num_iters):
        # Use the new gen_tensors_for_metadata_op which outputs shapes compatible with the operation
        (
            input_tokens,
            expert_indices,
            expert_scores,
            expert_mapping_old,
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

        if iter == 0:
            logger.info(f"input_tokens shape: {input_tokens.shape}")
            logger.info(f"expert_indices shape: {expert_indices.shape}")
            logger.info(f"expert_scores shape: {expert_scores.shape}")
            logger.info(f"expert_mapping_old shape: {expert_mapping_old.shape}")
            logger.info(f"expert_mapping_new shape: {expert_mapping_new.shape}")
            logger.info(f"sparse_output_token_tensor shape: {sparse_output_token_tensor.shape}")
            logger.info(f"metadata_tensor shape: {metadata_tensor.shape}")
            logger.info(f"scores_tensor shape: {scores_tensor.shape}")

        output_tensor_goldens_list.append(sparse_output_token_tensor)
        output_metadata_goldens_list.append(metadata_tensor)
        output_scores_goldens_list.append(scores_tensor)
        torch_expert_mappings.append(expert_mapping_old)
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

        # Use L1 height sharded memory for indices and scores
        # Height sharded with 1 row per core ensures 16B alignment and optimal memory access
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

        # Old format expert mapping: [1, 1, experts, devices] - one-hot encoding
        tt_expert_mapping = ttnn.from_torch(
            expert_mapping_old,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )

        # New format expert mapping: [devices, experts] - direct device ID lookup
        # Each entry expert_mapping_new[d, e] = device_id that owns expert e
        # Replicate across all devices so each device has the full mapping table
        tt_expert_mapping_new = ttnn.from_torch(
            expert_mapping_new,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )

        if iter == 0:
            logger.info(f"tt_input shape: {tt_input.shape}")
            logger.info(f"tt_expert_indices shape: {tt_expert_indices.shape}")
            logger.info(f"tt_expert_scores shape: {tt_expert_scores.shape}")
            logger.info(f"tt_expert_mapping (old) shape: {tt_expert_mapping.shape}")
            logger.info(f"tt_expert_mapping_new shape: {tt_expert_mapping_new.shape}")

        input_tensors.append(tt_input)
        expert_indices_tensors.append(tt_expert_indices)
        expert_scores_tensors.append(tt_expert_scores)
        expert_mapping_tensors.append(tt_expert_mapping)
        expert_mapping_new_tensors.append(tt_expert_mapping_new)

    tt_out_tensor_list = []

    # Create persistent output buffers and global semaphores for persistent mode
    # Only create 1 set of output buffers and 2 semaphores (for double-buffering) to save L1 space
    persistent_output_buffers = None
    cross_device_semaphores = []

    if use_persistent_mode:
        logger.info("Creating persistent output buffers for persistent mode (1 buffer set, 2 semaphores)")

        # Compute output shapes - use global shapes [devices, ...] for sharding across mesh
        output_tokens_shape = [devices, total_tokens, hidden_size]
        metadata_shape = [devices, total_tokens, select_experts_k]

        # Create core range set for worker cores (needed for global semaphore creation)
        worker_core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))})

        # Create sharded memory config for metadata/scores on drain core
        drain_core = ttnn.CoreCoord(0, 0)
        drain_core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(drain_core, drain_core)})
        metadata_shard_spec = ttnn.ShardSpec(
            drain_core_range_set,
            [total_tokens * devices, select_experts_k],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        metadata_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            metadata_shard_spec,
        )

        # Create single set of persistent output buffers (reused across all iterations)
        output_tokens_buffer = ttnn.from_torch(
            torch.zeros(output_tokens_shape, dtype=tt_to_torch_dtype(dtype)),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        logger.info(f"output_tokens_buffer shape: {output_tokens_buffer.shape}")

        metadata_buffer = ttnn.from_torch(
            torch.zeros(metadata_shape, dtype=torch.int16),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=metadata_sharded_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        logger.info(f"metadata_buffer shape: {metadata_buffer.shape}")
        scores_buffer = ttnn.from_torch(
            torch.zeros(metadata_shape, dtype=tt_to_torch_dtype(dtype)),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=metadata_sharded_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        logger.info(f"scores_buffer shape: {scores_buffer.shape}")
        persistent_output_buffers = (output_tokens_buffer, metadata_buffer, scores_buffer)

        # Create only 2 semaphores for double-buffering (rotate with i % 2)
        for _ in range(2):
            cross_device_semaphore = ttnn.create_global_semaphore(mesh_device, worker_core_range_set, 0)
            cross_device_semaphores.append(cross_device_semaphore)

        logger.info(f"Created 1 persistent buffer set and {len(cross_device_semaphores)} semaphores")
        ttnn.synchronize_device(mesh_device)

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []
        tt_metadata_list = []
        tt_scores_out_list = []

        for i in range(n_iters):
            buffer_index = i

            # Get optional persistent buffers and semaphore if in persistent mode
            output_tensors_arg = None
            cross_device_semaphore_arg = None
            if use_persistent_mode and persistent_output_buffers is not None:
                # Use single set of persistent buffers (reused across all iterations)
                output_tensors_arg = persistent_output_buffers
                # Rotate through 2 semaphores for double-buffering
                cross_device_semaphore_arg = cross_device_semaphores[i % 2]

            # Use the experimental all_to_all_dispatch_metadata op
            # Returns 3 tensors: output_tensor, indices_tensor, scores_tensor
            # When using persistent mode, drain_sync_tilizer_core is extracted from the tensor's shard spec
            # so we don't need to pass it explicitly
            output_tensor, indices_tensor, scores_tensor = ttnn.experimental.all_to_all_dispatch_metadata(
                input_tensors[buffer_index],
                expert_indices_tensors[buffer_index],
                expert_scores_tensors[buffer_index],
                expert_mapping_new_tensors[buffer_index],  # New format: [devices, experts]
                cluster_axis=cluster_axis,
                num_links=num_links,
                # Only pass drain_sync_tilizer_core when not using persistent mode
                # In persistent mode, it's extracted from the tensor's shard spec
                drain_sync_tilizer_core=None if use_persistent_mode else (0, 0),
                worker_mode=worker_mode,
                dispatch_algorithm=dispatch_algorithm,
                output_tensors=output_tensors_arg,
                cross_device_semaphore=cross_device_semaphore_arg,
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
        # compile run:
        logger.info("Compiling model")
        run_op(1, store_all_results=True)
        ttnn.synchronize_device(mesh_device)

        logger.info("Capturing Warmup")

        if warmup_iters > 0:
            logger.info(f"Capturing Warmup {warmup_iters} iterations")
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            run_op(warmup_iters, store_all_results=True)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)
        logger.info("Warmup done")

        logger.info("Capturing Trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_out_tensor_list, tt_metadata_list, tt_scores_out_list = run_op(num_iters, store_all_results=True)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        logger.info("Starting Trace perf test...")
        profiler.start("all-to-all-dispatch-metadata-trace-warmup")
        if warmup_iters > 0:
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)
        profiler.end("all-to-all-dispatch-metadata-trace-warmup")

        signpost("start")
        profiler.start("all-to-all-dispatch-metadata-trace")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)
        profiler.end("all-to-all-dispatch-metadata-trace")
        signpost("stop")

        time_taken = profiler.get_duration("all-to-all-dispatch-metadata-trace") - profiler.get_duration(
            "all-to-all-dispatch-metadata-trace-warmup"
        )
        logger.info(f"Time taken e2e: {time_taken} s")
    else:
        signpost("start")
        tt_out_tensor_list, tt_metadata_list, tt_scores_out_list = run_op(num_iters, store_all_results=True)
        signpost("stop")

    passed = True
    metadata_passed = True
    scores_passed = True
    first_failed_tensor_index = None
    first_failed_batch_index = None
    first_failed_expert_index = None
    first_failed_device_index = None
    first_failed_sequence_index = None

    first_failed_metadata_index = None
    first_failed_scores_index = None
    failed_indices = []
    failed_metadata_indices = []
    failed_scores_indices = []

    # In persistent mode, only verify the last iteration since we reuse the same output buffers
    # The non-persistent mode test will be used for testing accuracy for all iterations
    if use_persistent_mode:
        verification_indices = [len(tt_out_tensor_list) - 1]  # Only last iteration
        logger.info("Persistent mode: verifying only the last iteration for accuracy")
    else:
        verification_indices = range(len(tt_out_tensor_list))  # All iterations

    for tensor_index in verification_indices:
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

        # Log shapes for debugging
        if tensor_index == 0:
            logger.info(f"tt_torch_tensor shape: {tt_torch_tensor.shape}")
            logger.info(f"tt_metadata_tensor shape: {tt_metadata_tensor.shape}")
            logger.info(f"tt_scores_out_tensor shape: {tt_scores_out_tensor.shape}")
            logger.info(f"golden output shape: {output_tensor_goldens_list[tensor_index].shape}")
            logger.info(f"golden metadata shape: {output_metadata_goldens_list[tensor_index].shape}")
            logger.info(f"golden scores shape: {output_scores_goldens_list[tensor_index].shape}")

        # New shapes: [devices, total_tokens, ...] where total_tokens = batch * seq_len
        devices = tt_metadata_tensor.shape[0]
        total_tokens_out = tt_metadata_tensor.shape[1]
        selected_experts_k = tt_metadata_tensor.shape[2]

        # Verify metadata (indices)
        metadata_all_close = torch.allclose(tt_metadata_tensor, output_metadata_goldens_list[tensor_index])
        metadata_all_equal = torch.equal(tt_metadata_tensor, output_metadata_goldens_list[tensor_index])
        if not metadata_all_close or not metadata_all_equal:
            metadata_passed = False
            first_failed_metadata_index = tensor_index
            failed_metadata_indices = torch.where(tt_metadata_tensor != output_metadata_goldens_list[tensor_index])
            logger.info(f"All failed metadata devices: {failed_metadata_indices}")
            logger.info(f"Failing tt_metadata_tensor tensor {tt_metadata_tensor[failed_metadata_indices]}")
            logger.info(
                f"Relevant output_metadata_goldens_list tensor {output_metadata_goldens_list[tensor_index][failed_metadata_indices]}"
            )
            break

        # Verify scores
        scores_all_close = torch.allclose(
            tt_scores_out_tensor, output_scores_goldens_list[tensor_index], rtol=1e-2, atol=1e-2
        )
        if not scores_all_close:
            scores_passed = False
            first_failed_scores_index = tensor_index
            # Find indices where scores differ
            diff = torch.abs(tt_scores_out_tensor - output_scores_goldens_list[tensor_index])
            failed_scores_indices = torch.where(diff > 1e-2)
            logger.info(f"All failed scores indices: {failed_scores_indices}")
            logger.info(f"Failing tt_scores_out_tensor tensor {tt_scores_out_tensor[failed_scores_indices][:10]}")
            logger.info(
                f"Relevant output_scores_goldens_list tensor {output_scores_goldens_list[tensor_index][failed_scores_indices][:10]}"
            )
            break

        # Verify output tokens with new shape [devices, total_tokens, hidden_size]
        # Using new format expert mapping: expert_mapping_new[src_device, expert] = target_device_id
        # We use the source device's copy of the mapping for the lookup to future-proof
        # for expert masking where each device may have a different view of expert locations.
        # Token layout: tokens from each source device are concatenated sequentially
        tokens_per_src_device = total_tokens_out // devices
        for t in range(total_tokens_out):
            # Determine which source device this token came from
            src_device = t // tokens_per_src_device
            for k in range(selected_experts_k):
                expert_id = tt_metadata_tensor[src_device, t, k]
                # Use the source device's copy of the expert mapping for the lookup
                target_device = torch_expert_mappings_new[tensor_index][src_device, expert_id].item()
                is_all_equal = torch.equal(
                    tt_torch_tensor[target_device, t, :], output_tensor_goldens_list[tensor_index][target_device, t, :]
                )
                if not is_all_equal:
                    logger.info(
                        f"Output tensor {tensor_index} mismatch at token {t}, expert {expert_id}, src_device {src_device}, target_device {target_device}"
                    )
                    passed = False
                    first_failed_tensor_index = tensor_index
                    first_failed_batch_index = t  # Using token index instead
                    failed_indices = torch.where(
                        tt_torch_tensor[target_device, t, :]
                        != output_tensor_goldens_list[tensor_index][target_device, t, :]
                    )
                    first_10_fail_idx = failed_indices[0][:10]
                    logger.info(f"First 10 failing indices: {first_10_fail_idx}")
                    logger.info(
                        f"Failing tt_torch_tensor tensor (first 10) {tt_torch_tensor[target_device, t, first_10_fail_idx]}"
                    )
                    logger.info(
                        f"Relevant output_tensor_goldens_list tensor (first 10) {output_tensor_goldens_list[tensor_index][target_device, t, first_10_fail_idx]}"
                    )
                    first_failed_expert_index = expert_id
                    first_failed_device_index = target_device
                    first_failed_sequence_index = t
                    break
            if not passed:
                break
        if not passed:
            break

    logger.info(f"Device has {mesh_device.num_program_cache_entries()} program cache entries")
    assert (
        mesh_device.num_program_cache_entries() == 1
    ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"

    if not metadata_passed:
        logger.info(f"Failed metadata indices: {failed_metadata_indices}")
        assert metadata_passed, f"{first_failed_metadata_index} FAILED metadata indices: {failed_metadata_indices}"

    if not scores_passed:
        logger.info(f"Failed scores indices: {failed_scores_indices}")
        assert scores_passed, f"{first_failed_scores_index} FAILED scores indices: {failed_scores_indices}"

    if not passed:
        logger.info(f"Failed data indices: {failed_indices}")
        assert (
            passed
        ), f"First failing index: {first_failed_tensor_index} token {first_failed_batch_index} sequence {first_failed_sequence_index} expert {first_failed_expert_index} device {first_failed_device_index} FAILED data indices: {failed_indices}"


# Correctness test - single focused test case for pipeline validation
# Requires TT_MESH_GRAPH_DESC_PATH to be set to the 16x1 mesh descriptor before running
@pytest.mark.skipif(
    not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_16x1),
    reason=f"Requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_16x1}",
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": create_fabric_router_config(max_payload_size=4352),
            "trace_region_size": 500000,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device, cluster_axis",
    [
        pytest.param((16, 1), (16, 1), 0, id="16x1"),
    ],
    indirect=["mesh_device"],
)
def test_correctness(
    mesh_device,
    mesh_shape,
    cluster_axis,
):
    batches_per_device = 32
    experts = 2 * 16
    select_experts_k = 8
    hidden_size = 7168
    seq_len = 1
    num_iters = 20
    warmup_iters = 5
    num_links = 4
    dtype = ttnn.bfloat16
    congestion_scheme = "random_sequential_experts"
    worker_mode = ttnn.WorkerMode.DIRECT
    use_persistent_mode = False

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
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme=congestion_scheme,
        dtype=dtype,
        cluster_axis=cluster_axis,
        worker_mode=worker_mode,
        dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
        use_persistent_mode=use_persistent_mode,
    )


# Performance sweep test - disabled by default (too resource intensive for pipelines)
# Enable with: RUN_ALL_TO_ALL_PERF=1 and TT_MESH_GRAPH_DESC_PATH set to the appropriate mesh descriptor
@pytest.mark.skipif(
    os.environ.get("RUN_ALL_TO_ALL_PERF") != "1",
    reason="Resource intensive sweep test - enable with RUN_ALL_TO_ALL_PERF=1",
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": create_fabric_router_config(max_payload_size=7168),
            "trace_region_size": 500000,
        },
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": create_fabric_router_config(max_payload_size=4352),
            "trace_region_size": 500000,
        },
    ],
    indirect=True,
    ids=["double", "single"],
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device, cluster_axis",
    [
        pytest.param((16, 1), (16, 1), 0, id="16x1"),
        pytest.param((1, 16), (1, 16), 1, id="1x16"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("experts", [2 * 16])
@pytest.mark.parametrize(
    "select_experts_k", [8, 7, 6, 5, 4, 3, 2, 1], ids=["k8", "k7", "k6", "k5", "k4", "k3", "k2", "k1"]
)
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [
        (1, 40, 10),
    ],
    ids=[
        "decode",
    ],
)
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "congestion_scheme", ["random_sequential_experts", "worst_congestion_descending", "best_congestion"]
)
@pytest.mark.parametrize("use_persistent_mode", [True, False], ids=["persistent", "synced"])
@pytest.mark.parametrize(
    "worker_mode",
    [ttnn.WorkerMode.DIRECT, ttnn.WorkerMode.MUX_TOKEN_SPLIT, ttnn.WorkerMode.MUX_PAYLOAD_SPLIT],
    ids=["direct", "token_split", "payload_split"],
)
def test_decode_perf(
    mesh_device,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    dtype,
    congestion_scheme,
    use_persistent_mode,
    worker_mode,
):
    # Skip based on mesh shape and required mesh graph descriptor
    if mesh_shape == (16, 1):
        if not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_16x1):
            pytest.skip(f"16x1 mesh requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_16x1}")
    elif mesh_shape == (1, 16):
        if not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x16):
            pytest.skip(f"1x16 mesh requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_1x16}")

    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    trace_mode = True

    profiler = BenchmarkProfiler()
    step_name = "All2AllDispatchMetadataOp"
    profiler.start(step_name)
    signpost(header="start")

    run_all_to_all_dispatch_metadata_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme=congestion_scheme,
        dtype=dtype,
        cluster_axis=cluster_axis,
        worker_mode=worker_mode,
        dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
        use_persistent_mode=use_persistent_mode,
    )

    signpost(header="stop")
    profiler.end(step_name)

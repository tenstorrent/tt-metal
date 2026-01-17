# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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

from models.perf.benchmarking_utils import BenchmarkProfiler

from tracy import signpost


def gen_tensors_for_metadata_op(
    batch, experts, selected_experts_k, hidden_size, seq_len, mesh_shape, devices, scheme="random", dtype=torch.bfloat16
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
        expert_mapping: [1, 1, experts, devices] - expert to device mapping
        sparse_output_token_tensor: [devices, total_tokens, hidden_size] - golden output tokens
        metadata_tensor: [devices, total_tokens, selected_experts_k] - golden indices (all-gathered)
        scores_tensor: [devices, total_tokens, selected_experts_k] - golden scores (all-gathered)
    """
    # Use original gen_tensors to get base tensors
    input_tokens, expert_indices, expert_mapping, sparse_output_orig, metadata_orig = gen_tensors(
        batch, experts, selected_experts_k, hidden_size, seq_len, mesh_shape, devices, scheme=scheme, dtype=dtype
    )

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
        expert_mapping,
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
    scheme="random",
    dtype=ttnn.bfloat16,
    profiler=BenchmarkProfiler(),
    topology=None,
    cluster_axis=1,
    shard_dim=0,
):
    torch.manual_seed(2005)
    random.seed(2005)
    mesh_device.enable_program_cache()
    devices = mesh_shape[0] * mesh_shape[1]

    expert_indices_tensors = []
    expert_scores_tensors = []
    expert_mapping_tensors = []
    input_tensors = []

    torch_expert_mappings = []
    torch_expert_scores_list = []

    output_tensor_goldens_list = []
    output_metadata_goldens_list = []
    output_scores_goldens_list = []
    mesh_mapper = get_mesh_mapper(mesh_device, mesh_shape, cluster_axis, shard_dim)

    total_tokens = batch * seq_len

    for iter in range(num_iters):
        # Use the new gen_tensors_for_metadata_op which outputs shapes compatible with the operation
        (
            input_tokens,
            expert_indices,
            expert_scores,
            expert_mapping,
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
            scheme=scheme,
            dtype=tt_to_torch_dtype(dtype),
        )

        if iter == 0:
            logger.info(f"input_tokens shape: {input_tokens.shape}")
            logger.info(f"expert_indices shape: {expert_indices.shape}")
            logger.info(f"expert_scores shape: {expert_scores.shape}")
            logger.info(f"expert_mapping shape: {expert_mapping.shape}")
            logger.info(f"sparse_output_token_tensor shape: {sparse_output_token_tensor.shape}")
            logger.info(f"metadata_tensor shape: {metadata_tensor.shape}")
            logger.info(f"scores_tensor shape: {scores_tensor.shape}")

        output_tensor_goldens_list.append(sparse_output_token_tensor)
        output_metadata_goldens_list.append(metadata_tensor)
        output_scores_goldens_list.append(scores_tensor)
        torch_expert_mappings.append(expert_mapping)
        torch_expert_scores_list.append(expert_scores)

        tt_input = ttnn.from_torch(
            input_tokens,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # Use L1 memory for indices and scores to ensure 16B alignment
        # (DRAM uses 32B alignment which creates padding that doesn't match
        # the output metadata tensor's 16B aligned layout)
        tt_expert_indices = ttnn.from_torch(
            expert_indices,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        tt_expert_scores = ttnn.from_torch(
            expert_scores,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        tt_expert_mapping = ttnn.from_torch(
            expert_mapping,
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
            logger.info(f"tt_expert_mapping shape: {tt_expert_mapping.shape}")

        input_tensors.append(tt_input)
        expert_indices_tensors.append(tt_expert_indices)
        expert_scores_tensors.append(tt_expert_scores)
        expert_mapping_tensors.append(tt_expert_mapping)

    tt_out_tensor_list = []

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []
        tt_metadata_list = []
        tt_scores_out_list = []

        for i in range(n_iters):
            buffer_index = i
            # Use the experimental all_to_all_dispatch_metadata op
            # Returns 3 tensors: output_tensor, indices_tensor, scores_tensor
            output_tensor, indices_tensor, scores_tensor = ttnn.experimental.all_to_all_dispatch_metadata(
                input_tensors[buffer_index],
                expert_indices_tensors[buffer_index],
                expert_scores_tensors[buffer_index],
                expert_mapping_tensors[buffer_index],
                cluster_axis=cluster_axis,
                num_links=num_links,
                topology=topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                # Use a drain core that's NOT in the sender cores to avoid L1 address overlap
                # between the metadata tensor and the global semaphore
                drain_sync_tilizer_core=(0, 0),
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
        tt_out_tensor_list, tt_metadata_list, tt_scores_out_list = run_op(1, store_all_results=True)
        ttnn.synchronize_device(mesh_device)

        logger.info("Capturing Warmup")

        if warmup_iters > 0:
            logger.info(f"Capturing Warmup {warmup_iters} iterations")
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            tt_out_tensor_list, tt_metadata_list, tt_scores_out_list = run_op(warmup_iters, store_all_results=True)
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
        # scores_all_close = torch.allclose(
        #     tt_scores_out_tensor, output_scores_goldens_list[tensor_index], rtol=1e-2, atol=1e-2
        # )
        # if not scores_all_close:
        #     scores_passed = False
        #     first_failed_scores_index = tensor_index
        #     # Find indices where scores differ
        #     diff = torch.abs(tt_scores_out_tensor - output_scores_goldens_list[tensor_index])
        #     failed_scores_indices = torch.where(diff > 1e-2)
        #     logger.info(f"All failed scores indices: {failed_scores_indices}")
        #     logger.info(f"Failing tt_scores_out_tensor tensor {tt_scores_out_tensor[failed_scores_indices][:10]}")
        #     logger.info(
        #         f"Relevant output_scores_goldens_list tensor {output_scores_goldens_list[tensor_index][failed_scores_indices][:10]}"
        #     )
        #     break

        # Verify output tokens with new shape [devices, total_tokens, hidden_size]
        for t in range(total_tokens_out):
            for k in range(selected_experts_k):
                expert_id = tt_metadata_tensor[0, t, k]
                for d in range(devices):
                    if torch_expert_mappings[tensor_index][0, 0, expert_id, d] == 1:
                        is_all_equal = torch.equal(
                            tt_torch_tensor[d, t, :], output_tensor_goldens_list[tensor_index][d, t, :]
                        )
                        if not is_all_equal:
                            logger.info(
                                f"Output tensor {tensor_index} mismatch at token {t}, expert {expert_id}, device {d}"
                            )
                            passed = False
                            first_failed_tensor_index = tensor_index
                            first_failed_batch_index = t  # Using token index instead
                            failed_indices = torch.where(
                                tt_torch_tensor[d, t, :] != output_tensor_goldens_list[tensor_index][d, t, :]
                            )
                            first_10_fail_idx = failed_indices[0][:10]
                            logger.info(f"First 10 failing indices: {first_10_fail_idx}")
                            logger.info(
                                f"Failing tt_torch_tensor tensor (first 10) {tt_torch_tensor[d, t, first_10_fail_idx]}"
                            )
                            logger.info(
                                f"Relevant output_tensor_goldens_list tensor (first 10) {output_tensor_goldens_list[tensor_index][d, t, first_10_fail_idx]}"
                            )
                            first_failed_expert_index = expert_id
                            first_failed_device_index = d
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
        ), f"First failing index: {first_failed_tensor_index} token {first_failed_batch_index} expert {first_failed_expert_index} device {first_failed_device_index} FAILED data indices: {failed_indices}"


# Performance tests
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
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((1, 16), (1, 16), id="1x16_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("experts", [2 * 16])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [
        (1, 1, 1),
    ],
    ids=[
        "decode",
    ],
)
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
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
    topology,
    dtype,
):
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
        scheme="worst_congestion",
        topology=topology,
        dtype=dtype,
        cluster_axis=cluster_axis,
    )

    signpost(header="stop")
    profiler.end(step_name)

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import random
from loguru import logger
import ttnn


def get_max_links(cluster_axis, fabric_config):
    if fabric_config == ttnn.FabricConfig.FABRIC_2D:
        return 1
    elif cluster_axis is None:
        return 1
    else:
        return 2 if cluster_axis == 0 else 1


def tt_to_torch_dtype(tt_dtype):
    if tt_dtype == ttnn.bfloat16:
        return torch.bfloat16
    elif tt_dtype == ttnn.bfloat8_b:
        return torch.bfloat16
    elif tt_dtype == ttnn.float32:
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {tt_dtype}")


def gen_expert_mapping(experts, mesh_shape, cluster_axis):
    """
    Create expert mapping tensor that maps each expert to the device it belongs to.
    Shape: [devices, experts] where each entry is the device ID that owns that expert.
    All rows are identical (replicated across devices).

    For 256 experts and 128 devices (2 experts per device):
    expert_mapping[d, e] = e // experts_per_device
    So expert 0,1 -> device 0; expert 2,3 -> device 1; etc.
    """
    devices = mesh_shape[0] * mesh_shape[1]
    experts_per_device = experts // devices
    expert_mapping = torch.zeros(1, experts, dtype=torch.int16)
    for e in range(experts):
        expert_mapping[0, e] = e // experts_per_device
    expert_mapping = expert_mapping.repeat(devices, 1)
    return expert_mapping


def get_mesh_mapper(mesh_device, mesh_shape, cluster_axis, shard_dim):
    if cluster_axis is None:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim)
    elif cluster_axis == 1:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, shard_dim), mesh_shape=mesh_shape)
    elif cluster_axis == 0:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(shard_dim, None), mesh_shape=mesh_shape)
    else:
        raise ValueError(f"Invalid cluster_axis: {cluster_axis}")
    return mesh_mapper


def get_device_row_col(device_idx, mesh_shape):
    """Get (row, col) position of a device in the mesh."""
    row = device_idx // mesh_shape[1]
    col = device_idx % mesh_shape[1]
    return row, col


def get_axis_position(device_idx, mesh_shape, cluster_axis):
    """Get the device's position along the cluster axis."""
    row, col = get_device_row_col(device_idx, mesh_shape)
    if cluster_axis == 1:
        return col  # Position along columns
    else:  # cluster_axis == 0
        return row  # Position along rows


def get_other_axis_position(device_idx, mesh_shape, cluster_axis):
    """Get the device's position along the non-cluster axis (for determining which devices communicate)."""
    row, col = get_device_row_col(device_idx, mesh_shape)
    if cluster_axis == 1:
        return row  # Devices in the same row communicate
    else:  # cluster_axis == 0
        return col  # Devices in the same column communicate


def devices_on_same_dispatch_axis(device_a, device_b, mesh_shape, cluster_axis):
    """Check if two devices are on the same row/column for dispatching (can communicate)."""
    return get_other_axis_position(device_a, mesh_shape, cluster_axis) == get_other_axis_position(
        device_b, mesh_shape, cluster_axis
    )


def get_dte_intermediate(indices_tensor, scores_tensor, mapping_tensor, mesh_shape, cluster_axis):
    """
    Create DTE (Dispatch-To-Expert) intermediate tensor.

    This tensor tells each receiving device, for each (sending_device, token, local_expert) combination,
    what is the chunk position of the token being sent.

    Shape: [receiving_devices, dispatch_devices, tokens_per_device * experts_per_device]

    dte_intermediate[recv_dev, send_axis_idx, token_idx * experts_per_device + local_expert] = chunk_position

    where:
    - recv_dev: the receiving device's global index
    - send_axis_idx: the sending device's index along the dispatch axis (0 to dispatch_devices-1)
    - token_idx: the token's index on the sending device (0 to tokens_per_device-1)
    - local_expert: the expert's local index on the receiving device (0 to experts_per_device-1)
    - chunk_position: the i'th token sent from that source device (across all its sends),
                      or tokens_per_device if no token was sent for that slot

    Multiple tokens may select the same expert on a device. Some tokens may not select any expert
    on a particular device. Each (token, local_expert) pair has a unique entry.

    Args:
        indices_tensor: Expert indices tensor [batch, 1, seq_len, K] or similar
        mapping_tensor: Expert mapping tensor [devices, experts] where entry is device ID owning that expert
        mesh_shape: Tuple (rows, cols) of the mesh
        cluster_axis: Axis along which tokens are dispatched (0=rows, 1=cols)

    Returns:
        dte_intermediate tensor of shape [devices, dispatch_devices, tokens_per_device * experts_per_device]
    """
    devices = mesh_shape[0] * mesh_shape[1]
    dispatch_devices = mesh_shape[cluster_axis]
    experts = mapping_tensor.shape[-1]  # global experts
    selected_experts_k = indices_tensor.shape[-1]
    experts_per_device = experts // devices

    # Reshape indices to [devices, tokens_per_device, K]
    # Assuming indices_tensor can be reshaped this way (tokens are distributed across devices)
    indices_reshaped = indices_tensor.reshape(devices, -1, indices_tensor.shape[-1])
    scores_reshaped = scores_tensor.reshape(devices, -1, scores_tensor.shape[-1])
    tokens_per_device = indices_reshaped.shape[1]

    # Initialize: tokens_per_device means "no token sent for this slot"
    # Shape: [devices, dispatch_devices, tokens_per_device * experts_per_device]
    dte_intermediate = (
        torch.ones(devices, dispatch_devices, tokens_per_device * experts_per_device, dtype=indices_tensor.dtype)
        * tokens_per_device
    )

    dte_scores = torch.zeros_like(dte_intermediate, dtype=scores_tensor.dtype)
    ed_table = torch.zeros(
        devices, experts_per_device, devices, dtype=torch.int16
    )  # [target_device, expert_id, source_device]

    # Iterate over each source device
    for source_device_idx in range(devices):
        source_axis_pos = get_axis_position(source_device_idx, mesh_shape, cluster_axis)

        num_chunks = [tokens_per_device] * experts
        for t in range(tokens_per_device):
            for k in range(selected_experts_k):
                expert_id = indices_reshaped[source_device_idx, t, k].item()

                # Which device owns this expert?
                target_device = mapping_tensor[0, expert_id].item()  # All rows are identical

                # Check if target device is on the same dispatch axis as source
                # (i.e., they can communicate - same row for cluster_axis=1, same col for cluster_axis=0)
                if devices_on_same_dispatch_axis(source_device_idx, target_device, mesh_shape, cluster_axis):
                    # Record this in the receiving device's DTE intermediate
                    # Index = token_idx * experts_per_device + local_expert_idx
                    # This uniquely identifies each (token, local_expert) pair
                    local_expert_idx = expert_id % experts_per_device
                    flat_idx = t * experts_per_device + local_expert_idx
                    if num_chunks[expert_id] == tokens_per_device:
                        num_chunks[expert_id] = 0
                    dte_intermediate[target_device, source_axis_pos, flat_idx] = num_chunks[expert_id]
                    dte_scores[target_device, source_axis_pos, flat_idx] = scores_reshaped[source_device_idx, t, k]
                    num_chunks[expert_id] += 1
                    ed_table[target_device, local_expert_idx, source_device_idx] = 1

    return dte_intermediate, dte_scores, ed_table.reshape(devices, experts_per_device * devices)


def get_a2a_dispatch_golden(input_tokens, expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis):
    """
    Get the golden output of the all-to-all dispatch selective tilize operation.
    """
    devices = mesh_shape[0] * mesh_shape[1]
    dispatch_devices = mesh_shape[cluster_axis]
    # input_tokens shape is [dispatch_devices, batch, seq_len, hidden_size]
    # tokens_per_device = batch * seq_len

    selected_experts_k = expert_indices.shape[-1]

    expert_indices_reshaped = expert_indices.reshape(
        devices, -1, expert_indices.shape[-1]
    )  # [devices, tokens_per_device, selected_experts_k]
    input_tokens_reshaped = input_tokens.reshape(
        devices, -1, input_tokens.shape[-1]
    )  # [devices, tokens_per_device, hidden_size]
    tokens_per_device = input_tokens_reshaped.shape[1]
    output_tokens = torch.zeros_like(input_tokens_reshaped).repeat(
        1, dispatch_devices, 1
    )  # [devices, tokens_per_device*dispatch_devices, hidden_size]

    metadata = expert_indices.reshape(devices, -1, expert_indices.shape[-1]).repeat(
        dispatch_devices, 1, 1
    )  # all-gather of the expert indices
    gathered_scores = expert_scores.reshape(devices, -1, expert_scores.shape[-1]).repeat(
        dispatch_devices, 1, 1
    )  # all-gather of the expert scores

    # Track which (source_device, token, target_device) have already been dispatched
    # to avoid redundant writes when multiple experts map to the same device
    send_buffer = torch.zeros(devices, tokens_per_device, devices, dtype=torch.uint8)

    for source_device_idx in range(devices):
        source_axis_pos = get_axis_position(source_device_idx, mesh_shape, cluster_axis)
        for t in range(tokens_per_device):
            for k in range(selected_experts_k):
                expert_id = expert_indices_reshaped[source_device_idx, t, k].item()
                device_idx = expert_mapping[0, expert_id].item()
                if send_buffer[source_device_idx, t, device_idx] == 0:
                    if get_other_axis_position(source_device_idx, mesh_shape, cluster_axis) == get_other_axis_position(
                        device_idx, mesh_shape, cluster_axis
                    ):
                        output_tokens[device_idx, t + (source_axis_pos * tokens_per_device), :] = input_tokens_reshaped[
                            source_device_idx, t, :
                        ]  # source device can dispatch to the target device, offset at its own unique position in the output buffer, offset by the token
                    send_buffer[source_device_idx, t, device_idx] = 1

    return output_tokens, metadata, gathered_scores


def verify_a2a_dispatch_output_tokens(
    output_tokens, golden_output_tokens, expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
):
    """
    Verify the output of the all-to-all dispatch selective tilize operation against the golden.
    """
    logger.info(f"output_tokens shape: {output_tokens.shape}")
    logger.info(f"golden_output_tokens shape: {golden_output_tokens.shape}")
    output_tokens = output_tokens.reshape(golden_output_tokens.shape)
    devices = mesh_shape[0] * mesh_shape[1]

    selected_experts_k = expert_indices.shape[-1]

    expert_indices_reshaped = expert_indices.reshape(
        devices, -1, expert_indices.shape[-1]
    )  # [devices, tokens_per_device, selected_experts_k]
    logger.info(f"expert_indices_reshaped shape: {expert_indices_reshaped.shape}")
    tokens_per_device = expert_indices_reshaped.shape[1]

    # Track which (source_device, token, target_device) have already been verified
    # to avoid redundant checks when multiple experts map to the same device
    verified_buffer = torch.zeros(devices, tokens_per_device, devices, dtype=torch.uint8)

    for source_device_idx in range(devices):
        source_axis_pos = get_axis_position(source_device_idx, mesh_shape, cluster_axis)
        for t in range(tokens_per_device):
            for k in range(selected_experts_k):
                expert_id = expert_indices_reshaped[source_device_idx, t, k].item()
                device_idx = expert_mapping[0, expert_id].item()
                if verified_buffer[source_device_idx, t, device_idx] == 0:
                    if get_other_axis_position(source_device_idx, mesh_shape, cluster_axis) == get_other_axis_position(
                        device_idx, mesh_shape, cluster_axis
                    ):
                        page_idx = t + (source_axis_pos * tokens_per_device)
                        output_row = output_tokens[device_idx, page_idx, :]
                        golden_row = golden_output_tokens[device_idx, page_idx, :]

                        # Find exact mismatch locations
                        diff_mask = output_row != golden_row
                        if diff_mask.any():
                            diff_indices = diff_mask.nonzero(as_tuple=True)[0]
                            num_diffs = len(diff_indices)
                            logger.error(
                                f"Output token mismatch at target device {device_idx}, token {t}, "
                                f"source device {source_device_idx} (axis position {source_axis_pos}), "
                                f"selected expert {expert_id}, output page {page_idx}"
                            )
                            logger.error(f"  Total mismatches: {num_diffs} / {len(output_row)} elements")
                            logger.error(f"  First 10 diff indices: {diff_indices[:10].tolist()}")
                            logger.error(f"  Output values at diffs: {output_row[diff_indices[:10]].tolist()}")
                            logger.error(f"  Golden values at diffs: {golden_row[diff_indices[:10]].tolist()}")
                            assert False, f"Found {num_diffs} mismatches"

                        verified_buffer[source_device_idx, t, device_idx] = 1


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((1, 8), (1, 8), id="1x8_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1], ids=["cluster_col"])
@pytest.mark.parametrize("experts", [16])
@pytest.mark.parametrize("selected_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "batch, seq_len, shard_dim",
    [
        (1, 32, 0),
    ],
    ids=["b1s32"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
def test_all_to_all_dispatch_selective_tilize_no_trace(
    mesh_device,
    mesh_shape,
    cluster_axis,
    batch,
    shard_dim,
    experts,
    selected_experts_k,
    hidden_size,
    seq_len,
    dtype,
    input_memory_config,
    output_memory_config,
    device_params,
):
    torch.manual_seed(2005)
    devices = mesh_shape[0] * mesh_shape[1]

    input_tokens = torch.rand(
        mesh_shape[cluster_axis], batch, seq_len, hidden_size, dtype=tt_to_torch_dtype(dtype)
    ).repeat(mesh_shape[1 - cluster_axis], 1, 1, 1)
    expert_indices_along_dispatch_axis = []

    for _ in range(mesh_shape[cluster_axis]):
        per_batch_indices = []
        for b in range(batch):
            per_seq_indices = []
            for s in range(seq_len):
                per_seq_indices.append(
                    torch.randperm(experts)[:selected_experts_k].reshape(1, 1, 1, selected_experts_k)
                )
            per_seq_indices = torch.cat(per_seq_indices, dim=2)
            per_batch_indices.append(per_seq_indices)
        per_batch_indices = torch.cat(per_batch_indices, dim=1)

        expert_indices_along_dispatch_axis.append(per_batch_indices)

    expert_indices = (
        torch.cat(expert_indices_along_dispatch_axis, dim=0).repeat(mesh_shape[1 - cluster_axis], 1, 1, 1)
    ).to(torch.uint16)

    expert_scores = (
        torch.rand_like(expert_indices, dtype=tt_to_torch_dtype(dtype)) + 1e-5
    )  # add small epsilon to avoid division by zero
    expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)

    expert_mapping = gen_expert_mapping(experts, mesh_shape=mesh_shape, cluster_axis=cluster_axis)
    expert_mapping = expert_mapping.to(torch.uint16)

    # Shard along sequence dimension (dim 2) on cluster axis, replicate on other axis
    mesh_mapper = get_mesh_mapper(mesh_device, mesh_shape, cluster_axis, shard_dim)
    tt_input_tokens = ttnn.from_torch(
        input_tokens,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=input_memory_config,
        mesh_mapper=mesh_mapper,
    )

    tt_expert_indices = ttnn.from_torch(
        expert_indices,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=mesh_mapper,
    )

    tt_expert_scores = ttnn.from_torch(
        expert_scores,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,  # Use ttnn dtype, not torch dtype
        memory_config=input_memory_config,
        mesh_mapper=mesh_mapper,
    )
    # expert mapping is replicated across all devices
    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=mesh_mapper,
    )

    # print ttnn shapes
    logger.info(f"tt_input_tokens shape: {tt_input_tokens.shape}")
    logger.info(f"tt_expert_indices shape: {tt_expert_indices.shape}")
    logger.info(f"tt_expert_scores shape: {tt_expert_scores.shape}")
    logger.info(f"tt_expert_mapping shape: {tt_expert_mapping.shape}")

    logger.info(f"expert_mapping: {expert_mapping[2,:]}")
    logger.info(f"expert_indices: {expert_indices[2,:,:,:]}")

    output_tokens, metadata, gathered_scores = ttnn.experimental.all_to_all_dispatch_selective_tilize(
        tt_input_tokens,
        tt_expert_indices,
        tt_expert_scores,
        tt_expert_mapping,
        cluster_axis=cluster_axis,
    )

    logger.info(f"output_tokens shape: {output_tokens.shape}")
    logger.info(f"metadata shape: {metadata.shape}")
    logger.info(f"gathered_scores shape: {gathered_scores.shape}")

    logger.info(f"input_tokens: {input_tokens}")
    logger.info(f"output_tokens at device 2: {output_tokens[2,:,:]}")

    # logger.info(f"expert_scores: {expert_scores}")
    # logger.info(f"gathered_scores: {gathered_scores}")

    # logger.info(f"output_tokens: {output_tokens}")
    # logger.info(f"metadata: {metadata}")

    golden_output_tokens, golden_metadata, golden_gathered_scores = get_a2a_dispatch_golden(
        input_tokens, expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
    )
    output_tokens = ttnn.to_torch(output_tokens, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).to(
        tt_to_torch_dtype(dtype)
    )
    metadata = ttnn.to_torch(metadata, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).to(torch.uint16)
    gathered_scores = ttnn.to_torch(gathered_scores, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).to(
        tt_to_torch_dtype(dtype)
    )

    assert torch.allclose(metadata, golden_metadata), f"Metadata mismatch."
    assert torch.allclose(gathered_scores, golden_gathered_scores), f"Gathered scores mismatch."

    verify_a2a_dispatch_output_tokens(
        output_tokens, golden_output_tokens, expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
    )

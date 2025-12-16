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


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
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
    "batch, seq_len",
    [
        (2, 1),
    ],
    ids=["b32s1"],
)
@pytest.mark.parametrize("num_links", ["MAX_LINKS"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
def test_all_to_all_dispatch_selective_tilize_no_trace_batch32(
    mesh_device,
    mesh_shape,
    cluster_axis,
    batch,
    experts,
    selected_experts_k,
    hidden_size,
    seq_len,
    num_links,
    dtype,
    input_memory_config,
    output_memory_config,
    device_params,
):
    torch.manual_seed(2005)
    if num_links == "MAX_LINKS":
        num_links = get_max_links(cluster_axis, device_params["fabric_config"])
    devices = mesh_shape[0] * mesh_shape[1]

    input_tokens = torch.rand(1, batch, seq_len, hidden_size, dtype=tt_to_torch_dtype(dtype)).repeat(
        mesh_shape[1 - cluster_axis], 1, 1, 1
    )
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

    logger.info(f"Expert Indices: {expert_indices} with shape {expert_indices.shape}")
    logger.info(f"Expert Mapping: {expert_mapping} with shape {expert_mapping.shape}")

    dte_intermediate, dte_scores, ed_table = get_dte_intermediate(
        indices_tensor=expert_indices,
        mapping_tensor=expert_mapping,
        scores_tensor=expert_scores,
        mesh_shape=mesh_shape,
        cluster_axis=cluster_axis,
    )

    logger.info(f"DTE Intermediate: {dte_intermediate} with shape {dte_intermediate.shape}")
    logger.info(f"DTE Scores: {dte_scores} with shape {dte_scores.shape}")
    logger.info(f"ED Table: {ed_table} with shape {ed_table.shape}")

    mesh_mapper = get_mesh_mapper(mesh_device, mesh_shape, cluster_axis, 2)

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
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=mesh_mapper,
    )

    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    (
        tt_output_tensor,
        tt_metadata_tensor,
        tt_dte_intermediate,
        tt_dte_scores,
        tt_ed_table,
    ) = ttnn.experimental.all_to_all_dispatch_selective_tilize(
        tt_input_tokens,
        tt_expert_indices,
        tt_expert_mapping,
        cluster_axis=cluster_axis,
        num_links=num_links,
        memory_config=output_memory_config,
    )

    logger.info(f"tt_output_tensor per-device shape: {tt_output_tensor.shape}")
    logger.info(f"tt_metadata_tensor per-device shape: {tt_metadata_tensor.shape}")
    logger.info(f"tt_dte_intermediate per-device shape: {tt_dte_intermediate.shape}")
    logger.info(f"tt_dte_scores per-device shape: {tt_dte_scores.shape}")
    logger.info(f"tt_ed_table per-device shape: {tt_ed_table.shape}")

    torch_tt_output_tensor = ttnn.to_torch(
        tt_output_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
    )

    torch_tt_metadata_tensor = ttnn.to_torch(
        tt_metadata_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
    )

    torch_tt_dte_intermediate = ttnn.to_torch(
        tt_dte_intermediate,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
    )
    torch_tt_dte_scores = ttnn.to_torch(
        tt_dte_scores,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
    )
    torch_tt_ed_table = ttnn.to_torch(
        tt_ed_table,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
    )

    logger.info(f"torch_tt_output_tensor shape: {torch_tt_output_tensor.shape}")
    logger.info(f"torch_tt_metadata_tensor shape: {torch_tt_metadata_tensor.shape}")
    logger.info(f"torch_tt_dte_intermediate shape: {torch_tt_dte_intermediate.shape}")
    logger.info(f"torch_tt_dte_scores shape: {torch_tt_dte_scores.shape}")
    logger.info(f"torch_tt_ed_table shape: {torch_tt_ed_table.shape}")

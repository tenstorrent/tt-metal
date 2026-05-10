# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

OPTIMIZED_MOE_CLUSTER_AXIS = 0


def get_expert_owner_linearized_mesh_coord(
    mesh_shape: Sequence[int],
    cluster_axis: int,
    expert_id: int,
    experts_per_device: int,
) -> int:
    """Map a global expert id to the owning device for optimized MoE routing."""
    if len(mesh_shape) != 2:
        raise ValueError(f"Expected a 2D mesh shape, got {tuple(mesh_shape)}")
    if cluster_axis not in (0, 1):
        raise ValueError(f"Expected cluster_axis 0 or 1, got {cluster_axis}")
    if experts_per_device <= 0:
        raise ValueError(f"experts_per_device must be positive, got {experts_per_device}")

    rows, cols = (int(mesh_shape[0]), int(mesh_shape[1]))
    devices_per_cluster = rows if cluster_axis == 0 else cols
    num_clusters = cols if cluster_axis == 0 else rows
    experts_per_cluster = devices_per_cluster * experts_per_device

    cluster_id = expert_id // experts_per_cluster
    if cluster_id >= num_clusters:
        raise ValueError(
            f"Expert {expert_id} is outside mesh capacity for mesh_shape={tuple(mesh_shape)}, "
            f"cluster_axis={cluster_axis}, experts_per_device={experts_per_device}"
        )

    expert_id_within_cluster = expert_id % experts_per_cluster
    device_id_within_cluster = expert_id_within_cluster // experts_per_device

    if cluster_axis == 0:
        row = device_id_within_cluster
        col = cluster_id
    else:
        row = cluster_id
        col = device_id_within_cluster

    return row * cols + col


def create_expert_mapping_tensor(
    mesh_shape: Sequence[int],
    cluster_axis: int,
    num_experts: int,
    experts_per_device: int,
    dtype=None,
):
    """Create the replicated expert-to-owner mapping tensor consumed by optimized MoE kernels."""
    import torch

    if dtype is None:
        dtype = torch.uint16

    rows, cols = (int(mesh_shape[0]), int(mesh_shape[1]))
    num_devices = rows * cols
    expert_mapping = torch.empty((1, num_experts), dtype=dtype)
    for expert_id in range(num_experts):
        expert_mapping[0, expert_id] = get_expert_owner_linearized_mesh_coord(
            mesh_shape,
            cluster_axis,
            expert_id,
            experts_per_device,
        )
    return expert_mapping.repeat(num_devices, 1)

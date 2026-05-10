# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch

from models.demos.deepseek_v3.tt.moe_expert_mapping import (
    create_expert_mapping_tensor,
    get_expert_owner_linearized_mesh_coord,
)


def test_cluster_axis_zero_maps_experts_along_columns():
    mesh_shape = (16, 8)
    experts_per_device = 2

    assert get_expert_owner_linearized_mesh_coord(mesh_shape, 0, 0, experts_per_device) == 0
    assert get_expert_owner_linearized_mesh_coord(mesh_shape, 0, 1, experts_per_device) == 0
    assert get_expert_owner_linearized_mesh_coord(mesh_shape, 0, 2, experts_per_device) == 8
    assert get_expert_owner_linearized_mesh_coord(mesh_shape, 0, 31, experts_per_device) == 120
    assert get_expert_owner_linearized_mesh_coord(mesh_shape, 0, 32, experts_per_device) == 1
    assert get_expert_owner_linearized_mesh_coord(mesh_shape, 0, 255, experts_per_device) == 127


def test_cluster_axis_one_matches_row_major_expert_ownership():
    mesh_shape = (16, 8)
    experts_per_device = 2

    for expert_id in range(256):
        assert (
            get_expert_owner_linearized_mesh_coord(mesh_shape, 1, expert_id, experts_per_device)
            == expert_id // experts_per_device
        )


def test_create_expert_mapping_tensor_replicates_axis_aware_mapping():
    mapping = create_expert_mapping_tensor((16, 8), 0, 256, 2)

    assert mapping.shape == (128, 256)
    assert mapping.dtype == torch.uint16
    assert torch.equal(mapping[0], mapping[-1])
    assert mapping[0, 2].item() == 8
    assert mapping[0, 32].item() == 1

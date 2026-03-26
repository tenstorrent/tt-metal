# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for ttnn.masked_bincount operation in isolation.

Verifies that the TTNN masked_bincount (per-device expert histogram) matches
a simple PyTorch reference: torch.bincount masked by expert presence.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, extract_mesh_config


def torch_masked_bincount(
    indices: torch.Tensor, expert_dispatch_table: torch.Tensor, n_routed_experts: int
) -> torch.Tensor:
    """
    Reference implementation: count expert occurrences masked by expert_dispatch_table.

    Args:
        indices: [sp_dim, topk] int tensor of expert indices.
        expert_dispatch_table: [n_routed_experts] int32 tensor (>= 0 = present, -1 = absent).
        n_routed_experts: number of experts (output size).

    Returns:
        [n_routed_experts] int tensor of masked histogram counts.
    """
    counts = torch.bincount(indices.flatten().to(torch.int64), minlength=n_routed_experts).to(torch.int32)
    mask = (expert_dispatch_table >= 0).to(torch.int32)
    return counts[:n_routed_experts] * mask


@pytest.mark.parametrize(
    "sp_dim, topk, n_routed_experts",
    [
        (4096, 8, 256),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 1),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 1), topology="linear"),
            id="single",
        ),
        pytest.param(
            (1, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 2), topology="linear"),
            id="linear-1x2",
        ),
        pytest.param(
            (1, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="mesh-1x4"),
            id="linear-1x4",
        ),
        pytest.param(
            (2, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_masked_bincount(
    mesh_device,
    sp_dim,
    topk,
    n_routed_experts,
):
    """Test ttnn.masked_bincount against PyTorch reference on each device."""
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.info(
        f"Testing masked_bincount: {mesh_device.shape=}, {sp_dim=}, {topk=}, "
        f"{n_routed_experts=}, {num_dispatch_groups=}"
    )
    ttnn.visualize_mesh_device(mesh_device)

    torch.manual_seed(42)

    # Generate expert indices as 2D [total_sp, topk] — sharded across dispatch axis, each device sees [sp_dim, topk]
    total_sp = dispatch_group_size * sp_dim
    indices = torch.randint(0, n_routed_experts, (total_sp, topk), dtype=torch.int32)

    # Expert dispatch table: maps expert_id -> chip_id (>= 0) or -1 (absent)
    dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=n_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    # Compute torch reference per (dispatch_group, chip) pair
    # torch_histograms[group][chip] = histogram for that group/chip combination
    torch_histograms = {}
    for group in range(num_dispatch_groups):
        for chip in range(dispatch_group_size):
            chip_indices = indices[chip * sp_dim : (chip + 1) * sp_dim]
            hist = torch_masked_bincount(chip_indices, dispatch_table[group], n_routed_experts)
            torch_histograms[(group, chip)] = hist

    # Height-shard config matching the gate module pattern: 8x8 core grid
    num_cores = 64
    assert sp_dim % num_cores == 0, f"sp_dim={sp_dim} must be divisible by {num_cores}"
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(sp_dim // num_cores, topk),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    # Shard dim 0 across the dispatch axis, replicate across TP axis
    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None) if sp_axis == 0 else (None, 0),
    )

    # UINT16 as required by the kernel
    tt_indices = ttnn.from_torch(
        indices.to(torch.int16),
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint16,
    )
    tt_indices = ttnn.to_memory_config(tt_indices, sharded_mem_config)

    tt_dispatch_table = ttnn.from_torch(
        dispatch_table,
        device=mesh_device,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(None, 0),
        ),
    )

    # Run ttnn op
    tt_histograms = ttnn.experimental.deepseek_prefill.masked_bincount(
        tt_indices, tt_dispatch_table, n_routed_experts, topk
    )

    # Compare per-device
    device_tensors = ttnn.get_device_tensors(tt_histograms)
    n_cols = mesh_device.shape[1]
    all_passed = True

    for device_id in range(len(device_tensors)):
        tt_hist = ttnn.to_torch(device_tensors[device_id]).flatten().to(torch.int32)

        row = device_id // n_cols
        col = device_id % n_cols
        chip_idx = row if sp_axis == 0 else col
        group_idx = col if sp_axis == 0 else row
        ref_hist = torch_histograms[(group_idx, chip_idx)]

        matches = torch.equal(tt_hist[:n_routed_experts], ref_hist)
        if not matches:
            diff_mask = tt_hist[:n_routed_experts] != ref_hist
            num_diff = diff_mask.sum().item()
            logger.error(f"Device {device_id} (row={row}, col={col}): {num_diff}/{n_routed_experts} bins differ")
            logger.error(f"  tt:  {tt_hist[:n_routed_experts]}")
            logger.error(f"  ref: {ref_hist}")
            all_passed = False
        else:
            logger.info(f"Device {device_id} (row={row}, col={col}): PASS (sum={ref_hist.sum().item()})")

    assert all_passed, "masked_bincount output does not match torch reference on one or more chips"
    logger.info("masked_bincount matches torch reference!")

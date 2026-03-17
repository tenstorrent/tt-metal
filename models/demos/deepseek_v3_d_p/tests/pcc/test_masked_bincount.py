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
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import extract_mesh_config


def torch_masked_bincount(indices: torch.Tensor, expert_mask: torch.Tensor, n_routed_experts: int) -> torch.Tensor:
    """
    Reference implementation: count expert occurrences masked by expert_mask.

    Args:
        indices: [sp_dim, topk] int tensor of expert indices.
        expert_mask: [n_routed_experts] int tensor (nonzero = present).
        n_routed_experts: number of experts (output size).

    Returns:
        [n_routed_experts] int tensor of masked histogram counts.
    """
    counts = torch.bincount(indices.flatten().to(torch.int64), minlength=n_routed_experts).to(torch.int32)
    mask = (expert_mask > 0).to(torch.int32)
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
            (2, 1),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 1), topology="linear"),
            id="linear-2x1",
        ),
        pytest.param(
            (4, 1),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="mesh-4x1"),
            id="linear-4x1",
        ),
        pytest.param(
            (4, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("mask_all_present", [True, False], ids=["all_present", "sparse_mask"])
def test_masked_bincount(
    mesh_device,
    sp_dim,
    topk,
    n_routed_experts,
    mask_all_present,
):
    """Test ttnn.masked_bincount against PyTorch reference on each device."""
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size

    logger.info(
        f"Testing masked_bincount: {mesh_device.shape=}, {sp_dim=}, {topk=}, "
        f"{n_routed_experts=}, {mask_all_present=}"
    )
    ttnn.visualize_mesh_device(mesh_device)

    torch.manual_seed(42)

    # Generate expert indices as 2D [total_sp, topk] — sharded across dispatch axis, each device sees [sp_dim, topk]
    total_sp = dispatch_group_size * sp_dim
    indices = torch.randint(0, n_routed_experts, (total_sp, topk), dtype=torch.int32)

    # Expert mask
    if mask_all_present:
        expert_mask = torch.ones(n_routed_experts, dtype=torch.int32)
    else:
        expert_mask = torch.randint(0, 2, (n_routed_experts,), dtype=torch.int32)

    # Compute torch reference per chip (each chip owns a contiguous sp_dim slice)
    torch_histograms = []
    for chip in range(dispatch_group_size):
        chip_indices = indices[chip * sp_dim : (chip + 1) * sp_dim]
        hist = torch_masked_bincount(chip_indices, expert_mask, n_routed_experts)
        torch_histograms.append(hist)
    torch_histograms = torch.stack(torch_histograms)  # [dispatch_group_size, n_routed_experts]

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

    tt_expert_mask = ttnn.from_torch(
        expert_mask,
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Run ttnn op
    tt_histograms = ttnn.masked_bincount(tt_indices, tt_expert_mask, n_routed_experts)

    # Compare per-device (TP replicas in the same dispatch row should match)
    device_tensors = ttnn.get_device_tensors(tt_histograms)
    n_rows = mesh_device.shape[0]
    n_cols = mesh_device.shape[1]
    all_passed = True

    for device_id in range(len(device_tensors)):
        tt_hist = ttnn.to_torch(device_tensors[device_id]).flatten().to(torch.int32)

        row = device_id // n_cols
        col = device_id % n_cols
        chip_idx = row if sp_axis == 0 else col
        ref_hist = torch_histograms[chip_idx]

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

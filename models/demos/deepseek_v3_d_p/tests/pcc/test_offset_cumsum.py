# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for ttnn.offset_cumsum operation in isolation.

Verifies that the TTNN offset_cumsum (all_gather + shifted prefix sum of
per-device expert histograms) matches a PyTorch reference.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, extract_mesh_config


def torch_offset_cumsum(histograms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation: shifted prefix sum across devices.

    Given per-device histograms of shape [num_devices, n_routed_experts],
    returns:
      - dispatch_offsets: [num_devices, n_routed_experts] where row k = sum of rows 0..k-1
        (row 0 is all zeros).
      - total_counts_per_expert: [1, n_routed_experts] sum of all rows.

    Args:
        histograms: [num_devices, n_routed_experts] int tensor.

    Returns:
        Tuple of (dispatch_offsets, total_counts_per_expert).
    """
    cum = torch.cumsum(histograms, dim=0)
    zeros = torch.zeros(1, histograms.shape[1], dtype=histograms.dtype)
    full = torch.cat([zeros, cum], dim=0)
    return full[:-1, :], full[-1:, :]


@pytest.mark.parametrize(
    "n_routed_experts",
    [256],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 1), topology="linear"),
            id="linear-2x1",
        ),
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-4x1",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-4x2"),
            id="mesh-2x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_offset_cumsum(
    mesh_device,
    n_routed_experts,
    num_links,
    topology,
):
    """Test ttnn.offset_cumsum against PyTorch reference."""
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size

    logger.info(
        f"Testing offset_cumsum: {mesh_device.shape=}, {sp_axis=}, " f"{dispatch_group_size=}, {n_routed_experts=}"
    )
    ttnn.visualize_mesh_device(mesh_device)

    torch.manual_seed(42)

    # Simulate per-device expert histograms (output of masked_bincount)
    # Shape: [dispatch_group_size, n_routed_experts]
    histograms = torch.randint(0, 32, (dispatch_group_size, n_routed_experts), dtype=torch.int32)

    # Torch reference: shifted prefix sum
    torch_offsets, torch_totals = torch_offset_cumsum(histograms)
    logger.info(f"Torch reference shapes: offsets={torch_offsets.shape}, totals={torch_totals.shape}")

    # Shard histograms across devices along the SP axis
    # Each device gets its own 1D histogram of shape [n_routed_experts]
    if sp_axis == 0:
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(0, None),
        )
    else:
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(None, 0),
        )

    tt_histograms = ttnn.from_torch(
        histograms,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run ttnn op — now returns (dispatch_offsets, total_counts_per_expert)
    tt_offsets, tt_totals = ttnn.experimental.deepseek_prefill.offset_cumsum(
        tt_histograms,
        cluster_axis=sp_axis,
        num_links=num_links,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    all_passed = True

    mesh_rows, mesh_cols = mesh_device.shape

    def get_row_idx(dev_idx):
        coord_row = dev_idx // mesh_cols
        coord_col = dev_idx % mesh_cols
        return coord_row if sp_axis == 0 else coord_col

    # Verify dispatch_offsets on each device (each device holds only its own [1, W] row)
    for dev_idx, dt in enumerate(ttnn.get_device_tensors(tt_offsets)):
        tt_out = ttnn.to_torch(dt).to(torch.int32)
        while tt_out.dim() > 2:
            tt_out = tt_out.squeeze(0)

        row_idx = get_row_idx(dev_idx)
        ref_row = torch_offsets[row_idx : row_idx + 1, :]
        logger.info(f"Device {dev_idx} (row_idx={row_idx}) offsets: tt_shape={tt_out.shape}, ref_shape={ref_row.shape}")

        if tt_out.shape != ref_row.shape:
            logger.error(f"Device {dev_idx}: offsets shape mismatch tt={tt_out.shape} ref={ref_row.shape}")
            all_passed = False
            continue

        if not torch.equal(tt_out, ref_row):
            diff_mask = tt_out != ref_row
            num_diff = diff_mask.sum().item()
            logger.error(f"Device {dev_idx}: offsets {num_diff}/{ref_row.numel()} elements differ")
            logger.error(f"  Max abs diff: {(tt_out - ref_row).abs().max().item()}")
            all_passed = False
        else:
            logger.info(f"Device {dev_idx} offsets: PASS")

    # Verify total_counts_per_expert on each device
    for dev_idx, dt in enumerate(ttnn.get_device_tensors(tt_totals)):
        tt_out = ttnn.to_torch(dt).to(torch.int32)
        while tt_out.dim() > 2:
            tt_out = tt_out.squeeze(0)

        logger.info(f"Device {dev_idx} totals: tt_shape={tt_out.shape}, ref_shape={torch_totals.shape}")

        if tt_out.shape != torch_totals.shape:
            logger.error(f"Device {dev_idx}: totals shape mismatch tt={tt_out.shape} ref={torch_totals.shape}")
            all_passed = False
            continue

        if not torch.equal(tt_out, torch_totals):
            diff_mask = tt_out != torch_totals
            num_diff = diff_mask.sum().item()
            logger.error(f"Device {dev_idx}: totals {num_diff}/{torch_totals.numel()} elements differ")
            logger.error(f"  Max abs diff: {(tt_out - torch_totals).abs().max().item()}")
            all_passed = False
        else:
            logger.info(f"Device {dev_idx} totals: PASS")

    assert all_passed, "offset_cumsum output does not match torch reference on one or more devices"
    logger.info("offset_cumsum matches torch reference!")

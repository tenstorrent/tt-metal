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


def torch_offset_cumsum(histograms: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation: shifted prefix sum across devices.

    Given per-device histograms of shape [num_devices, n_routed_experts],
    returns [num_devices + 1, n_routed_experts] where row k = sum of rows 0..k-1
    (row 0 is all zeros).

    Args:
        histograms: [num_devices, n_routed_experts] int tensor.

    Returns:
        [num_devices + 1, n_routed_experts] int tensor of shifted prefix sums.
    """
    cum = torch.cumsum(histograms, dim=0)
    zeros = torch.zeros(1, histograms.shape[1], dtype=histograms.dtype)
    return torch.cat([zeros, cum], dim=0)


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
    torch_result = torch_offset_cumsum(histograms)  # [dispatch_group_size + 1, n_routed_experts]
    logger.info(f"Torch reference shape: {torch_result.shape}")

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

    # Run ttnn op
    tt_result = ttnn.offset_cumsum(
        tt_histograms,
        cluster_axis=sp_axis,
        num_links=num_links,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Each device should have the same full [dispatch_group_size + 1, n_routed_experts] result
    device_tensors = ttnn.get_device_tensors(tt_result)
    all_passed = True

    for dev_idx, dt in enumerate(device_tensors):
        tt_out = ttnn.to_torch(dt).to(torch.int32)
        # Squeeze any extra leading dims added by ttnn
        while tt_out.dim() > 2:
            tt_out = tt_out.squeeze(0)

        logger.info(f"Device {dev_idx}: tt_shape={tt_out.shape}, ref_shape={torch_result.shape}")

        if tt_out.shape != torch_result.shape:
            logger.error(f"Device {dev_idx}: shape mismatch tt={tt_out.shape} ref={torch_result.shape}")
            all_passed = False
            continue

        matches = torch.equal(tt_out, torch_result)
        if not matches:
            diff_mask = tt_out != torch_result
            num_diff = diff_mask.sum().item()
            total = torch_result.numel()
            logger.error(f"Device {dev_idx}: {num_diff}/{total} elements differ")
            logger.error(f"  Max abs diff: {(tt_out - torch_result).abs().max().item()}")
            all_passed = False
        else:
            logger.info(f"Device {dev_idx}: PASS")

    assert all_passed, "offset_cumsum output does not match torch reference on one or more devices"
    logger.info("offset_cumsum matches torch reference!")

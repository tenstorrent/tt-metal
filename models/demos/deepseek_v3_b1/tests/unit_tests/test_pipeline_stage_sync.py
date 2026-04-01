# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for PipelineStageSync operation.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.micro_ops.pipeline_stage_sync.op import PipelineStageSync


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "stalling_device_mesh_coord, stalling_core, run_stalling_kernel_on_brisc, signalling_device_mesh_coord, signalling_core, run_signalling_kernel_on_brisc",
    [
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((1, 1)),
            ttnn.CoreCoord(1, 1),
            False,
        ),
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            False,
            ttnn.MeshCoordinate((1, 1)),
            ttnn.CoreCoord(1, 1),
            True,
        ),
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((1, 1)),
            ttnn.CoreCoord(1, 1),
            True,
        ),
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            False,
            ttnn.MeshCoordinate((1, 1)),
            ttnn.CoreCoord(1, 1),
            False,
        ),
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(1, 1),
            True,
        ),
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            False,
        ),
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            False,
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            True,
        ),
    ],
)
@pytest.mark.parametrize("num_iterations", [50])
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    indirect=["device_params"],
)
def test_pipeline_stage_sync_2d(
    bh_2d_mesh_device,
    stalling_device_mesh_coord,
    stalling_core,
    run_stalling_kernel_on_brisc,
    signalling_device_mesh_coord,
    signalling_core,
    run_signalling_kernel_on_brisc,
    num_iterations,
    num_devices,
):
    """Test pipeline_stage_sync with 2D fabric."""
    # Validate mesh size
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    logger.info(f"\n=== Testing pipeline_stage_sync (num_iterations={num_iterations}) ===")

    # Pseudo input/output tensors
    pseudo_input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        device=submesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            submesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh_device.shape),
        ),
    )
    pseudo_output_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        device=submesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            submesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh_device.shape),
        ),
    )

    # Run pipeline_stage_sync with looping inside the kernel
    PipelineStageSync.op(
        pseudo_input_tensor=pseudo_input_tensor,
        pseudo_output_tensor=pseudo_output_tensor,
        mesh_device=submesh_device,
        stalling_device_mesh_coord=stalling_device_mesh_coord,
        stalling_core=stalling_core,
        run_stalling_kernel_on_brisc=run_stalling_kernel_on_brisc,
        signalling_device_mesh_coord=signalling_device_mesh_coord,
        signalling_core=signalling_core,
        run_signalling_kernel_on_brisc=run_signalling_kernel_on_brisc,
        num_iterations=num_iterations,
    )
    ttnn.synchronize_device(submesh_device)

    logger.info("Test passed!")

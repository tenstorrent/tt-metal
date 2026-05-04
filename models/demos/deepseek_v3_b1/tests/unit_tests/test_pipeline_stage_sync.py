# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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
    "src_device_mesh_coord, signalling_core, run_signalling_kernel_on_ncrisc, dst_device_mesh_coord, stalling_core, run_stalling_kernel_on_ncrisc",
    [
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((1, 1)),
            ttnn.CoreCoord(1, 1),
            False,
        ),  # nc signaller, b staller
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            False,
            ttnn.MeshCoordinate((1, 1)),
            ttnn.CoreCoord(1, 1),
            True,
        ),  # b signaller, nc staller
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((1, 1)),
            ttnn.CoreCoord(1, 1),
            True,
        ),  # nc signaller, nc staller
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            False,
            ttnn.MeshCoordinate((1, 1)),
            ttnn.CoreCoord(1, 1),
            False,
        ),  # b signaller, b staller
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((1, 1)),
            ttnn.CoreCoord(0, 0),
            False,
        ),  # same core, different risc
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((1, 1)),
            ttnn.CoreCoord(0, 0),
            True,
        ),  # same core, same risc
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((2, 1)),
            ttnn.CoreCoord(1, 1),
            False,
        ),  # multiple intermediate, first col to second col
        (
            ttnn.MeshCoordinate((2, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((0, 1)),
            ttnn.CoreCoord(1, 1),
            False,
        ),  # multiple intermediate, second col to first col
        (
            ttnn.MeshCoordinate((3, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(1, 1),
            False,
        ),  # wrap around
        (
            ttnn.MeshCoordinate((1, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(1, 1),
            False,
        ),  # backwards across rows
        (
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((0, 1)),
            ttnn.CoreCoord(1, 1),
            False,
        ),  # just col traversal, left to right
        (
            ttnn.MeshCoordinate((0, 1)),
            ttnn.CoreCoord(0, 0),
            True,
            ttnn.MeshCoordinate((0, 0)),
            ttnn.CoreCoord(1, 1),
            False,
        ),  # just col traversal, right to left
    ],
)
@pytest.mark.parametrize("num_iterations", [50])
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X})],
    indirect=["device_params"],
)
def test_pipeline_stage_sync_2d(
    bh_2d_mesh_device,
    src_device_mesh_coord,
    signalling_core,
    run_signalling_kernel_on_ncrisc,
    dst_device_mesh_coord,
    stalling_core,
    run_stalling_kernel_on_ncrisc,
    num_iterations,
    num_devices,
):
    """Test pipeline_stage_sync with 2D fabric."""
    # Validate mesh size
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    logger.info(f"=== Testing pipeline_stage_sync (num_iterations={num_iterations}) ===")

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

    compute_grid_size = submesh_device.compute_with_storage_grid_size()
    num_cores = compute_grid_size.x * compute_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
    semaphore = ttnn.create_global_semaphore(submesh_device, available_cores, 0)

    # Run pipeline_stage_sync with looping inside the kernel
    ttnn.synchronize_device(submesh_device)
    PipelineStageSync.op(
        pseudo_input_tensor=pseudo_input_tensor,
        pseudo_output_tensor=pseudo_output_tensor,
        mesh_device=submesh_device,
        semaphore=semaphore,
        src_device_mesh_coord=src_device_mesh_coord,
        signalling_core=signalling_core,
        run_signalling_kernel_on_ncrisc=run_signalling_kernel_on_ncrisc,
        dst_device_mesh_coord=dst_device_mesh_coord,
        stalling_core=stalling_core,
        run_stalling_kernel_on_ncrisc=run_stalling_kernel_on_ncrisc,
        num_iterations=num_iterations,
    )
    ttnn.synchronize_device(submesh_device)

    logger.info("Test passed!")

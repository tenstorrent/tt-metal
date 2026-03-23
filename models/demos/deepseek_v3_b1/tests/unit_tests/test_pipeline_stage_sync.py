# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for PipelineStageSyncB1 operation.
"""

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.micro_ops.ccl_pipeline_stage_sync.op import PipelineStageSyncB1


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "stalling_device_mesh_coord, signalling_device_mesh_coord",
    [
        (ttnn.MeshCoordinate((0, 0)), ttnn.MeshCoordinate((1, 1))),
        (ttnn.MeshCoordinate((0, 0)), ttnn.MeshCoordinate((0, 0))),
    ],
)
@pytest.mark.parametrize(
    "stalling_core, signalling_core",
    [
        (ttnn.CoreCoord(), ttnn.CoreCoord()),
    ],
)
@pytest.mark.parametrize("num_iterations", [50])
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    indirect=["device_params"],
)
def test_reduce_to_one_2d(
    bh_2d_mesh_device,
    stalling_device_mesh_coord,
    signalling_device_mesh_coord,
    stalling_core,
    signalling_core,
    num_iterations,
):
    """Test pipeline_stage_sync with 2D fabric."""
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    logger.info(f"\n=== Testing pipeline_stage_sync (num_iterations={num_iterations}) ===")

    # Run pipeline_stage_sync with looping inside the kernel
    PipelineStageSyncB1.op(
        mesh_device=submesh_device,
        stalling_device_mesh_coord=stalling_device_mesh_coord,
        stalling_core=stalling_core,
        signalling_device_mesh_coord=signalling_device_mesh_coord,
        signalling_core=signalling_core,
        num_iterations=num_iterations,
    )
    ttnn.synchronize_device(submesh_device)

    logger.info("Test passed!")

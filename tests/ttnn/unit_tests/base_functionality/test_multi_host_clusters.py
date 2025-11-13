# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch


# Multi host trace tests


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((8, 16), id="8x16_dev_grid"),
        pytest.param((16, 8), id="16x8_dev_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "tensor_shard_shape",
    [
        pytest.param((8, 4), id="8x4_shard"),
        pytest.param((4, 8), id="4x8_shard"),
        pytest.param((8, 8), id="8x8_shard"),
        pytest.param((8, 16), id="8x16_shard"),
        pytest.param((16, 8), id="16x8_shard"),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 500000}], indirect=True)
def test_quad_galaxy_mesh_device_trace(mesh_device, tensor_shard_shape):
    torch_input_0 = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch.uint16)
    input_0_dev = ttnn.from_torch(
        torch_input_0,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=tensor_shard_shape),
    )
    output_dev = ttnn.relu(input_0_dev)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    ttnn.relu(input_0_dev)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((8, 8), id="8x8_dev_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "tensor_shard_shape",
    [
        pytest.param((8, 4), id="8x4_shard"),
        pytest.param((4, 8), id="4x8_shard"),
        pytest.param((8, 8), id="8x8_shard"),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 500000}], indirect=True)
def test_dual_galaxy_mesh_device_trace(mesh_device, tensor_shard_shape):
    torch_input_0 = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch.uint16)
    input_0_dev = ttnn.from_torch(
        torch_input_0,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=tensor_shard_shape),
    )
    output_dev = ttnn.relu(input_0_dev)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    ttnn.relu(input_0_dev)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.release_trace(mesh_device, trace_id)

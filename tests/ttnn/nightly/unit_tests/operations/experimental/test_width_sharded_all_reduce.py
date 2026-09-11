# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for ttnn.experimental.deepseek.width_sharded_all_reduce.

ROW_MAJOR WIDTH_SHARDED input is line-mcast across ``cluster_axis`` and reduced
as 1x32 RM faces. Output layout and shard spec match the input.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _width_sharded_l1_config(device, height: int, width: int, num_cores: int) -> ttnn.MemoryConfig:
    grid_size = device.compute_with_storage_grid_size()
    device_cores = grid_size.x * grid_size.y
    if num_cores >= device_cores:
        pytest.skip(f"need spare fabric-worker cores: {num_cores} tensor cores on a {grid_size.x}x{grid_size.y} grid")
    if width % num_cores != 0:
        pytest.skip(f"width {width} does not split evenly over {num_cores} cores")
    shard_width = width // num_cores
    if shard_width % 32 != 0:
        pytest.skip(f"shard width {shard_width} is not a multiple of 32")
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)
    shard_spec = ttnn.ShardSpec(core_grid, (height, shard_width), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "logical_shape, num_cores",
    [
        ((1, 1, 1, 256), 4),
        ((1, 1, 2, 128), 4),
        ((1, 1, 1, 4096), 64),  # decode o_b: shard [1, 64] on 8x8
    ],
    ids=["1x256-4c", "2x128-4c", "decode-4096-64c"],
)
def test_width_sharded_all_reduce(mesh_device, logical_shape, num_cores):
    torch.manual_seed(0)

    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[1] < 2:
        pytest.skip(f"need a line of at least 2 devices, got mesh {mesh_shape}")
    cluster_axis = 1
    ring_size = mesh_shape[1]

    height, width = logical_shape[-2], logical_shape[-1]
    mem_config = _width_sharded_l1_config(mesh_device, height, width, num_cores)

    torch_per_device = torch.randn((ring_size, *logical_shape), dtype=torch.bfloat16)
    expected = torch_per_device.float().sum(dim=0).to(torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_per_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=mem_config,
    )

    tt_output = ttnn.experimental.deepseek.width_sharded_all_reduce(
        tt_input,
        cluster_axis=cluster_axis,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )

    assert tt_output.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert tt_output.memory_config().shard_spec.shape == mem_config.shard_spec.shape
    assert tt_output.memory_config().shard_spec.grid == mem_config.shard_spec.grid

    got = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert got.shape[0] == ring_size
    for device_id in range(ring_size):
        assert_with_pcc(expected, got[device_id], pcc=0.999)

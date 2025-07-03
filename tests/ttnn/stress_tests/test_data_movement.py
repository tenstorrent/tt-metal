# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from ttnn.device import get_device_core_grid, is_blackhole, is_wormhole_b0

NUM_REPEATS = 5
NUM_DEVICES = ttnn.distributed.get_num_pcie_devices()


##### WORMMHOLE #######
L1_INPUT_SHAPE_WH = (30_000, 8, 8)
L1_OUTPUT_SHAPE_WH = (3_000, 16, 40)

DRAM_INPUT_SHAPE_WH = (3_125_000, 8, 8)
DRAM_OUTPUT_SHAPE_WH = (312_500, 16, 40)

SHARDING_INPUT_SHAPE_WH = (1, 6_000, 2_000)

##### BLACKHOLE #######
L1_INPUT_SHAPE_BH = (60_000, 8, 8)
L1_OUTPUT_SHAPE_BH = (6_000, 16, 40)

DRAM_INPUT_SHAPE_BH = (7_500_000, 8, 8)
DRAM_OUTPUT_SHAPE_BH = (750_000, 16, 40)

SHARDING_INPUT_SHAPE_BH = (1, 6_000, 6_000)


def reshape_input_shapes(test_case: str):
    if is_blackhole():
        return (
            (L1_INPUT_SHAPE_BH, L1_OUTPUT_SHAPE_BH)
            if test_case == "l1"
            else (DRAM_INPUT_SHAPE_BH, DRAM_OUTPUT_SHAPE_BH)
        )
    elif is_wormhole_b0():
        return (
            (L1_INPUT_SHAPE_WH, L1_OUTPUT_SHAPE_WH)
            if test_case == "l1"
            else (DRAM_INPUT_SHAPE_WH, DRAM_OUTPUT_SHAPE_WH)
        )
    else:
        raise RuntimeError("Unidentifiable device")


@pytest.mark.parametrize("mesh_device", [(1, NUM_DEVICES)], indirect=True)
@pytest.mark.parametrize(
    "shapes_memory_config",
    [
        (*reshape_input_shapes("l1"), ttnn.L1_MEMORY_CONFIG),
        (*reshape_input_shapes("dram"), ttnn.DRAM_MEMORY_CONFIG),
    ],
)
def test_stress_reshape(mesh_device, shapes_memory_config):
    input_shape, output_shape, memory_config = shapes_memory_config
    for _ in range(NUM_REPEATS):
        torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input_tensor, memory_config=memory_config, device=mesh_device)
        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        output_tensor = ttnn.reshape(input_tensor, output_shape)

        del input_tensor
        del output_tensor

    assert True


def sharding_input_shape():
    if is_blackhole():
        return SHARDING_INPUT_SHAPE_BH
    elif is_wormhole_b0():
        return SHARDING_INPUT_SHAPE_WH
    else:
        raise RuntimeError("Unidentifiable device")


@pytest.mark.parametrize("mesh_device", [(1, NUM_DEVICES)], indirect=True)
def test_stress_reshard(mesh_device):
    input_tensor_shape = sharding_input_shape()
    core_grid = get_device_core_grid(mesh_device)

    l1_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
    )

    shard_shape = input_tensor_shape[1] // core_grid.x, input_tensor_shape[2] // core_grid.y
    shard_shape = tuple(s + 32 - s % 32 for s in shard_shape)

    shard_spec = ttnn.ShardSpec(l1_shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)

    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    for _ in range(NUM_REPEATS):
        input_tensor_torch = torch.randn(input_tensor_shape)
        input_tensor = ttnn.from_torch(input_tensor_torch, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        output_tensor = ttnn.interleaved_to_sharded(input_tensor, sharded_mem_config)

        del input_tensor
        del output_tensor

    assert True

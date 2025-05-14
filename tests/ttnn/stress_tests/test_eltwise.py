# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from ttnn.device import get_device_core_grid, is_blackhole, is_wormhole_b0

NUM_REPEATS = 5
NUM_DEVICES = ttnn.distributed.get_num_pcie_devices()

##### WORMMHOLE #######
L1_INPUT_SHAPE_WH = (10_000, 8, 8)

DRAM_INPUT_SHAPE_WH = (1_500_000, 8, 8)

##### BLACKHOLE #######
L1_INPUT_SHAPE_BH = (25_000, 8, 8)

DRAM_INPUT_SHAPE_BH = (3_000_000, 8, 8)


def eltwise_input_shapes(test_case: str):
    if is_blackhole():
        return L1_INPUT_SHAPE_BH if test_case == "l1" else DRAM_INPUT_SHAPE_BH
    elif is_wormhole_b0():
        return L1_INPUT_SHAPE_WH if test_case == "l1" else DRAM_INPUT_SHAPE_WH
    else:
        raise RuntimeError("Unidentifiable device")


@pytest.mark.parametrize("mesh_device", [(1, NUM_DEVICES)], indirect=True)
@pytest.mark.parametrize(
    "shape_memory_config",
    [
        (eltwise_input_shapes("l1"), ttnn.L1_MEMORY_CONFIG),
        (eltwise_input_shapes("dram"), ttnn.DRAM_MEMORY_CONFIG),
    ],
)
def test_stress_binary(mesh_device, use_program_cache, shape_memory_config):
    input_shape, memory_config = shape_memory_config
    for _ in range(NUM_REPEATS):
        torch_input_tensor1 = torch.randn(input_shape, dtype=torch.bfloat16)
        torch_input_tensor2 = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor1 = ttnn.from_torch(
            torch_input_tensor1, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=mesh_device
        )
        input_tensor2 = ttnn.from_torch(
            torch_input_tensor2, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=mesh_device
        )

        output_tensor = ttnn.add(input_tensor1, input_tensor2)

        del torch_input_tensor1
        del torch_input_tensor2
        del input_tensor1
        del input_tensor2
        del output_tensor

    assert True


@pytest.mark.parametrize("mesh_device", [(1, NUM_DEVICES)], indirect=True)
@pytest.mark.parametrize(
    "shape_memory_config",
    [
        (eltwise_input_shapes("l1"), ttnn.L1_MEMORY_CONFIG),
        (eltwise_input_shapes("dram"), ttnn.DRAM_MEMORY_CONFIG),
    ],
)
def test_stress_unary(mesh_device, use_program_cache, shape_memory_config):
    input_shape, memory_config = shape_memory_config
    for _ in range(NUM_REPEATS):
        torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=mesh_device
        )

        output_tensor = ttnn.silu(input_tensor)

        del torch_input_tensor
        del input_tensor
        del output_tensor

    assert True

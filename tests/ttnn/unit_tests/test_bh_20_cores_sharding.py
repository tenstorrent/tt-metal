# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from models.utility_functions import is_blackhole
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def skip_if_not_blackhole_20_cores(device):
    if not is_blackhole():
        pytest.skip("This test is intended to run only on Blackhole devices with 20 cores.")
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(
            f"This test is intended to run only on Blackhole devices with 20 cores. Core grid [{compute_grid.x},{compute_grid.y}] must be [5, 4]."
        )


def test_sharding_on_bh_20_cores(device):
    skip_if_not_blackhole_20_cores(device)

    H = 1024 * 20
    W = 32
    in0_shape = (1, 1, H, W)
    torch_input_tensor = torch.rand(in0_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    compute_grid = device.compute_with_storage_grid_size()
    print("Compute grid is: ", compute_grid)
    num_cores = compute_grid.x * compute_grid.y
    assert num_cores > 0, "Number of cores must be greater than zero"
    assert (
        compute_grid.x == 5 and compute_grid.y == 4
    ), f"Compute grid x={compute_grid.x=}, expected 5, y={compute_grid.y=}, expected 4"

    core_grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)

    core_range_set = ttnn.num_cores_to_corerangeset(num_cores, compute_grid)
    height_sharded_memory_config = ttnn.create_sharded_memory_config(
        in0_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    print("Height sharded memory config is: ", height_sharded_memory_config)
    print(f"Number of cores: {num_cores}, Core range set: {core_range_set}")
    ttnn_input_tensor_hs = ttnn.to_memory_config(ttnn_input_tensor, height_sharded_memory_config)

    output_tensor = ttnn.to_torch(ttnn_input_tensor_hs)
    assert_with_pcc(output_tensor, torch_input_tensor, 1.0)

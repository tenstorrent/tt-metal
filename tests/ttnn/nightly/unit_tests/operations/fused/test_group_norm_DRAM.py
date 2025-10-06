# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import is_blackhole

from tests.ttnn.unit_tests.operations.fused.test_group_norm_DRAM import get_welford_params

welford_flavors = get_welford_params()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x",
    [
        # Only SDXL/sd35 tests with 512x512 or larger sizes moved to nightly
        #  SDXL VAE
        (1, 128, 1024, 1024, 32, 32, 8, 8),
        (1, 128, 512, 512, 32, 8, 8, 8),
        (1, 256, 1024, 1024, 32, 48, 8, 8),
        (1, 256, 515, 512, 32, 12, 8, 8),
        (1, 512, 512, 512, 32, 12, 8, 8),
        # SDXL Refiner
        (1, 256, 512, 512, 32, 16, 8, 8),  # SD 1.4 VAE
        (1, 128, 512, 512, 32, 22, 4, 4),  # SD 1.4 VAE
        # sd35. 4 indicates the number of device.
        (1, 512 // 4, 512, 512, 32 // 4, 8, 8, 8),
        (1, 256 // 4, 512, 512, 32 // 4, 4, 8, 8),
        (1, 256 // 4, 1024, 1024, 32 // 4, 16, 8, 8),
        (1, 128 // 4, 1024, 1024, 32 // 4, 8, 8, 8),
        # mochi
        # (21, 128, 480, 848, 32, 140, 8, 8), Failing on single device CI.
    ],
)
@pytest.mark.parametrize("welford_mode", welford_flavors)
def test_group_norm_DRAM(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x)

    # Determine welford and reciprocals settings
    use_welford = welford_mode in ("welford_normal", "welford_reciprocal")
    use_reciprocals = welford_mode == "welford_reciprocal"

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias, eps=1e-12
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor_row_major = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_tilized = ttnn.tilize_with_zero_padding(input_tensor_row_major, use_multicore=True)

    # Create dram group norm params
    [gamma_t, beta_t], input_mask_tensor = ttnn.dram_group_norm_params_from_torch(
        [torch_weight, torch_bias], C, num_groups, device, core_grid=grid_size, return_mask=True
    )

    # Create reciprocals tensor if needed
    reciprocals_tensor = None
    if use_reciprocals:
        # Generate reciprocals tensor
        torch_reciprocals = ttnn.create_group_norm_reciprocals(N, C, H, W, num_groups, grid_size)
        reciprocals_tensor = ttnn.from_torch(
            torch_reciprocals,
            dtype=ttnn.DataType.FLOAT32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
                    ),
                    (torch_reciprocals.shape[0] // (grid_size.x * grid_size.y), torch_reciprocals.shape[1]),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )

    # groupnorm
    num_itr = 2  # second iteration to help catch potential runtime args issue.
    for _ in range(num_itr):
        output_tensor = ttnn.group_norm(
            input_tensor_tilized,
            num_groups=num_groups,
            input_mask=input_mask_tensor,
            weight=gamma_t,
            bias=beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=grid_size,
            inplace=False,
            num_out_blocks=num_out_blocks,
            use_welford=use_welford,
            reciprocals=reciprocals_tensor,
        )
        ttnn.synchronize_device(device)

    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9996)

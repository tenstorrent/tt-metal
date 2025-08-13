# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.utility_functions import skip_for_wormhole_b0, skip_for_blackhole


def calculate_max_valid_cores_for_group_norm(num_groups: int, num_channels: int, tile_width: int = 32) -> int:
    """
    Calculate the maximum number of cores per dimension that can be used for group norm operations.

    Args:
        num_groups: Number of groups for group normalization
        num_channels: Number of channels in the tensor
        tile_width: Width of a tile (default 32)

    Returns:
        Maximum valid number of cores per dimension, or 0 if no valid configuration exists

    Raises:
        ValueError: If input parameters are invalid
    """
    if num_channels <= 0 or num_groups <= 0 or tile_width <= 0:
        raise ValueError("All parameters must be positive integers")

    if num_channels % tile_width != 0:
        logger.error(
            f"Invalid configuration: num_channels ({num_channels}) must be divisible by tile_width ({tile_width})"
        )
        return 0

    if num_channels % num_groups != 0:
        logger.error(
            f"Invalid configuration: num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        )
        return 0

    num_tiles = num_channels // tile_width
    group_width = num_channels // num_groups

    logger.info(
        f"Group norm core calculation: channels={num_channels}, groups={num_groups}, "
        f"tiles={num_tiles}, group_width={group_width}"
    )

    max_cores_to_test = 8
    for num_cores in range(max_cores_to_test, 0, -1):
        # Check if tiles can be evenly distributed across cores
        if num_tiles % num_cores != 0:
            logger.debug(f"  cores={num_cores}: SKIP - tiles ({num_tiles}) not evenly divisible")
            continue

        tiles_per_core = num_tiles // num_cores
        channels_per_core = tiles_per_core * tile_width

        if channels_per_core % group_width != 0:
            logger.debug(
                f"  cores={num_cores}: SKIP - channels_per_core ({channels_per_core}) "
                f"not divisible by group_width ({group_width})"
            )
            continue

        groups_per_core = channels_per_core // group_width
        logger.info(
            f"  cores={num_cores}: VALID - tiles_per_core={tiles_per_core}, "
            f"channels_per_core={channels_per_core}, groups_per_core={groups_per_core}"
        )
        return num_cores

    logger.error("No valid core configuration found")
    return 0


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x",
    [
        (8, 768, 1, 512, 32, 2, 8, 8),  # base case
        (9, 768, 1, 512, 32, 2, 8, 8),  # test batch size 9 (uneven batch sizes)
        (1, 768, 1, 512, 32, 2, 8, 8),  # test group channel count is less than tile size
        (1, 480, 1, 64, 8, 1, 1, 1),  # test last group ends less than max tile span
        (1, 2560, 1, 512, 32, 2, 8, 8),  # test mcast num_out_blocks 2
        (1, 2560, 1, 1024, 32, 4, 8, 8),  # test mcast num_out_blocks 4
        (1, 768, 1, 512, 32, 2, 8, 8),  # test group channel count is less than tile size
        (2, 768, 1, 512, 32, 2, 8, 8),  # test batch size 2 (still multicast)
        (8, 768, 1, 512, 32, 2, 8, 8),  # test batch size 8 (no multicast)
        (8, 768, 1, 512, 32, 3, 8, 8),  # test batch size 8 (no multicast), but uneven num_out_blocks divisor
        (9, 768, 1, 512, 32, 2, 8, 8),  # test batch size 9 (uneven batch sizes)
        (
            1,
            128,
            1,
            512,
            32,
            2,
            4,
            4,
        ),  # test all groups on core fit in less than one tile, so need to reduce col core count
        # # # SDXL 1024x1024 resoultion
        (1, 640, 128, 128, 32, 3, 4, 4),
        (1, 960, 128, 128, 32, 6, 2, 2),
        # VAE
        # tensor is too large, but good example
        (1, 256, 1024, 1024, 32, 128, 8, 4),
        (1, 256, 512, 512, 32, 32, 4, 4),
        (1, 512, 128, 128, 32, 4, 1, 4),
        (1, 512, 256, 256, 32, 16, 4, 4),
        (1, 512, 512, 512, 32, 32, 4, 4),
        (1, 512, 64, 64, 32, 1, 8, 8),  # SD 1.4 VAE
        (1, 512, 128, 128, 32, 1, 8, 8),  # SD 1.4 VAE
        (1, 512, 256, 256, 32, 4, 8, 8),  # SD 1.4 VAE
        (1, 256, 256, 256, 32, 8, 8, 8),  # SD 1.4 VAE
        (1, 256, 512, 512, 32, 16, 8, 8),  # SD 1.4 VAE
        (1, 128, 512, 512, 32, 22, 4, 4),  # SD 1.4 VAE
        # SDXL Refiner
        (1, 1152, 128, 128, 32, 3, 4, 4),
        (1, 1152, 64, 64, 32, 1, 4, 4),
        (1, 1536, 16, 16, 32, 1, 8, 8),
        (1, 1536, 32, 32, 32, 1, 8, 8),
        (1, 1536, 64, 64, 32, 1, 8, 8),
        (1, 2304, 32, 32, 32, 1, 8, 8),
        (1, 2304, 64, 64, 32, 1, 8, 8),
        (1, 3072, 16, 16, 32, 1, 8, 8),
        (1, 3072, 32, 32, 32, 1, 8, 8),
        (1, 384, 128, 128, 32, 3, 4, 4),
        (1, 384, 64, 64, 32, 1, 4, 4),
        (1, 768, 128, 128, 32, 2, 8, 8),
        (1, 768, 32, 32, 32, 1, 8, 8),
        (1, 768, 64, 64, 32, 1, 8, 8),
    ],
)
def test_group_norm_DRAM(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    # Calculate and log the maximum valid cores for this configuration
    max_valid_cores = calculate_max_valid_cores_for_group_norm(num_groups, C)
    logger.info(
        f"Test case: N={N}, C={C}, H={H}, W={W}, groups={num_groups}, "
        f"requested_grid={cores_y}x{cores_x}, max_valid_cores_per_dim={max_valid_cores}"
    )

    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x)

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
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

    # input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # gamma/beta
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # groupnorm
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
    )

    ttnn.synchronize_device(device)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9996)

# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.utility_functions import skip_for_wormhole_b0


# for debug purpose
def manual_group_norm(input_tensor, num_groups, eps=1e-2):
    N, C, H, W = input_tensor.shape
    assert C % num_groups == 0, "Number of channels must be divisible by number of groups"

    # Reshape into groups
    group_channels = C // num_groups
    input_tensor = input_tensor.view(N, num_groups, group_channels, H, W)

    # Calculate mean and variance
    mean = input_tensor.mean(dim=(2, 3, 4), keepdim=True)
    var = input_tensor.var(dim=(2, 3, 4), keepdim=True)

    # Normalize
    input_tensor = (input_tensor - mean) / torch.sqrt(var + eps)

    # Reshape back to original dimensions
    input_tensor = input_tensor.view(N, C, H, W)
    return input_tensor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x",
    [
        # (1, 2560, 1, 512, 32, 1, 8, 8), # base case
        # (1, 2560, 1, 512, 32, 2, 8, 8), # test mcast num_out_blocks 2
        # (1, 2560, 1, 1024, 32, 4, 8, 8), # test mcast num_out_blocks 4
        # (1, 768, 1, 512, 32, 2, 8, 8), # test group channel count is less than tile size
        # (2, 768, 1, 512, 32, 2, 8, 8), # test batch size 2 (still multicast)
        # (8, 768, 1, 512, 32, 2, 8, 8), # test batch size 8 (no multicast)
        # (8, 768, 1, 512, 32, 3, 8, 8), # test batch size 8 (no multicast), but uneven num_out_blocks divisor
        # (9, 768, 1, 512, 32, 2, 8, 8), # test batch size 9 (uneven batch sizes)
        # (1, 128, 1, 512, 32, 2, 4, 8), # test all groups on core fit in less than one tile, so need to reduce col core count
        # (4, 768, 60, 106, 32, 8, 8, 4),  # Mochi VAE variant 1 (sharded, so T=1/8th the full tensor)
        # (3, 768, 60, 106, 32, 8, 8, 3),  # Mochi VAE variant 1 (sharded, so T=1/8th the full tensor)
        # (11, 512, 120, 212, 32, 10, 8, 8), # Mochi VAE variant 2 (sharded, so T=1/8th the full tensor)
        # (10, 512, 120, 212, 32, 10, 8, 8), # Mochi VAE variant 2 (sharded, so T=1/8th the full tensor)
        # (21, 256, 240, 424, 32, 40, 8, 8), # Mochi VAE variant 3 (sharded, so T=1/8th the full tensor)
        # (20, 256, 240, 424, 32, 40, 8, 8), # Mochi VAE variant 3 (sharded, so T=1/8th the full tensor)
        # (21, 128, 480, 848, 32, 135, 4, 8), # Mochi VAE variant 4 (sharded, so T=1/8th the full tensor)
        # (20, 128, 480, 848, 32, 135, 4, 8), # Mochi VAE variant 4 (sharded, so T=1/8th the full tensor)
        # (1,640,128,128,32,2,4,8),#Stable Diffusion XL Variant 1
        # (1,960,128,128,32,2,2,8),#Stable Diffusion XL Variant 2
    ],
)
def test_group_norm_DRAM(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

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

    # output tensor
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9996)

# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utils_for_testing import assert_numeric_metrics


def run_sdxl_base_group_norm_test(
    device, N, C, H, W, use_welford, layout, inplace, specify_grid=True, perf_test_mode=False
):
    num_groups = 32  #  always 32 for SDXL Base
    if layout == ttnn.TILE_LAYOUT and inplace:
        pytest.skip("Tile layout requires non-inplace tensors.")
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    core_grid = ttnn.CoreGrid(y=8, x=8)

    # Generate torch tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)

    if not perf_test_mode:
        # Execute torch group_norm
        torch_output_tensor = torch.nn.functional.group_norm(torch_input_tensor, num_groups)
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Generate ttnn tensor
    dummy_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    tt_input_tensor = ttnn.from_torch(
        dummy_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=layout,
        memory_config=ttnn.create_sharded_memory_config(
            shape=dummy_tensor.shape,
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        device=device,
    )

    # Generate input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, core_grid.x, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    # Execute ttnn group_norm
    tt_output_tensor = ttnn.group_norm(
        tt_input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        memory_config=tt_input_tensor.memory_config(),
        core_grid=core_grid if specify_grid else None,
        inplace=inplace,
        use_welford=use_welford,
    )
    ttnn.synchronize_device(device)

    if not perf_test_mode:
        tt_output_tensor = ttnn.from_device(tt_output_tensor)
        tt_output_tensor = ttnn.to_torch(tt_output_tensor)

        if use_welford:
            pcc_threshold = 0.9995
            rtol = 0.14
            atol = 0.085
            frobenius_threshold = 0.04
        else:
            pcc_threshold = 0.9999
            rtol = 0.065
            atol = 0.065
            frobenius_threshold = 0.04
        assert_numeric_metrics(
            torch_output_tensor,
            tt_output_tensor,
            pcc_threshold=pcc_threshold,
            rtol=rtol,
            atol=atol,
            frobenius_threshold=frobenius_threshold,
        )


def test_sdxl_base_group_norm(device, input_shape, use_welford, specify_grid=True, perf_test_mode=False):
    # Only one test case has C == 512, which has TILE_LAYOUT and inplace False
    # ALL other inputs have ROW_MAJOR_LAYOUT and inplace True
    N, C, H, W = input_shape
    layout = ttnn.TILE_LAYOUT if C == 512 else ttnn.ROW_MAJOR_LAYOUT
    inplace = layout != ttnn.TILE_LAYOUT
    run_sdxl_base_group_norm_test(device, N, C, H, W, use_welford, layout, inplace, specify_grid, perf_test_mode)


def test_sdxl_base_group_norm_negative_mask(device, input_shape, specify_grid=True, perf_test_mode=False):
    num_groups = 32  #  always 32 for SDXL Base 1024x1024
    N, C, H, W = input_shape
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    core_x = 8
    core_y = 8
    grid_size = ttnn.CoreGrid(y=core_y, x=core_x)

    # Generate torch tensor
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)

    if not perf_test_mode:
        # Execute torch group_norm
        torch_output_tensor = torch.nn.functional.group_norm(
            torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
        )
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Generate ttnn tensor
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    tt_input_tensor = ttnn.from_torch(
        tt_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Generate input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.x, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    input_negative_mask_tensor = ttnn.create_group_norm_input_negative_mask(
        C, num_groups, grid_size.x, ttnn.DataType.BFLOAT8_B
    )
    input_negative_mask_tensor = ttnn.to_device(input_negative_mask_tensor, device)

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.x)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.x)

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

    # Generate shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.y, C // grid_size.x
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input_tensor = ttnn.to_device(tt_input_tensor, device, memory_config=sharded_mem_config)

    # Execute ttnn group_norm
    tt_output_tensor = ttnn.group_norm(
        tt_input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        negative_mask=input_negative_mask_tensor,
        memory_config=sharded_mem_config,
        core_grid=grid_size if specify_grid else None,
        weight=gamma_t,
        bias=beta_t,
    )
    ttnn.synchronize_device(device)

    if not perf_test_mode:
        tt_output_tensor = ttnn.from_device(tt_output_tensor)
        tt_output_tensor = ttnn.to_torch(tt_output_tensor)

        pcc_threshold = 0.9999
        rtol = 0.065
        atol = 0.065
        frobenius_threshold = 0.016
        assert_numeric_metrics(
            torch_output_tensor,
            tt_output_tensor,
            pcc_threshold=pcc_threshold,
            rtol=rtol,
            atol=atol,
            frobenius_threshold=frobenius_threshold,
        )

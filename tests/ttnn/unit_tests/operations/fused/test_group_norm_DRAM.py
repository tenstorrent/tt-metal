# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.utility_functions import skip_for_wormhole_b0, skip_for_blackhole


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x",
    [
        # Original tests
        # (1, 256, 1024, 1024, 32, 128, 8, 4),
        # (1, 256, 512, 512, 32, 32, 8, 4),
        # (1, 512, 128, 128, 32, 4, 8, 4),
        # (1, 512, 256, 256, 32, 16, 8, 4),
        # (1, 512, 512, 512, 32, 32, 8, 4),
        # Current in model
        (1, 256, 1024, 1024, 32, 48, 8, 8),  # 62ms
        (1, 256, 512, 512, 32, 12, 8, 8),  # 15.7ms
        (1, 512, 128, 128, 32, 4, 8, 8),  # 1.57 ms
        (1, 512, 256, 256, 32, 4, 8, 8),  # 6.1ms
        (1, 512, 512, 512, 32, 12, 8, 8),  # 24ms
        (1, 128, 1024, 1024, 32, 96, 4, 4),  # 173ms (6 of these in total) can this be on more cores?
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

    ttnn.synchronize_device(device)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9996)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_splits",
    [
        # (1, 256, 1024, 1024, 32, 32) # does not fit -> input is [16384, 8] per core (~260kB) gets tilized internally to [16384, 32] which is ~1MB, and 2 buffers are of that size (cb_x and cb_in)
        # (1, 256, 512, 512, 32, 8) # Can fit in 8 slices, each slice does: (0,8ms for split, 0.3ms for interleavedToSharded + 0.68ms for GN) = 1.78ms, 8 slices x 1.78ms = 14.24ms + 4.6ms for concat = 18.84ms (original is 15.7ms)
        # (1, 512, 128, 128, 32, 1) # Can fit in 1 slice, i2s = 0.09ms GN = 1.5ms, s2i = 0.135ms = 1.725ms against 1.57ms of dram GN. Is block shardable as well, in that case it GN takes 0.35ms
        # (1, 512, 256, 256, 32, 4) # Can fit in 4 slice, split= 0.3ms i2s = 0.1ms GN = 0.6ms, s2i = 0.421ms = 5.6ms + 1ms for concat = 6.6ms (original is 6.1ms)
        # (
        #     1, 512, 512, 512, 32, 16,
        # )  # Can fit in 16 slice, split= 0.8ms i2s = 0.33ms GN = 0.38ms, s2i = 1.58ms = 49.44ms + 7.806ms for concat = 57.246ms (original is 24ms)
        # (1, 128, 1024, 1024, 32, 32) # does not fit -> input is [16384, 4] per core (~130kB) gets tilized internally to [16384, 32] which is ~1MB, and 2 buffers are of that size (cb_x and cb_in). in addition to that, RM stick of size 4 is not L1 aligned
    ],
)
def test_sdxl_base_group_norm_split(device, N, C, H, W, num_groups, num_splits):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    grid_size = ttnn.CoreGrid(y=8, x=8)

    # Generate torch tensor
    torch_input_tensor = torch.rand([N, C, H, W], dtype=torch.bfloat16)

    # Execute torch group_norm
    torch_output_tensor = torch.nn.functional.group_norm(torch_input_tensor, num_groups)
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Generate ttnn tensor
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    tt_input_tensor = ttnn.from_torch(
        tt_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    print("tt_input_tensor shape: ", tt_input_tensor.shape)

    # Generate input mask
    num_groups_per_split = num_groups // num_splits  # 16
    C_per_split = C // num_splits
    input_mask_tensor = ttnn.create_group_norm_input_mask(C_per_split, num_groups_per_split, 1)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_negative_mask_tensor = ttnn.create_group_norm_input_negative_mask(C_per_split, num_groups_per_split, 1)
    input_negative_mask_tensor = ttnn.from_torch(
        input_negative_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_input_tensor = ttnn.to_device(tt_input_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Generate shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // (grid_size.y * grid_size.x), C_per_split
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config_per_split = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    print("sharded_mem_config_per_split: ", sharded_mem_config_per_split)
    print("Shard shape: ", shard_shape)

    per_group_results = []
    for i in range(num_splits):
        print("Starting split: ", i)
        tt_input_tensor_per_group_split = tt_input_tensor[:, :, :, i * C_per_split : (i + 1) * C_per_split]
        print("tt_input_tensor_per_group_split shape: ", tt_input_tensor_per_group_split.shape)
        tt_input_tensor_per_group_split_sharded = ttnn.to_memory_config(
            tt_input_tensor_per_group_split, sharded_mem_config_per_split
        )
        tt_output_tensor_per_group_split = ttnn.group_norm(
            tt_input_tensor_per_group_split_sharded,
            num_groups=num_groups_per_split,
            input_mask=input_mask_tensor,
            memory_config=sharded_mem_config_per_split,
            core_grid=grid_size,
            inplace=False,
            negative_mask=input_negative_mask_tensor,
        )
        tt_output_tensor_per_group_split = ttnn.to_memory_config(
            tt_output_tensor_per_group_split, ttnn.DRAM_MEMORY_CONFIG
        )
        per_group_results.append(tt_output_tensor_per_group_split)

    tt_output_tensor = ttnn.concat(per_group_results, dim=-1)
    # tt_output_tensor = per_group_results[0]

    # tt_input_tensor = ttnn.to_device(tt_input_tensor, device, memory_config=sharded_mem_config)
    # Execute ttnn group_norm
    # tt_output_tensor = ttnn.group_norm(
    #     tt_input_tensor,
    #     num_groups=num_groups,
    #     input_mask=input_mask_tensor,
    #     memory_config=sharded_mem_config,
    #     core_grid=grid_size,
    # )

    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    print("Out tensor: ", tt_output_tensor)
    print("Torch out tensor: ", torch_output_tensor)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9997)


@pytest.mark.parametrize("input_shape, core_x, core_y", [((1, 4096, 640), 5, 8), ((1, 1024, 1280), 8, 8)])
def test_sharded_layer_norm(device, input_shape, core_x, core_y, reset_seeds):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.rand((input_shape[-1],), dtype=torch.bfloat16)
    torch_bias = torch.rand((input_shape[-1],), dtype=torch.bfloat16)

    H = input_shape[-2]
    W = input_shape[-1]

    ln_eps = 1e-5
    ln_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[input_shape[-1]], weight=torch_weight, bias=torch_bias, eps=ln_eps
    )

    mem_config = ttnn.create_sharded_memory_config(
        shape=(input_shape[-2] // core_y, input_shape[-1] // core_x),
        core_grid=ttnn.CoreGrid(y=core_y, x=core_x),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=True,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config
    )
    ttnn_weight = ttnn.from_torch(
        torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_bias = ttnn.from_torch(
        torch_bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    ln_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 8],
        subblock_w=1,
        block_h=H // core_y // 32,
        block_w=W // core_x // 32,
        inplace=True,
    )

    output_tensor = ttnn.layer_norm(
        ttnn_input_tensor,
        weight=ttnn_weight,
        bias=ttnn_bias,
        epsilon=ln_eps,
        compute_kernel_config=ln_compute_kernel_config,
        program_config=ln_config,
    )

    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)

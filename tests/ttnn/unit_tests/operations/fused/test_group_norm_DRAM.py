# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_blackhole


# Helper function to get welford parameters based on device type
def get_welford_params():
    """Return welford parameters - only legacy mode for Blackhole, both modes for other devices"""
    if is_blackhole():
        return ("legacy",)
    else:
        return ("legacy", "welford_normal", "welford_reciprocal")


welford_flavors = get_welford_params()


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
        #  SDXL VAE
        (1, 128, 1024, 1024, 32, 32, 8, 8),
        (1, 128, 512, 512, 32, 8, 8, 8),
        (1, 256, 1024, 1024, 32, 48, 8, 8),
        (1, 256, 515, 512, 32, 12, 8, 8),
        (1, 256, 256, 256, 32, 4, 8, 8),
        (1, 512, 512, 512, 32, 12, 8, 8),
        (1, 512, 256, 256, 32, 4, 8, 8),
        # SDXL Refiner
        (1, 1152, 128, 128, 32, 2, 8, 4),
        (1, 512, 64, 64, 32, 1, 8, 8),  # SD 1.4 VAE
        (1, 512, 128, 128, 32, 1, 8, 8),  # SD 1.4 VAE
        (1, 512, 256, 256, 32, 4, 8, 8),  # SD 1.4 VAE
        (1, 256, 256, 256, 32, 8, 8, 8),  # SD 1.4 VAE
        (1, 256, 512, 512, 32, 16, 8, 8),  # SD 1.4 VAE
        (1, 128, 512, 512, 32, 22, 4, 4),  # SD 1.4 VAE
        # sd35. 4 indicates the number of device.
        (1, 256 // 4, 256, 256, 32 // 4, 1, 8, 8),
        (1, 512 // 4, 128, 128, 32 // 4, 1, 8, 8),
        (1, 512 // 4, 256, 256, 32 // 4, 2, 8, 8),
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_splits",
    [
        # (1, 256, 1024, 1024, 32, 32), # does not fit -> input is [16384, 8] per core (~260kB) gets tilized internally to [16384, 32] which is ~1MB, and 2 buffers are of that size (cb_x and cb_in)
        (
            1,
            256,
            512,
            512,
            32,
            8,
        ),  # Can fit in 8 slices, each slice does: (0,8ms for split, 0.3ms for interleavedToSharded + 0.68ms for GN) = 1.78ms, 8 slices x 1.78ms = 14.24ms + 4.6ms for concat = 18.84ms (original is 15.7ms)
        (
            1,
            512,
            128,
            128,
            32,
            1,
        ),  # Can fit in 1 slice, i2s = 0.09ms GN = 1.5ms, s2i = 0.135ms = 1.725ms against 1.57ms of dram GN. Is block shardable as well, in that case it GN takes 0.35ms
        (
            1,
            512,
            256,
            256,
            32,
            4,
        ),  # Can fit in 4 slice, split= 0.3ms i2s = 0.1ms GN = 0.6ms, s2i = 0.421ms = 5.6ms + 1ms for concat = 6.6ms (original is 6.1ms)
        (
            1,
            512,
            512,
            512,
            32,
            16,
        ),  # Can fit in 16 slice, split= 0.8ms i2s = 0.33ms GN = 0.38ms, s2i = 1.58ms = 49.44ms + 7.806ms for concat = 57.246ms (original is 24ms)
        # (1, 128, 1024, 1024, 32, 32), # does not fit -> input is [16384, 4] per core (~130kB) gets tilized internally to [16384, 32] which is ~1MB, and 2 buffers are of that size (cb_x and cb_in). in addition to that, RM stick of size 4 is not L1 aligned
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
    per_group_results = []
    for i in range(num_splits):
        tt_input_tensor_per_group_split = tt_input_tensor[:, :, :, i * C_per_split : (i + 1) * C_per_split]
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

    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9997)

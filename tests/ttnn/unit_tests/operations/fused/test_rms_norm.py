# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("h", [32, 384])
@pytest.mark.parametrize("w", [64, 1024])
def test_rms_norm(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.rms_norm(input_tensor, weight=weight)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [32])
def test_rms_norm_row_major(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.rms_norm(input_tensor, weight=weight)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_large_rms_norm(device, batch_size, h, w, dtype):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=dtype)
    torch_residual_input_tensor = torch.rand((batch_size, h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor + torch_residual_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.rms_norm(input_tensor, residual_input_tensor=residual_input_tensor, weight=weight)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2048])
@pytest.mark.parametrize("epsilon", [9.99999974e-6])
def test_rms_norm_sharded_repro(device, h, w, epsilon):
    """
    REPRO: Crashes with division by zero in layernorm_op_multi_core_sharded.cpp:208
    because sharded RMS norm without program_config leaves block_wt/subblock_wt as 0.

    Test RMS norm with width-sharded input on L1 and interleaved weight on DRAM.
    Matches MLIR layout:
      Input: tensor<32x2048xbf16> width_sharded on L1, shard shape 32x32, 64 cores (8x8 grid)
      Weight: tensor<2048xbf16> interleaved on DRAM
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)

    # Golden function computation
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight, epsilon=epsilon)

    # Create width-sharded memory config for input on L1
    # 64 cores (8x8 grid), each shard is 32x32 (1 tile)
    # Wormhole has 8x8 compute grid, not 1x64
    num_cores_x = 8
    num_cores_y = 8
    num_cores_total = num_cores_x * num_cores_y  # 64
    shard_height = h  # 32
    shard_width = w // num_cores_total  # 2048 / 64 = 32

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1),
                )
            }
        ),
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Input tensor: width-sharded on L1
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
    )

    # Weight tensor: interleaved on DRAM
    weight = ttnn.from_torch(
        torch_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # This crashes without program_config!
    output_tensor = ttnn.rms_norm(input_tensor, weight=weight, epsilon=epsilon)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2048])
@pytest.mark.parametrize("epsilon", [9.99999974e-6])
def test_rms_norm_sharded(device, h, w, epsilon):
    """
    Test RMS norm with width-sharded input on L1 and interleaved weight on DRAM.
    Matches MLIR layout:
      Input: tensor<32x2048xbf16> width_sharded on L1, shard shape 32x32, 64 cores (8x8 grid)
      Weight: tensor<2048xbf16> interleaved on DRAM

    Uses LayerNormShardedMultiCoreProgramConfig to avoid division by zero crash.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)

    # Golden function computation
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight, epsilon=epsilon)

    # Create width-sharded memory config for input on L1
    # 64 cores (8x8 grid), each shard is 32x32 (1 tile)
    num_cores_x = 8
    num_cores_y = 8
    num_cores_total = num_cores_x * num_cores_y  # 64
    shard_height = h  # 32
    shard_width = w // num_cores_total  # 2048 / 64 = 32

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1),
                )
            }
        ),
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Input tensor: width-sharded on L1
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
    )

    # Weight tensor: interleaved on DRAM
    weight = ttnn.from_torch(
        torch_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Program config for sharded RMS norm
    # block_h = 1 tile (32 rows), block_w = 1 tile (32 cols), subblock_w = 1
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(num_cores_x, num_cores_y),
        block_h=shard_height // 32,  # 1 tile
        block_w=shard_width // 32,  # 1 tile
        subblock_w=1,
        use_welford=False,
        inplace=False,
    )

    output_tensor = ttnn.rms_norm(
        input_tensor,
        weight=weight,
        epsilon=epsilon,
        memory_config=sharded_mem_config,
        program_config=program_config,
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "first_dtype, second_dtype",
    [
        (ttnn.bfloat8_b, ttnn.bfloat16),
        (ttnn.bfloat8_b, ttnn.float32),
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat8_b),
        (ttnn.float32, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize("input_in_l1", [True, False])
@pytest.mark.parametrize("keep_l1_aligned", [True, False])
def test_interleaved_to_sharded_hash(device, first_dtype, second_dtype, input_in_l1, keep_l1_aligned):
    # Sample tensor size and shard config
    input_tensor_shape = (1, 1, 512, 512)
    input_memory_config = ttnn.L1_MEMORY_CONFIG if input_in_l1 else ttnn.DRAM_MEMORY_CONFIG

    # L1 grid and calculations
    l1_core_grid_x = 4
    l1_core_grid_y = 4
    l1_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(l1_core_grid_x - 1, l1_core_grid_y - 1))}
    )
    l1_h, l1_w = input_tensor_shape[2] // l1_core_grid_x, input_tensor_shape[3] // l1_core_grid_y

    # Shard spec and mem config
    shard_spec = ttnn.ShardSpec(l1_shard_grid, [l1_h, l1_w], ttnn.ShardOrientation.ROW_MAJOR)

    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    # Create input tensors and send to device
    input_tensor_torch = torch.randn(input_tensor_shape)
    input_tensor = ttnn.from_torch(input_tensor_torch, first_dtype, layout=ttnn.TILE_LAYOUT)
    input_tensor_device = ttnn.allocate_tensor_on_device(
        input_tensor.shape, input_tensor.dtype, input_tensor.layout, device, input_memory_config
    )
    ttnn.copy_host_to_device_tensor(input_tensor, input_tensor_device)

    # Do interleaved to sharded op on device several times to load the program from cache
    for iter in range(0, 5):
        output_tensor = ttnn.interleaved_to_sharded(
            input_tensor_device, sharded_mem_config, first_dtype, keep_l1_aligned=keep_l1_aligned
        )
        pcc_passed_a, pcc_message_a = assert_with_pcc(input_tensor_torch, ttnn.to_torch(output_tensor), pcc=0.9999)

        output_tensor = ttnn.interleaved_to_sharded(
            input_tensor_device, sharded_mem_config, second_dtype, keep_l1_aligned=keep_l1_aligned
        )
        pcc_passed_b, pcc_message_b = assert_with_pcc(input_tensor_torch, ttnn.to_torch(output_tensor), pcc=0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, shard_grid",
    [
        [
            [2, 2, 128, 64],
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        [
            [1, 1, 416, 64],
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_interleaved_to_dram_height_sharded(
    device, dtype, layout, tensor_shape, shard_shape, shard_grid, shard_orientation
):
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b not supported for i2s row-major")

    # Output memory config
    output_shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, output_shard_spec
    )

    # Test
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)
    ttnn_output_tensor = ttnn.interleaved_to_sharded(ttnn_input_tensor, output_mem_config)

    assert_with_pcc(torch_input_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, shard_grid",
    [
        [
            [2, 1, 32, 512],
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        [
            [1, 1, 64, 416],
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_interleaved_to_dram_width_sharded(
    device, dtype, layout, tensor_shape, shard_shape, shard_grid, shard_orientation
):
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b not supported for i2s row-major")

    # Output memory config
    output_shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, output_shard_spec
    )

    # Test
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)
    ttnn_output_tensor = ttnn.interleaved_to_sharded(ttnn_input_tensor, output_mem_config)

    assert_with_pcc(torch_input_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        [ttnn.bfloat8_b, ttnn.float32],
        [ttnn.float32, ttnn.bfloat8_b],
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "tensor_shape, shard_type, shard_shape, shard_grid",
    [
        [
            [1, 1, 416, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        [
            [1, 1, 64, 416],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_interleaved_to_dram_sharded_convert_dtype(
    device, in_dtype, out_dtype, layout, tensor_shape, shard_type, shard_shape, shard_grid, shard_orientation
):
    # Output memory config
    output_shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(shard_type, ttnn.BufferType.DRAM, output_shard_spec)

    # Test
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=in_dtype, layout=layout)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)
    ttnn_output_tensor = ttnn.interleaved_to_sharded(ttnn_input_tensor, output_mem_config, out_dtype)

    assert_with_pcc(torch_input_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "tensor_shape, shard_type, shard_shape, shard_grid",
    [
        [
            [1, 1, 416, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        [
            [1, 1, 64, 416],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_interleaved_to_dram_sharded_via_to_memory_layout(
    device, dtype, layout, input_mem_config, tensor_shape, shard_type, shard_shape, shard_grid, shard_orientation
):
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b not supported for i2s row-major")

    # Output memory config
    output_shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(shard_type, ttnn.BufferType.DRAM, output_shard_spec)

    # Test
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=dtype, layout=layout, device=device, memory_config=input_mem_config
    )
    ttnn_output_tensor = ttnn.to_memory_config(ttnn_input_tensor, output_mem_config)

    assert_with_pcc(torch_input_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)

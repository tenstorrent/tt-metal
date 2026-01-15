# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
import ttnn
import math
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores

TILE_WIDTH = 32
L1_ALIGNMENT = 16


def run_upsample_nearest_float_sharded_max_grid(
    device, input_shape, scale_factor_h, scale_factor_w, shard_strategy, dtype=torch.bfloat16, debug=False
):
    """
    Creates shards that maximize grid usage without requiring even division.
    Shard shape is calculated to fit the entire tensor while respecting alignment.
    """
    torch.manual_seed(42)

    batch, height, width, channels = input_shape
    input_nhwc = torch.randn(input_shape, dtype=dtype)

    # Calculate expected output shape
    output_height = int(math.floor(height * scale_factor_h))
    output_width = int(math.floor(width * scale_factor_w))

    # PyTorch reference (uses NCHW)
    input_nchw = input_nhwc.permute(0, 3, 1, 2)
    torch_result_nchw = F.interpolate(input_nchw, scale_factor=(scale_factor_h, scale_factor_w), mode="nearest")
    torch_result = torch_result_nchw.permute(0, 2, 3, 1)

    ttnn_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
    num_bytes = 2  # bfloat16

    # Calculate grid usage
    max_grid_size = (device.compute_with_storage_grid_size().y, device.compute_with_storage_grid_size().x)
    total_nhw = batch * height * width
    min_shard_width = L1_ALIGNMENT // num_bytes  # 8 for bfloat16

    if shard_strategy == ttnn.ShardStrategy.HEIGHT:
        # Height sharding
        max_num_shards = min(total_nhw, max_grid_size[0] * max_grid_size[1])
        ncores = max_num_shards
        shard_height = math.ceil(total_nhw / max_num_shards)
        # Round up channels to L1 alignment
        shard_width = math.ceil(channels / min_shard_width) * min_shard_width
        tensor_memory_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    elif shard_strategy == ttnn.ShardStrategy.WIDTH:
        # Width sharding
        max_num_shards = min(channels // min_shard_width, max_grid_size[0] * max_grid_size[1])
        if max_num_shards == 0:
            max_num_shards = 1
        shard_width = math.ceil(channels / max_num_shards)
        # Round up to alignment
        shard_width = math.ceil(shard_width / min_shard_width) * min_shard_width
        ncores = math.ceil(channels / shard_width)
        shard_height = total_nhw
        tensor_memory_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    elif shard_strategy == ttnn.ShardStrategy.BLOCK:
        # Block sharding
        max_num_shards_height = min(total_nhw, max_grid_size[0])
        max_num_shards_width = min(channels // min_shard_width, max_grid_size[1])
        if max_num_shards_width == 0:
            max_num_shards_width = 1

        shard_height = math.ceil(total_nhw / max_num_shards_height)
        shard_width = math.ceil(channels / max_num_shards_width)
        # Round up to alignment
        shard_width = math.ceil(shard_width / min_shard_width) * min_shard_width

        ncores_h = math.ceil(total_nhw / shard_height)
        ncores_w = math.ceil(channels / shard_width)
        ncores = (ncores_h, ncores_w)
        tensor_memory_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    else:
        raise ValueError(f"Unsupported shard strategy: {shard_strategy}")

    shard_grid = get_shard_grid_from_num_cores(ncores, device)
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    shard_shape = (shard_height, shard_width)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.BufferType.L1, shard_spec)

    if debug:
        logger.info(f"Input shape: {input_shape}, scale_h={scale_factor_h}, scale_w={scale_factor_w}")
        logger.info(f"Shard strategy: {shard_strategy}")
        logger.info(f"Shard shape: {shard_shape}, ncores: {ncores}")
        logger.info(f"Max grid: {max_grid_size}, Total NHW: {total_nhw}, Channels: {channels}")
        logger.info(f"Memory config: {sharded_mem_config}")

    # Create input tensor and convert to sharded layout
    input_tensor = ttnn.from_torch(
        input_nhwc,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)

    # Run upsample operation (operation handles output memory config)
    output_tensor = ttnn.upsample(input_tensor, [scale_factor_h, scale_factor_w], mode="nearest")
    output_torch = ttnn.to_torch(output_tensor)

    # Verify shape
    assert list(output_torch.shape) == list(
        torch_result.shape
    ), f"Shape mismatch: expected {list(torch_result.shape)}, got {list(output_torch.shape)}"

    if debug:
        logger.info("=" * 80)
        logger.info(f"Torch golden (batch 0, channel 0, full H×W):\n{torch_result[0, :, :, 0]}")
        logger.info(f"TTNN output (batch 0, channel 0, full H×W):\n{output_torch[0, :, :, 0]}")
        logger.info("=" * 80)

    # Check correctness
    is_equal = torch.equal(output_torch, torch_result)
    if not is_equal:
        max_diff = (output_torch - torch_result).abs().max().item()
        num_diffs = (output_torch != torch_result).sum().item()
        total_elements = torch_result.numel()
        logger.warning(
            f"Not exactly equal: max_diff={max_diff}, "
            f"num_diffs={num_diffs}/{total_elements} ({100*num_diffs/total_elements:.2f}%)"
        )
        pcc_passed, pcc_message = assert_with_pcc(torch_result, output_torch, pcc=0.9999)
        logger.info(pcc_message)
        assert pcc_passed, f"PCC check failed: {pcc_message}"
    else:
        logger.info("Results are exactly equal")

    return is_equal


def run_upsample_nearest_float_sharded(
    device, input_shape, scale_factor_h, scale_factor_w, shard_strategy, dtype=torch.bfloat16, debug=False
):
    torch.manual_seed(42)

    batch, height, width, channels = input_shape
    input_nhwc = torch.randn(input_shape, dtype=dtype)

    # Create sequential input data (1, 2, 3, ...) across H and W
    flat_size = height * width
    seq_hw = torch.arange(1, flat_size + 1, dtype=dtype).reshape(height, width)
    # Replicate across batch and channels
    input_nhwc = seq_hw.unsqueeze(0).unsqueeze(-1).repeat(batch, 1, 1, channels)

    # Calculate expected output shape
    output_height = int(math.floor(height * scale_factor_h))
    output_width = int(math.floor(width * scale_factor_w))

    # PyTorch reference (uses NCHW)
    input_nchw = input_nhwc.permute(0, 3, 1, 2)
    torch_result_nchw = F.interpolate(input_nchw, scale_factor=(scale_factor_h, scale_factor_w), mode="nearest")
    torch_result = torch_result_nchw.permute(0, 2, 3, 1)

    ttnn_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
    num_bytes = 2  # bfloat16

    # Calculate sharding configuration
    max_grid_size = (device.compute_with_storage_grid_size().y, device.compute_with_storage_grid_size().x)
    min_shard_width = L1_ALIGNMENT // num_bytes  # 8 for bfloat16

    if shard_strategy == ttnn.ShardStrategy.HEIGHT:
        # Height sharding: shard along N*H*W dimension
        max_nshards = min(batch * height * width, max_grid_size[0] * max_grid_size[1])
        nshards = max_nshards
        while nshards > 0:
            if (batch * height * width) % nshards == 0:
                break
            nshards -= 1
        ncores = nshards
        shard_height = math.ceil(batch * height * width / ncores)
        # Round up channels to L1 alignment
        shard_width = math.ceil(channels / min_shard_width) * min_shard_width
        tensor_memory_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    elif shard_strategy == ttnn.ShardStrategy.WIDTH:
        # Width sharding: shard along C dimension

        max_nshards = min(channels // min_shard_width, max_grid_size[0] * max_grid_size[1])
        nshards = max_nshards
        while nshards > 0:
            shard_width_candidate = channels // nshards
            if channels % nshards == 0 and (shard_width_candidate * num_bytes) % L1_ALIGNMENT == 0:
                break
            nshards -= 1
        if nshards == 0:
            pytest.skip("Cannot find valid width sharding configuration with L1 alignment")
        ncores = nshards
        shard_height = batch * height * width
        shard_width = channels // ncores
        tensor_memory_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    elif shard_strategy == ttnn.ShardStrategy.BLOCK:
        # Block sharding: shard along both N*H*W and C dimensions
        max_nshards_h = min(batch * height * width, max_grid_size[0])
        max_nshards_w = min(channels // min_shard_width, max_grid_size[1])

        # Find nshards_h along N*H*W
        nshards_h = max_nshards_h
        while nshards_h > 0:
            if (batch * height * width) % nshards_h == 0:
                break
            nshards_h -= 1

        # Find nshards_w along C with L1 alignment
        nshards_w = max_nshards_w
        while nshards_w > 0:
            shard_width_candidate = channels // nshards_w
            if channels % nshards_w == 0 and (shard_width_candidate * num_bytes) % L1_ALIGNMENT == 0:
                break
            nshards_w -= 1

        if nshards_w == 0 or nshards_h == 0:
            pytest.skip("Cannot find valid block sharding configuration with L1 alignment")

        ncores = (nshards_h, nshards_w)
        shard_height = batch * height * width // ncores[0]
        shard_width = channels // ncores[1]
        tensor_memory_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    else:
        raise ValueError(f"Unsupported shard strategy: {shard_strategy}")

    shard_grid = get_shard_grid_from_num_cores(ncores, device)
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    shard_shape = (shard_height, shard_width)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.BufferType.L1, shard_spec)

    if debug:
        logger.info(f"Input shape: {input_shape}, scale_h={scale_factor_h}, scale_w={scale_factor_w}")
        logger.info(f"Shard strategy: {shard_strategy}")
        logger.info(f"Shard shape: {shard_shape}, ncores: {ncores}")
        logger.info(f"Memory config: {sharded_mem_config}")

    # Create input tensor and convert to sharded layout
    input_tensor = ttnn.from_torch(
        input_nhwc,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)

    # Run upsample operation (operation handles output memory config)
    output_tensor = ttnn.upsample(input_tensor, [scale_factor_h, scale_factor_w], mode="nearest")
    output_torch = ttnn.to_torch(output_tensor)

    # Verify shape
    assert list(output_torch.shape) == list(
        torch_result.shape
    ), f"Shape mismatch: expected {list(torch_result.shape)}, got {list(output_torch.shape)}"

    if debug:
        # Temporarily disable torch print options to show full tensors
        torch.set_printoptions(threshold=100000, linewidth=200, edgeitems=100000)
        if True:
            # logger.info("=" * 80)
            # logger.info(f"Torch golden (batch 0, channel 0, full H×W):\n{torch_result[0, :, :, 0]}")
            # logger.info(f"TTNN output (batch 0, channel 0, full H×W):\n{output_torch[0, :, :, 0]}")
            # logger.info("=" * 80)

            print("Torch golden (batch 0, channel 0, full H×W):")
            print(torch_result[0, :, :, 0])
            print("TTNN output (batch 0, channel 0, full H×W):")
            print(output_torch[0, :, :, 0])
    # if debug:
    #     logger.info("=" * 80)
    #     logger.info(f"Torch golden (batch 0, channel 0, full H×W):\n{torch_result[0, :, :, 0]}")
    #     logger.info(f"TTNN output (batch 0, channel 0, full H×W):\n{output_torch[0, :, :, 0]}")
    #     logger.info("=" * 80)

    # Check correctness
    is_equal = torch.equal(output_torch, torch_result)
    if not is_equal:
        max_diff = (output_torch - torch_result).abs().max().item()
        num_diffs = (output_torch != torch_result).sum().item()
        total_elements = torch_result.numel()
        logger.warning(
            f"Not exactly equal: max_diff={max_diff}, "
            f"num_diffs={num_diffs}/{total_elements} ({100*num_diffs/total_elements:.2f}%)"
        )
        pcc_passed, pcc_message = assert_with_pcc(torch_result, output_torch, pcc=0.9999)
        logger.info(pcc_message)
        assert pcc_passed, f"PCC check failed: {pcc_message}"
    else:
        logger.info("Results are exactly equal")

    return is_equal


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        # Basic upscaling
        # ((1, 8, 8, 67), 2.0, 2.0),
        # ((1, 16, 16, 128), 2.0, 2.0),
        # ((2, 8, 8, 64), 2.0, 2.0),
        # ((1, 4, 4, 32), 3.0, 3.0),
        # # Fractional upscaling
        # ((1, 8, 8, 64), 1.5, 1.5),
        # ((1, 16, 16, 128), 1.5, 1.5),
        # ((1, 4, 4, 32), 2.5, 2.5),
        # # Downscaling
        # ((1, 16, 16, 64), 0.5, 0.5),
        # ((1, 32, 32, 128), 0.5, 0.5),
        # ((1, 16, 16, 64), 0.75, 0.75),
        # # Asymmetric scaling
        # ((1, 8, 16, 64), 2.0, 1.5),
        # ((1, 16, 8, 128), 1.5, 2.0),
        # # Mixed upscale/downscale
        # ((1, 8, 16, 64), 2.0, 0.5),
        # ((1, 16, 8, 128), 0.5, 2.0),
        # # Large spatial dimensions
        # ((1, 32, 32, 64), 2.0, 2.0),
        # ((1, 56, 56, 128), 2.0, 2.0),
        # # Various channel counts
        # ((1, 8, 8, 32), 2.0, 2.0),
        # ((1, 8, 8, 256), 2.0, 2.0),
        # ((1, 8, 8, 512), 2.0, 2.0),
    ],
)
def test_upsample_nearest_float_height_sharded(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_sharded(device, input_shape, scale_factor_h, scale_factor_w, ttnn.ShardStrategy.HEIGHT)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        # # Basic upscaling
        # ((1, 8, 8, 64), 2.0, 2.0),
        # ((1, 16, 16, 128), 2.0, 2.0),
        # ((2, 8, 8, 64), 2.0, 2.0),
        # ((1, 4, 4, 32), 3.0, 3.0),
        # # # Fractional upscaling
        # ((1, 8, 8, 64), 1.5, 1.5),
        # ((1, 16, 16, 128), 1.5, 1.5),
        # ((1, 4, 4, 32), 2.5, 2.5),
        # # # Downscaling
        # ((1, 16, 16, 64), 0.5, 0.5),
        # ((2, 32, 32, 128), 0.5, 0.5),
        # ((2, 16, 16, 64), 0.75, 0.75),
        # # # Asymmetric scaling
        # ((1, 8, 16, 64), 2.0, 1.5),
        # ((1, 16, 8, 128), 1.5, 2.0),
        # # # Mixed upscale/downscale
        # ((1, 8, 16, 64), 2.0, 0.5),
        # ((1, 16, 8, 128), 0.5, 2.0),
        # # # Large spatial dimensions
        # ((1, 32, 32, 64), 2.0, 2.0),
        # ((1, 56, 56, 128), 2.0, 2.0),
        # # Various channel counts
        # ((1, 16, 16, 32), 2.0, 2.0),
    ],
)
def test_upsample_nearest_float_width_sharded(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_sharded(
        device, input_shape, scale_factor_h, scale_factor_w, ttnn.ShardStrategy.WIDTH, debug=False
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        # Basic upscaling
        ((1, 8, 8, 64), 2.0, 2.0),
        ((1, 16, 16, 128), 2.0, 2.0),
        ((2, 8, 8, 64), 2.0, 2.0),
        ((1, 4, 4, 32), 3.0, 3.0),
        # Fractional upscaling
        ((1, 8, 8, 64), 1.5, 1.5),
        ((1, 16, 16, 128), 1.5, 1.5),
        ((1, 4, 4, 32), 2.5, 2.5),
        # Downscaling
        ((1, 16, 16, 64), 0.5, 0.5),
        ((1, 32, 32, 128), 0.5, 0.5),
        ((1, 16, 16, 64), 0.75, 0.75),
        # Asymmetric scaling
        ((1, 8, 16, 64), 2.0, 1.5),
        ((1, 16, 8, 128), 1.5, 2.0),
        # Mixed upscale/downscale
        ((1, 8, 16, 64), 2.0, 0.5),
        ((1, 16, 8, 128), 0.5, 2.0),
        # Large spatial dimensions
        ((1, 32, 32, 64), 2.0, 2.0),
        ((1, 56, 56, 128), 2.0, 2.0),
        # Various channel counts
        ((1, 8, 8, 32), 2.0, 2.0),
        ((1, 8, 8, 256), 2.0, 2.0),
        ((1, 8, 8, 512), 2.0, 2.0),
        ((1, 24, 24, 512), 1.9, 1.9),
        ((1, 32, 32, 512), 2.2, 2.2),
        ((1, 20, 20, 256), 2.5, 2.5),
        ((1, 40, 40, 256), 2.5, 2.5),
        ((1, 16, 16, 1024), 2.0, 2.0),
        ((1, 32, 32, 512), 0.67, 0.67),
    ],
)
def test_upsample_nearest_float_block_sharded(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_sharded(device, input_shape, scale_factor_h, scale_factor_w, ttnn.ShardStrategy.BLOCK)


# # Tests with maximized grid usage (uneven sharding)
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
# @pytest.mark.parametrize(
#     "input_shape, scale_factor_h, scale_factor_w",
#     [
#         # Odd dimensions that don't divide evenly
#         ((1, 7, 7, 63), 2.0, 2.0),
#         ((1, 13, 13, 127), 2.0, 2.0),
#         ((1, 11, 11, 77), 1.5, 1.5),
#         ((2, 9, 9, 99), 2.0, 2.0),
#         # Prime number dimensions
#         ((1, 17, 17, 97), 2.0, 2.0),
#         ((1, 19, 19, 113), 1.5, 1.5),
#         # Large tensors with odd dimensions
#         ((1, 33, 33, 511), 2.0, 2.0),
#         ((1, 45, 45, 333), 2.0, 2.0),
#         # Downscaling with odd results
#         ((1, 37, 37, 255), 0.5, 0.5),
#         ((1, 51, 51, 129), 0.67, 0.67),
#     ],
# )
# def test_upsample_nearest_float_height_sharded_max_grid(device, input_shape, scale_factor_h, scale_factor_w):
#     run_upsample_nearest_float_sharded_max_grid(
#         device, input_shape, scale_factor_h, scale_factor_w, ttnn.ShardStrategy.HEIGHT
#     )


# @pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
# @pytest.mark.parametrize(
#     "input_shape, scale_factor_h, scale_factor_w",
#     [
#         # Odd channel counts with L1 alignment
#         ((1, 8, 8, 72), 2.0, 2.0),  # 72 channels
#         ((2, 8, 8, 88), 2.0, 2.0),  # 88 channels
#         ((3, 16, 16, 136), 2.0, 2.0),  # 136 channels
#         ((4, 8, 8, 200), 1.5, 1.5),  # 200 channels
#         # Larger odd channel counts
#         ((1, 16, 16, 312), 2.0, 2.0),
#         ((2, 24, 24, 456), 2.0, 2.0),
#         # Downscaling with odd channels
#         ((1, 32, 32, 184), 0.5, 0.5),
#         ((2, 24, 24, 248), 0.75, 0.75),
#     ],
# )
# def test_upsample_nearest_float_width_sharded_max_grid(device, input_shape, scale_factor_h, scale_factor_w):
#     run_upsample_nearest_float_sharded_max_grid(
#         device, input_shape, scale_factor_h, scale_factor_w, ttnn.ShardStrategy.WIDTH
#     )


# @pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
# @pytest.mark.parametrize(
#     "input_shape, scale_factor_h, scale_factor_w",
#     [
#         # Both dimensions odd
#         ((1, 13, 13, 88), 2.0, 2.0),
#         ((1, 17, 17, 120), 2.0, 2.0),
#         ((1, 21, 21, 152), 1.5, 1.5),
#         # Larger odd dimensions
#         ((1, 29, 29, 248), 2.0, 2.0),
#         ((1, 37, 37, 312), 2.0, 2.0),
#         # Mixed scales
#         ((1, 19, 19, 184), 2.0, 1.5),
#         ((3, 23, 23, 200), 1.5, 2.0),
#         # Downscaling
#         ((2, 41, 41, 264), 0.5, 0.5),
#         ((2, 33, 33, 216), 0.67, 0.67),
#     ],
# )
# def test_upsample_nearest_float_block_sharded_max_grid(device, input_shape, scale_factor_h, scale_factor_w):
#     run_upsample_nearest_float_sharded_max_grid(
#         device, input_shape, scale_factor_h, scale_factor_w, ttnn.ShardStrategy.BLOCK
#     )

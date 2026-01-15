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

L1_ALIGNMENT = 16


def run_upsample_nearest_float_ND_sharded(
    device, input_shape, scale_factor_h, scale_factor_w, shard_dims, dtype=torch.bfloat16, debug=False
):
    """
    Tests upsample with ND sharding across specified dimensions.

    Args:
        shard_dims: tuple of dimensions to shard across, e.g., (3,) for C only, (1,2,3) for H,W,C
                    Dimensions: 0=N, 1=H, 2=W, 3=C
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
    min_shard_width = L1_ALIGNMENT // num_bytes  # 8 for bfloat16

    # Get device grid
    max_grid_size = (device.compute_with_storage_grid_size().y, device.compute_with_storage_grid_size().x)
    max_cores = max_grid_size[0] * max_grid_size[1]

    # Map dimension indices to sizes
    dim_sizes = [batch, height, width, channels]

    # Calculate shard shape
    # For ND sharding, we need to determine how to split each dimension
    shard_shape = list(input_shape)
    num_shards_per_dim = [1, 1, 1, 1]

    # Calculate total shards needed across sharded dimensions
    sharded_elements = 1
    for dim in shard_dims:
        sharded_elements *= dim_sizes[dim]

    # Distribute shards across the specified dimensions
    if len(shard_dims) == 1:
        # Single dimension sharding
        dim = shard_dims[0]
        if dim == 3:  # Channel dimension - must respect alignment
            max_possible_shards = min(channels // min_shard_width, max_cores)
            if max_possible_shards == 0:
                max_possible_shards = 1
            num_shards_per_dim[dim] = max_possible_shards
            shard_shape[dim] = math.ceil(channels / max_possible_shards)
            # Round up to alignment
            shard_shape[dim] = math.ceil(shard_shape[dim] / min_shard_width) * min_shard_width
        else:
            # Non-channel dimensions
            max_possible_shards = min(dim_sizes[dim], max_cores)
            num_shards_per_dim[dim] = max_possible_shards
            shard_shape[dim] = math.ceil(dim_sizes[dim] / max_possible_shards)

    elif len(shard_dims) == 2:
        # Two-dimensional sharding
        dim0, dim1 = shard_dims

        # Allocate shards across both dimensions
        if 3 in shard_dims:  # One of them is channels
            ch_dim = 3
            other_dim = dim0 if dim1 == 3 else dim1

            # For channels, respect alignment
            max_ch_shards = min(channels // min_shard_width, max_grid_size[1])
            if max_ch_shards == 0:
                max_ch_shards = 1
            max_other_shards = min(dim_sizes[other_dim], max_grid_size[0])

            num_shards_per_dim[ch_dim] = max_ch_shards
            num_shards_per_dim[other_dim] = max_other_shards

            shard_shape[ch_dim] = math.ceil(channels / max_ch_shards)
            shard_shape[ch_dim] = math.ceil(shard_shape[ch_dim] / min_shard_width) * min_shard_width
            shard_shape[other_dim] = math.ceil(dim_sizes[other_dim] / max_other_shards)
        else:
            # Neither is channels - distribute evenly
            max_shards_0 = min(dim_sizes[dim0], max_grid_size[0])
            max_shards_1 = min(dim_sizes[dim1], max_grid_size[1])

            num_shards_per_dim[dim0] = max_shards_0
            num_shards_per_dim[dim1] = max_shards_1

            shard_shape[dim0] = math.ceil(dim_sizes[dim0] / max_shards_0)
            shard_shape[dim1] = math.ceil(dim_sizes[dim1] / max_shards_1)

    elif len(shard_dims) == 3:
        # Three-dimensional sharding
        # Distribute across 3D grid (use grid_y for one dim, grid_x for another, and split third)
        if 3 in shard_dims:  # Channels is one of them
            ch_dim = 3
            other_dims = [d for d in shard_dims if d != 3]

            max_ch_shards = min(channels // min_shard_width, max_grid_size[1])
            if max_ch_shards == 0:
                max_ch_shards = 1

            # Combine other two dimensions
            combined_size = dim_sizes[other_dims[0]] * dim_sizes[other_dims[1]]
            max_combined_shards = min(combined_size, max_grid_size[0])

            num_shards_per_dim[ch_dim] = max_ch_shards
            # Distribute combined shards across the two dimensions
            num_shards_per_dim[other_dims[0]] = min(dim_sizes[other_dims[0]], max_combined_shards)
            num_shards_per_dim[other_dims[1]] = math.ceil(max_combined_shards / num_shards_per_dim[other_dims[0]])
            num_shards_per_dim[other_dims[1]] = min(num_shards_per_dim[other_dims[1]], dim_sizes[other_dims[1]])

            shard_shape[ch_dim] = math.ceil(channels / max_ch_shards)
            shard_shape[ch_dim] = math.ceil(shard_shape[ch_dim] / min_shard_width) * min_shard_width
            shard_shape[other_dims[0]] = math.ceil(dim_sizes[other_dims[0]] / num_shards_per_dim[other_dims[0]])
            shard_shape[other_dims[1]] = math.ceil(dim_sizes[other_dims[1]] / num_shards_per_dim[other_dims[1]])
        else:
            # No channels - combine all three spatially
            combined_size = dim_sizes[shard_dims[0]] * dim_sizes[shard_dims[1]] * dim_sizes[shard_dims[2]]
            max_combined_shards = min(combined_size, max_cores)

            # Simple distribution
            num_shards_per_dim[shard_dims[0]] = min(dim_sizes[shard_dims[0]], max_combined_shards)
            remaining = math.ceil(max_combined_shards / num_shards_per_dim[shard_dims[0]])
            num_shards_per_dim[shard_dims[1]] = min(dim_sizes[shard_dims[1]], remaining)

            for dim in shard_dims:
                shard_shape[dim] = math.ceil(dim_sizes[dim] / num_shards_per_dim[dim])

    elif len(shard_dims) == 4:
        # All four dimensions sharded
        # Combine N*H*W and distribute with C
        combined_nhw = batch * height * width
        max_nhw_shards = min(combined_nhw, max_grid_size[0])
        max_ch_shards = min(channels // min_shard_width, max_grid_size[1])
        if max_ch_shards == 0:
            max_ch_shards = 1

        # Distribute NHW shards
        num_shards_per_dim[0] = min(batch, max_nhw_shards)
        remaining = math.ceil(max_nhw_shards / num_shards_per_dim[0])
        num_shards_per_dim[1] = min(height, remaining)
        remaining = math.ceil(remaining / num_shards_per_dim[1])
        num_shards_per_dim[2] = min(width, remaining)
        num_shards_per_dim[3] = max_ch_shards

        shard_shape[0] = math.ceil(batch / num_shards_per_dim[0])
        shard_shape[1] = math.ceil(height / num_shards_per_dim[1])
        shard_shape[2] = math.ceil(width / num_shards_per_dim[2])
        shard_shape[3] = math.ceil(channels / max_ch_shards)
        shard_shape[3] = math.ceil(shard_shape[3] / min_shard_width) * min_shard_width

    # Calculate total cores used
    total_shards = 1
    for num_shards in num_shards_per_dim:
        total_shards *= num_shards

    if total_shards > max_cores:
        pytest.skip(f"Required shards {total_shards} exceeds available cores {max_cores}")

    # Create shard grid
    # Map num_shards_per_dim to a 2D grid
    if total_shards == 1:
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    else:
        grid_y = min(total_shards, max_grid_size[0])
        grid_x = math.ceil(total_shards / grid_y)
        if grid_x > max_grid_size[1]:
            grid_x = max_grid_size[1]
            grid_y = math.ceil(total_shards / grid_x)

        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})

    # Validate shard shape doesn't exceed input shape
    for i in range(4):
        if shard_shape[i] > dim_sizes[i]:
            # This can happen with alignment - cap it to input size
            shard_shape[i] = dim_sizes[i]

    # Use NdShardSpec for 4D sharding
    nd_shard_spec = ttnn.NdShardSpec(shard_shape, shard_grid)

    sharded_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)

    if debug:
        logger.info(f"Input shape: {input_shape}, scale_h={scale_factor_h}, scale_w={scale_factor_w}")
        logger.info(f"Shard dims: {shard_dims}")
        logger.info(f"Shard shape: {tuple(shard_shape)}")
        logger.info(f"Num shards per dim: {num_shards_per_dim}")
        logger.info(f"Total shards: {total_shards}")
        logger.info(f"Grid: {grid_x}x{grid_y}")
        logger.info(f"Memory config: {sharded_mem_config}")

    input_tensor = ttnn.from_torch(
        input_nhwc,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )

    logger.info("Input tensor succesfully created with ND sharding.")

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


# Shard only last dimension (C)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        ((1, 8, 8, 64), 2.0, 2.0),
        # ((1, 16, 16, 128), 2.0, 2.0),
        # ((2, 8, 8, 256), 1.5, 1.5),
        # ((1, 32, 32, 512), 0.5, 0.5),
        # ((1, 8, 16, 64), 2.0, 1.5),
    ],
)
def test_upsample_nearest_float_ND_shard_C_only(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_ND_sharded(device, input_shape, scale_factor_h, scale_factor_w, shard_dims=(3,))


# Shard last two dimensions (W, C)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        ((1, 8, 8, 64), 2.0, 2.0),
        ((1, 16, 16, 128), 2.0, 2.0),
        ((2, 8, 8, 256), 1.5, 1.5),
        ((1, 16, 32, 512), 0.5, 0.5),
        ((1, 16, 32, 512), 0.75, 0.25),
        ((1, 16, 32, 512), 0.33, 0.33),
        ((1, 16, 32, 512), 1.0, 1.0),
        ((1, 16, 32, 512), 3.0, 2.0),
    ],
)
def test_upsample_nearest_float_ND_shard_WC(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_ND_sharded(device, input_shape, scale_factor_h, scale_factor_w, shard_dims=(2, 3))


# Shard last three dimensions (H, W, C)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        ((1, 8, 8, 64), 2.0, 2.0),
        ((1, 16, 16, 128), 2.0, 2.0),
        ((2, 8, 8, 256), 1.5, 1.5),
        ((1, 32, 32, 512), 0.5, 0.5),
    ],
)
def test_upsample_nearest_float_ND_shard_HWC(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_ND_sharded(device, input_shape, scale_factor_h, scale_factor_w, shard_dims=(1, 2, 3))


# Shard all four dimensions (N, H, W, C)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        ((2, 8, 8, 64), 2.0, 2.0),
        # ((2, 16, 16, 128), 2.0, 2.0),
        # ((4, 8, 8, 256), 1.5, 1.5),
        # ((2, 32, 32, 512), 0.5, 0.5),
    ],
)
def test_upsample_nearest_float_ND_shard_NHWC(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_ND_sharded(device, input_shape, scale_factor_h, scale_factor_w, shard_dims=(0, 1, 2, 3))


# Shard first dimension only (N)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        ((4, 8, 8, 64), 2.0, 2.0),
        ((8, 16, 16, 128), 2.0, 2.0),
        ((2, 8, 8, 256), 1.5, 1.5),
        ((4, 32, 32, 512), 0.5, 0.5),
    ],
)
def test_upsample_nearest_float_ND_shard_N_only(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_ND_sharded(device, input_shape, scale_factor_h, scale_factor_w, shard_dims=(0,))


# Shard first two dimensions (N, H)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        ((2, 8, 8, 64), 2.0, 2.0),
        ((4, 16, 16, 128), 2.0, 2.0),
        ((2, 32, 32, 256), 1.5, 1.5),
        ((4, 16, 16, 512), 0.5, 0.5),
    ],
)
def test_upsample_nearest_float_ND_shard_NH(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_ND_sharded(device, input_shape, scale_factor_h, scale_factor_w, shard_dims=(0, 1))


# Shard first three dimensions (N, H, W)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        ((2, 8, 8, 64), 2.0, 2.0),
        ((4, 16, 16, 128), 2.0, 2.0),
        ((2, 32, 32, 256), 1.5, 1.5),
    ],
)
def test_upsample_nearest_float_ND_shard_NHW(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_ND_sharded(device, input_shape, scale_factor_h, scale_factor_w, shard_dims=(0, 1, 2))


# Shard batch and channels (N, C)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        ((4, 8, 8, 64), 2.0, 2.0),
        ((8, 16, 16, 128), 2.0, 2.0),
        ((2, 32, 32, 256), 1.5, 1.5),
        ((4, 16, 16, 512), 0.5, 0.5),
    ],
)
def test_upsample_nearest_float_ND_shard_NC(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_ND_sharded(device, input_shape, scale_factor_h, scale_factor_w, shard_dims=(0, 3))


# Shard height and channels (H, C)
@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        ((1, 16, 8, 64), 2.0, 2.0),
        ((1, 32, 16, 128), 2.0, 2.0),
        ((2, 16, 16, 256), 1.5, 1.5),
        ((1, 64, 32, 512), 0.5, 0.5),
    ],
)
def test_upsample_nearest_float_ND_shard_HC(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float_ND_sharded(device, input_shape, scale_factor_h, scale_factor_w, shard_dims=(1, 3))

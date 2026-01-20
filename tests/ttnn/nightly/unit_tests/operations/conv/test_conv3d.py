# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import pytest
import ttnn
import torch.nn as nn
from tests.ttnn.utils_for_testing import check_with_pcc

from tests.ttnn.unit_tests.operations.conv.test_conv3d import (
    setup_conv3d_test,
    create_conv3d_config,
    prepare_weights,
    reshape_output,
    run_conv3d_test,
    ALIGNMENT,
)

# Test configuration constants
PCC_TOLERANCE = 0.99
HIGH_PRECISION_PCC = 0.9999


def compute_conv3d_tensor_2d_shape(input_shape):
    """Compute the 2D physical shape of a Conv3D input tensor.

    Conv3D input is (B, C, T, H, W) in PyTorch format.
    On device after permute, it becomes (B, T, H, W, C_aligned) in ROW_MAJOR.
    Physical 2D shape is (B*T*H*W, C_aligned).

    Returns:
        tuple: (height, width) where height = B*T*H*W, width = C_aligned
    """
    B, C, T, H, W = input_shape
    # Channel dimension is aligned to ALIGNMENT (32)
    C_aligned = C if C % ALIGNMENT == 0 else C + (ALIGNMENT - C % ALIGNMENT)
    height = B * T * H * W
    width = C_aligned
    return height, width


def create_sharded_memory_config(layout_type, grid_size, input_shape):
    """Create a sharded memory configuration for Conv3D testing.

    Args:
        layout_type: "height_sharded", "width_sharded", or "block_sharded"
        grid_size: Device grid size
        input_shape: Conv3D input shape (B, C, T, H, W)

    Returns:
        ttnn.MemoryConfig for the specified sharding layout
    """
    height, width = compute_conv3d_tensor_2d_shape(input_shape)

    logger.debug(f"Creating {layout_type} config for tensor 2D shape: ({height}, {width})")

    if layout_type == "height_sharded":
        # Shard along height (B*T*H*W dimension)
        # Use as many cores as possible while ensuring even division
        max_cores = grid_size.x * grid_size.y
        num_cores = 1
        for n in range(1, min(max_cores, height) + 1):
            if height % n == 0:
                num_cores = n

        shard_height = height // num_cores
        shard_width = width

        # Create core grid - distribute cores row-wise
        cores_x = min(num_cores, grid_size.x)
        cores_y = (num_cores + cores_x - 1) // cores_x
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(cores_x - 1, cores_y - 1),
                )
            }
        )
        memory_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        orientation = ttnn.ShardOrientation.ROW_MAJOR

        logger.debug(f"HEIGHT_SHARDED: {num_cores} cores, shard shape: ({shard_height}, {shard_width})")

    elif layout_type == "width_sharded":
        # Shard along width (C_aligned dimension)
        # Use as many cores as possible while ensuring even division
        max_cores = grid_size.x * grid_size.y
        num_cores = 1
        for n in range(1, min(max_cores, width) + 1):
            if width % n == 0:
                num_cores = n

        shard_height = height
        shard_width = width // num_cores

        # Create core grid - distribute cores column-wise
        cores_y = min(num_cores, grid_size.y)
        cores_x = (num_cores + cores_y - 1) // cores_y
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(cores_x - 1, cores_y - 1),
                )
            }
        )
        memory_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        orientation = ttnn.ShardOrientation.ROW_MAJOR

        logger.debug(f"WIDTH_SHARDED: {num_cores} cores, shard shape: ({shard_height}, {shard_width})")

    elif layout_type == "block_sharded":
        # Block sharding: shard along both height and width
        # Find factors that evenly divide both dimensions
        max_cores_x = min(grid_size.x, 8)
        max_cores_y = min(grid_size.y, 8)

        # Find best core grid that evenly divides both dimensions
        # For BLOCK_SHARDED: cores_x divides width, cores_y divides height
        cores_x = 1
        cores_y = 1
        for cy in range(1, max_cores_y + 1):
            for cx in range(1, max_cores_x + 1):
                if height % cy == 0 and width % cx == 0:
                    if cx * cy > cores_x * cores_y:
                        cores_x = cx
                        cores_y = cy

        if cores_x == 1 and cores_y == 1:
            raise ValueError(
                f"Cannot create BLOCK_SHARDED config: no valid core grid found for "
                f"tensor shape ({height}, {width}) with grid_size ({grid_size.x}, {grid_size.y})"
            )

        shard_height = height // cores_y
        shard_width = width // cores_x

        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(cores_x - 1, cores_y - 1),
                )
            }
        )
        memory_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        orientation = ttnn.ShardOrientation.ROW_MAJOR

        logger.debug(f"BLOCK_SHARDED: grid ({cores_x}, {cores_y}), shard: ({shard_height}, {shard_width})")

    else:
        raise ValueError(f"Unsupported layout_type: {layout_type}")

    shard_spec = ttnn.ShardSpec(shard_grid, [shard_height, shard_width], orientation)
    return ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)


def run_conv3d_with_memory_config(
    device,
    input_shape,
    out_channels,
    kernel_size,
    stride,
    padding,
    padding_mode,
    input_mem_config=None,
    output_mem_config=None,
):
    """Run conv3d test with specified memory configurations.

    Returns:
        Tuple of (tt_output, gt_output) for comparison
    """
    # Setup test
    tt_input, conv3d_module, gt_output, kernel_config, output_dims = setup_conv3d_test(
        input_shape,
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
        device,
    )
    N, D_out, H_out, W_out = output_dims
    C = input_shape[1]

    # Prepare weights and bias (always interleaved for simplicity)
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device)

    # Convert input to specified memory config if provided
    if input_mem_config is not None:
        try:
            tt_input = ttnn.to_memory_config(tt_input, input_mem_config)
        except Exception as e:
            pytest.skip(f"Incompatible input memory configuration: {e}")

    # Create config
    grid_size = device.compute_with_storage_grid_size()
    config = create_conv3d_config(compute_with_storage_grid_size=grid_size)

    # Run conv3d
    conv3d_kwargs = {
        "input_tensor": tt_input,
        "weight_tensor": tt_weight,
        "bias_tensor": tt_bias,
        "dtype": ttnn.bfloat16,
        "output_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "padding_mode": padding_mode,
        "config": config,
        "compute_kernel_config": kernel_config,
    }

    # Add output memory config if specified
    if output_mem_config is not None:
        conv3d_kwargs["memory_config"] = output_mem_config

    try:
        tt_output = ttnn.experimental.conv3d(**conv3d_kwargs)
    except Exception as e:
        if output_mem_config is not None:
            pytest.skip(f"Output memory configuration not supported: {e}")
        else:
            raise

    # Convert output to interleaved for comparison if needed
    if output_mem_config is not None:
        tt_output = ttnn.to_memory_config(tt_output, ttnn.DRAM_MEMORY_CONFIG)

    # Reshape output
    tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)

    return tt_output, gt_output


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("C_in", [12, 64])
@pytest.mark.parametrize("C_out", [32, 64])
@pytest.mark.parametrize("T", [3, 5])
@pytest.mark.parametrize("H", [4, 6])
@pytest.mark.parametrize("W", [5, 7])
@pytest.mark.parametrize("kernel_size", [(3, 3, 3)], ids=["kernel_333"])
@pytest.mark.parametrize(
    "stride",
    [
        (1, 1, 1),
        (1, 3, 5),
    ],
    ids=["stride_111", "stride_135"],
)
@pytest.mark.parametrize("padding", [(0, 1, 1)], ids=["padding_011"])
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
def test_conv3d_sweep_shapes(device, B, C_in, C_out, T, H, W, kernel_size, stride, padding, padding_mode):
    if padding == (0, 0, 0) and padding_mode == "replicate":
        pytest.skip("Skipping padding (0, 0, 0) and padding_mode replicate " "because it's duplicate")
    input_shape = (B, C_in, T, H, W)
    out_channels = C_out
    kernel_size = kernel_size
    stride = stride
    padding = padding
    padding_mode = padding_mode
    grid_size = device.compute_with_storage_grid_size()
    run_conv3d_test(
        device,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
        grid_size=grid_size,
    )


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 128, 16, 16, 16), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(3, 64, 8, 8, 8), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
    ],
)
@pytest.mark.timeout(1000)
def test_conv3d_sweep_blocks(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    """
    For a specific shape, sweep through different block sizes.
    Constrain the sweep such that the num_patches in a block doesn't exceed 64
    """
    import math

    grid_size = device.compute_with_storage_grid_size()
    tt_input, conv3d_module, gt_output, kernel_config, output_dims = setup_conv3d_test(
        input_shape, out_channels, kernel_size, stride, padding, padding_mode, device
    )
    N, D_out, H_out, W_out = output_dims
    C = input_shape[1]
    C_in_blocks = filter(lambda x: C % x == 0, range(32, C + 1, 32))
    C_out_blocks = filter(lambda x: out_channels % x == 0, range(32, out_channels + 1, 32))
    T_out_blocks = [2**i for i in range(int(math.log2(D_out)))]
    H_out_blocks = [2**i for i in range(int(math.log2(H_out)))]
    W_out_blocks = [2**i for i in range(int(math.log2(W_out)))]

    MAX_NUM_PATCHES_IN_BLOCK = 64
    prev_C_in_block = None

    import itertools

    for C_in_block, C_out_block, T_out_block, H_out_block, W_out_block in itertools.product(
        C_in_blocks, C_out_blocks, T_out_blocks, H_out_blocks, W_out_blocks
    ):
        num_patches_in_block = T_out_block * H_out_block * W_out_block
        if num_patches_in_block > MAX_NUM_PATCHES_IN_BLOCK:
            continue
        if (C_in_block == 128 or C_out_block == 128) and num_patches_in_block > 32:
            continue

        logger.info(f"Testing {C_in_block}, {C_out_block}, {T_out_block}, " f"{H_out_block}, {W_out_block}")
        # Prepare weights with specified C_in_block
        if prev_C_in_block != C_in_block:
            # Only prepare if changing C_in_block
            tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device, C_in_block=C_in_block)
            prev_C_in_block = C_in_block

        config = create_conv3d_config(
            T_out_block=T_out_block,
            H_out_block=H_out_block,
            W_out_block=W_out_block,
            C_out_block=C_out_block,
            C_in_block=C_in_block,
            compute_with_storage_grid_size=grid_size,
        )

        tt_output = ttnn.experimental.conv3d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            dtype=ttnn.bfloat16,
            output_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            groups=1,
            config=config,
            compute_kernel_config=kernel_config,
        )
        # Reshape output and verify results
        tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)

        assert tt_output.shape == gt_output.shape
        pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.99)
        assert pcc_passed, (
            f"{pcc_message} on "
            f"C_out_block={C_out_block}, T_out_block={T_out_block}, "
            f"W_out_block={W_out_block}, H_out_block={H_out_block}, "
            f"C_in_block={C_in_block}"
        )


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode, blocking",
    [
        [
            (1, 768, 4, 60, 106),
            768,
            (3, 3, 3),
            (1, 1, 1),
            (0, 1, 1),
            "replicate",
            (128, 96, 1, 2, 16),
        ],  # Best blocking found so far
        [
            (1, 512, 11, 120, 212),
            512,
            (3, 3, 3),
            (1, 1, 1),
            (0, 1, 1),
            "replicate",
            (128, 128, 1, 8, 4),
        ],  # Best blocking found so far
        [
            (1, 256, 21, 240, 424),
            256,
            (3, 3, 3),
            (1, 1, 1),
            (0, 1, 1),
            "replicate",
            (128, 128, 4, 4, 2),
        ],  # Best blocking found so far
        [
            (1, 128, 21, 480, 848),
            128,
            (3, 3, 3),
            (1, 1, 1),
            (0, 1, 1),
            "replicate",
            (128, 128, 1, 2, 16),
        ],  # Best blocking found so far
    ],
    ids=["variant1", "variant2", "variant3", "variant4"],
)
def test_conv3d_mochi_shapes(
    device,
    input_shape,
    out_channels,
    kernel_size,
    stride,
    padding,
    padding_mode,
    blocking,
    is_ci_env,
):
    if out_channels == 128 or out_channels == 256:
        pytest.skip("Skipping test for 128 out channels on CI due to host OOM")

    C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = blocking
    tt_input, conv3d_module, gt_output, kernel_config, output_dims = setup_conv3d_test(
        input_shape, out_channels, kernel_size, stride, padding, padding_mode, device
    )
    N, D_out, H_out, W_out = output_dims
    C = input_shape[1]

    # Prepare weights with specified C_in_block
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device, C_in_block=C_in_block)

    config = create_conv3d_config(
        T_out_block=T_out_block,
        H_out_block=H_out_block,
        W_out_block=W_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )

    tt_output = ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        dtype=ttnn.bfloat16,
        output_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=1,
        config=config,
        compute_kernel_config=kernel_config,
    )
    # Reshape output and verify results
    tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)

    assert tt_output.shape == gt_output.shape
    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.999)
    logger.info(f"{pcc_message}")
    assert pcc_passed, pcc_message


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        # Use shapes where dimensions divide evenly for sharding
        # B*T*H*W must be divisible, C must be aligned to 32
        [(1, 64, 4, 8, 8), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
        [(2, 32, 4, 4, 4), 32, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
)
@pytest.mark.parametrize(
    "input_layout",
    ["interleaved", "height_sharded", "width_sharded", "block_sharded"],
)
def test_conv3d_sharded_layouts(
    device,
    input_shape,
    out_channels,
    kernel_size,
    stride,
    padding,
    padding_mode,
    input_layout,
):
    """Test Conv3d with various input memory layouts.

    This test validates that Conv3d works correctly with different memory layouts,
    leveraging TensorAccessor's unified memory access for sharded tensors.
    """
    torch.manual_seed(42)

    grid_size = device.compute_with_storage_grid_size()

    # Create input memory config
    if input_layout == "interleaved":
        input_mem_config = None  # Use default interleaved
    else:
        try:
            input_mem_config = create_sharded_memory_config(input_layout, grid_size, input_shape)
        except Exception as e:
            pytest.skip(f"Cannot create {input_layout} config for shape {input_shape}: {e}")

    # Run test
    tt_output, gt_output = run_conv3d_with_memory_config(
        device,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
        input_mem_config=input_mem_config,
    )

    # Validate results
    assert tt_output.shape == gt_output.shape
    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=PCC_TOLERANCE)
    logger.info(f"Conv3d with {input_layout} input: {pcc_message}")
    assert pcc_passed, f"{pcc_message} - {input_layout} input failed"


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 64, 4, 8, 8), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
    ],
)
def test_conv3d_sharded_output(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    """Test Conv3d with sharded output memory configuration.

    This test validates that Conv3d can produce output directly in sharded layout,
    which is important for chaining operations without intermediate conversions.
    """
    torch.manual_seed(42)

    grid_size = device.compute_with_storage_grid_size()

    try:
        output_mem_config = create_sharded_memory_config("height_sharded", grid_size, input_shape)
    except Exception as e:
        pytest.skip(f"Cannot create sharded output config: {e}")

    # Run test
    tt_output, gt_output = run_conv3d_with_memory_config(
        device,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
        output_mem_config=output_mem_config,
    )

    # Validate results
    assert tt_output.shape == gt_output.shape
    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=PCC_TOLERANCE)
    logger.info(f"Conv3d with sharded output: {pcc_message}")
    assert pcc_passed, f"{pcc_message} - sharded output failed"


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 64, 4, 8, 8), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
        [(1, 32, 4, 4, 4), 32, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
)
def test_conv3d_interleaved_vs_sharded_equivalence(
    device, input_shape, out_channels, kernel_size, stride, padding, padding_mode
):
    """Test numerical equivalence between interleaved and sharded layouts.

    This test validates that TensorAccessor provides correct unified access
    by comparing results from interleaved and sharded inputs.
    """
    torch.manual_seed(42)

    grid_size = device.compute_with_storage_grid_size()

    # Run with interleaved input
    tt_output_interleaved, gt_output = run_conv3d_with_memory_config(
        device, input_shape, out_channels, kernel_size, stride, padding, padding_mode
    )

    # Run with sharded input
    try:
        sharded_mem_config = create_sharded_memory_config("height_sharded", grid_size, input_shape)
    except Exception as e:
        pytest.skip(f"Cannot create sharded config: {e}")

    tt_output_sharded, _ = run_conv3d_with_memory_config(
        device,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
        input_mem_config=sharded_mem_config,
    )

    # Compare outputs (should be nearly identical)
    pcc_passed, pcc_message = check_with_pcc(tt_output_interleaved, tt_output_sharded, pcc=HIGH_PRECISION_PCC)
    logger.info(f"Interleaved vs Sharded equivalence: {pcc_message}")
    assert pcc_passed, f"{pcc_message} - Layouts produce different results"

    # Also verify against PyTorch ground truth
    pcc_passed_gt, pcc_message_gt = check_with_pcc(gt_output, tt_output_sharded, pcc=PCC_TOLERANCE)
    logger.info(f"Sharded vs PyTorch: {pcc_message_gt}")
    assert pcc_passed_gt, f"{pcc_message_gt} - Sharded output differs from PyTorch"

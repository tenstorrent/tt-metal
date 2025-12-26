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
)


@pytest.mark.parametrize("B", [1])
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
        pytest.skip("Skipping padding (0, 0, 0) and padding_mode replicate because it's duplicate")
    input_shape = (B, C_in, T, H, W)
    out_channels = C_out
    kernel_size = kernel_size
    stride = stride
    padding = padding
    padding_mode = padding_mode
    grid_size = device.compute_with_storage_grid_size()
    run_conv3d_test(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, grid_size=grid_size)


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 128, 16, 16, 16), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
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

        logger.info(f"Testing {C_in_block}, {C_out_block}, {T_out_block}, {H_out_block}, {W_out_block}")
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


def create_sharded_tensor(torch_tensor, device, shard_layout, shard_grid_shape, shard_orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """
    Helper function to create a sharded tensor from a torch tensor.
    
    Args:
        torch_tensor: Input torch tensor (already in TTNN format: N, T, H, W, C)
        device: TTNN device
        shard_layout: One of HEIGHT_SHARDED, WIDTH_SHARDED, or BLOCK_SHARDED
        shard_grid_shape: Tuple (grid_height, grid_width) for shard grid
        shard_orientation: ShardOrientation (default: ROW_MAJOR)
    
    Returns:
        Sharded TTNN tensor
    """
    N, T, H, W, C = torch_tensor.shape
    shard_grid = ttnn.CoreGrid(*shard_grid_shape)
    n_cores = shard_grid_shape[0] * shard_grid_shape[1]
    
    if shard_layout == ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED:
        # Shard along height dimension (T*H dimension)
        shard_shape = [N * T * H // n_cores, W, C]
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    elif shard_layout == ttnn.types.TensorMemoryLayout.WIDTH_SHARDED:
        # Shard along width dimension
        shard_shape = [N * T * H, W // n_cores, C]
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    elif shard_layout == ttnn.types.TensorMemoryLayout.BLOCK_SHARDED:
        # Block sharding: shard both height and width
        grid_h, grid_w = shard_grid_shape
        shard_shape = [N * T * H // grid_h, W // grid_w, C]
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    else:
        raise ValueError(f"Unsupported shard layout: {shard_layout}")
    
    memory_config = ttnn.MemoryConfig(
        shard_layout, ttnn.types.BufferType.L1, shard_spec
    )
    
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 64, 8, 8, 8), 32, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
        [(1, 64, 8, 8, 8), 32, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
)
@pytest.mark.parametrize(
    "shard_layout",
    [
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED,
    ],
    ids=["height_sharded", "width_sharded", "block_sharded"],
)
def test_conv3d_sharded_input(
    device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, shard_layout
):
    """Test Conv3d with sharded input tensors."""
    torch.manual_seed(42)
    
    # Setup test
    tt_input, conv3d_module, gt_output, kernel_config, output_dims = setup_conv3d_test(
        input_shape, out_channels, kernel_size, stride, padding, padding_mode, device
    )
    N, D_out, H_out, W_out = output_dims
    C = input_shape[1]
    
    # Convert input to torch format for sharding
    tt_input_torch = ttnn.to_torch(tt_input, device=device, dtype=torch.float32)
    
    # Create sharded input tensor
    grid_size = device.compute_with_storage_grid_size()
    # Use a smaller grid for testing (e.g., 2x2 or 4x1)
    shard_grid_shape = (min(4, grid_size.y), min(4, grid_size.x))
    sharded_input = create_sharded_tensor(
        tt_input_torch, device, shard_layout, shard_grid_shape
    )
    
    # Prepare weights and bias (interleaved for now)
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device, C_in_block=0)
    
    # Create config
    config = create_conv3d_config(compute_with_storage_grid_size=grid_size)
    
    # Run Conv3d with sharded input
    tt_output = ttnn.experimental.conv3d(
        input_tensor=sharded_input,
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
    logger.info(f"Sharded input test ({shard_layout}): {pcc_message}")
    assert pcc_passed, f"Sharded input test failed: {pcc_message}"


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 64, 8, 8, 8), 32, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
    ],
)
def test_conv3d_sharded_output(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    """Test Conv3d with sharded output tensor."""
    torch.manual_seed(42)
    
    # Setup test
    tt_input, conv3d_module, gt_output, kernel_config, output_dims = setup_conv3d_test(
        input_shape, out_channels, kernel_size, stride, padding, padding_mode, device
    )
    N, D_out, H_out, W_out = output_dims
    C = input_shape[1]
    
    # Prepare weights and bias
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device, C_in_block=0)
    
    # Create config
    grid_size = device.compute_with_storage_grid_size()
    config = create_conv3d_config(compute_with_storage_grid_size=grid_size)
    
    # Create sharded output memory config
    shard_grid_shape = (min(4, grid_size.y), min(4, grid_size.x))
    shard_grid = ttnn.CoreGrid(*shard_grid_shape)
    n_cores = shard_grid_shape[0] * shard_grid_shape[1]
    shard_shape = [N * D_out * H_out // n_cores, W_out, out_channels]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    
    # Run Conv3d with sharded output
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
        memory_config=output_mem_config,
    )
    
    # Reshape output and verify results
    tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)
    
    assert tt_output.shape == gt_output.shape
    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.99)
    logger.info(f"Sharded output test: {pcc_message}")
    assert pcc_passed, f"Sharded output test failed: {pcc_message}"


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 64, 8, 8, 8), 32, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
    ],
)
def test_conv3d_sharded_input_and_output(
    device, input_shape, out_channels, kernel_size, stride, padding, padding_mode
):
    """Test Conv3d with both sharded input and output tensors."""
    torch.manual_seed(42)
    
    # Setup test
    tt_input, conv3d_module, gt_output, kernel_config, output_dims = setup_conv3d_test(
        input_shape, out_channels, kernel_size, stride, padding, padding_mode, device
    )
    N, D_out, H_out, W_out = output_dims
    C = input_shape[1]
    
    # Convert input to torch format for sharding
    tt_input_torch = ttnn.to_torch(tt_input, device=device, dtype=torch.float32)
    
    # Create sharded input tensor
    grid_size = device.compute_with_storage_grid_size()
    shard_grid_shape = (min(4, grid_size.y), min(4, grid_size.x))
    sharded_input = create_sharded_tensor(
        tt_input_torch, device, ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, shard_grid_shape
    )
    
    # Prepare weights and bias
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device, C_in_block=0)
    
    # Create config
    config = create_conv3d_config(compute_with_storage_grid_size=grid_size)
    
    # Create sharded output memory config
    shard_grid = ttnn.CoreGrid(*shard_grid_shape)
    n_cores = shard_grid_shape[0] * shard_grid_shape[1]
    shard_shape = [N * D_out * H_out // n_cores, W_out, out_channels]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    
    # Run Conv3d with sharded input and output
    tt_output = ttnn.experimental.conv3d(
        input_tensor=sharded_input,
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
        memory_config=output_mem_config,
    )
    
    # Reshape output and verify results
    tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)
    
    assert tt_output.shape == gt_output.shape
    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.99)
    logger.info(f"Sharded input and output test: {pcc_message}")
    assert pcc_passed, f"Sharded input and output test failed: {pcc_message}"

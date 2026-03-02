# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Conv3D with sharded memory layout support.

This test module verifies that conv3d now supports sharded memory configuration
for input, weight, and bias tensors, as well as sharded output.
"""

from loguru import logger
import torch
import pytest
import ttnn
import torch.nn as nn
from tests.ttnn.utils_for_testing import check_with_pcc
from models.common.utility_functions import skip_for_blackhole, skip_with_watcher


def _out_size(in_size, pad, stride, k):
    return (in_size + 2 * pad - k) // stride + 1


ALIGNMENT = 32  # Valid L1 alignment for Wormhole and Blackhole


def prepare_input_tensor(input_tensor, C, device, alignment=ALIGNMENT):
    """Prepare input tensor for TTNN by permuting and padding."""
    tt_input = input_tensor.permute(0, 2, 3, 4, 1)
    ALIGN_PAD = alignment - C % alignment
    if C % alignment != 0:
        tt_input = torch.nn.functional.pad(tt_input, (0, ALIGN_PAD))
    return ttnn.from_torch(tt_input, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT)


def prepare_input_tensor_sharded(
    input_tensor, C, device, shard_scheme, alignment=ALIGNMENT, core_grid=None
):
    """Prepare input tensor with sharded memory configuration.
    
    Args:
        input_tensor: Input tensor in [N, C, D, H, W] format
        C: Number of input channels
        device: TTNN device
        shard_scheme: Sharding scheme (HEIGHT_SHARDED, WIDTH_SHARDED, or BLOCK_SHARDED)
        alignment: Memory alignment requirement
        core_grid: Core grid for sharding (default: device compute grid)
    """
    tt_input = input_tensor.permute(0, 2, 3, 4, 1)
    ALIGN_PAD = alignment - C % alignment
    if C % alignment != 0:
        tt_input = torch.nn.functional.pad(tt_input, (0, ALIGN_PAD))
    
    # Calculate shard shape based on sharding scheme
    N, D, H, W, C_aligned = tt_input.shape
    total_elements = N * D * H * W * C_aligned
    
    if core_grid is None:
        core_grid = device.compute_with_storage_grid_size()
    
    num_cores = core_grid.x * core_grid.y
    
    if shard_scheme == ttnn.ShardStrategy.HEIGHT:
        # Height sharding: shard along the H dimension (or combined spatial dims)
        shard_shape = [1, 1, (N * D * H * W + num_cores - 1) // num_cores, C_aligned]
    elif shard_scheme == ttnn.ShardStrategy.WIDTH:
        # Width sharding: shard along the C dimension
        shard_shape = [N * D * H * W, (C_aligned + num_cores - 1) // num_cores]
    else:  # BLOCK_SHARDED
        # Block sharding: shard along both dimensions
        shard_shape = [(N * D * H * W + num_cores - 1) // num_cores, C_aligned]
    
    shard_config = ttnn.ShardConfig(
        shard_shape=shard_shape,
        shard_strategy=shard_scheme,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    
    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.SHARDED,
        shard_spec=shard_config,
        buffer_type=ttnn.BufferType.L1,
    )
    
    return ttnn.from_torch(
        tt_input, 
        device=device, 
        dtype=ttnn.DataType.BFLOAT16, 
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )


def prepare_weights(conv3d_module, C, out_channels, device, C_in_block=0, alignment=ALIGNMENT):
    """Prepare weights and bias for TTNN."""
    w = conv3d_module.weight.data  # out_chan, C, kD, kH, kW
    w = w.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C, out_chan
    ALIGN_PAD = alignment - C % alignment
    if C % alignment != 0:
        w = torch.nn.functional.pad(w, (0, 0, 0, ALIGN_PAD))

    # Reshape weights so that num_C_in_blocks is the first dimension
    kD, kH, kW, C_in_aligned, out_channels = w.shape
    C_in_block = C_in_aligned if C_in_block == 0 else C_in_block
    num_C_in_blocks = C_in_aligned // C_in_block
    assert num_C_in_blocks * C_in_block == C_in_aligned
    w = w.reshape(kD, kH, kW, num_C_in_blocks, C_in_block, out_channels)
    w = w.permute(3, 0, 1, 2, 4, 5)
    w = w.reshape(-1, out_channels)

    tt_weight = ttnn.from_torch(w, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, pad_value=0)
    tt_bias = ttnn.from_torch(
        conv3d_module.bias.data.reshape(1, -1),
        device=device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
    )
    return tt_weight, tt_bias


def reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device):
    """Reshape and permute TTNN output to match PyTorch format."""
    tt_output = ttnn.to_torch(tt_output, device=device, dtype=torch.float32)
    tt_output = tt_output.reshape(N, D_out, H_out, W_out, out_channels)
    return tt_output.permute(0, 4, 1, 2, 3)


def create_conv3d_config(
    T_out_block=1,
    W_out_block=1,
    H_out_block=1,
    C_out_block=0,
    C_in_block=0,
    compute_with_storage_grid_size=(1, 1),
):
    """Create Conv3d configuration."""
    return ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=T_out_block,
        W_out_block=W_out_block,
        H_out_block=H_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=compute_with_storage_grid_size,
    )


def setup_conv3d_test(input_shape, out_channels, kernel_size, stride, padding, padding_mode, device):
    """Common setup for Conv3D tests, preparing inputs and ground truth."""
    torch.manual_seed(42)

    # Define input dimensions
    N, C, D, H, W = input_shape
    D_out = _out_size(D, padding[0], stride[0], kernel_size[0])
    H_out = _out_size(H, padding[1], stride[1], kernel_size[1])
    W_out = _out_size(W, padding[2], stride[2], kernel_size[2])

    # Create input tensor and PyTorch Conv3d module
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)

    conv3d_module = nn.Conv3d(
        C,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=(1, 1, 1),
        bias=True,
        padding_mode=padding_mode,
    )

    gt_output = conv3d_module(input_tensor)

    # Prepare input for TTNN
    tt_input = prepare_input_tensor(input_tensor, C, device)

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    return tt_input, conv3d_module, gt_output, kernel_config, (N, D_out, H_out, W_out)


def run_conv3d_test_sharded_input(
    device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, 
    shard_scheme, grid_size=(1, 1)
):
    """Test conv3d with sharded input tensor."""
    torch.manual_seed(42)

    # Define input dimensions
    N, C, D, H, W = input_shape
    D_out = _out_size(D, padding[0], stride[0], kernel_size[0])
    H_out = _out_size(H, padding[1], stride[1], kernel_size[1])
    W_out = _out_size(W, padding[2], stride[2], kernel_size[2])

    # Create input tensor and PyTorch Conv3d module
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)

    conv3d_module = nn.Conv3d(
        C,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=(1, 1, 1),
        bias=True,
        padding_mode=padding_mode,
    )

    gt_output = conv3d_module(input_tensor)

    # Prepare sharded input for TTNN
    tt_input = prepare_input_tensor_sharded(input_tensor, C, device, shard_scheme)
    
    # Verify input is sharded
    assert tt_input.is_sharded(), "Input tensor should be sharded"
    logger.info(f"Input tensor memory config: {tt_input.memory_config()}")

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # Prepare weights and bias for TTNN
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device, C_in_block=0)

    # Create config and run TTNN conv3d
    config = create_conv3d_config(compute_with_storage_grid_size=grid_size)

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
        config=config,
        compute_kernel_config=kernel_config,
    )

    # Reshape output and verify results
    tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)

    print(f"gt output shape = {gt_output.shape}")
    print(f"tt output shape = {tt_output.shape}")
    assert tt_output.shape == gt_output.shape

    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.999)
    logger.info(f"Compare conv3d torch vs ttnn (sharded input): {pcc_message}")
    assert pcc_passed, pcc_message


# Test sharded input with height sharding
@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 32, 4, 8, 8), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
        [(2, 16, 4, 8, 8), 32, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
)
@skip_with_watcher("Skipping test with watcher enabled due to failure, see github issue #37184")
def test_conv3d_sharded_input_height(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    """Test Conv3d with height-sharded input tensor."""
    grid_size = device.compute_with_storage_grid_size()
    run_conv3d_test_sharded_input(
        device, input_shape, out_channels, kernel_size, stride, padding, padding_mode,
        ttnn.ShardStrategy.HEIGHT, grid_size=grid_size
    )


# Test sharded input with width sharding
@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 32, 4, 8, 8), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
    ],
)
@skip_with_watcher("Skipping test with watcher enabled due to failure, see github issue #37184")
def test_conv3d_sharded_input_width(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    """Test Conv3d with width-sharded input tensor."""
    grid_size = device.compute_with_storage_grid_size()
    run_conv3d_test_sharded_input(
        device, input_shape, out_channels, kernel_size, stride, padding, padding_mode,
        ttnn.ShardStrategy.WIDTH, grid_size=grid_size
    )


# Test sharded input with block sharding
@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 32, 4, 8, 8), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
    ],
)
@skip_with_watcher("Skipping test with watcher enabled due to failure, see github issue #37184")
def test_conv3d_sharded_input_block(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    """Test Conv3d with block-sharded input tensor."""
    grid_size = device.compute_with_storage_grid_size()
    run_conv3d_test_sharded_input(
        device, input_shape, out_channels, kernel_size, stride, padding, padding_mode,
        ttnn.ShardStrategy.BLOCK, grid_size=grid_size
    )


def run_conv3d_test_sharded_output(
    device, input_shape, out_channels, kernel_size, stride, padding, padding_mode,
    output_shard_scheme, grid_size=(1, 1)
):
    """Test conv3d with sharded output memory configuration."""
    torch.manual_seed(42)

    # Define input dimensions
    N, C, D, H, W = input_shape
    D_out = _out_size(D, padding[0], stride[0], kernel_size[0])
    H_out = _out_size(H, padding[1], stride[1], kernel_size[1])
    W_out = _out_size(W, padding[2], stride[2], kernel_size[2])

    # Create input tensor and PyTorch Conv3d module
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)

    conv3d_module = nn.Conv3d(
        C,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=(1, 1, 1),
        bias=True,
        padding_mode=padding_mode,
    )

    gt_output = conv3d_module(input_tensor)

    # Prepare input for TTNN (interleaved)
    tt_input = prepare_input_tensor(input_tensor, C, device)

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # Prepare weights and bias for TTNN
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device, C_in_block=0)

    # Create sharded output memory config
    core_grid = device.compute_with_storage_grid_size()
    num_cores = core_grid.x * core_grid.y
    
    if output_shard_scheme == ttnn.ShardStrategy.HEIGHT:
        shard_shape = [1, 1, (N * D_out * H_out * W_out + num_cores - 1) // num_cores, out_channels]
    elif output_shard_scheme == ttnn.ShardStrategy.WIDTH:
        shard_shape = [N * D_out * H_out * W_out, (out_channels + num_cores - 1) // num_cores]
    else:  # BLOCK_SHARDED
        shard_shape = [(N * D_out * H_out * W_out + num_cores - 1) // num_cores, out_channels]
    
    shard_config = ttnn.ShardConfig(
        shard_shape=shard_shape,
        shard_strategy=output_shard_scheme,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    
    output_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.SHARDED,
        shard_spec=shard_config,
        buffer_type=ttnn.BufferType.L1,
    )

    # Create config and run TTNN conv3d with sharded output
    config = create_conv3d_config(compute_with_storage_grid_size=grid_size)

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
        config=config,
        compute_kernel_config=kernel_config,
        memory_config=output_memory_config,
    )

    # Verify output is sharded
    assert tt_output.is_sharded(), "Output tensor should be sharded"
    logger.info(f"Output tensor memory config: {tt_output.memory_config()}")

    # Reshape output and verify results
    tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)

    print(f"gt output shape = {gt_output.shape}")
    print(f"tt output shape = {tt_output.shape}")
    assert tt_output.shape == gt_output.shape

    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.999)
    logger.info(f"Compare conv3d torch vs ttnn (sharded output): {pcc_message}")
    assert pcc_passed, pcc_message


# Test sharded output with height sharding
@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 32, 4, 8, 8), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
    ],
)
@skip_with_watcher("Skipping test with watcher enabled due to failure, see github issue #37184")
def test_conv3d_sharded_output_height(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    """Test Conv3d with height-sharded output memory config."""
    grid_size = device.compute_with_storage_grid_size()
    run_conv3d_test_sharded_output(
        device, input_shape, out_channels, kernel_size, stride, padding, padding_mode,
        ttnn.ShardStrategy.HEIGHT, grid_size=grid_size
    )


# Compare interleaved vs sharded performance/accuracy
@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 64, 8, 16, 16), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
    ],
)
@skip_with_watcher("Skipping test with watcher enabled due to failure, see github issue #37184")
def test_conv3d_interleaved_vs_sharded_accuracy(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    """Compare numerical accuracy between interleaved and sharded memory layouts."""
    torch.manual_seed(42)
    
    N, C, D, H, W = input_shape
    D_out = _out_size(D, padding[0], stride[0], kernel_size[0])
    H_out = _out_size(H, padding[1], stride[1], kernel_size[1])
    W_out = _out_size(W, padding[2], stride[2], kernel_size[2])

    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)
    conv3d_module = nn.Conv3d(
        C, out_channels, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=(1, 1, 1), bias=True, padding_mode=padding_mode,
    )
    gt_output = conv3d_module(input_tensor)

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False,
    )

    # Test with interleaved input
    tt_input_interleaved = prepare_input_tensor(input_tensor, C, device)
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device)
    config = create_conv3d_config(compute_with_storage_grid_size=device.compute_with_storage_grid_size())
    
    tt_output_interleaved = ttnn.experimental.conv3d(
        input_tensor=tt_input_interleaved, weight_tensor=tt_weight, bias_tensor=tt_bias,
        dtype=ttnn.bfloat16, output_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, padding_mode=padding_mode,
        config=config, compute_kernel_config=kernel_config,
    )
    tt_output_interleaved = reshape_output(tt_output_interleaved, N, D_out, H_out, W_out, out_channels, device)

    # Test with sharded input
    tt_input_sharded = prepare_input_tensor_sharded(input_tensor, C, device, ttnn.ShardStrategy.HEIGHT)
    tt_output_sharded = ttnn.experimental.conv3d(
        input_tensor=tt_input_sharded, weight_tensor=tt_weight, bias_tensor=tt_bias,
        dtype=ttnn.bfloat16, output_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, padding_mode=padding_mode,
        config=config, compute_kernel_config=kernel_config,
    )
    tt_output_sharded = reshape_output(tt_output_sharded, N, D_out, H_out, W_out, out_channels, device)

    # Compare interleaved and sharded results
    pcc_passed, pcc_message = check_with_pcc(tt_output_interleaved, tt_output_sharded, pcc=0.999)
    logger.info(f"Compare interleaved vs sharded: {pcc_message}")
    assert pcc_passed, f"Interleaved and sharded results should match: {pcc_message}"
    
    # Both should match ground truth
    pcc_interleaved, _ = check_with_pcc(gt_output, tt_output_interleaved, pcc=0.999)
    pcc_sharded, _ = check_with_pcc(gt_output, tt_output_sharded, pcc=0.999)
    assert pcc_interleaved, "Interleaved output should match ground truth"
    assert pcc_sharded, "Sharded output should match ground truth"

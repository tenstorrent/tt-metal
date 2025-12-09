# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Tests for config_tensors_in_dram functionality in pool operations.

When config_tensors_in_dram is True, configuration tensors (reader indices and scalar config)
are stored in DRAM instead of L1_SMALL. This is useful for large CNNs where L1_SMALL
persistent storage gets quickly used up.
"""

import torch
import ttnn
import pytest
import math


@pytest.fixture(scope="module")
def tensor_map(request):
    """Cache map used for torch tensor reuse."""
    return {}


def randomize_torch_tensor(tensor_map, tensor_shape):
    """Generate or retrieve a random torch tensor with the given shape."""
    tensor_shape = tuple(tensor_shape)
    if tensor_shape in tensor_map:
        return tensor_map[tensor_shape]
    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    tensor_map[tensor_shape] = torch_tensor
    return torch_tensor


def run_max_pool2d_with_config_tensors_in_dram(
    input_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    tensor_map,
    config_tensors_in_dram,
    shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ceil_mode=False,
):
    """
    Run max_pool2d with the config_tensors_in_dram flag and verify correctness.

    Args:
        input_shape: (N, C, H, W) tuple
        kernel_size: (kernel_h, kernel_w) tuple
        padding: (pad_h, pad_w) tuple
        stride: (stride_h, stride_w) tuple
        dilation: (dilation_h, dilation_w) tuple
        device: TT device
        tensor_map: Tensor cache map
        config_tensors_in_dram: Whether to store config tensors in DRAM
        shard_scheme: Sharding scheme to use
        ceil_mode: Whether to use ceiling mode for output shape
    """
    in_n, in_c, in_h, in_w = input_shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_h, pad_w = padding

    # Calculate output dimensions
    if ceil_mode:
        out_h = math.ceil((in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.ceil((in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1
    else:
        out_h = math.floor((in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.floor((in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1

    # Generate input tensor
    torch.manual_seed(42)
    torch_input = randomize_torch_tensor(tensor_map, input_shape)
    torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # N, H, W, C
    ttnn_input = ttnn.from_torch(torch_input_permuted, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Run TTNN max_pool2d with config_tensors_in_dram
    ttnn_output = ttnn.max_pool2d(
        input_tensor=ttnn_input,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        applied_shard_scheme=shard_scheme,
        deallocate_input=True,
        reallocate_halo_output=True,
        config_tensors_in_dram=config_tensors_in_dram,
    )

    # Run PyTorch max_pool2d
    torch_output = torch.nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=ceil_mode,
    )(torch_input)

    # Convert TTNN output back to comparable format
    ttnn_output = ttnn.to_torch(ttnn_output)
    out_n, out_c = in_n, in_c
    ttnn_output = ttnn_output.reshape(out_n, out_h, out_w, out_c)  # N, H, W, C
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))  # N, C, H, W

    # Verify correctness
    assert torch.equal(ttnn_output, torch_output), "Max pool output mismatch"


def run_avg_pool2d_with_config_tensors_in_dram(
    input_shape,
    kernel_size,
    padding,
    stride,
    device,
    tensor_map,
    config_tensors_in_dram,
    shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ceil_mode=False,
    count_include_pad=True,
):
    """
    Run avg_pool2d with the config_tensors_in_dram flag and verify correctness.

    Args:
        input_shape: (N, C, H, W) tuple
        kernel_size: (kernel_h, kernel_w) tuple
        padding: (pad_h, pad_w) tuple
        stride: (stride_h, stride_w) tuple
        device: TT device
        tensor_map: Tensor cache map
        config_tensors_in_dram: Whether to store config tensors in DRAM
        shard_scheme: Sharding scheme to use
        ceil_mode: Whether to use ceiling mode for output shape
        count_include_pad: Whether to include padding in average calculation
    """
    in_n, in_c, in_h, in_w = input_shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    # Calculate output dimensions
    if ceil_mode:
        out_h = math.ceil((in_h + 2 * pad_h - kernel_h) / stride_h) + 1
        out_w = math.ceil((in_w + 2 * pad_w - kernel_w) / stride_w) + 1
    else:
        out_h = math.floor((in_h + 2 * pad_h - kernel_h) / stride_h) + 1
        out_w = math.floor((in_w + 2 * pad_w - kernel_w) / stride_w) + 1

    # Generate input tensor
    torch.manual_seed(42)
    torch_input = randomize_torch_tensor(tensor_map, input_shape)
    torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # N, H, W, C
    ttnn_input = ttnn.from_torch(torch_input_permuted, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Run TTNN avg_pool2d with config_tensors_in_dram
    ttnn_output = ttnn.avg_pool2d(
        input_tensor=ttnn_input,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        applied_shard_scheme=shard_scheme,
        deallocate_input=True,
        reallocate_halo_output=True,
        config_tensors_in_dram=config_tensors_in_dram,
    )

    # Run PyTorch avg_pool2d
    torch_output = torch.nn.AvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )(torch_input)

    # Convert TTNN output back to comparable format
    ttnn_output = ttnn.to_torch(ttnn_output)
    out_n, out_c = in_n, in_c
    ttnn_output = ttnn_output.reshape(out_n, out_h, out_w, out_c)  # N, H, W, C
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))  # N, C, H, W

    # Verify correctness with tolerance for avg pool
    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    assert torch.allclose(ttnn_output, torch_output, atol=atol, rtol=rtol), "Avg pool output mismatch"


# Test parameters for config_tensors_in_dram tests
max_pool_test_configs = [
    # Basic tests with height sharding
    # [in_n, in_c, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w]
    [1, 64, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1],  # ResNet-style pool
    [1, 128, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1],  # Simple 2x2 pool
    [1, 32, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1],  # Larger input
]

avg_pool_test_configs = [
    # Basic tests for avg pool
    # [in_n, in_c, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w]
    [1, 64, 56, 56, 3, 3, 2, 2, 1, 1],  # ResNet-style pool
    [1, 128, 28, 28, 2, 2, 2, 2, 0, 0],  # Simple 2x2 pool
]


@pytest.mark.parametrize("config_tensors_in_dram", [False, True])
@pytest.mark.parametrize("input_spec", max_pool_test_configs)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_config_tensors_in_dram(device, input_spec, config_tensors_in_dram, tensor_map):
    """
    Test max_pool2d with config_tensors_in_dram flag.

    Verifies that the pool operation produces correct results both when
    config tensors are stored in L1_SMALL (False) and DRAM (True).
    """
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
    ) = input_spec

    run_max_pool2d_with_config_tensors_in_dram(
        input_shape=[in_n, in_c, in_h, in_w],
        kernel_size=(kernel_h, kernel_w),
        padding=(pad_h, pad_w),
        stride=(stride_h, stride_w),
        dilation=(dilation_h, dilation_w),
        device=device,
        tensor_map=tensor_map,
        config_tensors_in_dram=config_tensors_in_dram,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )


@pytest.mark.parametrize("config_tensors_in_dram", [False, True])
@pytest.mark.parametrize("input_spec", avg_pool_test_configs)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_avg_pool2d_config_tensors_in_dram(device, input_spec, config_tensors_in_dram, tensor_map):
    """
    Test avg_pool2d with config_tensors_in_dram flag.

    Verifies that the pool operation produces correct results both when
    config tensors are stored in L1_SMALL (False) and DRAM (True).
    """
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    ) = input_spec

    run_avg_pool2d_with_config_tensors_in_dram(
        input_shape=[in_n, in_c, in_h, in_w],
        kernel_size=(kernel_h, kernel_w),
        padding=(pad_h, pad_w),
        stride=(stride_h, stride_w),
        device=device,
        tensor_map=tensor_map,
        config_tensors_in_dram=config_tensors_in_dram,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )


@pytest.mark.parametrize(
    "shard_scheme",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        # BLOCK_SHARDED with config_tensors_in_dram=True is not yet supported due to
        # complex core-to-page mapping that depends on shard orientation. Use
        # config_tensors_in_dram=False for BLOCK_SHARDED configurations.
        pytest.param(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            marks=pytest.mark.skip(reason="BLOCK_SHARDED not supported with config_tensors_in_dram=True"),
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_config_tensors_in_dram_sharding_schemes(device, shard_scheme, tensor_map):
    """
    Test max_pool2d with config_tensors_in_dram=True across different sharding schemes.

    This ensures the DRAM config tensor feature works correctly with all supported
    sharding configurations.
    """
    # Use appropriate test configs for each sharding scheme
    if shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        input_spec = [1, 64, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1]
    elif shard_scheme == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        input_spec = [1, 8192, 8, 8, 2, 2, 1, 1, 0, 0, 1, 1]
    else:  # BLOCK_SHARDED
        input_spec = [1, 2048, 16, 16, 2, 2, 1, 1, 0, 0, 1, 1]

    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
    ) = input_spec

    run_max_pool2d_with_config_tensors_in_dram(
        input_shape=[in_n, in_c, in_h, in_w],
        kernel_size=(kernel_h, kernel_w),
        padding=(pad_h, pad_w),
        stride=(stride_h, stride_w),
        dilation=(dilation_h, dilation_w),
        device=device,
        tensor_map=tensor_map,
        config_tensors_in_dram=True,
        shard_scheme=shard_scheme,
    )

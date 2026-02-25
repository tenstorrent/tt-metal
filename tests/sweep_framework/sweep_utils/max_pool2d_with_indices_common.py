# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import itertools
import random
import torch
import math

import ttnn
from ttnn.operations.pool import validate_maxpool2d_indices


def run_max_pool2d_with_indices(
    in_n,
    in_c,
    in_h,
    in_w,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_tb,
    pad_lr,
    dilation_h,
    dilation_w,
    ttnn_dtype,
    device,
    sharding=None,
    ceil_mode=False,
    memory_config=None,
    run_twice=False,
    dram_slice_config=None,
    config_tensor_in_dram=False,
):
    kernel_size = [kernel_h, kernel_w]
    stride = [stride_h, stride_w]
    padding = [pad_tb, pad_lr]
    dilation = [dilation_h, dilation_w]
    pad_h = pad_tb * 2  # total padding
    pad_w = pad_lr * 2  # total padding

    if ceil_mode:
        out_h = math.ceil((in_h + pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.ceil((in_w + pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1
        if ((out_h - 1) * stride_h) >= (in_h + pad_tb):
            ceil_mode_out_shape_adj = True
            out_h -= 1
        if ((out_w - 1) * stride_w) >= (in_w + pad_lr):
            ceil_mode_out_shape_adj = True
            out_w -= 1
    else:
        out_h = math.floor((in_h + pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.floor((in_w + pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    tensor_shape = (in_n, in_c, in_h, in_w)
    ttnn_input_shape = (1, 1, in_n * in_h * in_w, in_c)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    # torch_input = torch.zeros(tensor_shape, dtype=torch.bfloat16)
    # for n in range(in_n):
    #     for c in range(in_c):
    #         for h in range(in_h):
    #             for w in range(in_w):
    #                 torch_input[n, c, h, w] = h * in_w + w
    torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # N, H, W, C
    torch_input_reshaped = torch_input_permuted.reshape(ttnn_input_shape)  # NHW, C
    ttnn_layout = ttnn.ROW_MAJOR_LAYOUT
    if ttnn_dtype == ttnn.bfloat8_b:
        ttnn_layout = ttnn.TILE_LAYOUT

    ttnn_input = ttnn.from_torch(
        torch_input_reshaped, ttnn_dtype, layout=ttnn_layout, memory_config=memory_config, device=device
    )

    ttnn_output, ttnn_indices = ttnn.max_pool2d(
        input_tensor=ttnn_input,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        applied_shard_scheme=sharding,
        ceil_mode=ceil_mode,
        deallocate_input=False,
        reallocate_halo_output=True,
        return_indices=True,
        dram_slice_config=dram_slice_config,
        config_tensor_in_dram=config_tensor_in_dram,
    )

    if run_twice:
        ttnn.deallocate(ttnn_output, True)
        ttnn.deallocate(ttnn_indices, True)
        ttnn_output, ttnn_indices = ttnn.max_pool2d(
            input_tensor=ttnn_input,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            applied_shard_scheme=sharding,
            ceil_mode=ceil_mode,
            deallocate_input=False,
            reallocate_halo_output=True,
            return_indices=True,
            config_tensor_in_dram=config_tensor_in_dram,
        )

    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # convert indexes to int64 for compatibility with torch
    ttnn_indices_torch = ttnn.to_torch(ttnn_indices, dtype=torch.int64)

    # manually fix the wrapping since TTNN uint16/uint32 tensors get converted to int16/int32 torch tensors
    # even when data type is specified as int64
    if ttnn_indices.dtype == ttnn.uint16:
        # uint16: wraps at 65536 (2^16)
        ttnn_indices_torch = torch.where(ttnn_indices_torch < 0, ttnn_indices_torch + 65536, ttnn_indices_torch)
    elif ttnn_indices.dtype == ttnn.uint32:
        # uint32: wraps at 4294967296 (2^32)
        ttnn_indices_torch = torch.where(ttnn_indices_torch < 0, ttnn_indices_torch + 4294967296, ttnn_indices_torch)

    torch_output, torch_indices = torch.nn.functional.max_pool2d(
        torch_input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )

    # Reshape torch output to match TTNN format (NCHW -> NHWC)
    torch_output_reshaped = torch_output.permute(0, 2, 3, 1)  # N, H, W, C
    torch_indices_reshaped = torch_indices.permute(0, 2, 3, 1)  # N, H, W, C

    # Reshape TTNN outputs to match PyTorch shape for comparison
    # TTNN output is in shape (1, 1, in_n * out_h * out_w, channels)
    ttnn_output_reshaped = ttnn_output_torch.reshape(in_n, out_h, out_w, in_c)
    ttnn_indices_reshaped = ttnn_indices_torch.reshape(in_n, out_h, out_w, in_c)

    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if ttnn_dtype == ttnn.bfloat8_b:
        atol = 0.35
    output_match = torch.allclose(ttnn_output_reshaped, torch_output_reshaped, atol=atol, rtol=rtol)

    (
        indices_valid,
        tie_breaking_differences,
        actual_errors,
        value_differences,
        window_violations,
    ) = validate_maxpool2d_indices(
        torch_input_permuted,
        torch_indices_reshaped,
        ttnn_indices_reshaped,
        kernel_size,
        stride,
        padding,
        dilation,
        ttnn_dtype,
    )

    test_passed = output_match and indices_valid

    print(
        "Results: ",
        {
            "test_passed": test_passed,
            "output_match": output_match,
            "indices_valid": indices_valid,
            "tie_breaking_differences": tie_breaking_differences,
            "actual_errors": actual_errors,
            "value_differences": value_differences,
            "window_violations": window_violations,
        },
    )

    assert test_passed

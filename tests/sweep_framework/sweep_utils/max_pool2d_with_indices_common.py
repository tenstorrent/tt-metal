# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import itertools
import random
import torch
import math

import ttnn


def validate_indices(input_tensor, torch_indices, ttnn_indices, kernel_size, stride, padding, dilation, dtype):
    """
    Validate indices using logic from test_mpwi.py
    Note input tensors should be in [N, H, W, C] format
    Returns (indices_valid, tie_breaking_differences, actual_errors, value_differences, window_violations)
    """
    batch_size, input_h, input_w, channels = input_tensor.shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_tb, pad_lr = padding
    dilation_h, dilation_w = dilation

    # Check if indices are exactly equal first
    indices_match = torch.equal(torch_indices, ttnn_indices)
    if indices_match:
        return True, 0, 0, 0, 0

    # Find positions where indices don't match
    diff = torch.abs(torch_indices - ttnn_indices)
    mismatch_positions = torch.nonzero(diff, as_tuple=False)
    num_mismatches = len(mismatch_positions)

    tie_breaking_differences = 0
    actual_errors = 0
    value_differences = 0
    window_violations = 0
    for pos in mismatch_positions:
        n, h, w, c = pos
        torch_idx = torch_indices[n, h, w, c]
        ttnn_idx = ttnn_indices[n, h, w, c]

        # Convert linear indices to spatial coordinates
        torch_h = torch_idx // input_w
        torch_w = torch_idx % input_w
        ttnn_h = ttnn_idx // input_w
        ttnn_w = ttnn_idx % input_w

        # Get input values at these positions
        if ttnn_h >= 0 and ttnn_w >= 0 and ttnn_h < input_h and ttnn_w < input_w:
            torch_input_val = input_tensor[n, torch_h, torch_w, c]
            ttnn_input_val = input_tensor[n, ttnn_h, ttnn_w, c]
        else:
            actual_errors += 1
            window_violations += 1
            continue

        # Check if this is a valid tie-breaking difference
        # Two conditions must be satisfied:
        # 1. The values must be the same
        # 2. Both indices must be within the same kernel window

        # Check if values are the same
        atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
        if dtype == ttnn.bfloat8_b:
            atol = 0.35
            values_same = math.isclose(torch_input_val, ttnn_input_val, abs_tol=atol, rel_tol=rtol)
        else:
            values_same = torch_input_val == ttnn_input_val

        # Check if both indices are within the same kernel window
        def is_in_dilated_kernel_window(
            input_h, input_w, kernel_top_left_h, kernel_top_left_w, kernel_h, kernel_w, dilation_h, dilation_w
        ):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    kernel_pos_h = kernel_top_left_h + kh * dilation_h
                    kernel_pos_w = kernel_top_left_w + kw * dilation_w
                    if kernel_pos_h == input_h and kernel_pos_w == input_w:
                        return True
            return False

        kernel_top_left_h = h * stride_h - pad_tb
        kernel_top_left_w = w * stride_w - pad_lr
        ttnn_in_window = is_in_dilated_kernel_window(
            ttnn_h, ttnn_w, kernel_top_left_h, kernel_top_left_w, kernel_h, kernel_w, dilation_h, dilation_w
        )

        if values_same and ttnn_in_window:
            tie_breaking_differences += 1
        elif not ttnn_in_window:
            actual_errors += 1
            window_violations += 1
        else:
            actual_errors += 1
            value_differences += 1

    # Indices are valid if there are no actual errors
    assert num_mismatches == (
        tie_breaking_differences + actual_errors
    ), "Total mismatches should equal sum of tie-breaking differences and actual errors"
    if actual_errors > 0:
        assert actual_errors == (
            value_differences + window_violations
        ), "Actual errors should equal sum of value differences and window violations"
    else:
        assert actual_errors == 0 and value_differences == 0 and window_violations == 0, "No errors should be present"
    return (actual_errors == 0), tie_breaking_differences, actual_errors, value_differences, window_violations


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
    # convert indexes to int64 for compatability with torch
    ttnn_indices_torch = ttnn.to_torch(ttnn_indices, dtype=torch.int64)
    # manually fix the wrapping since TTNN uint16 tensors get converted to int16 torch tensors, even when data type is specified as int64
    ttnn_indices_torch = torch.where(ttnn_indices_torch < 0, ttnn_indices_torch + 65536, ttnn_indices_torch)

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

    indices_valid, tie_breaking_differences, actual_errors, value_differences, window_violations = validate_indices(
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

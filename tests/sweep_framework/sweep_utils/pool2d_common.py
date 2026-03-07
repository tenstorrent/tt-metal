# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Shared utility functions for max_pool2d and avg_pool2d tests.
"""

import math
import torch
import ttnn


def randomize_tensor(tensor_map, tensor_shape):
    """Get or create a random tensor, caching by shape to avoid regeneration."""
    tensor_shape = tuple(tensor_shape)
    if tensor_shape in tensor_map:
        return tensor_map[tensor_shape]
    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    tensor_map[tensor_shape] = torch_tensor
    return torch_tensor


def parse_padding(padding):
    """
    Parse 2D or 4D padding into individual pad values.

    Args:
        padding: Tuple of 2 (pad_h, pad_w) or 4 (pad_t, pad_b, pad_l, pad_r) values.

    Returns:
        (pad_t, pad_b, pad_l, pad_r, pad_h, pad_w, is_4d)
        where pad_h = total vertical padding, pad_w = total horizontal padding
    """
    if len(padding) == 2:
        pad_t = pad_b = padding[0]
        pad_l = pad_r = padding[1]
        pad_h = int(padding[0] * 2)
        pad_w = int(padding[1] * 2)
        is_4d = False
    elif len(padding) == 4:
        pad_t, pad_b, pad_l, pad_r = padding
        pad_h = pad_t + pad_b
        pad_w = pad_l + pad_r
        is_4d = True
    else:
        raise ValueError(f"Padding must be 2D or 4D tuple, got {len(padding)}D")
    return pad_t, pad_b, pad_l, pad_r, pad_h, pad_w, is_4d


def compute_output_shape(
    in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w, pad_h, pad_w, pad_t, pad_l, ceil_mode
):
    """
    Compute the output spatial dimensions for a pool2d operation.

    Args:
        in_h, in_w: Input spatial dimensions
        kernel_h, kernel_w: Kernel size
        stride_h, stride_w: Stride
        dilation_h, dilation_w: Dilation
        pad_h, pad_w: Total padding (vertical, horizontal)
        pad_t, pad_l: Top and left padding (for ceil_mode adjustment)
        ceil_mode: Whether to use ceiling for output shape

    Returns:
        (out_h, out_w, ceil_mode_out_shape_adj)
    """
    ceil_mode_out_shape_adj = False
    if ceil_mode:
        out_h = math.ceil((in_h + pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.ceil((in_w + pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1
        if ((out_h - 1) * stride_h) >= (in_h + pad_t):
            ceil_mode_out_shape_adj = True
            out_h -= 1
        if ((out_w - 1) * stride_w) >= (in_w + pad_l):
            ceil_mode_out_shape_adj = True
            out_w -= 1
    else:
        out_h = math.floor((in_h + pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.floor((in_w + pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1
    return out_h, out_w, ceil_mode_out_shape_adj


def prepare_torch_pool_input(input_tensor, batch_size, input_h, input_w, channels, padding, pad_fill_value):
    """
    Shared input preparation for pool2d golden functions.

    Reshapes (1, 1, N*H*W, C) -> (N, C, H, W) and converts padding:
    - 4D padding [pad_t, pad_b, pad_l, pad_r]: applies torch.nn.functional.pad manually, returns torch_padding=0
    - 2D padding (pad_h, pad_w): passes through as-is

    Returns:
        (input_nchw, torch_padding)
    """
    input_nchw = input_tensor.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)

    if isinstance(padding, (list, tuple)) and len(padding) == 4:
        pad_t, pad_b, pad_l, pad_r = padding
        input_nchw = torch.nn.functional.pad(
            input_nchw, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=pad_fill_value
        )
        torch_padding = 0
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        torch_padding = padding
    else:
        torch_padding = padding

    return input_nchw, torch_padding


def pool_output_to_flat_nhwc(output_tensor):
    """Convert pool output from (N, C, H, W) -> (1, 1, N*H*W, C)."""
    N, C, H, W = output_tensor.shape
    return output_tensor.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)


def validate_maxpool2d_indices(
    input_tensor, torch_indices, ttnn_indices, kernel_size, stride, padding, dilation, dtype
):
    """
    Validate indices by checking if differences are due to valid tie-breaking.
    Note: input tensors should be in [N, H, W, C] format.
    Supports both uint16 and uint32 index tensors (indices should be converted to int64 before calling).
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

    out_h, out_w, _ = compute_output_shape(
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        pad_h,
        pad_w,
        pad_tb,
        pad_lr,
        ceil_mode,
    )

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    tensor_shape = (in_n, in_c, in_h, in_w)
    ttnn_input_shape = (1, 1, in_n * in_h * in_w, in_c)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
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

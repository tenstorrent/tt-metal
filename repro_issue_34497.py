#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reproduction script for issue #34497: Grouped transposed convolution error
https://github.com/tenstorrent/tt-metal/issues/34497

Error: RuntimeError: TT_FATAL @ conv2d_utils.cpp:350: channels % num_cores_channels == 0
Channels: 1088, num core channels: 7
"""

import torch
import ttnn


def repro_grouped_conv_transpose2d_issue():
    """
    Reproduces the grouped transposed convolution error with the following parameters:
    - Input: N=1, H=W=7, C_in=1088
    - Weights: groups=17, kernel 3×3, stride 2, padding 1, output_padding 1
    - Output: N=1, H=W=14, C_out=1088
    - Data types: bfloat16 and bfloat8_b
    - Layout: ROW_MAJOR_LAYOUT
    """

    # Parameters from the issue
    batch_size = 1
    input_height = 7
    input_width = 7
    input_channels = 1088
    output_channels = 1088
    filter_height = 3
    filter_width = 3
    stride_h = 2
    stride_w = 2
    pad_h = 1
    pad_w = 1
    out_pad_h = 1
    out_pad_w = 1
    groups = 17

    print(f"Testing grouped transposed convolution with:")
    print(f"  Input: N={batch_size}, H={input_height}, W={input_width}, C_in={input_channels}")
    print(f"  Weights: groups={groups}, kernel {filter_height}×{filter_width}")
    print(f"  Stride: {stride_h}, Padding: {pad_h}, Output padding: {out_pad_h}")
    print(f"  Expected output: N={batch_size}, H=14, W=14, C_out={output_channels}")
    print()

    # Initialize device with l1_small_size (required for conv operations)
    device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)

    try:
        # Create input tensor
        torch.manual_seed(0)
        conv_input_shape = [batch_size, input_channels, input_height, input_width]
        conv_weight_shape = [input_channels, output_channels // groups, filter_height, filter_width]
        conv_bias_shape = [1, 1, 1, output_channels]

        torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
        torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))  # NHWC

        torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
        torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()

        # Convert to ttnn tensors
        tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
        tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16)
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

        # Setup conv config
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
        )

        print("Calling ttnn.conv_transpose2d...")

        # This should trigger the error
        result = ttnn.conv_transpose2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight_tensor,
            in_channels=input_channels,
            out_channels=output_channels,
            device=device,
            bias_tensor=tt_bias_tensor,
            kernel_size=(filter_height, filter_width),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            output_padding=(out_pad_h, out_pad_w),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            groups=groups,
            dtype=ttnn.bfloat16,
        )

        print("SUCCESS: Conv transpose2d completed without error!")
        print(f"Result: {result}")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        ttnn.close_device(device)

    return True


def repro_secondary_issue():
    """
    Reproduces the secondary issue where C_in and C_out are 119 (divisible by 7)
    This should produce "Fatal Python error: Floating point exception"
    """

    # Parameters modified to be divisible by 7
    batch_size = 1
    input_height = 7
    input_width = 7
    input_channels = 119  # 17 * 7
    output_channels = 119
    filter_height = 3
    filter_width = 3
    stride_h = 2
    stride_w = 2
    pad_h = 1
    pad_w = 1
    out_pad_h = 1
    out_pad_w = 1
    groups = 17

    print(f"\nTesting secondary issue with divisible channels:")
    print(f"  Input: N={batch_size}, H={input_height}, W={input_width}, C_in={input_channels}")
    print(f"  Weights: groups={groups}, kernel {filter_height}×{filter_width}")
    print(f"  Note: {input_channels} % 7 = {input_channels % 7} (divisible by 7)")
    print()

    # Initialize device with l1_small_size (required for conv operations)
    device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)

    try:
        # Create input tensor
        torch.manual_seed(0)
        conv_input_shape = [batch_size, input_channels, input_height, input_width]
        conv_weight_shape = [input_channels, output_channels // groups, filter_height, filter_width]
        conv_bias_shape = [1, 1, 1, output_channels]

        torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
        torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))  # NHWC

        torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
        torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()

        # Convert to ttnn tensors
        tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
        tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16)
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

        # Setup conv config
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
        )

        print("Calling ttnn.conv_transpose2d...")

        # This should trigger the floating point exception
        result = ttnn.conv_transpose2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight_tensor,
            in_channels=input_channels,
            out_channels=output_channels,
            device=device,
            bias_tensor=tt_bias_tensor,
            kernel_size=(filter_height, filter_width),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            output_padding=(out_pad_h, out_pad_w),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            groups=groups,
            dtype=ttnn.bfloat16,
        )

        print("SUCCESS: Conv transpose2d completed without error!")
        print(f"Result: {result}")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        ttnn.close_device(device)

    return True


if __name__ == "__main__":
    print("=" * 80)
    print("Reproducing GitHub Issue #34497")
    print("=" * 80)
    print()

    # Test the main issue
    print("TEST 1: Main issue (C_in=1088, groups=17)")
    print("-" * 80)
    success1 = repro_grouped_conv_transpose2d_issue()

    print()
    print("=" * 80)

    # Test the secondary issue
    print("TEST 2: Secondary issue (C_in=119, divisible by 7)")
    print("-" * 80)
    success2 = repro_secondary_issue()

    print()
    print("=" * 80)
    print("SUMMARY:")
    print(f"  Test 1 (C_in=1088): {'PASSED' if success1 else 'FAILED'}")
    print(f"  Test 2 (C_in=119): {'PASSED' if success2 else 'FAILED'}")
    print("=" * 80)

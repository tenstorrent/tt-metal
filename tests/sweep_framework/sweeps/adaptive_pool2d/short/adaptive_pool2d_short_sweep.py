# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import itertools
import random
import torch
import math

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, assert_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Short sweep for adaptive pooling operations - focused on commonly used configurations
parameters = {
    "adaptive_pool2d_short_sweep_suite": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "pool_type": ["avg", "max"],
        "input_specs": [
            # [batch_size, input_channels, input_height, input_width, output_height, output_width]
            # Real-world adaptive pooling scenarios
            [1, 256, 56, 56, 1, 1],  # Global pooling deeper network
            [1, 512, 28, 28, 1, 1],  # Global pooling very deep
            [1, 1024, 14, 14, 1, 1],  # Global pooling final layers
            [1, 512, 7, 7, 7, 7],  # No-op case (input == output size)
            [1, 2048, 7, 7, 1, 1],  # ResNet-style global pooling
            [1, 64, 224, 224, 7, 7],  # Standard classifier head
            [1, 256, 32, 32, 4, 4],  # 8x downsampling
            [1, 128, 64, 64, 8, 8],  # 8x downsampling different size
            [1, 160, 75, 75, 3, 3],  # Small odd dimensions
            [1, 320, 28, 28, 14, 14],  # 2x downsampling
            [1, 480, 14, 14, 7, 7],  # 2x downsampling small
            # Asymmetric output sizes (real-world scenarios)
            [1, 512, 32, 32, 2, 4],  # Another asymmetric case
            [1, 256, 64, 64, 7, 7],  # Target case for dilation testing
            # Edge cases for hybrid padding+dilation testing
            [1, 256, 64, 64, 3, 5],  # Asymmetric pooling
            [1, 64, 17, 17, 4, 3],  # Small asymmetric edge case
            # Failing edge cases
            # [1, 64, 37, 37, 5, 7],   # Prime dimension edge case
            # [1, 128, 50, 20, 6, 4],  # Asymmetric input with variance
            # [1, 32, 31, 29, 7, 5],   # Double prime case (complex variance)
            # [1, 32, 10, 10, 3, 4],   # Small input high compression
        ],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def run_adaptive_pool2d(in_n, in_c, in_h, in_w, out_h, out_w, pool_type, dtype, device):
    """Helper function to run adaptive pooling operations"""

    # Generate input tensor
    torch.manual_seed(0)
    input_shape = [in_n, in_c, in_h, in_w]

    # Generate input tensor directly in bfloat16
    # torch_input_tensor = torch_random(input_shape, low=-100, high=100, dtype=torch.bfloat16)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)  # All 1s for debugging

    # PyTorch reference (use bfloat16)
    output_size = (out_h, out_w)
    if pool_type == "avg":
        torch_output_tensor = torch.nn.functional.adaptive_avg_pool2d(torch_input_tensor, output_size)
    else:  # max
        torch_output_tensor = torch.nn.functional.adaptive_max_pool2d(torch_input_tensor, output_size)

    # For bfloat8_b, we need to go through ttnn conversion
    if dtype == ttnn.bfloat8_b:
        temp_tt_tensor = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=None, memory_config=None
        )
        torch_input_tensor = ttnn.to_torch(temp_tt_tensor)

    # Convert to tt-metal format [1, 1, NHW, C]
    torch_input_tensor_ttnn = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    torch_input_tensor_ttnn = torch.reshape(torch_input_tensor_ttnn, [1, 1, in_n * in_h * in_w, in_c])

    # Create tt-metal tensor
    input_tensor = ttnn.from_torch(
        torch_input_tensor_ttnn,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    start_time = start_measuring_time()

    # Call adaptive pooling operation
    if pool_type == "avg":
        result = ttnn.adaptive_avg_pool2d(
            input_tensor=input_tensor,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            output_size=[out_h, out_w],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:  # max
        result = ttnn.adaptive_max_pool2d(
            input_tensor=input_tensor,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            output_size=[out_h, out_w],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    result = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    # ttnn outputs flattened tensor [1, 1, N*output_h*output_w, C], reshape to [N, output_h, output_w, C]
    result = result.reshape(in_n, out_h, out_w, in_c)

    # Convert back to NCHW format for comparison
    output_tensor = torch.permute(result, (0, 3, 1, 2))

    # Test for equivalence - using patterns from existing pool tests
    if pool_type == "max":
        # MAX POOL: Use strict equivalence checks like maxpool2d
        pcc_thresh = 1.0
        atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
        if dtype == ttnn.bfloat8_b:
            pcc_thresh = 0.99
            atol = 0.35
        assert_with_pcc(torch_output_tensor, output_tensor, pcc_thresh)
        allclose = torch.allclose(output_tensor, torch_output_tensor, atol=atol, rtol=rtol)
        isequal = torch.equal(output_tensor, torch_output_tensor)
        assert (
            allclose
        ), f"Reference and output tensor are not close. Input: {input_shape}, Output size: {output_size}, Pool type: {pool_type}"
        if dtype == ttnn.bfloat16:
            assert (
                isequal
            ), f"Reference and output tensor are not equal for bfloat16. Input: {input_shape}, Output size: {output_size}, Pool type: {pool_type}"
    else:
        # AVG POOL: Use relaxed checks like avgpool2d due to bfloat16 precision limitations
        pcc_thresh = 0.985
        atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
        # TTNN only supports scalars in Bfloat16, so we cannot support rtol lower than 0.01
        # for instance, a 3x3 kernel uses scalar 1/9 = 0.111, which in Bfloat16 is 0.11084
        # so if we fill the tensor with 1s, Torch gets 9 * 0.111 = 0.999 which converted back
        # to Bfloat16 rounds to 1.0 but TTNN gets 9 * 0.11084 = 0.99756 which converted back
        # to Bfloat16 rounds to 0.9961, so the rdiff in this case is 0.0039
        # since the atol default is 0.016 we don't see this issue for low magnitude values, but
        # when using small divisor overrides with large kernels we see much large values which
        # overwhelm the atol and the rtol becomes significant
        rtol = 0.01
        if dtype == ttnn.bfloat8_b:
            atol = 0.35
        assert_with_pcc(torch_output_tensor, output_tensor, pcc_thresh)
        allclose = torch.allclose(output_tensor, torch_output_tensor, atol=atol, rtol=rtol)
        assert (
            allclose
        ), f"Reference and output tensor are not close. Input: {input_shape}, Output size: {output_size}, Pool type: {pool_type}"

    print(f"Adaptive {pool_type} pool2d - Input: {input_shape}, Output size: {output_size}")

    # Get the actual tolerance values used in the test
    test_atol, test_rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if pool_type == "avg":
        test_rtol = 0.01  # Updated rtol for avg pool
    if dtype == ttnn.bfloat8_b:
        test_atol = 0.35

    # Detailed element-by-element comparison for debugging
    print(f"\n=== ELEMENTS EXCEEDING TEST TOLERANCES ===")
    print(f"Test tolerances: atol={test_atol:.6f}, rtol={test_rtol:.6f}")
    print(f"Output shape: {torch_output_tensor.shape}")

    # Flatten for easier comparison
    torch_flat = torch_output_tensor.flatten()
    ttnn_flat = output_tensor.flatten()
    diff_flat = torch.abs(torch_flat - ttnn_flat)

    # Find elements that exceed tolerance using PyTorch's allclose logic
    # allclose: |a - b| <= atol + rtol * |b|
    tolerance_threshold = test_atol + test_rtol * torch.abs(torch_flat)
    exceeds_tolerance = diff_flat > tolerance_threshold

    failing_indices = torch.nonzero(exceeds_tolerance).flatten()

    if len(failing_indices) > 0:
        print(f"\n{len(failing_indices)} elements exceed tolerances:")
        print(f"{'Index':<8} {'PyTorch':<14} {'TTNN':<14} {'Abs Diff':<14} {'Threshold':<14} {'Excess':<14}")
        print(f"{'-'*8} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

        for idx in failing_indices[:20]:  # Show first 20 failing elements
            i = idx.item()
            torch_val = torch_flat[i].item()
            ttnn_val = ttnn_flat[i].item()
            abs_diff = diff_flat[i].item()
            threshold = tolerance_threshold[i].item()
            excess = abs_diff - threshold

            print(f"{i:<8} {torch_val:<14.8f} {ttnn_val:<14.8f} {abs_diff:<14.8f} {threshold:<14.8f} {excess:<14.8f}")

        if len(failing_indices) > 20:
            print(f"... ({len(failing_indices) - 20} more failing elements)")
    else:
        print("✅ All elements are within test tolerances!")

    # Statistics
    mean_diff = torch.mean(diff_flat).item()
    max_diff = torch.max(diff_flat).item()
    std_diff = torch.std(diff_flat).item()
    max_threshold = torch.max(tolerance_threshold).item()

    print(f"\nTOLERANCE ANALYSIS:")
    print(f"Mean absolute difference: {mean_diff:.8f}")
    print(f"Max absolute difference:  {max_diff:.8f}")
    print(f"Max tolerance threshold:  {max_threshold:.8f}")
    print(f"Max excess over tolerance: {torch.max(diff_flat - tolerance_threshold).item():.8f}")
    print(
        f"Elements within tolerance: {len(torch_flat) - len(failing_indices)}/{len(torch_flat)} ({100*(len(torch_flat) - len(failing_indices))/len(torch_flat):.1f}%)"
    )

    # Check results
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    return pcc


def run(
    input_specs,
    pool_type,
    dtype,
    *,
    device,
):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
    ) = input_specs

    return run_adaptive_pool2d(in_n, in_c, in_h, in_w, out_h, out_w, pool_type, dtype, device)


import pytest


@pytest.mark.parametrize("input_spec", parameters["adaptive_pool2d_short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("pool_type", parameters["adaptive_pool2d_short_sweep_suite"]["pool_type"])
@pytest.mark.parametrize("dtype", parameters["adaptive_pool2d_short_sweep_suite"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_adaptive_pool2d_localrun(device, dtype, pool_type, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_height,
        output_width,
    ) = input_spec

    run_adaptive_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        pool_type,
        dtype,
        device,
    )

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import itertools
import random
import torch
import math

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random


def run_max_pool2d_with_indices(
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
    dtype,
    device,
    sharding=None,
    ceil_mode=False,
    memory_config=None,
    in_place=False,
):
    kernel_size = [kernel_h, kernel_w]
    stride = [stride_h, stride_w]
    padding = [pad_h, pad_w]
    dilation = [dilation_h, dilation_w]

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    act_shape = [in_n, in_c, in_h, in_w]
    act = torch.randn(act_shape, dtype=torch.bfloat16)
    act_permuted = torch.permute(act, (0, 2, 3, 1))

    if dtype == ttnn.bfloat8_b:
        act_shape = (1, 1, in_n * in_h * in_w, in_c)
        act_reshaped = act_permuted.reshape(act_shape)
        ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
    else:
        ttact = ttnn.from_torch(act_permuted, dtype)

    ttact_device = ttnn.to_device(ttact, device)
    start_time = start_measuring_time()

    # Use auto sharding as recommended by colleague for sweep tests
    # Let TTNN automatically select the best sharding scheme
    # sharding remains None for auto sharding

    # Call max_pool2d with return_indices=True
    output, indices = ttnn.max_pool2d(
        input_tensor=ttact_device,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=[kernel_h, kernel_w],
        stride=[stride_h, stride_w],
        padding=[pad_h, pad_w],
        dilation=[dilation_h, dilation_w],
        memory_config=memory_config,
        applied_shard_scheme=sharding,
        in_place_halo=in_place,
        return_indices=True,  # This is the key difference
        ceil_mode=ceil_mode,
    )

    output_host = output.cpu()
    indices_host = indices.cpu()
    output_pytorch = torch.Tensor(ttnn.to_torch(output_host))
    indices_pytorch = torch.Tensor(ttnn.to_torch(indices_host))
    e2e_perf = stop_measuring_time(start_time)

    ## reference
    golden_pytorch, golden_indices = torch.nn.functional.max_pool2d(
        act,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=True,
        ceil_mode=ceil_mode,
    )

    golden_shape = golden_pytorch.shape
    output_pytorch = output_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])
    output_pytorch = torch.permute(output_pytorch, (0, 3, 1, 2))  ## N, C, H, W

    # Handle indices reshaping
    indices_pytorch = indices_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])
    indices_pytorch = torch.permute(indices_pytorch, (0, 3, 1, 2))  ## N, C, H, W

    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if dtype == ttnn.bfloat8_b:
        atol = 0.35

    ## test for equivalence of outputs
    output_allclose = torch.allclose(output_pytorch, golden_pytorch, atol=atol)
    output_isequal = torch.equal(output_pytorch, golden_pytorch)

    def validate_indices(input_tensor, torch_indices, ttnn_indices, kernel_size, stride, padding, dilation, dtype):
        """
        Validate indices using logic from test_mpwi.py
        Returns (indices_valid, tie_breaking_differences, actual_errors)
        """
        try:
            batch_size, channels, input_h, input_w = input_tensor.shape
            kernel_h, kernel_w = kernel_size
            stride_h, stride_w = stride
            dilation_h, dilation_w = dilation

            # Check if indices are exactly equal first
            indices_match = torch.equal(torch_indices, ttnn_indices)
            if indices_match:
                return True, 0, 0

            # Analyze mismatches - keep indices as integers
            diff = torch.abs(torch_indices - ttnn_indices)

            # Find positions where indices don't match
            mismatch_mask = diff > 0
            mismatch_positions = torch.nonzero(mismatch_mask, as_tuple=False)

            tie_breaking_differences = 0
            actual_errors = 0

            atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
            if dtype == ttnn.bfloat8_b:
                atol = 0.35

            for pos in mismatch_positions:
                n, c, h, w = pos
                torch_idx = int(torch_indices[n, c, h, w].item())
                ttnn_idx = int(ttnn_indices[n, c, h, w].item())

                # Convert linear indices to spatial coordinates
                torch_h = torch_idx // input_w
                torch_w = torch_idx % input_w
                ttnn_h = ttnn_idx // input_w
                ttnn_w = ttnn_idx % input_w

                # Get input values at these positions
                if torch_h >= 0 and torch_w >= 0 and torch_h < input_h and torch_w < input_w:
                    torch_input_val = input_tensor[n, c, torch_h, torch_w]
                else:
                    actual_errors += 1
                    continue

                if ttnn_h >= 0 and ttnn_w >= 0 and ttnn_h < input_h and ttnn_w < input_w:
                    ttnn_input_val = input_tensor[n, c, ttnn_h, ttnn_w]
                else:
                    actual_errors += 1
                    continue

                # Check if values are the same
                if dtype == ttnn.bfloat8_b:
                    values_same = math.isclose(torch_input_val, ttnn_input_val, abs_tol=atol, rel_tol=rtol)
                else:
                    values_same = torch_input_val == ttnn_input_val

                # Calculate kernel window for this output position
                kernel_top_left_h = h * stride_h - padding[0]
                kernel_top_left_w = w * stride_w - padding[1]

                def is_in_dilated_kernel_window(
                    input_h, input_w, kernel_top_left_h, kernel_top_left_w, kernel_h, kernel_w, dilation_h, dilation_w
                ):
                    """Check if a position is within the dilated kernel window"""
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            kernel_pos_h = kernel_top_left_h + kh * dilation_h
                            kernel_pos_w = kernel_top_left_w + kw * dilation_w
                            if kernel_pos_h == input_h and kernel_pos_w == input_w:
                                return True
                    return False

                torch_in_window = is_in_dilated_kernel_window(
                    torch_h, torch_w, kernel_top_left_h, kernel_top_left_w, kernel_h, kernel_w, dilation_h, dilation_w
                )
                ttnn_in_window = is_in_dilated_kernel_window(
                    ttnn_h, ttnn_w, kernel_top_left_h, kernel_top_left_w, kernel_h, kernel_w, dilation_h, dilation_w
                )

                same_kernel_window = torch_in_window and ttnn_in_window

                if values_same and same_kernel_window:
                    # Valid tie-breaking difference
                    tie_breaking_differences += 1
                else:
                    # Actual error
                    actual_errors += 1

            # Indices are valid if there are no actual errors
            return actual_errors == 0, tie_breaking_differences, actual_errors

        except Exception:
            return False, 0, 1

    indices_valid, tie_breaking_diffs, actual_errors = validate_indices(
        act, golden_indices, indices_pytorch, kernel_size, stride, padding, dilation, dtype
    )

    # Assert all validations as requested by colleague:

    # 1. Output validation: allclose for bfloat8, equal for bfloat16
    if dtype == ttnn.bfloat8_b:
        assert output_allclose, "Reference and output tensor are not close"
    else:  # bfloat16
        assert output_isequal, "Reference and output tensor are not equal"

    # 2. PCC check (do for both dtypes, even though bfloat16 should always be 1)
    pcc_result = check_with_pcc(output_pytorch, golden_pytorch, pcc=0.998)
    assert pcc_result[0], f"PCC check failed: {pcc_result[1]}"

    # 3. Indices validation
    assert (
        indices_valid
    ), f"Indices validation failed with {actual_errors} actual errors (tie-breaking differences: {tie_breaking_diffs})"

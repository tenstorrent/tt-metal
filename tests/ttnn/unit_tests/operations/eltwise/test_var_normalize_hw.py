# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        # Standard tile-aligned shapes
        (1, 1, 32, 32),
        (1, 1, 64, 64),
        (2, 3, 32, 32),
        # Non-tile-aligned shapes from real demo vision/segmentation models
        (1, 1, 14, 14),  # MobileNetV2 Stage 4
        (1, 1, 28, 28),  # MobileNetV2 Stage 3
        (1, 1, 56, 56),  # ResNet Stage 2
        (1, 1, 30, 40),  # UNet bottleneck
        (1, 1, 7, 7),  # MobileNetV2 Stage 5
        (2, 4, 17, 23),  # Arbitrary unaligned prime shapes
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_var_hw_parity(device, shape, dtype):
    """
    Verifies that ttnn.var_hw accurately computes variance over H and W
    dimensions without underestimating variance or leaking tile padding on
    non-tile-aligned tensors.
    """
    torch.manual_seed(2026)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randn(shape, dtype=torch.float32)

    # Ground truth reference matching registered golden in ttnn/operations/unary.py:1056
    expected_var = torch.var(torch_input, dim=[-2, -1], keepdim=True, correction=0).to(torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input.to(torch_dtype),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )

    actual_ttnn = ttnn.var_hw(ttnn_input)
    actual_torch = ttnn.to_torch(actual_ttnn)

    assert_with_pcc(expected_var.float(), actual_torch.float(), pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 14, 14),  # MobileNetV2 Stage 4
        (1, 1, 28, 28),  # MobileNetV2 Stage 3
        (1, 1, 30, 40),  # UNet bottleneck
        (2, 3, 15, 25),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_std_hw_parity(device, shape, dtype):
    """
    Verifies that ttnn.std_hw produces parity with torch.std over H and W.
    """
    torch.manual_seed(2026)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randn(shape, dtype=torch.float32)

    expected_std = torch.std(torch_input, dim=[-2, -1], keepdim=True, correction=0).to(torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input.to(torch_dtype),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )

    actual_ttnn = ttnn.std_hw(ttnn_input)
    actual_torch = ttnn.to_torch(actual_ttnn)

    assert_with_pcc(expected_std.float(), actual_torch.float(), pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 14, 14),  # MobileNetV2 Stage 4
        (1, 1, 28, 28),  # MobileNetV2 Stage 3
        (1, 1, 30, 40),  # UNet bottleneck
        (2, 3, 21, 19),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_normalize_hw_parity(device, shape, dtype):
    """
    Verifies that ttnn.normalize_hw accurately normalizes activations without
    artificial inflation caused by padded tile divisor mismatch.
    """
    torch.manual_seed(2026)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randn(shape, dtype=torch.float32)

    mean_hw = torch.mean(torch_input, dim=[-2, -1], keepdim=True)
    std_hw = torch.std(torch_input, dim=[-2, -1], keepdim=True, correction=0)
    expected_norm = ((torch_input - mean_hw) / std_hw).to(torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input.to(torch_dtype),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )

    actual_ttnn = ttnn.normalize_hw(ttnn_input)
    actual_torch = ttnn.to_torch(actual_ttnn)

    assert_with_pcc(expected_norm.float(), actual_torch.float(), pcc=0.999)

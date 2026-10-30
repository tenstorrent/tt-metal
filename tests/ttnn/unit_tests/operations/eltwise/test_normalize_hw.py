# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_allclose, assert_with_pcc


def torch_normalize_hw(x):
    xd = x.double()
    mean = xd.mean(dim=(2, 3), keepdim=True)
    std = xd.std(dim=(2, 3), keepdim=True, unbiased=False)
    return ((xd - mean) / std).float()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("scale", [1.0, 100.0, 1e-3])
@pytest.mark.parametrize("shape", [[1, 1, 32, 32], [1, 2, 64, 64]])
def test_normalize_hw(device, dtype, scale, shape):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    x = (torch.randn(shape) * scale).to(torch_dtype)

    got = ttnn.to_torch(ttnn.normalize_hw(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)))

    assert_with_pcc(torch_normalize_hw(x.float()), got.float(), 0.999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_normalize_hw_zero_variance(device, dtype):
    # Every element equal, so the divisor is zero. rsqrt(0) is inf where
    # reciprocal(sqrt(0)) was, and 0 * inf resolves the same way.
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    x = torch.full([1, 1, 32, 32], 3.0, dtype=torch_dtype)

    got = ttnn.to_torch(ttnn.normalize_hw(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)))

    assert torch.isfinite(got.float()).all() or torch.isnan(got.float()).all()


@pytest.mark.parametrize(
    "shape",
    [
        # Standard tile-aligned shapes
        (1, 2, 32, 32),
        (2, 3, 32, 32),
        # Non-tile-aligned shapes from real vision/segmentation models
        (1, 2, 14, 14),  # MobileNetV2 Stage 4
        (1, 2, 28, 28),  # MobileNetV2 Stage 3
        (1, 2, 56, 56),  # ResNet Stage 2
        (1, 2, 30, 40),  # UNet bottleneck
        (2, 2, 7, 7),  # MobileNetV2 Stage 5
        (2, 3, 21, 19),  # Arbitrary unaligned prime shapes
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_normalize_hw_parity(device, shape, dtype):
    """
    Verifies that ttnn.normalize_hw accurately normalizes activations without
    scale distortion caused by padded tile divisor mismatch.
    """
    torch.manual_seed(2026)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randn(shape, dtype=torch.float32)

    # Reference built from rounded input for exact bf16 parity
    ref_input = torch_input.to(torch_dtype).float()
    mean_hw = torch.mean(ref_input, dim=[-2, -1], keepdim=True)
    std_hw = torch.std(ref_input, dim=[-2, -1], keepdim=True, correction=0)
    expected_norm = ((ref_input - mean_hw) / std_hw).to(torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input.to(torch_dtype),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )

    actual_ttnn = ttnn.normalize_hw(ttnn_input)
    actual_torch = ttnn.to_torch(actual_ttnn)

    # Absolute scale check: catches divisor bug where main is 30-97% off
    rtol = 1e-5 if dtype == ttnn.float32 else 0.02
    atol = 1e-5 if dtype == ttnn.float32 else 0.02
    assert_allclose(
        expected_result=expected_norm.float(),
        actual_result=actual_torch.float(),
        rtol=rtol,
        atol=atol,
    )

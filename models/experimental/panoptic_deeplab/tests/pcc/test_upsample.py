# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ...tt.tt_upsample import BilinearUpsampleMatmulTTNN
from ...tt.tt_upsample import BilinearUpsampleTorch


@pytest.mark.parametrize(
    "b, h, w, c, scale, channels_first",
    [
        (1, 128, 64, 32, 2, False),  # single batch, channels-last, scale 2
        # (1, 128, 64, 32, 4, False),  # single batch, channels-last, scale 4
        # (1, 128, 64, 32, 2, True),  # single batch, channels-first, scale 2
        # (1, 128, 64, 32, 4, True),  # single batch, channels-first, scale 4
        # (3, 64, 32, 16, 2, False),  # multi-batch, channels-last, scale 2
        # (3, 64, 32, 16, 4, True),  # multi-batch, channels-first, scale 4
        # (8, 32, 32, 8, 3, False),  # larger batch, channels-last, scale 3
    ],
)
def test_bilinear_upsample_matmul_vs_torch(b, h, w, c, scale, channels_first):
    torch.manual_seed(0)

    if channels_first:
        # Generate channels-first input: (B, C, H, W)
        img_torch = torch.rand(b, c, h, w, dtype=torch.float32)
        img_nchw = img_torch  # Already (B, C, H, W)
    else:
        # Generate channels-last input: (B, H, W, C)
        img_torch = torch.rand(b, h, w, c, dtype=torch.float32)
        img_nchw = img_torch.permute(0, 3, 1, 2)  # (B, C, H, W)

    # PyTorch matmul upsampling using the class
    upsampler = BilinearUpsampleTorch(h, w, scale=scale, channels_first=channels_first)
    out_torch = upsampler.forward(img_torch)

    # Torch bilinear upsampling (always works with channels-first)
    out_t = torch.nn.functional.interpolate(img_nchw, scale_factor=scale, mode="bilinear", align_corners=True)

    if channels_first:
        # Keep channels-first format
        out_t_formatted = out_t  # (B, C, H_out, W_out)
    else:
        # Convert back to channels-last format
        out_t_formatted = out_t.permute(0, 2, 3, 1)  # (B, H_out, W_out, C)

    # Validate numerical closeness
    torch.testing.assert_close(out_torch, out_t_formatted, rtol=1e-3, atol=1e-3)

    # Verify output shapes
    if channels_first:
        expected_shape = (b, c, h * scale, w * scale)
    else:
        expected_shape = (b, h * scale, w * scale, c)

    assert out_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {out_torch.shape}"


@pytest.mark.parametrize(
    "b, h, w, c, scale, channels_first",
    [
        # (1, 128, 64, 32, 2, False),  # single batch, channels-last, scale 2
        # (1, 128, 64, 32, 4, False),  # single batch, channels-last, scale 4
        # (1, 128, 64, 32, 2, True),  # single batch, channels-first, scale 2
        # (1, 128, 64, 32, 4, True),  # single batch, channels-first, scale 4
        # (3, 64, 32, 16, 2, False),  # multi-batch, channels-last, scale 2
        # (3, 64, 32, 16, 4, True),  # multi-batch, channels-first, scale 4
        # (8, 32, 32, 8, 3, False),  # larger batch, channels-last, scale 3
        (1, 128, 256, 32, 4, False)
    ],
)
def test_bilinear_upsample_ttnn_matmul_vs_ttnn_upsample(device, b, h, w, c, scale, channels_first):
    torch.manual_seed(0)

    # Generate input in NHWC format
    img_torch = torch.rand(b, h, w, c, dtype=torch.bfloat16)

    # Create TTNN tensor
    input_tensor = ttnn.from_torch(
        img_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Method 1: Custom matrix multiplication implementation (align_corners=True)
    upsampler = BilinearUpsampleMatmulTTNN(device, b, c, h, w, scale=scale)
    output_matmul = upsampler.forward(input_tensor)
    output_matmul_torch = ttnn.to_torch(output_matmul)

    # Method 2: PyTorch reference with align_corners=True
    img_torch_nchw = img_torch.permute(0, 3, 1, 2)
    torch_result_nchw = torch.nn.functional.interpolate(
        img_torch_nchw, scale_factor=scale, mode="bilinear", align_corners=True
    )
    torch_result_nhwc = torch_result_nchw.permute(0, 2, 3, 1)

    # Compare matmul implementation with PyTorch reference
    pcc_passed, pcc_message = assert_with_pcc(torch_result_nhwc, output_matmul_torch, pcc=0.99)
    assert pcc_passed, f"Matmul implementation differs from PyTorch: {pcc_message}"

    # Verify output shapes
    expected_shape = (b, h * scale, w * scale, c)
    assert output_matmul_torch.shape == expected_shape

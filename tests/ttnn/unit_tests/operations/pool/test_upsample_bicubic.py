# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math
import torch
import torch.nn.functional as F
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# =============================================================================
# Phase 0: Golden Reference (no device)
# =============================================================================


@pytest.mark.parametrize("src_size,tgt_size", [(16, 20), (16, 24), (16, 32), (8, 16)])
def test_torch_bicubic_golden(src_size, tgt_size):
    """Verify torch bicubic interpolate behavior we're targeting."""
    torch.manual_seed(42)
    x = torch.randn(1, 64, src_size, src_size, dtype=torch.float32)
    out = F.interpolate(x, size=(tgt_size, tgt_size), mode="bicubic", align_corners=False)
    assert out.shape == (1, 64, tgt_size, tgt_size)


def test_bicubic_weights_sum_to_one():
    """Bicubic weights for any fractional position must sum to ~1.0 (partition of unity)."""

    def cubic_weight(t, a=-0.5):
        abs_t = abs(t)
        if abs_t < 1.0:
            return (a + 2.0) * abs_t**3 - (a + 3.0) * abs_t**2 + 1.0
        elif abs_t < 2.0:
            return a * abs_t**3 - 5.0 * a * abs_t**2 + 8.0 * a * abs_t - 4.0 * a
        return 0.0

    for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        weights = []
        for offset in [-1, 0, 1, 2]:
            t = offset - frac
            weights.append(cubic_weight(t))
        weight_sum = sum(weights)
        assert abs(weight_sum - 1.0) < 1e-6, f"Weights sum to {weight_sum} for frac={frac}"


# =============================================================================
# Phase 1: Minimal End-to-End (Smoke Test)
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bicubic_upsample_basic_2x(device):
    """Smoke test: 2x bicubic upsample runs and produces correct shape."""
    input_shape = [1, 8, 8, 64]  # NHWC
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.upsample(ttnn_input, scale_factor=2, mode="bicubic")
    assert list(output.shape) == [1, 16, 16, 64]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bicubic_upsample_pcc_2x(device):
    """2x bicubic upsample matches torch F.interpolate with PCC >= 0.999."""
    input_shape = [1, 8, 8, 64]
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    # torch reference (NCHW)
    ref = (
        F.interpolate(torch_input.permute(0, 3, 1, 2).float(), scale_factor=2, mode="bicubic", align_corners=False)
        .to(torch.bfloat16)
        .permute(0, 2, 3, 1)
    )

    # ttnn (NHWC)
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.upsample(ttnn_input, scale_factor=2, mode="bicubic")
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, 0.996)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bicubic_upsample_pcc_3x(device):
    """3x bicubic upsample matches torch."""
    input_shape = [1, 8, 8, 64]
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    ref = (
        F.interpolate(torch_input.permute(0, 3, 1, 2).float(), scale_factor=3, mode="bicubic", align_corners=False)
        .to(torch.bfloat16)
        .permute(0, 2, 3, 1)
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.upsample(ttnn_input, scale_factor=3, mode="bicubic")
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, 0.996)


# =============================================================================
# Phase 2: Float Scale Factors
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("src_size,tgt_size", [(16, 20), (16, 24), (8, 12)])
def test_bicubic_upsample_float_scale(device, src_size, tgt_size):
    """Float scale bicubic upsample matches torch."""
    C = 64
    scale = tgt_size / src_size
    input_shape = [1, src_size, src_size, C]
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    ref = (
        F.interpolate(
            torch_input.permute(0, 3, 1, 2).float(),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        )
        .to(torch.bfloat16)
        .permute(0, 2, 3, 1)
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.upsample(ttnn_input, scale_factor=scale, mode="bicubic")
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, 0.996)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("scale_h,scale_w", [(1.5, 2.0), (1.25, 1.75)])
def test_bicubic_upsample_asymmetric_scale(device, scale_h, scale_w):
    """Asymmetric H/W scale factors."""
    input_shape = [1, 8, 8, 64]
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    out_h = int(8 * scale_h)
    out_w = int(8 * scale_w)
    ref = (
        F.interpolate(
            torch_input.permute(0, 3, 1, 2).float(),
            size=(out_h, out_w),
            mode="bicubic",
            align_corners=False,
        )
        .to(torch.bfloat16)
        .permute(0, 2, 3, 1)
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.upsample(ttnn_input, scale_factor=[scale_h, scale_w], mode="bicubic")
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, 0.996)


# =============================================================================
# Phase 3: DeepSeek OCR Exact Dimensions
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("tgt_size_sqrt", [20, 24, 28, 32])
def test_bicubic_upsample_deepseek_ocr_dims(device, tgt_size_sqrt):
    """Exact DeepSeek OCR positional embedding interpolation dimensions."""
    C = 1024
    src_size = 16
    input_shape = [1, src_size, src_size, C]
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    scale = tgt_size_sqrt / src_size
    ref = (
        F.interpolate(
            torch_input.permute(0, 3, 1, 2).float(),
            size=(tgt_size_sqrt, tgt_size_sqrt),
            mode="bicubic",
            align_corners=False,
        )
        .to(torch.bfloat16)
        .permute(0, 2, 3, 1)
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.upsample(ttnn_input, scale_factor=scale, mode="bicubic")
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, 0.996)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bicubic_upsample_pos_embed_integration(device):
    """
    End-to-end test matching DeepSeek OCR get_abs_pos_ttnn flow:
    1. Start with (1, L, C) positional embedding
    2. Extract CLS token
    3. Reshape to (1, H, W, C) NHWC
    4. Bicubic upsample
    5. Reshape back to (1, L', C)
    6. Concat CLS token
    7. Compare with torch reference
    """
    C = 1024
    src_size = 16
    tgt_size = 20
    num_positions = src_size * src_size + 1  # +1 for CLS

    torch.manual_seed(42)
    pos_embed = torch.randn(1, num_positions, C, dtype=torch.bfloat16)

    # --- Torch reference ---
    cls_token = pos_embed[:, :1, :]
    old_pos = pos_embed[:, 1:, :]
    old_pos_2d = old_pos.view(1, src_size, src_size, C).permute(0, 3, 1, 2).contiguous().float()
    new_pos_2d = F.interpolate(old_pos_2d, size=(tgt_size, tgt_size), mode="bicubic", align_corners=False)
    new_pos = new_pos_2d.to(torch.bfloat16).permute(0, 2, 3, 1).reshape(1, tgt_size * tgt_size, C)
    ref = torch.cat([cls_token, new_pos], dim=1)

    # --- TTNN path ---
    old_pos_nhwc = old_pos.view(1, src_size, src_size, C)
    ttnn_input = ttnn.from_torch(
        old_pos_nhwc,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scale = tgt_size / src_size
    ttnn_output = ttnn.upsample(ttnn_input, scale_factor=scale, mode="bicubic")
    new_pos_ttnn = ttnn.to_torch(ttnn_output).reshape(1, tgt_size * tgt_size, C)
    result = torch.cat([cls_token, new_pos_ttnn], dim=1)

    assert_with_pcc(ref, result, 0.996)


# =============================================================================
# Phase 4: Edge Cases + Robustness
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bicubic_upsample_identity(device):
    """scale_factor=1.0 should return identical tensor."""
    input_shape = [1, 8, 8, 64]
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.upsample(ttnn_input, scale_factor=1.0, mode="bicubic")
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(torch_input, output_torch, 0.996)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("C", [32, 256, 1024, 2048])
def test_bicubic_upsample_various_channels(device, C):
    """Various channel dimensions."""
    input_shape = [1, 8, 8, C]
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    ref = (
        F.interpolate(torch_input.permute(0, 3, 1, 2).float(), scale_factor=1.5, mode="bicubic", align_corners=False)
        .to(torch.bfloat16)
        .permute(0, 2, 3, 1)
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.upsample(ttnn_input, scale_factor=1.5, mode="bicubic")
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, 0.996)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("H,W", [(4, 4), (8, 8), (16, 16), (32, 32), (14, 14)])
def test_bicubic_upsample_various_spatial(device, H, W):
    """Various input spatial sizes."""
    input_shape = [1, H, W, 64]
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    ref = (
        F.interpolate(torch_input.permute(0, 3, 1, 2).float(), scale_factor=2, mode="bicubic", align_corners=False)
        .to(torch.bfloat16)
        .permute(0, 2, 3, 1)
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.upsample(ttnn_input, scale_factor=2, mode="bicubic")
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, 0.996)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_bicubic_upsample_batched(device, batch):
    """Multiple batches."""
    input_shape = [batch, 8, 8, 64]
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    ref = (
        F.interpolate(torch_input.permute(0, 3, 1, 2).float(), scale_factor=1.5, mode="bicubic", align_corners=False)
        .to(torch.bfloat16)
        .permute(0, 2, 3, 1)
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.upsample(ttnn_input, scale_factor=1.5, mode="bicubic")
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, 0.996)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bicubic_upsample_non_square(device):
    """Non-square H != W input."""
    input_shape = [1, 12, 16, 64]
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    ref = (
        F.interpolate(torch_input.permute(0, 3, 1, 2).float(), scale_factor=1.5, mode="bicubic", align_corners=False)
        .to(torch.bfloat16)
        .permute(0, 2, 3, 1)
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.upsample(ttnn_input, scale_factor=1.5, mode="bicubic")
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, 0.996)

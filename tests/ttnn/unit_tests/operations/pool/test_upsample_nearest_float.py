# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_upsample_nearest_float(device, input_shape, scale_factor_h, scale_factor_w, dtype=torch.bfloat16, debug=False):
    torch.manual_seed(42)

    batch, height, width, channels = input_shape

    input_nhwc = torch.randn(input_shape, dtype=dtype)

    input_nchw = input_nhwc.permute(0, 3, 1, 2)
    torch_result_nchw = F.interpolate(input_nchw, scale_factor=(scale_factor_h, scale_factor_w), mode="nearest")
    torch_result = torch_result_nchw.permute(0, 2, 3, 1)

    ttnn_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
    input_tensor = ttnn.from_torch(
        input_nhwc,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    output_tensor = ttnn.upsample(input_tensor, [scale_factor_h, scale_factor_w], mode="nearest")
    output_torch = ttnn.to_torch(output_tensor)

    assert list(output_torch.shape) == list(
        torch_result.shape
    ), f"Shape mismatch: expected {list(torch_result.shape)}, got {list(output_torch.shape)}"

    if debug:
        logger.info("=" * 80)
        logger.info(f"Debug mode: shape={input_shape}, scale_h={scale_factor_h}, scale_w={scale_factor_w}")
        logger.info(f"Torch golden (batch 0, channel 0, full H×W):\n{torch_result[0, :, :, 0]}")
        logger.info(f"TTNN output (batch 0, channel 0, full H×W):\n{output_torch[0, :, :, 0]}")
        logger.info("=" * 80)

    is_equal = torch.equal(output_torch, torch_result)
    if not is_equal:
        max_diff = (output_torch - torch_result).abs().max().item()
        num_diffs = (output_torch != torch_result).sum().item()
        total_elements = torch_result.numel()
        logger.warning(
            f"Not exactly equal: max_diff={max_diff}, "
            f"num_diffs={num_diffs}/{total_elements} ({100*num_diffs/total_elements:.2f}%)"
        )
        pcc_passed, pcc_message = assert_with_pcc(torch_result, output_torch, pcc=0.9999)
        logger.info(pcc_message)
        assert pcc_passed, f"PCC check failed: {pcc_message}"
    else:
        logger.info("Results are exactly equal")

    return is_equal


@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        # Basic integer upscaling (2x, 3x, 4x)
        ((1, 16, 16, 64), 2.0, 2.0),
        ((1, 4, 4, 8), 3.0, 3.0),
        ((1, 4, 4, 16), 4.0, 4.0),
        ((1, 8, 8, 32), 2.0, 2.0),
        ((1, 16, 16, 64), 2.0, 2.0),
        # Identity scale (1.0)
        ((1, 8, 8, 4), 1.0, 1.0),
        ((1, 16, 16, 32), 1.0, 1.0),
        ((2, 8, 8, 64), 1.0, 1.0),
        # Fractional upscaling (1.25x, 1.5x, 1.75x, 2.5x)
        ((1, 4, 4, 3), 1.5, 1.5),
        ((1, 8, 8, 4), 1.25, 1.25),
        ((1, 4, 4, 8), 2.5, 2.5),
        ((1, 8, 8, 16), 1.75, 1.75),
        ((1, 16, 16, 32), 1.5, 1.5),
        ((2, 8, 8, 64), 1.25, 1.25),
        # Asymmetric float scales
        ((1, 4, 8, 16), 1.5, 2.0),
        ((1, 8, 4, 32), 2.0, 1.5),
        ((1, 6, 10, 8), 1.5, 2.5),
        ((2, 10, 8, 16), 2.0, 1.5),
        # Downscaling (0.25x, 0.33x, 0.5x, 0.75x)
        ((1, 8, 8, 4), 0.5, 0.5),
        ((1, 8, 8, 4), 0.75, 0.75),
        ((1, 12, 12, 8), 0.33, 0.33),
        ((1, 16, 16, 16), 0.25, 0.25),
        ((1, 16, 16, 32), 0.5, 0.5),
        ((2, 32, 32, 64), 0.5, 0.5),
        # Asymmetric downscaling
        ((1, 8, 16, 4), 0.5, 0.75),
        ((1, 16, 8, 8), 0.75, 0.5),
        ((2, 16, 32, 16), 0.5, 0.25),
        # Mixed upscale/downscale
        ((1, 4, 8, 8), 2.0, 0.5),
        ((1, 8, 4, 8), 0.5, 2.0),
        ((1, 8, 16, 16), 1.5, 0.5),
        ((1, 16, 8, 32), 0.5, 1.5),
        ((2, 8, 16, 64), 2.0, 0.5),
        # Various batch sizes
        ((2, 4, 4, 8), 2.0, 2.0),
        ((4, 4, 4, 4), 1.5, 1.5),
        ((8, 4, 4, 4), 2.0, 2.0),
        ((16, 4, 4, 8), 2.0, 2.0),
        ((2, 8, 8, 16), 1.5, 1.5),
        ((4, 8, 8, 32), 2.0, 2.0),
        # Various channel counts
        ((1, 8, 8, 1), 2.0, 2.0),
        ((1, 8, 8, 3), 1.5, 1.5),
        ((1, 4, 4, 32), 2.0, 2.0),
        ((1, 4, 4, 64), 2.0, 2.0),
        ((1, 4, 4, 128), 2.0, 2.0),
        ((1, 4, 4, 256), 2.0, 2.0),
        ((1, 4, 4, 512), 2.0, 2.0),
        ((1, 8, 8, 1024), 2.0, 2.0),
        ((2, 4, 4, 128), 1.5, 1.5),
        ((2, 8, 8, 256), 2.0, 2.0),
        # Large spatial dimensions
        ((1, 16, 16, 8), 2.0, 2.0),
        ((1, 32, 32, 16), 2.0, 2.0),
        ((1, 64, 64, 8), 2.0, 2.0),
        ((1, 16, 32, 8), 1.5, 2.0),
        ((1, 32, 16, 16), 2.0, 1.5),
        ((2, 8, 8, 16), 3.0, 3.0),
        # Edge cases: single pixel, single row, single column
        ((1, 1, 1, 4), 4.0, 4.0),
        ((1, 1, 1, 8), 8.0, 8.0),
        ((1, 1, 8, 4), 3.0, 2.0),
        ((1, 8, 1, 4), 2.0, 3.0),
        ((1, 1, 16, 8), 4.0, 1.5),
        ((1, 16, 1, 8), 1.5, 4.0),
        # Large scale factors
        ((1, 2, 2, 4), 8.0, 8.0),
        ((1, 2, 2, 8), 16.0, 16.0),
        ((1, 4, 4, 4), 8.0, 8.0),
        # Non-power-of-2 dimensions
        ((1, 5, 7, 3), 2.0, 2.0),
        ((1, 7, 5, 8), 1.5, 1.5),
        ((1, 9, 11, 4), 2.0, 2.0),
        ((1, 11, 9, 16), 1.5, 2.0),
        ((1, 13, 17, 8), 2.0, 1.5),
        ((2, 7, 11, 32), 2.0, 2.0),
        # Typical ML shapes
        ((1, 28, 28, 64), 2.0, 2.0),
        ((1, 14, 14, 128), 2.0, 2.0),
        ((1, 7, 7, 256), 2.0, 2.0),
        ((1, 56, 56, 64), 0.5, 0.5),
        ((1, 112, 112, 32), 0.5, 0.5),
        ((2, 28, 28, 128), 2.0, 2.0),
        ((4, 14, 14, 256), 2.0, 2.0),
        # Wide/tall aspect ratios
        ((1, 4, 32, 8), 2.0, 2.0),
        ((1, 32, 4, 8), 2.0, 2.0),
        ((1, 2, 64, 4), 2.0, 2.0),
        ((1, 64, 2, 4), 2.0, 2.0),
        ((1, 8, 64, 16), 1.5, 0.5),
        ((1, 64, 8, 16), 0.5, 1.5),
        # Fine-grained fractional scales
        ((1, 8, 8, 16), 1.2, 1.2),
        ((1, 8, 8, 32), 1.3, 1.3),
        ((1, 8, 8, 64), 1.4, 1.4),
        ((1, 8, 8, 8), 1.6, 1.6),
        ((1, 8, 8, 16), 1.7, 1.7),
        ((1, 8, 8, 32), 1.8, 1.8),
        ((1, 8, 8, 64), 1.9, 1.9),
        # Very small downscales
        ((1, 32, 32, 8), 0.1, 0.1),
        ((1, 64, 64, 8), 0.15, 0.15),
        ((1, 32, 32, 16), 0.2, 0.2),
        # Combinations: large batch + many channels
        ((8, 8, 8, 128), 2.0, 2.0),
        ((4, 16, 16, 64), 1.5, 1.5),
        ((16, 4, 4, 256), 2.0, 2.0),
    ],
)
def test_upsample_nearest_float_interleaved(device, input_shape, scale_factor_h, scale_factor_w):
    run_upsample_nearest_float(device, input_shape, scale_factor_h, scale_factor_w)

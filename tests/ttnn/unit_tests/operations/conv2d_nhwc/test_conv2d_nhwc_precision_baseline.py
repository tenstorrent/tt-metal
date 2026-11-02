# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for conv2d_nhwc (Phase 0: fp32, ROW_MAJOR, DRAM interleaved).

Measures PCC, max / mean abs error, relative RMS error, fp32 ULP error (p50 / p99 / max)
and the got/true ratio spread (scale-bug detector) on a few shapes. Prints a
`PRECISION_BASELINE` line per shape; asserts only the PCC floor.
"""

import pytest
import torch
import ttnn

from models.common.metrics import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.conv2d_nhwc import conv2d_nhwc, prepare_conv2d_bias, prepare_conv2d_weights

# (input NHWC, C_out, k, padding)
SHAPES = [
    pytest.param((1, 8, 8, 32), 64, 3, 1, id="small_8x8x32_k3"),
    pytest.param((1, 28, 28, 64), 64, 3, 1, id="medium_28x28x64_k3"),
    pytest.param((1, 14, 14, 256), 512, 3, 0, id="deepK_14x14x256_to512_k3"),
    pytest.param((1, 112, 112, 32), 64, 7, 3, id="large_112x112x32_k7"),
    pytest.param((1, 128, 128, 512), 128, 3, 1, id="wide_row_128x128x512_k3_colclip"),
]


def _ulp_err(actual, expected):
    """|actual - expected| in units of the fp32 spacing at |expected|."""
    exp = expected.float()
    spacing = (torch.nextafter(exp.abs(), torch.tensor(float("inf"))) - exp.abs()).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return ((actual.float() - exp).abs() / spacing).flatten()


@pytest.mark.parametrize("input_shape, C_out, k, padding", SHAPES)
def test_conv2d_nhwc_precision_baseline(device, input_shape, C_out, k, padding):
    torch.manual_seed(0)
    N, H, W, C_in = input_shape
    x = torch.randn(N, H, W, C_in, dtype=torch.float32)
    w = torch.randn(C_out, C_in, k, k, dtype=torch.float32) / (C_in * k * k) ** 0.5
    b = torch.randn(C_out, dtype=torch.float32)
    expected = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, b, padding=padding).permute(0, 2, 3, 1).contiguous()

    tt_x = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    common = dict(
        kernel_size=k,
        padding=padding,
        input_dtype=ttnn.float32,
        weight_dtype=ttnn.float32,
        input_memory_config=tt_x.memory_config(),
        device=device,
    )
    tt_w = prepare_conv2d_weights(w, **common)
    tt_b = prepare_conv2d_bias(b, **common)
    actual = ttnn.to_torch(conv2d_nhwc(tt_x, tt_w, k, bias=tt_b, padding=padding)).float()

    diff = (actual - expected).abs()
    rel_rms = (torch.sqrt(torch.mean((actual - expected) ** 2)) / torch.sqrt(torch.mean(expected**2))).item()
    ulp = _ulp_err(actual, expected)
    mask = expected.abs() > 1e-3
    ratio = (actual[mask] / expected[mask]).flatten()
    q = torch.quantile(ratio[: 1 << 20], torch.tensor([0.05, 0.5, 0.95])).tolist()
    _, allclose_msg = comp_allclose(actual, expected)
    pcc = torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1].item()
    print(
        f"\nPRECISION_BASELINE shape={input_shape} C_out={C_out} k={k} p={padding} "
        f"pcc={pcc:.7f} max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e} "
        f"rel_rms={rel_rms:.3e} ulp_p50={ulp.median().item():.1f} "
        f"ulp_p99={torch.quantile(ulp[: 1 << 20], 0.99).item():.1f} ulp_max={ulp.max().item():.1f} "
        f"ratio_p5/p50/p95={q[0]:.5f}/{q[1]:.5f}/{q[2]:.5f} | {allclose_msg}"
    )
    assert_with_pcc(expected, actual, 0.999)

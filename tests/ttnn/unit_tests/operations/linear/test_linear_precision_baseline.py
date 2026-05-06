# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for ttnn.operations.linear.linear (Phase 0).

Measures PCC, max/mean abs error, and relative RMS error against the PyTorch
reference across a small but representative set of shapes. The tolerances
asserted here reflect what the operation can reliably achieve today; the
output of this test seeds the verification report's Precision Baseline table.

Run from repo root:
    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/linear/test_linear_precision_baseline.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.operations.linear import linear

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import check_with_pcc


torch.manual_seed(2026)


def _to_device(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _reference(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    out = x @ w
    if b is not None:
        out = out + b[..., 0:1, :]
    return out


def _make_bias(N: int, device):
    b_torch = torch.zeros((1, 1, 32, N), dtype=torch.bfloat16)
    b_torch[..., 0, :] = torch.randn(N, dtype=torch.bfloat16)
    return b_torch, _to_device(b_torch, device)


# (M, K, N) — small / medium / multi-tile / larger.
_BASELINE_SHAPES = [
    pytest.param((32, 32, 32), id="tiny_M32_K32_N32"),
    pytest.param((128, 128, 128), id="small_M128_K128_N128"),
    pytest.param((128, 256, 128), id="medium_M128_K256_N128"),
    pytest.param((256, 256, 256), id="large_M256_K256_N256"),
]


@pytest.mark.parametrize("shape", _BASELINE_SHAPES)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
def test_linear_precision_baseline(device, shape, with_bias):
    M, K, N = shape

    x_torch = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    w_torch = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
    x = _to_device(x_torch, device)
    w = _to_device(w_torch, device)

    if with_bias:
        b_torch, b = _make_bias(N, device)
        y = linear(x, w, bias=b)
    else:
        b_torch, b = None, None
        y = linear(x, w)

    y_torch = ttnn.to_torch(y).to(torch.float32)
    expected = _reference(x_torch, w_torch, b_torch).to(torch.float32)

    # PCC — primary precision metric.
    pcc_passed, pcc_message = check_with_pcc(expected, y_torch, pcc=0.999)

    # Abs error metrics.
    diff = (y_torch - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    # Relative RMS — robust to sign/scale.
    rms_err = (y_torch - expected).pow(2).mean().sqrt().item()
    rms_ref = expected.pow(2).mean().sqrt().item()
    rel_rms = rms_err / rms_ref if rms_ref > 0 else float("nan")

    # comp_allclose for a redundant atol/rtol delta read-out.
    _, allclose_msg = comp_allclose(expected, y_torch, rtol=0.02, atol=0.15)

    logger.info(
        "linear precision[shape={} bias={}]: pcc={} max_abs={:.6f} " "mean_abs={:.6f} rel_rms={:.6f} | {} | {}",
        shape,
        with_bias,
        pcc_message,
        max_abs,
        mean_abs,
        rel_rms,
        allclose_msg,
        "ok" if pcc_passed else "FAIL",
    )

    assert pcc_passed, f"PCC below 0.999 for shape={shape} bias={with_bias}: {pcc_message}"

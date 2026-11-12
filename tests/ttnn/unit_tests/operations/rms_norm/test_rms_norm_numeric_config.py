# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 2: fp32_dest_acc_en=False (16-bit DEST, bf16 intermediates) and bfloat8_b input / gamma.

Accuracy gate: PCC + relative RMS from the golden suite's helpers.TOLERANCES (keyed by input dtype);
the HiFi2 wide-prefill case additionally checks the perf cases' soft PCC gate (0.9995).
"""

import pytest
import torch
import ttnn

from eval.metrics import check_output
from eval.golden_tests.rms_norm.helpers import TOLERANCES

from ttnn.operations.rms_norm import rms_norm


def _reference(x, gamma, epsilon=1e-6):
    xf = x.float()
    out = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    return out * gamma.float().reshape(-1) if gamma is not None else out


def _run(device, shape, dtype, layout, gamma_dtype, fp32_dest, fidelity=ttnn.MathFidelity.HiFi4, pcc=None):
    torch.manual_seed(7)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_g = None
    g_ref = None
    if gamma_dtype is not None:
        g = torch.randn(W, dtype=torch.float32).to(torch.bfloat16).reshape(1, 1, 1, W)
        g_layout = ttnn.TILE_LAYOUT if gamma_dtype == ttnn.bfloat8_b else layout
        tt_g = ttnn.from_torch(g, dtype=gamma_dtype, layout=g_layout, device=device)
        g_ref = ttnn.to_torch(tt_g).float().reshape(-1)[:W]  # quantized gamma as the device sees it
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = fidelity
    cfg.fp32_dest_acc_en = fp32_dest
    cfg.math_approx_mode = False
    out = rms_norm(tt_x, gamma=tt_g, compute_kernel_config=cfg)
    # bf8b input: reference from the quantized input the device actually reads.
    x_ref = ttnn.to_torch(tt_x).float() if dtype == ttnn.bfloat8_b else x
    expected = _reference(x_ref, g_ref).to(torch.bfloat16)
    tol = TOLERANCES[dtype] if pcc is None else (pcc, TOLERANCES[dtype][1])
    check_output(out, expected, shape=list(shape), dtype=dtype, expected_layout=layout, tolerance=tol)


SHAPES = [(1, 1, 32, 32), (2, 3, 64, 96), (1, 1, 256, 1024), (1, 1, 32, 7168)]
NON_ALIGNED = [(1, 1, 17, 50), (1, 17, 100)]


@pytest.mark.parametrize("shape", SHAPES + NON_ALIGNED, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, None], ids=["gamma", "no_gamma"])
def test_bf16_dest_off(device, shape, layout, gamma_dtype):
    _run(device, shape, ttnn.bfloat16, layout, gamma_dtype, fp32_dest=False)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("fp32_dest", [True, False], ids=["fp32_dest", "bf16_dest"])
@pytest.mark.parametrize(
    "gamma_dtype", [ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32, None], ids=["g_bf8b", "g_bf16", "g_fp32", "no_gamma"]
)
def test_bf8b_input(device, shape, fp32_dest, gamma_dtype):
    _run(device, shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, gamma_dtype, fp32_dest)


@pytest.mark.parametrize("shape", [(1, 1, 64, 1024), (1, 1, 17, 50)], ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_bf8b_gamma(device, shape, dtype):
    _run(device, shape, dtype, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, fp32_dest=True)


@pytest.mark.parametrize("shape", [(1, 1, 8192, 1024), (1, 1, 8192, 7168)], ids=lambda s: "x".join(map(str, s)))
def test_perf_corner_hifi2(device, shape):
    """The _PERF_BASE precision corner: bf16, HiFi2, 16-bit DEST, bf16 TILE gamma, soft PCC 0.9995."""
    _run(device, shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, ttnn.MathFidelity.HiFi2, pcc=0.9995)

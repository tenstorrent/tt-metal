# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for ttnn.operations.layer_norm_rm.layer_norm.

Measures PCC, absolute error (max/mean), relative RMS error, and the
fp32 ULP P99 distance against a torch f64 reference. Parametrised over
4 shapes spanning the Phase-0 envelope (small / multi-tile / batched /
wide-W chunk loop). Prints BASELINE summary lines for each cell so the
verifier can copy/paste them into the report.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.layer_norm_rm import layer_norm


# --------------------------------------------------------------------------
# Reference (mirrors test_layer_norm_rm.py::pytorch_reference + helpers).
# --------------------------------------------------------------------------
def _pytorch_reference(
    input_tensor: torch.Tensor,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    x = input_tensor.to(torch.float64)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    y = (x - mean) / torch.sqrt(var + epsilon)
    if gamma is not None:
        y = y * gamma.reshape(-1).to(torch.float64)
    if beta is not None:
        y = y + beta.reshape(-1).to(torch.float64)
    return y.to(input_tensor.dtype)


def _ulp_p99_fp32(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Per-element ULP distance at fp32 readback granularity, P99."""
    a_bits = actual.to(torch.float32).view(torch.int32).to(torch.int64)
    e_bits = expected.to(torch.float32).view(torch.int32).to(torch.int64)
    sign_offset = 1 << 31

    def _ordered(bits: torch.Tensor) -> torch.Tensor:
        sign = bits < 0
        magnitude = torch.where(sign, bits + sign_offset, bits)
        return torch.where(sign, -magnitude, magnitude)

    ulp = (_ordered(a_bits) - _ordered(e_bits)).abs()
    return float(torch.quantile(ulp.float(), 0.99).item())


SHAPES = [
    pytest.param((1, 1, 32, 32), id="32x32_single_tile"),
    pytest.param((1, 1, 64, 128), id="64x128_multi_tile"),
    pytest.param((2, 4, 32, 256), id="batched_32x256"),
    pytest.param((1, 1, 32, 2048), id="wide_W_2048"),
]


AFFINE_MODES = [
    pytest.param("no_affine", id="affine=none"),
    pytest.param("gamma_beta", id="affine=gamma_beta"),
]


def _make_inputs(shape, affine_mode, device, seed=42):
    torch.manual_seed(seed)
    torch_input = torch.randn(shape, dtype=torch.float32)
    W = shape[-1]
    torch_gamma = None
    torch_beta = None
    ttnn_gamma = None
    ttnn_beta = None
    if affine_mode in ("gamma_only", "gamma_beta"):
        torch_gamma = torch.randn((1, 1, 1, W), dtype=torch.float32) * 0.5 + 1.0
        ttnn_gamma = ttnn.from_torch(
            torch_gamma,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if affine_mode == "gamma_beta":
        torch_beta = torch.randn((1, 1, 1, W), dtype=torch.float32) * 0.1
        ttnn_beta = ttnn.from_torch(
            torch_beta,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
def test_layer_norm_rm_precision_baseline(device, shape, affine_mode):
    """Measure PCC + abs error + relative-RMS + ULP at fp32 readback granularity."""
    epsilon = 1e-5
    (
        torch_input,
        torch_gamma,
        torch_beta,
        ttnn_input,
        ttnn_gamma,
        ttnn_beta,
    ) = _make_inputs(shape, affine_mode, device)

    torch_expected = _pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=epsilon)
    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta, epsilon=epsilon)
    torch_actual = ttnn.to_torch(ttnn_output)

    # --- Metrics ---
    actual_f64 = torch_actual.to(torch.float64)
    expected_f64 = torch_expected.to(torch.float64)

    a_flat = actual_f64.flatten()
    e_flat = expected_f64.flatten()
    pcc = float(torch.corrcoef(torch.stack([a_flat, e_flat]))[0, 1].item())

    abs_diff = (actual_f64 - expected_f64).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    abs_rms = float(torch.nn.functional.mse_loss(actual_f64, expected_f64).sqrt().item())
    ref_std = float(expected_f64.std().item())
    rel_rms = abs_rms / ref_std if ref_std > 1e-12 else abs_rms
    ulp_p99 = _ulp_p99_fp32(torch_actual, torch_expected)

    # Use comp_allclose to get the (atol, rtol) verdict the team usually quotes.
    _, allclose_msg = comp_allclose(torch_expected, torch_actual)

    # Emit a one-line summary for the verifier to copy into the report.
    print(
        f"\nBASELINE shape={tuple(shape)} affine={affine_mode}: "
        f"PCC={pcc:.7f} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"rel_rms={rel_rms:.3e} ulp_p99={ulp_p99:.1f}  ({allclose_msg})"
    )

    # Gate: PCC > 0.9999 and rel_rms < 0.01 — well above Phase 0 acceptance threshold.
    assert pcc >= 0.9999, f"PCC {pcc} below 0.9999 floor"
    assert rel_rms <= 0.01, f"relative RMS {rel_rms} above 0.01 ceiling"
    # Sanity: also check the canonical assert_with_pcc gate at the same value.
    assert_with_pcc(torch_expected, torch_actual, 0.9999)

# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for layer_norm_tile: PCC, abs error, relative RMS, bf16 ULP and the got/true ratio spread.

tests/ttnn/utils_for_testing.assert_with_pcc is not present in this checkout; PCC and allclose come from
models.common.metrics (the host path of the same helpers).
"""

import pytest
import torch
import ttnn

from models.common.metrics import comp_allclose, compute_pcc
from ttnn.operations.layer_norm_tile import layer_norm

W = 768
PCC_FLOOR = {True: 0.999, False: 0.998}
REL_RMS_CEIL = {True: 0.02, False: 0.05}


def bf16_ulp(reference):
    """Spacing of bf16 values at |reference| (8 mantissa bits incl. implicit)."""
    magnitude = reference.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.pow(2.0, torch.floor(torch.log2(magnitude)) - 7)


def measure(expected, actual):
    diff = (actual - expected).abs()
    rel_rms = torch.sqrt(torch.mean((actual - expected) ** 2)) / expected.std()
    ulp = diff / bf16_ulp(expected)
    mask = torch.isfinite(expected) & (expected.abs() > 1e-3)
    ratio = actual[mask] / expected[mask]
    return {
        "pcc": compute_pcc(actual, expected),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_rms": rel_rms.item(),
        "max_ulp": ulp.max().item(),
        "mean_ulp": ulp.mean().item(),
        "ratio_median": ratio.median().item(),
        "ratio_p5": torch.quantile(ratio[:1_000_000], 0.05).item(),
        "ratio_p95": torch.quantile(ratio[:1_000_000], 0.95).item(),
    }


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize("with_residual", [True, False], ids=["residual", "no_residual"])
@pytest.mark.parametrize(
    "shape",
    [(32, 1, 32, W), (64, 1, 17, W), (3, 1, 500, W), (8, 1, 512, W)],
    ids=["32x32", "64x17", "3x500", "8x512"],
)
def test_precision_baseline(device, shape, with_residual, fp32_dest_acc_en):
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    res = torch.randn(shape).to(torch.bfloat16) if with_residual else None
    gamma = torch.randn(W).to(torch.bfloat16)
    beta = torch.randn(W).to(torch.bfloat16)

    z = x.float() + (res.float() if res is not None else 0.0)
    expected = torch.nn.functional.layer_norm(z, (W,), weight=gamma.float(), bias=beta.float(), eps=1e-5)

    def dev(t, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)

    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.HiFi4
    config.fp32_dest_acc_en = fp32_dest_acc_en
    config.math_approx_mode = False

    out = layer_norm(
        dev(x),
        dev(gamma.reshape(1, 1, W // 32, 32), ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG),
        dev(beta.reshape(1, 1, W // 32, 32), ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG),
        dev(res) if res is not None else None,
        compute_kernel_config=config,
    )
    actual = ttnn.to_torch(out).float()

    stats = measure(expected, actual)
    _, allclose_msg = comp_allclose(actual, expected)
    print(f"\nPRECISION {shape} residual={with_residual} fp32acc={fp32_dest_acc_en}: {stats} | {allclose_msg}")

    assert torch.isfinite(actual).all()
    assert stats["pcc"] >= PCC_FLOOR[fp32_dest_acc_en]
    assert stats["rel_rms"] <= REL_RMS_CEIL[fp32_dest_acc_en]
    assert abs(stats["ratio_median"] - 1.0) < 0.01, "got/true ratio centered off 1.0: scale bug"

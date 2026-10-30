# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for layer_norm (layer_norm_tile): PCC, abs / relative-RMS error, bf16 ULP error and the
got/true ratio spread (scale-bug detector) against an fp32 torch reference."""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.comparison import ulp
from ttnn.operations.layer_norm import layer_norm

WIDTH = 768
BASELINE_PCC = 0.9995
SAMPLE = 1_000_000

SHAPES = [
    (32, 1, 32, WIDTH),  # 32 tile-rows: one per core, 32 of 110 cores
    (11, 1, 100, WIDTH),  # 44 tile-rows, S not tile-aligned
    (8, 1, 512, WIDTH),  # 128 tile-rows: two per core on some cores
    (124, 1, 33, WIDTH),  # 248 tile-rows, heavy padding; multi-block with residual at fp32 DEST
]

CONFIGS = {
    "fp32_acc": ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    "bf16_acc": ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False),
}

# Affine forms give bit-identical gamma/beta rows, so each residual mode is paired with one form.
VARIANTS = {"tile_dram-no_residual": ("tile_dram", False), "rm_l1-residual": ("rm_l1", True)}


def make_affine(vec, affine_form, device):
    if affine_form == "tile_dram":
        return ttnn.from_torch(
            vec, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return ttnn.from_torch(
        vec.reshape(1, 1, WIDTH // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def to_device(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def error_metrics(expected, actual):
    diff = (actual - expected).abs()
    rel_rms = (torch.sqrt(torch.mean((actual - expected) ** 2)) / torch.sqrt(torch.mean(expected**2))).item()
    ulp_err = diff / ulp(expected.to(torch.bfloat16)).to(torch.float32)
    mask = expected.abs() > 1e-3
    ratio = actual[mask] / expected[mask]
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_rms": rel_rms,
        # Max ULP is dominated by references near zero (beta crossing), so report the p99.
        "mean_ulp": ulp_err.mean().item(),
        "p99_ulp": torch.quantile(ulp_err.flatten()[:SAMPLE], 0.99).item(),
        "ratio_median": ratio.median().item(),
        "ratio_p5": torch.quantile(ratio[:SAMPLE], 0.05).item(),
        "ratio_p95": torch.quantile(ratio[:SAMPLE], 0.95).item(),
    }


@pytest.mark.parametrize("config_name", list(CONFIGS))
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(str(d) for d in s))
def test_layer_norm_tile_precision_baseline(device, shape, variant, config_name):
    affine_form, with_residual = VARIANTS[variant]
    torch.manual_seed(1234)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    residual = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16) if with_residual else None
    gamma = torch.randn(WIDTH, dtype=torch.float32).to(torch.bfloat16)
    beta = torch.randn(WIDTH, dtype=torch.float32).to(torch.bfloat16)

    out = layer_norm(
        to_device(x, device),
        make_affine(gamma, affine_form, device),
        make_affine(beta, affine_form, device),
        to_device(residual, device) if with_residual else None,
        compute_kernel_config=CONFIGS[config_name],
    )
    actual = ttnn.to_torch(out).to(torch.float32)
    x_ref = x.float() + residual.float() if with_residual else x.float()
    expected = torch.nn.functional.layer_norm(x_ref, (WIDTH,), weight=gamma.float(), bias=beta.float(), eps=1e-5)

    _, pcc_message = assert_with_pcc(expected, actual, BASELINE_PCC)
    _, allclose_message = comp_allclose(expected, actual)
    metrics = error_metrics(expected, actual)
    print(
        f"\nBASELINE shape={shape} variant={variant} config={config_name} pcc={pcc_message} "
        f"{allclose_message} " + " ".join(f"{k}={v:.5g}" for k, v in metrics.items())
    )
    # A uniform scale bug keeps PCC high while the got/true ratio clusters away from 1.0.
    assert abs(metrics["ratio_median"] - 1.0) < 0.01

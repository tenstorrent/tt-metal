# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for nomic_residual_layer_norm (verifier, Phase 0).

Measures, against a float64 reference, over shapes that pin rows_per_shard 1 / 2 / 3 / 4 (and a partial last shard):
PCC, max / mean absolute error, relative RMS error (RMS / std of the reference), the bf16 ULP distance to the
reference rounded to bf16, and the got/true ratio spread (scale-bug detector: a tight cluster around a non-1.0
constant means a structural scale error, a broad spread centred on 1.0 is rounding noise).

Run with `-s` to see the table rows (one `PRECISION_BASELINE` line a case).
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.nomic_residual_layer_norm import (
    default_compute_kernel_config,
    input_memory_config,
    nomic_residual_layer_norm,
)
from ttnn.operations.nomic_residual_layer_norm.nomic_residual_layer_norm import model_memory_config

HIDDEN = 768
EPSILON = 1e-5

# Gates: the golden suite's (PCC >= 0.99999, RMS / std <= 0.005); the ULP and ratio bounds sit about 2x above the
# Phase-0 measurement. The ratio median sits at ~0.9987: design risk P3 (normalize stages re-read fp32 y through srcA,
# TF32 truncation shrinks |out| by ~0.13%), at the bf16 output-rounding floor (RMS 0.0022 vs ~0.0023 for rounding).
PCC_GATE = 0.99999
RMS_GATE = 0.005
MAX_ULP_GATE = 8
RATIO_MEDIAN_TOL = 2e-3


def _round_to_tf32(tensor):
    bits = tensor.float().contiguous().view(torch.int32)
    bits = (bits + 0xFFF + ((bits >> 13) & 1)) & ~0x1FFF
    return bits.view(torch.float32)


def _bf16_ulp_distance(got_bf16, ref_bf16):
    """|ordinal(got) - ordinal(ref)| over bf16 bit patterns (sign-magnitude mapped to a monotone integer line)."""

    def ordinal(t):
        bits = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordinal(got_bf16) - ordinal(ref_bf16)).abs()


SHAPES = [
    (2, 1, 512, 768),  # M = 32, 1 row a core
    (4, 1, 512, 768),  # M = 64, 2 rows a core
    (5, 1, 512, 768),  # M = 80, 3 rows a core, partial last shard
    (8, 1, 512, 768),  # M = 128, 4 rows a core (perf focus)
    (11, 1, 352, 768),  # M = 121, 4 rows a core, partial last shard, wrapping groups
]
CASES = [(shape, "op_layout") for shape in SHAPES]

# Refinement 1: one model_layout shape per bh = ceil(M / 11) class of the golden INPUTS (G = 8, 3 hidden tiles a
# core, odd-width self-add row sum).
MODEL_SHAPES = [
    (2, 1, 512, 768),  # M = 32, bh 3
    (4, 1, 288, 768),  # M = 36, bh 4
    (3, 1, 512, 768),  # M = 48, bh 5
    (4, 1, 512, 768),  # M = 64, bh 6
    (5, 1, 512, 768),  # M = 80, bh 8
    (6, 1, 512, 768),  # M = 96, bh 9
    (7, 1, 512, 768),  # M = 112, bh 11
    (8, 1, 512, 768),  # M = 128, bh 12
]
CASES += [(shape, "model_layout") for shape in MODEL_SHAPES]


@pytest.mark.parametrize("shape,placement", CASES, ids=lambda c: f"{c[0]}x{c[2]}" if isinstance(c, tuple) else c)
@pytest.mark.parametrize("with_residual", [True, False], ids=["residual", "no_residual"])
def test_precision_baseline(device, shape, placement, with_residual):
    torch.manual_seed(1234)
    x = torch.randn(shape).to(torch.bfloat16)
    residual = torch.randn(shape).to(torch.bfloat16) if with_residual else None
    gamma = _round_to_tf32(1.0 + 0.25 * torch.randn(HIDDEN))
    beta = (0.1 * torch.randn(HIDDEN)).to(torch.bfloat16)

    y = x.double() + (residual.double() if with_residual else 0.0)
    expected = torch.nn.functional.layer_norm(y, (HIDDEN,), gamma.double(), beta.double(), eps=EPSILON)

    layout = (model_memory_config if placement == "model_layout" else input_memory_config)(shape, device)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout)
    tt_r = (
        ttnn.from_torch(residual, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout)
        if with_residual
        else None
    )
    tt_g = ttnn.from_torch(gamma.reshape(1, 1, 1, HIDDEN), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(beta.reshape(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = nomic_residual_layer_norm(
        tt_x, tt_g, tt_b, residual=tt_r, epsilon=EPSILON, compute_kernel_config=default_compute_kernel_config()
    )
    got_bf16 = ttnn.to_torch(out)
    got = got_bf16.double()

    # PCC (library helper; returns (passed, message-or-value)).
    pcc_passed, pcc_value = assert_with_pcc(expected, got, PCC_GATE)
    _, allclose_msg = comp_allclose(expected, got)

    err = (got - expected).abs()
    rel_rms = float(((got - expected) ** 2).mean().sqrt() / expected.std())
    # Non-negligible reference elements (|ref| > 0.1 std): the ULP distance and the ratio are only meaningful away
    # from zero (a sign flip around 0 is ~30k bf16 ordinals apart for a ~1e-4 absolute error).
    mask = torch.isfinite(expected) & (expected.abs() > 0.1 * expected.std())
    ulp = _bf16_ulp_distance(got_bf16.to(torch.bfloat16)[mask], expected.to(torch.bfloat16)[mask])

    # got/true ratio over the same elements.
    ratio = got[mask] / expected[mask]
    q = torch.quantile(ratio.float(), torch.tensor([0.05, 0.5, 0.95]))
    ratio_p5, ratio_median, ratio_p95 = (float(v) for v in q)

    print(
        f"\nPRECISION_BASELINE shape={shape[0]}x{shape[2]} placement={placement} residual={with_residual} pcc={pcc_value} "
        f"max_abs={err.max().item():.5f} mean_abs={err.mean().item():.6f} rel_rms={rel_rms:.5f} "
        f"ulp_max={int(ulp.max())} ulp_mean={ulp.float().mean().item():.4f} "
        f"ulp_le1={(ulp <= 1).float().mean().item():.4f} "
        f"ratio_p5={ratio_p5:.5f} ratio_median={ratio_median:.6f} ratio_p95={ratio_p95:.5f} | {allclose_msg}"
    )

    assert torch.isfinite(got).all()
    assert pcc_passed, f"PCC {pcc_value} < {PCC_GATE}"
    assert rel_rms <= RMS_GATE, f"relative RMS {rel_rms:.5f} > {RMS_GATE}"
    assert int(ulp.max()) <= MAX_ULP_GATE, f"max bf16 ULP {int(ulp.max())} > {MAX_ULP_GATE}"
    assert abs(ratio_median - 1.0) <= RATIO_MEDIAN_TOL, f"got/true ratio median {ratio_median:.6f} != 1 (scale bug?)"

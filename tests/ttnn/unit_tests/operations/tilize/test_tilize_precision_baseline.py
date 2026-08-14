# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for `ttnn.operations.tilize.tilize` (verifier-authored).

tilize is a pure permutation of byte positions, so the numeric reference is the
IDENTITY: every measured error here is either exactly zero (no cast) or purely
the cast's representation error (`dtype=` narrowing to bf16 / bf8b).

Recorded per (shape, dtype pair): PCC, max abs error, mean abs error, relative
RMS error, and the got/true RATIO spread — the scale-bug detector. A tight
cluster of `r = got/true` around a non-1.0 constant is a uniform scale /
structural bug (fix the kernel, do not file it as precision); a broad spread
centred on 1.0 is ordinary rounding noise. For this op the expected reading is
`r == 1.0` exactly on every no-cast pair.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.tilize import tilize


SHAPES = [
    pytest.param([1, 1, 32, 32], id="small_1x1x32x32"),
    pytest.param([1, 1, 256, 256], id="medium_1x1x256x256"),
    pytest.param([2, 3, 64, 128], id="batched_2x3x64x128"),
    pytest.param([1, 1, 1024, 1024], id="large_1x1x1024x1024"),
]

# (input dtype, output dtype, pcc floor). The no-cast pairs must be EXACT.
DTYPE_PAIRS = [
    pytest.param(ttnn.bfloat16, None, 1.0, id="bf16"),
    pytest.param(ttnn.float32, None, 1.0, id="fp32"),
    pytest.param(ttnn.float32, ttnn.bfloat16, 0.995, id="fp32_to_bf16"),
    pytest.param(ttnn.bfloat16, ttnn.bfloat8_b, 0.99, id="bf16_to_bf8b"),
]

RESULTS = []


def _ratio_spread(expected, actual):
    """Median and p5/p95 of got/true over finite, non-zero-reference elements."""
    mask = (expected != 0) & torch.isfinite(expected) & torch.isfinite(actual)
    if mask.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    r = (actual[mask] / expected[mask]).to(torch.float64)
    return (
        r.median().item(),
        torch.quantile(r, 0.05).item(),
        torch.quantile(r, 0.95).item(),
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype, out_dtype, pcc_floor", DTYPE_PAIRS)
def test_tilize_precision_baseline(device, shape, dtype, out_dtype, pcc_floor):
    torch.manual_seed(7)
    reference = torch.randn(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        reference = reference.to(torch.bfloat16).to(torch.float32)

    tt_input = ttnn.from_torch(
        reference,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input, dtype=out_dtype)
    actual = ttnn.to_torch(tt_output).to(torch.float32)

    # assert_with_pcc returns (passed, pcc) — the second element is the measured
    # coefficient (a float in this tree, a message string in older ones).
    _, pcc_result = assert_with_pcc(reference, actual, pcc_floor)
    measured_pcc = f"{pcc_result:.6f}" if isinstance(pcc_result, float) else str(pcc_result).splitlines()[0].strip()

    diff = (actual - reference).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_rms = (
        torch.sqrt((actual - reference).pow(2).mean()) / torch.sqrt(reference.pow(2).mean().clamp_min(1e-30))
    ).item()
    pcc_str = comp_allclose(reference, actual, rtol=1e-2, atol=1e-2)
    r_med, r_p5, r_p95 = _ratio_spread(reference, actual)

    RESULTS.append(
        {
            "shape": tuple(shape),
            "pair": f"{dtype}->{out_dtype or dtype}",
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "rel_rms": rel_rms,
            "ratio": (r_med, r_p5, r_p95),
            "pcc": measured_pcc,
        }
    )
    print(
        f"\n[precision] shape={tuple(shape)} {dtype}->{out_dtype or dtype}: "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_rms={rel_rms:.3e} "
        f"ratio med={r_med:.6f} p5={r_p5:.6f} p95={r_p95:.6f} | {measured_pcc} | {pcc_str}"
    )

    if out_dtype is None or out_dtype == dtype:
        # No cast: tilize moves bytes, so the round trip must be BIT-exact.
        assert max_abs == 0.0, f"no-cast tilize is not bit-exact (max_abs={max_abs})"
        assert r_med == 1.0 and r_p5 == 1.0 and r_p95 == 1.0, "got/true ratio is not identically 1.0"


def test_report_precision_table():
    """Print the collected table (runs last — pytest keeps file order)."""
    if not RESULTS:
        pytest.skip("no measurements collected")
    print("\n| shape | dtypes | pcc | max_abs | mean_abs | rel_rms | ratio med (p5..p95) |")
    print("|---|---|---|---|---|---|---|")
    for row in RESULTS:
        med, p5, p95 = row["ratio"]
        print(
            f"| {row['shape']} | {row['pair']} | {row['pcc']} | {row['max_abs']:.3e} | {row['mean_abs']:.3e} | "
            f"{row['rel_rms']:.3e} | {med:.6f} ({p5:.6f}..{p95:.6f}) |"
        )

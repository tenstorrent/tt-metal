# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for `tilize` (verifier-authored, Phase 0).

`tilize` performs **no arithmetic** — it re-lays bytes from ROW_MAJOR into the
four-face TILE geometry. So this file is not an error-budget measurement in the
usual sense: the expected result is bit-identity, and any deviation at all is a
structural bug (a wrong face permutation, a stale CB page, a dropped tile-row),
never rounding.

It is still worth measuring, and measured the same way as every other op, for
two reasons:

1. It fixes the Phase 0 numbers a refinement is not allowed to regress. When
   the dtype refinement lands, `bfloat8_b` / `bfloat4_b` outputs WILL have a
   real error budget, and the bf16 rows here are the "no-cast diagonal" they
   are compared against.
2. The got/true **ratio spread** is the scale-bug detector the verifier
   protocol asks for. On a pure passthrough a uniform non-1.0 median ratio
   would mean a scaler leaked into the datapath; a broad spread around 1.0
   would mean rounding. For Phase 0 both must be exactly 1.0.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.tilize import tilize

# Phase 0 SUPPORTED: bfloat16, rank 4, tile-aligned, interleaved DRAM->DRAM.
DTYPE = ttnn.bfloat16
PCC_FLOOR = 0.9999

SHAPES = [
    ((1, 1, 32, 32), "small__single_tile"),
    ((1, 1, 64, 128), "medium__multi_tile"),
    ((2, 3, 128, 256), "medium__leading_fold"),
    ((1, 1, 1024, 1024), "large__square_large"),
    ((1, 1, 32, 16384), "large__short_wide_perf_focus"),
]


def _ratio_stats(actual: torch.Tensor, expected: torch.Tensor):
    """Median and spread of r = actual / expected over finite, non-zero refs.

    A tight cluster around a non-1.0 constant is a uniform scale / structural
    bug (fix the kernel); a broad spread centred on 1.0 is ordinary precision
    noise. For a byte-relaying op both must read exactly 1.0.
    """
    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    keep = torch.isfinite(a) & torch.isfinite(e) & (e.abs() > 0)
    if keep.sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = a[keep] / e[keep]
    p5, p95 = torch.quantile(r, torch.tensor([0.05, 0.95], dtype=torch.float64))
    return float(r.median()), float(p5), float(p95), float(r.std())


@pytest.mark.parametrize("shape", [s for s, _ in SHAPES], ids=[i for _, i in SHAPES])
def test_tilize_precision_baseline(device, shape):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=DTYPE,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.dtype == DTYPE
    assert list(tt_output.shape) == list(shape)

    expected = torch_input.float()
    actual = ttnn.to_torch(tt_output).float()

    diff = (actual - expected).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rel_rms = float(torch.sqrt((diff**2).mean()) / (torch.sqrt((expected**2).mean()) + 1e-30))
    med, p5, p95, std = _ratio_stats(actual, expected)

    passing, message = comp_allclose(expected, actual)
    print(
        f"\n[precision-baseline] shape={tuple(shape)} dtype={DTYPE}"
        f"\n  max_abs_err  = {max_abs:.6g}"
        f"\n  mean_abs_err = {mean_abs:.6g}"
        f"\n  rel_rms_err  = {rel_rms:.6g}"
        f"\n  ratio median = {med:.9f}   p5 = {p5:.9f}   p95 = {p95:.9f}   std = {std:.3g}"
        f"\n  comp_allclose: {message}"
    )

    assert_with_pcc(expected, actual, PCC_FLOOR)

    # tilize moves bytes; it does not compute. Bit-identity is the contract, so
    # these are equalities and not tolerances.
    assert max_abs == 0.0, f"tilize is a byte re-lay — max_abs_err must be 0, got {max_abs}"
    assert rel_rms == 0.0, f"tilize is a byte re-lay — rel_rms_err must be 0, got {rel_rms}"
    assert med == 1.0 and p5 == 1.0 and p95 == 1.0, (
        f"got/true ratio is not identically 1.0 (median={med}, p5={p5}, p95={p95}) — "
        f"a clustered non-1.0 ratio means a scale/structural bug, not rounding"
    )
    assert passing, message

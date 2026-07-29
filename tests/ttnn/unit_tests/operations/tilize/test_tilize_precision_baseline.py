# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for `ttnn.operations.tilize.tilize`.

tilize does no arithmetic — it re-lays bytes — so the reference is the identity
function and the *only* precision surface is the optional output-dtype cast
(`dtype=`). This file measures, per (shape, transition):

* PCC (`assert_with_pcc` from tests.ttnn.utils_for_testing)
* max abs error / mean abs error, and the allclose verdict
  (`comp_allclose` from models.common.utility_functions)
* relative RMS error (`||actual - expected||_2 / ||expected||_2`)
* exact-match fraction (how many elements are bit-identical) — the meaningful
  number for a value-preserving op

Transitions measured:
  bf16 -> bf16   identity, must be exact
  fp32 -> fp32   identity, must be exact (needs Fp32Mode::Lossless)
  bf16 -> fp32   widening, must be exact
  fp32 -> bf16   narrowing cast (mantissa truncation)
  bf16 -> bf8b   block-float pack (shared exponent, lossy)

Run:
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc


# (in_dtype, out_dtype, pcc_threshold, must_be_exact)
TRANSITIONS = [
    pytest.param(ttnn.bfloat16, ttnn.bfloat16, 0.9999, True, id="bf16_to_bf16"),
    pytest.param(ttnn.float32, ttnn.float32, 0.9999, True, id="fp32_to_fp32"),
    pytest.param(ttnn.bfloat16, ttnn.float32, 0.9999, True, id="bf16_to_fp32"),
    pytest.param(ttnn.float32, ttnn.bfloat16, 0.999, False, id="fp32_to_bf16"),
    pytest.param(ttnn.bfloat16, ttnn.bfloat8_b, 0.99, False, id="bf16_to_bf8b"),
]

SHAPES = [
    pytest.param((1, 1, 32, 32), id="1x1x32x32"),
    pytest.param((1, 1, 64, 128), id="1x1x64x128"),
    pytest.param((1, 1, 256, 512), id="1x1x256x512"),
    pytest.param((2, 3, 128, 256), id="2x3x128x256"),
]

_READBACK_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # bf8b reads back as bf16
    ttnn.float32: torch.float32,
}


def _metrics(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    exp = expected.float()
    act = actual.float()
    diff = (act - exp).abs()
    denom = exp.pow(2).sum().sqrt().item()
    exact = act == exp
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_rms": (diff.pow(2).sum().sqrt().item() / denom) if denom else 0.0,
        "exact_frac": exact.float().mean().item(),
        "n_mismatch": int((~exact).sum().item()),
        "n_elem": exact.numel(),
    }


@pytest.mark.parametrize("in_dtype,out_dtype,pcc,must_be_exact", TRANSITIONS)
@pytest.mark.parametrize("shape", SHAPES)
def test_tilize_precision_baseline(device, shape, in_dtype, out_dtype, pcc, must_be_exact):
    torch.manual_seed(42)
    if in_dtype == ttnn.float32:
        torch_input = torch.randn(shape, dtype=torch.float32)
    else:
        torch_input = torch.randn(shape).bfloat16()

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=in_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input, dtype=out_dtype)

    actual = ttnn.to_torch(tt_output)
    expected = torch_input.to(_READBACK_TORCH_DTYPE[out_dtype])

    stats = _metrics(expected, actual)
    allclose_pass, allclose_msg = comp_allclose(expected.float(), actual.float(), rtol=1e-2, atol=1e-2)
    print(
        f"\nPRECISION shape={tuple(shape)} {in_dtype}->{out_dtype}: "
        f"max_abs={stats['max_abs']:.3e} mean_abs={stats['mean_abs']:.3e} "
        f"rel_rms={stats['rel_rms']:.3e} exact={stats['exact_frac']:.6f} "
        f"mismatch={stats['n_mismatch']}/{stats['n_elem']} "
        f"allclose={allclose_pass} ({allclose_msg})"
    )

    if must_be_exact:
        assert stats["exact_frac"] == 1.0, (
            f"value-preserving transition {in_dtype}->{out_dtype} must be bit-exact, "
            f"exact fraction was {stats['exact_frac']}"
        )
    assert_with_pcc(expected.float(), actual.float(), pcc)

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 0 precision baseline for rms_norm.

Measures PCC, max abs error, mean abs error, and relative RMS error on a
small fixed shape matrix. Goes alongside the acceptance test
(`test_rms_norm.py`) — those tests check correctness with broad pytest-style
assertions; this file pins the *numerical* baseline so refinements can be
compared against it.

Coverage (intentionally small — full coverage lives in the golden suite):
  - 4 shapes: single-tile, 2-tile-wide, multi-row, wide-W
  - 2 dtypes: bfloat16, float32
  - both TILE_LAYOUT and ROW_MAJOR_LAYOUT (the two input layouts in
    SUPPORTED).
  - gamma supplied (mirrors typical use-case in LLMs).

Numbers emitted by parametrize ids — use `pytest --tb=no -q -v` to read
them off without running the full test list. The "Precision Baseline"
section of verification_report.md aggregates results from this file.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm


# --------------------------------------------------------------------------- #
#  Reference (computed in fp32)                                                #
# --------------------------------------------------------------------------- #


def _torch_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32, returned in fp32 (caller casts back if needed)."""
    x_fp = x.float()
    rms = torch.sqrt(x_fp.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x_fp / rms
    out = out * gamma.float().view(*([1] * (x.dim() - 1)), -1)
    return out


# --------------------------------------------------------------------------- #
#  Per-dtype PCC targets (matched to eval/golden_tests/rms_norm/helpers.py)   #
# --------------------------------------------------------------------------- #

_PCC_BY_DTYPE = {ttnn.bfloat16: 0.995, ttnn.float32: 0.999}
_TTNN_TO_TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


_SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile_32x32"),
    pytest.param((1, 1, 64, 128), id="2x4_tiles_64x128"),
    pytest.param((2, 1, 128, 256), id="multibatch_2x128x256"),
    pytest.param((1, 1, 32, 1024), id="wide_W_1024"),
]


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_rms_norm_precision_baseline(device, shape, dtype, layout):
    """Compute and report PCC, max_abs_err, mean_abs_err, rms_err.

    Asserts PCC threshold by dtype (matches golden suite tolerances) and
    pyhamcrest-style allclose with relaxed atol/rtol — failure here is a
    real numerical regression, not flaky noise.
    """
    torch.manual_seed(2026)
    torch_dtype = _TTNN_TO_TORCH[dtype]
    torch_input = torch.randn(shape, dtype=torch_dtype)
    torch_gamma_4d = torch.randn(1, 1, 1, shape[-1], dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma_4d,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_out = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=1e-6)
    torch_out = ttnn.to_torch(ttnn_out).float()

    torch_expected = _torch_rms_norm(torch_input, torch_gamma_4d, eps=1e-6)
    # Reference at fp32 then back to the kernel's dtype before comparison.
    torch_expected_q = torch_expected.to(torch_dtype).float()

    err = (torch_out - torch_expected_q).abs()
    max_abs = err.max().item()
    mean_abs = err.mean().item()
    # Relative RMS: rms(err) / std(reference) — same metric the eval helpers use.
    abs_rms = ((torch_out - torch_expected_q) ** 2).mean().sqrt().item()
    ref_std = torch_expected_q.std().item()
    rel_rms = abs_rms / ref_std if ref_std > 1e-12 else abs_rms

    pcc_target = _PCC_BY_DTYPE[dtype]
    # PCC check (Pearson correlation) — bf16 0.995, fp32 0.999.
    assert_with_pcc(torch_expected_q, torch_out, pcc=pcc_target)

    # comp_allclose with dtype-appropriate tolerances.
    rtol = 0.05 if dtype == ttnn.bfloat16 else 0.005
    atol = 0.05 if dtype == ttnn.bfloat16 else 0.005
    ok, msg = comp_allclose(torch_expected_q, torch_out, rtol=rtol, atol=atol)
    print(
        f"\n[rms_norm precision baseline] shape={shape} dtype={dtype} "
        f"layout={layout} | max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} "
        f"rel_rms={rel_rms:.4g} pcc_target={pcc_target} | {msg}"
    )
    assert ok, msg

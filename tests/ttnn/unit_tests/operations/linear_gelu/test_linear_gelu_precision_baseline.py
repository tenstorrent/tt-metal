# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for linear_gelu at the production corner (x / output in L1, default config).

Records PCC, abs error, relative RMS, bf16 ULP error and the got/true ratio spread against the
fp64 reference on model-distributed inputs. The ratio spread separates a uniform scale bug (tight
cluster around a non-1.0 constant) from rounding noise (spread centered on 1.0).
"""

import pytest
import torch
import ttnn

from eval.golden_tests.linear_gelu.helpers import make_inputs, pytorch_linear_gelu
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.linear_gelu import linear_gelu

PCC_TARGET = 0.9999
REL_RMS_TARGET = 1.4e-2
# |ref| below this is GELU's negative tail, where ULP error is dominated by fp32 cancellation in 1 + tanh.
ULP_MAGNITUDE_FLOOR = 0.1


def bf16_ulp(values):
    """Spacing of the bf16 grid at |values| (8-bit mantissa incl. the implicit bit)."""
    magnitude = values.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.pow(2.0, torch.floor(torch.log2(magnitude)) - 7)


def to_device(tensor, device, memory_config):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )


@pytest.mark.parametrize("batch, seq", [(1, 32), (2, 512), (8, 384), (8, 512)])
def test_precision_baseline(device, batch, seq):
    x, w, b = make_inputs(batch, seq)
    expected = pytorch_linear_gelu(x, w, b)
    out = linear_gelu(
        to_device(x, device, ttnn.L1_MEMORY_CONFIG),
        to_device(w, device, ttnn.DRAM_MEMORY_CONFIG),
        to_device(b, device, ttnn.DRAM_MEMORY_CONFIG),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(out).double()

    _, pcc = assert_with_pcc(expected, actual, PCC_TARGET)
    _, allclose_msg = comp_allclose(actual, expected)
    diff = (actual - expected).abs()
    rel_rms = (diff.pow(2).mean().sqrt() / expected.pow(2).mean().sqrt()).item()

    ulp = diff / bf16_ulp(expected)
    body = expected.abs() >= ULP_MAGNITUDE_FLOOR
    ratio = (actual[body] / expected[body]).float()
    ratio_p5, ratio_p50, ratio_p95 = torch.quantile(ratio[: 1 << 24], torch.tensor([0.05, 0.5, 0.95])).tolist()

    print(
        f"\nPRECISION {batch}x{seq}: pcc={float(pcc):.6f} max_abs={diff.max().item():.4e} "
        f"mean_abs={diff.mean().item():.4e} rel_rms={rel_rms:.4e} "
        f"ulp_max(|ref|>={ULP_MAGNITUDE_FLOOR})={ulp[body].max().item():.1f} "
        f"ulp_mean(|ref|>={ULP_MAGNITUDE_FLOOR})={ulp[body].mean().item():.3f} "
        f"ratio(|ref|>={ULP_MAGNITUDE_FLOOR}) p5/p50/p95={ratio_p5:.4f}/{ratio_p50:.4f}/{ratio_p95:.4f} ({allclose_msg})"
    )
    assert rel_rms <= REL_RMS_TARGET, f"relative rms {rel_rms:.4e} > {REL_RMS_TARGET}"

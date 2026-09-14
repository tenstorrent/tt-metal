# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for ttnn.operations.rms_norm (Phase 0 corner: HiFi4, fp32 DEST accumulation).

Measures, per shape x dtype: PCC, max / mean absolute error, relative RMS error, and the got/true
ratio spread (median, p5, p95). The ratio spread is the scale-bug detector: a tight cluster around a
non-1.0 constant means a uniformly mis-scaled output (a structural bug that a high PCC would hide);
a broad spread centred on 1.0 is ordinary rounding noise.

Shapes pin every blocking regime: single tile (R1), multi-batch rows (R1, 32 cores), one wide
tile-row (R2 by occupancy), two very wide tile-rows (R2 by residency).
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm

PCC_BY_DTYPE = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((2, 4, 128, 512), id="multi_batch_R1"),
    pytest.param((1, 1, 32, 4096), id="wide_row_R2_occupancy"),
    pytest.param((1, 1, 64, 12288), id="very_wide_R2_residency"),
]


def torch_rms_norm(x, gamma, epsilon=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    return xf / rms * gamma.to(torch.float32).reshape(-1)


def precision_metrics(expected, actual):
    """expected/actual in fp32. Returns a dict of the baseline metrics."""
    err = (actual - expected).abs()
    rel_rms = torch.sqrt(torch.mean((actual - expected) ** 2)) / torch.sqrt(torch.mean(expected**2))
    mask = torch.isfinite(expected) & (expected.abs() > 1e-3)
    ratio = (actual[mask] / expected[mask]).flatten()
    return {
        "max_abs_err": err.max().item(),
        "mean_abs_err": err.mean().item(),
        "rel_rms_err": rel_rms.item(),
        "ratio_median": ratio.median().item(),
        "ratio_p5": torch.quantile(ratio[: 2**24], 0.05).item(),
        "ratio_p95": torch.quantile(ratio[: 2**24], 0.95).item(),
    }


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rms_norm_precision_baseline(device, shape, dtype):
    torch.manual_seed(1234)
    x = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    g = torch.randn(shape[-1], dtype=torch.float32).to(TORCH_DTYPE[dtype])
    expected = torch_rms_norm(x, g)

    ttnn_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_g = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    actual = ttnn.to_torch(rms_norm(ttnn_x, gamma=ttnn_g)).to(torch.float32)

    m = precision_metrics(expected, actual)
    _, pcc_msg = assert_with_pcc(expected, actual, PCC_BY_DTYPE[dtype])
    _, allclose_msg = comp_allclose(expected, actual, rtol=1e-2, atol=1e-2)
    print(
        f"\nPRECISION shape={tuple(shape)} dtype={dtype} pcc={pcc_msg} "
        f"max_abs={m['max_abs_err']:.4g} mean_abs={m['mean_abs_err']:.4g} rel_rms={m['rel_rms_err']:.4g} "
        f"ratio(median/p5/p95)={m['ratio_median']:.5f}/{m['ratio_p5']:.5f}/{m['ratio_p95']:.5f} | {allclose_msg}"
    )
    # Scale-bug guard: the got/true ratio must cluster on 1.0, not on some other constant.
    assert abs(m["ratio_median"] - 1.0) < 5e-3, f"uniform scale error: median got/true = {m['ratio_median']}"
    assert not torch.isnan(actual).any() and not torch.isinf(actual).any()

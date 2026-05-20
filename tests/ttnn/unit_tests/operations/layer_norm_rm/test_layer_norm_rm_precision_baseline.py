# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for layer_norm_rm — measured PCC, abs error, RMS error.

Records the Phase-0 numerical envelope for the supported shapes. The numbers
in verification_report.md and changelog.md were collected from this test.
The thresholds here are loose floors (catastrophic-failure tripwires); the
tighter, per-dtype PCC thresholds are enforced by the acceptance test and
the golden suite.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.layer_norm import layer_norm


# 4 shapes — small (single-tile), medium (multi-tile-W), batched, partial-W
# (kept within the supported Phase-0 SUPPORTED rectangle: no fp32-rm + w_partial).
SHAPES = [
    pytest.param((1, 1, 32, 64), id="small_single_tile"),
    pytest.param((1, 1, 64, 256), id="medium_multi_tile_W"),
    pytest.param((2, 4, 32, 128), id="batched_4d"),
    pytest.param((1, 1, 32, 100), id="W_partial"),
]


_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}


def _pytorch_layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    x32 = x.to(torch.float32)
    mean = x32.mean(dim=-1, keepdim=True)
    var = x32.var(dim=-1, keepdim=True, unbiased=False)
    out = (x32 - mean) / torch.sqrt(var + eps)
    out = out * gamma.to(torch.float32).reshape(-1)
    out = out + beta.to(torch.float32).reshape(-1)
    return out.to(x.dtype)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a = actual.detach().to(torch.float64).flatten()
    e = expected.detach().to(torch.float64).flatten()
    finite = torch.isfinite(a) & torch.isfinite(e)
    a, e = a[finite], e[finite]
    diff = (a - e).abs()
    rms = torch.sqrt(((a - e) ** 2).mean()).item()
    # Pearson correlation
    a_c = a - a.mean()
    e_c = e - e.mean()
    denom = torch.sqrt((a_c * a_c).sum() * (e_c * e_c).sum())
    pcc = (a_c * e_c).sum().item() / denom.item() if denom.item() > 0 else 1.0
    return {
        "pcc": pcc,
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "rms_err": rms,
        "rel_rms_err": rms / (e.std().item() + 1e-12),
    }


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(ttnn.bfloat16, id="bf16"), pytest.param(ttnn.float32, id="fp32")],
)
def test_precision_baseline_tile(device, shape, dtype):
    """Per-shape × per-dtype baseline (TILE layout, gamma+beta affine).

    Asserts loose PCC floors and prints exact metrics for the report.
    """
    torch.manual_seed(0)
    torch_dtype = _TORCH_DTYPE[dtype]
    W = shape[-1]

    x = torch.randn(shape, dtype=torch_dtype)
    gamma = torch.randn(W, dtype=torch_dtype)
    beta = torch.randn(W, dtype=torch_dtype)

    expected = _pytorch_layer_norm(x, gamma, beta)

    ttnn_x = ttnn.from_torch(
        x,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        gamma.reshape(1, 1, 1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        beta.reshape(1, 1, 1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_y = layer_norm(ttnn_x, ttnn_gamma, ttnn_beta)
    y = ttnn.to_torch(ttnn_y)

    m = _metrics(y.to(torch.float32), expected.to(torch.float32))
    print(
        f"\nlayer_norm precision baseline shape={tuple(shape)} dtype={dtype.name}:"
        f" PCC={m['pcc']:.6f}"
        f" max_abs={m['max_abs_err']:.5f}"
        f" mean_abs={m['mean_abs_err']:.5f}"
        f" rms={m['rms_err']:.5f}"
        f" rel_rms={m['rel_rms_err']:.5f}"
    )

    # Loose floors — golden / acceptance tests enforce tighter per-dtype PCC.
    floor = 0.99 if dtype == ttnn.float32 else 0.98
    assert_with_pcc(y.to(torch.float32), expected.to(torch.float32), pcc=floor)
    passing, allclose_str = comp_allclose(expected.to(torch.float32), y.to(torch.float32), atol=0.5, rtol=0.1)
    print(f"  comp_allclose(atol=0.5, rtol=0.1): {allclose_str}")
    assert passing, f"comp_allclose failed: {allclose_str}"

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 2 — bfloat16 input support.

Verifies that glu_fused accepts bfloat16 input end-to-end and that the
dtype-aware compute config produces correct results at bf16's precision
regime. Loosened tolerances vs the fp32 precision baseline — bf16 quantises
the input itself to ~3 decimal digits, so PCC ≈ 0.999 and max_abs in the
0.02 range are expected.
"""

import pytest
import torch
import ttnn

from ttnn.operations.glu_fused import glu_fused


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    am = a - a.mean()
    bm = b - b.mean()
    denom = (am.norm() * bm.norm()).item()
    return (am * bm).sum().item() / denom if denom > 0 else 1.0


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 64),
        (1, 1, 32, 128),
        (1, 1, 256, 128),
        (2, 4, 64, 128),
    ],
)
def test_glu_fused_bf16_correctness(device, shape):
    torch.manual_seed(0)
    x_t = torch.randn(*shape, dtype=torch.float32)
    expected = torch.nn.functional.glu(x_t, dim=-1)

    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = glu_fused(x)

    assert y.dtype == ttnn.bfloat16, f"output dtype should be bf16, got {y.dtype}"
    assert tuple(y.shape) == tuple(expected.shape)

    actual = ttnn.to_torch(y).to(torch.float32)
    pcc = _pcc(actual, expected)
    max_abs = (actual - expected).abs().max().item()

    assert pcc >= 0.999, f"PCC too low at bf16: {pcc:.6f} (shape={shape})"
    assert max_abs <= 0.05, f"max_abs too high at bf16: {max_abs:.4e} (shape={shape})"


def test_glu_fused_bf16_validation_rejects_other_dtypes(device):
    """The validator accepts fp32 and bf16; other dtypes must still raise."""
    x_t = torch.randn(1, 1, 32, 64, dtype=torch.float32)
    x = ttnn.from_torch(x_t, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(ValueError, match="only float32 and bfloat16"):
        glu_fused(x)

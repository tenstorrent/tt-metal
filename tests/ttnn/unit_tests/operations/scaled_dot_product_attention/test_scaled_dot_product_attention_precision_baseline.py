# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for scaled_dot_product_attention (Flash Attention).

Measures PCC, max/mean absolute error, and relative RMS error across a small
set of representative tile-aligned bf16 shapes (single-tile, multi-tile,
multi-head, longer-sequence). Establishes the Phase 0 numerical baseline the
refinement loop tracks against.

Uses assert_with_pcc (tests.ttnn.utils_for_testing) and comp_allclose
(models.common.utility_functions) — no hand-rolled metrics.
"""

import math

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose, comp_pcc

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _torch_reference(q, k, v, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


# (B, H, S, D): single-tile, multi-tile multi-head, batched, longer-sequence.
SHAPES = [
    (1, 1, 32, 32),
    (1, 8, 128, 64),
    (2, 8, 256, 64),
    (1, 4, 1024, 64),
]


@pytest.mark.parametrize("shape", SHAPES, ids=[f"B{b}_H{h}_S{s}_D{d}" for (b, h, s, d) in SHAPES])
def test_precision_baseline(device, shape):
    torch.manual_seed(0)
    b, h, s, d = shape

    q = torch.randn(shape, dtype=torch.float32)
    k = torch.randn(shape, dtype=torch.float32)
    v = torch.randn(shape, dtype=torch.float32)

    torch_out = _torch_reference(q, k, v)

    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = scaled_dot_product_attention(tt_q, tt_k, tt_v)
    out = ttnn.to_torch(tt_out).to(torch.float32)

    max_abs = (out - torch_out).abs().max().item()
    mean_abs = (out - torch_out).abs().mean().item()
    rms = ((out - torch_out).pow(2).mean().sqrt() / torch_out.std()).item()

    _, allclose_str = comp_allclose(torch_out, out)
    _, pcc_str = comp_pcc(torch_out, out, 0.995)
    print(
        f"\n[precision-baseline] shape={shape} {pcc_str} max_abs={max_abs:.5f} "
        f"mean_abs={mean_abs:.5f} rel_rms={rms:.5f} | {allclose_str}"
    )

    assert_with_pcc(torch_out, out, 0.995)

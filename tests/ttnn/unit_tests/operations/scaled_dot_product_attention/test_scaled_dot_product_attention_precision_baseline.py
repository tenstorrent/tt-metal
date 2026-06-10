# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for Flash-Attention SDPA (Phase 0, bfloat16).

Measures PCC, max abs error, mean abs error, and relative RMS error
against an fp32 torch reference across 4 shapes. Results recorded in
verification_report.md.
"""

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 4, 128, 64),  # medium multi-head
    (2, 4, 256, 64),  # batched, multi-block KV loop
    (1, 8, 1024, 128),  # long context, wide head
]
IDS = [f"B{s[0]}_H{s[1]}_S{s[2]}_D{s[3]}" for s in SHAPES]

PCC_THRESHOLD = 0.995


def torch_sdpa(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    return torch.matmul(torch.softmax(scores, dim=-1), v)


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_precision_baseline(device, shape):
    torch.manual_seed(0)
    q = torch.randn(shape)
    k = torch.randn(shape)
    v = torch.randn(shape)
    scale = 1.0 / math.sqrt(shape[-1])
    golden = torch_sdpa(q, k, v, scale)

    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tt(q), tt(k), tt(v))).float()

    abs_err = (out - golden).abs()
    rel_rms = (abs_err.pow(2).mean().sqrt() / golden.float().std()).item()
    print(
        f"\nPRECISION {shape}: max_abs={abs_err.max().item():.5f} "
        f"mean_abs={abs_err.mean().item():.6f} rel_rms={rel_rms:.5f}"
    )
    print(comp_allclose(golden, out, rtol=0.05, atol=0.05))
    _, pcc_msg = assert_with_pcc(golden, out, PCC_THRESHOLD)
    print(f"PCC {shape}: {pcc_msg}")

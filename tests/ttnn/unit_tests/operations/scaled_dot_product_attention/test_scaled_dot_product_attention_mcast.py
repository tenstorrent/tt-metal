# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness for the NoC-multicast (KV read-once + broadcast) SDPA variant.

The mcast path (KV-outer restructure + Mcast1D row-family broadcast) activates
only for its guarded shape class: bf16, self-attn, MHA, mask none, tile-aligned,
fp32_dest_acc_en=False, B*H <= grid rows, n_q_chunks >= grid cols, subchunk <= 4.
These shapes are picked to HIT that predicate on the 11x10 Blackhole grid
(n_q_chunks >= 11 needs S_q >= ~2816). DO NOT DELETE — this is the correctness
gate for the mcast path (the golden suite only reaches it via the long-context
fp32_dest=False cells).

Env A/B (measurement, not asserted here): SDPA_MCAST=0 forces baseline;
SDPA_MCAST_NO_BCAST=1 runs KV-outer without the broadcast. All must match torch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _fa_rand(*shape):
    n1 = torch.randn(shape)
    n2 = torch.randn(shape) * 10
    bern = torch.bernoulli(torch.full(shape, 0.001))
    return n1 + n2 * bern


# Shapes chosen to HIT the mcast predicate on the 11x10 grid.
#   (1,2,3072,64):  B*H=2 rows, Sq_t=96, n_q_chunks=12 (>=11), subchunk<=2
#   (1,1,4096,64):  B*H=1 row,  Sq_t=128, n_q_chunks=16, subchunk<=2
#   (1,4,4096,64):  B*H=4 rows, n_q_chunks=16
#   (1,3,3072,128): B*H=3 rows, Dt=4 (target head-dim), n_q_chunks=12
MCAST_SHAPES = [
    (1, 2, 3072, 64),
    (1, 1, 4096, 64),
    (1, 4, 4096, 64),
    (1, 3, 3072, 128),
]


@pytest.mark.parametrize("shape", MCAST_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_sdpa_mcast_correctness(device, shape):
    torch.manual_seed(1234)
    B, H, S, D = shape
    Q = _fa_rand(B, H, S, D)
    K = _fa_rand(B, H, S, D)
    V = _fa_rand(B, H, S, D)
    scale = 1.0 / math.sqrt(D)
    expected = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), attn_mask=None, is_causal=False, scale=scale
    )

    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )

    out = scaled_dot_product_attention(tq, tk, tv, scale=scale, compute_kernel_config=cfg)
    result = ttnn.to_torch(out).to(torch.float32)
    a, e = result.flatten().numpy(), expected.flatten().numpy()
    pcc = np.corrcoef(a, e)[0, 1]
    logger.info(f"mcast {shape}: PCC={pcc:.5f} max_abs={np.abs(a - e).max():.4f}")
    assert_with_pcc(expected, result, pcc=0.99)

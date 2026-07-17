# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness for the Q-OUTER LOCKSTEP multicast SDPA variant (SDPA_LOCKSTEP=1).

This variant keeps the Q-outer loop (one (m,l,O) per core) and reuses the shipped
compute kernel verbatim — so it's a measured ~7% SPEEDUP that leaves the compute
floor unchanged (perf_findings.md § Q-outer lockstep). It activates only for its
valid shape class (bf16, self-attn, MHA, mask none, tile-aligned,
fp32_dest_acc_en=False, B*H <= grid rows). DO NOT DELETE — correctness gate for
the lockstep path (the golden suite reaches it via long-context fp32_dest=False
cells if it is ever made default-on).
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


@pytest.fixture(autouse=True)
def _enable_lockstep(monkeypatch):
    monkeypatch.setenv("SDPA_LOCKSTEP", "1")


def _fa_rand(*shape):
    n1 = torch.randn(shape)
    n2 = torch.randn(shape) * 10
    bern = torch.bernoulli(torch.full(shape, 0.001))
    return n1 + n2 * bern


LOCKSTEP_SHAPES = [
    (1, 2, 3072, 64),
    (1, 1, 4096, 64),  # golden long-context cell
    (1, 1, 8192, 64),  # golden long-context cell
    (1, 4, 4096, 64),  # golden long-context cell
    (1, 3, 3072, 128),  # target head-dim (Dt=4)
    (1, 8, 512, 64),  # small: n_q_chunks=2 < cores_per_row -> clamp-pad path
    (1, 10, 9472, 128),  # THE perf-target shape
]


@pytest.mark.parametrize("shape", LOCKSTEP_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_sdpa_lockstep_correctness(device, shape):
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
    logger.info(f"lockstep {shape}: PCC={pcc:.5f} max_abs={np.abs(a - e).max():.4f}")
    assert_with_pcc(expected, result, pcc=0.99)

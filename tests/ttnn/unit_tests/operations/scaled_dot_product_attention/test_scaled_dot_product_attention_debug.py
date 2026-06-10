# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug tests for Flash-Attention SDPA. DO NOT DELETE.

Documents the multi-KV-block debugging session (see probes/probe_001..004):

  probe_001 (uniform softmax across 2 blocks) -> alpha=1 path correct.
  probe_002 (max jumps at block 1) -> caught alpha = exp(m_new - m_prev):
      output 1/(1+e) instead of e/(1+e). DEST_TO_SRCB ELWSUB computes CB - DEST.
  probe_004 (DPRINT m/m_prev/alpha) -> with DEST_TO_SRCA, alpha was correct only
      at tile row 0 (0.369) and 1.0 for rows > 0: dest-reuse Sub broke rows > 0.
      Fix: compute alpha with BinaryFpu(prev_max, running_max, Sub) — both
      operands are CBs anyway.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def test_uniform_softmax_two_blocks(device):
    """Q=K=ones -> uniform softmax over 256 kv; O = mean(V) = 1.275 everywhere (alpha=1 path)."""
    B, H, Sq, Skv, D = 1, 1, 32, 256, 32
    q = torch.ones(B, H, Sq, D)
    k = torch.ones(B, H, Skv, D)
    v = torch.zeros(B, H, Skv, D)
    for n in range(Skv):
        v[:, :, n, :] = n / 100.0
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv)).float()
    assert torch.allclose(out, torch.full_like(out, 1.275), rtol=0.02, atol=0.02), f"mean={out.mean()}"


def test_max_jump_two_blocks(device):
    """Block0 scores=0, block1 scores=1 -> O = e/(1+e) = 0.731 (alpha=exp(-1) rescale path).

    Wrong alpha direction gives 1/(1+e)=0.269; alpha broken for rows>0 gives 0.496.
    """
    B, H, Sq, Skv, D = 1, 1, 32, 256, 32
    q = torch.zeros(B, H, Sq, D)
    q[..., 0] = 1.0
    k = torch.zeros(B, H, Skv, D)
    k[:, :, 128:, 0] = math.sqrt(D)
    v = torch.zeros(B, H, Skv, D)
    v[:, :, 128:, :] = 1.0
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv)).float()
    expected = math.e / (1 + math.e)
    assert torch.allclose(
        out, torch.full_like(out, expected), rtol=0.02, atol=0.02
    ), f"min={out.min()} max={out.max()} expected {expected:.3f}"

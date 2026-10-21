# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the reference attention in query blocks matches the all-at-once version (no device needed).

Small random shapes keep it fast; the block size is patched down so each prompt spans many blocks."""

import pytest
import torch
import torch.nn.functional as F

from models.experimental.mimo_v2_6_flash.reference import model as ref
from models.experimental.mimo_v2_6_flash.reference.config import AttentionDims


def attention_all_at_once(x, w, dims, v_scale, positions, history=None):
    """reference.model.attention as it was before query blocks: scores for every query at once."""
    S = x.shape[0]
    q = (x @ w["q_proj"].T).view(S, dims.num_heads, dims.head_dim).transpose(0, 1)
    k = (x @ w["k_proj"].T).view(S, dims.num_kv_heads, dims.head_dim).transpose(0, 1)
    v = (x @ w["v_proj"].T).view(S, dims.num_kv_heads, dims.v_head_dim).transpose(0, 1)
    if v_scale is not None:
        v = v * v_scale
    cos, sin = ref.rope_cos_sin(positions, dims.rope_dim, dims.rope_theta)
    q = ref.apply_partial_rope(q, cos, sin, dims.rope_dim)
    k = ref.apply_partial_rope(k, cos, sin, dims.rope_dim)
    if history is not None:
        k_all, v_all, k_pos = history.append(k, v, positions)
    else:
        k_all, v_all, k_pos = k, v, positions
    groups = dims.num_heads // dims.num_kv_heads
    k_rep = k_all.repeat_interleave(groups, dim=0)
    v_rep = v_all.repeat_interleave(groups, dim=0)
    scores = (q @ k_rep.transpose(1, 2)) * dims.scaling
    allowed = k_pos[None, :] <= positions[:, None]
    if dims.sliding_window is not None:
        allowed &= (positions[:, None] - k_pos[None, :]) < dims.sliding_window
    scores = scores.masked_fill(~allowed[None], float("-inf"))
    if dims.has_sink:
        sinks = w["attention_sink_bias"].reshape(-1, 1, 1).expand(-1, S, 1)
        scores = torch.cat((scores, sinks), dim=-1)
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = F.softmax(scores, dim=-1, dtype=torch.float32)
    if dims.has_sink:
        probs = probs[..., :-1]
    out = (probs @ v_rep).transpose(0, 1).reshape(S, dims.num_heads * dims.v_head_dim)
    return out @ w["o_proj"].T


def make_layer(swa, seed=0):
    """MiMo's shape ratios at a small size: 8 Q heads, 2 KV heads, QK 48 / V 32, partial RoPE 16."""
    g = torch.Generator().manual_seed(seed)
    dims = AttentionDims(
        is_swa=swa,
        num_heads=8,
        num_kv_heads=2,
        head_dim=48,
        v_head_dim=32,
        rope_dim=16,
        rope_theta=1e4 if swa else 1e7,
        sliding_window=40 if swa else None,
        has_sink=swa,
    )
    hidden = 64
    w = {
        "q_proj": torch.randn(dims.num_heads * dims.head_dim, hidden, generator=g) * 0.1,
        "k_proj": torch.randn(dims.num_kv_heads * dims.head_dim, hidden, generator=g) * 0.1,
        "v_proj": torch.randn(dims.num_kv_heads * dims.v_head_dim, hidden, generator=g) * 0.1,
        "o_proj": torch.randn(hidden, dims.num_heads * dims.v_head_dim, generator=g) * 0.1,
    }
    if swa:
        w["attention_sink_bias"] = torch.randn(dims.num_heads, generator=g)
    return dims, w, hidden


@pytest.mark.parametrize("swa", [True, False], ids=["swa", "global"])
@pytest.mark.parametrize("block", [16, 64, 1024])
def test_query_blocks_match_all_at_once(monkeypatch, swa, block):
    monkeypatch.setattr(ref, "QUERY_BLOCK", block)
    dims, w, hidden = make_layer(swa)
    x = torch.randn(300, hidden, generator=torch.Generator().manual_seed(1))

    # One pass over the whole prompt.
    got = ref.attention(x, w, dims, 0.707, torch.arange(300))
    want = attention_all_at_once(x, w, dims, 0.707, torch.arange(300))
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)

    # The same prompt in chunks, each reading the earlier ones from the history, like chunked prefill.
    h_new, h_old = ref.KVHistory(), ref.KVHistory()
    for s in range(0, 300, 70):
        pos = torch.arange(s, min(s + 70, 300))
        got = ref.attention(x[pos], w, dims, 0.707, pos, h_new)
        want = attention_all_at_once(x[pos], w, dims, 0.707, pos, h_old)
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)

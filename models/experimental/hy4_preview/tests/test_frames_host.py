# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The cache frames (tt/frames.py) against the references: what the device modules compute, in torch.

The device attention and indexer load reordered weights, apply interleaved RoPE and (the indexer)
rotate by a Hadamard matrix. Here the same steps run in torch on the reordered weights and must
give the reference's tensors mapped by ``frames.*_to_device``, and the same q . k, scores and
selection as the reference. Pure CPU, no device.
"""

import pytest
import torch
import torch.nn.functional as F

from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention, rms_norm, rope_cos_sin
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig
from models.experimental.hy4_preview.tt import frames

ATTENTION = {
    "small": AttentionConfig(
        hidden_size=96,
        num_heads=4,
        q_lora_rank=48,
        kv_lora_rank=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=8,
        v_head_dim=16,
    ),
    "hy4-dims-narrow": AttentionConfig(hidden_size=128, num_heads=2, q_lora_rank=64, kv_lora_rank=512),
}
INDEXER = {
    "small": IndexerConfig(hidden_size=96, q_lora_rank=48, num_heads=4, head_dim=32, qk_rope_head_dim=8, topk=16),
    "hy4-dims-narrow": IndexerConfig(hidden_size=128, q_lora_rank=64, num_heads=4),
}


def _interleaved_rope(x, cos, sin):
    """What tt/attention.py and tt/indexer.py _rope compute: x * cos + (x @ swap) * sin."""
    return x * cos + (x @ frames.rope_swap(x.shape[-1])) * sin


def test_perms_and_constants():
    assert frames.interleave_perm(8).tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
    # [pass 4 | rope 4] -> [rope interleaved | pass]
    assert frames.index_key_perm(8, 4).tolist() == [4, 6, 5, 7, 0, 1, 2, 3]
    x = torch.arange(1.0, 9.0)
    assert (x @ frames.rope_swap(8)).tolist() == [-2, 1, -4, 3, -6, 5, -8, 7]
    h = frames.hadamard(128)
    assert torch.equal(h, h.to(torch.bfloat16).float()), "the entries are bf16 values"
    assert torch.allclose(h @ h.T, torch.eye(128), atol=2e-3), "orthonormal up to the bf16 rounding of 128^-0.5"
    v = torch.randn(3, 8)
    # the *_from_device maps are the inverses
    latent, key = torch.randn(5, 40), torch.randn(5, 3, 128)
    assert torch.equal(frames.rope_from_device(frames.rope_to_device(v)), v)
    assert torch.equal(frames.tail_from_device(frames.tail_to_device(latent, 8), 8), latent)
    assert torch.allclose(frames.index_from_device(frames.index_to_device(key, 64), 64), key, atol=1e-5)
    # the swap is the half-split rotation in the interleaved order
    half = torch.cat([-v[:, 4:], v[:, :4]], dim=-1)
    assert torch.equal(frames.rope_to_device(v) @ frames.rope_swap(8), frames.rope_to_device(half))


@pytest.mark.parametrize("name", list(ATTENTION))
def test_attention_frame(name):
    cfg = ATTENTION[name]
    torch.manual_seed(0)
    ref = Hy4Attention(cfg)
    with torch.no_grad():
        ref.learnable_sink_param.normal_()
        ref.q_a_layernorm.uniform_(0.5, 1.5)
        ref.kv_a_layernorm.uniform_(0.5, 1.5)
    S, L, N, R = 40, cfg.kv_lora_rank, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim
    x = torch.randn(S, cfg.hidden_size)
    cos, sin = rope_cos_sin(torch.arange(S) + 7, cfg)
    with torch.no_grad():
        r_q_resid, r_k_pass, r_k_rope = ref.latent(x, cos, sin)
        r_q = ref.queries(r_q_resid, cos, sin)  # [H, S, qk]
        r_keys, _ = ref.keys_values(r_k_pass, r_k_rope)  # [H, S, qk]

    w = frames.attention_weights(ref.weights(), cfg)
    assert torch.equal(w["kv_a_proj_with_mqa"][:L], ref.kv_a_proj_with_mqa[:L]), "the latent's kv_pass rows do not move"
    assert sorted(w) == sorted(ref.weights())
    d_cos, d_sin = frames.rope_tables(cos, sin)
    with torch.no_grad():
        kv = F.linear(x, w["kv_a_proj_with_mqa"])
        d_k_pass = rms_norm(kv[:, :L], ref.kv_a_layernorm, cfg.norm_eps)
        d_k_rope = _interleaved_rope(kv[:, L:], d_cos, d_sin)
        q = F.linear(r_q_resid, w["q_b_proj"]).view(S, cfg.num_heads, cfg.qk_head_dim).transpose(0, 1)
        d_q = torch.cat([q[..., :N], _interleaved_rope(q[..., N:], d_cos, d_sin)], dim=-1)

    assert torch.allclose(d_k_pass, r_k_pass, atol=1e-6)
    assert torch.allclose(d_k_rope, frames.rope_to_device(r_k_rope), atol=1e-5), "cached k_rope: reference, reordered"
    assert torch.allclose(d_q, frames.tail_to_device(r_q, R), atol=1e-5)
    latent = torch.cat([r_k_pass, r_k_rope], dim=-1)
    assert torch.allclose(torch.cat([d_k_pass, d_k_rope], dim=-1), frames.tail_to_device(latent, R), atol=1e-5)
    # the attention logits are the reference's
    d_keys = torch.cat([r_keys[..., :N], d_k_rope.expand(cfg.num_heads, -1, -1)], dim=-1)
    assert torch.allclose(d_q @ d_keys.transpose(-1, -2), r_q @ r_keys.transpose(-1, -2), atol=1e-4)


@pytest.mark.parametrize("name", list(INDEXER))
def test_indexer_frame(name):
    cfg = INDEXER[name]
    torch.manual_seed(1)
    ref = Hy4Indexer(cfg)
    with torch.no_grad():
        ref.k_norm_weight.uniform_(0.5, 1.5)
        ref.k_norm_bias.normal_(std=0.1)
    T, R = 300, cfg.qk_rope_head_dim
    x = torch.randn(T, cfg.hidden_size)
    q_resid = torch.randn(T, cfg.q_lora_rank)
    acfg = AttentionConfig(qk_rope_head_dim=R, rope_theta=cfg.rope_theta)
    cos, sin = rope_cos_sin(torch.arange(T), acfg)
    with torch.no_grad():
        r_keys, r_q, r_w = ref.keys(x, cos, sin), ref.queries(q_resid, cos, sin), ref.head_weights(x)
        r_scores = ref.scores(r_q, r_w, r_keys)

    w = frames.indexer_weights(ref.weights(), cfg)
    d_cos, d_sin = frames.rope_tables(cos, sin)
    had = frames.hadamard(cfg.head_dim)
    with torch.no_grad():
        k = F.layer_norm(
            F.linear(x, w["wk.weight"]), (cfg.head_dim,), w["k_norm.weight"], w["k_norm.bias"], cfg.norm_eps
        )
        d_keys = torch.cat([_interleaved_rope(k[:, :R], d_cos, d_sin), k[:, R:]], dim=-1) @ had
        q = F.linear(q_resid, w["wq_b.weight"]).view(T, cfg.num_heads, cfg.head_dim)
        d_q = torch.cat([_interleaved_rope(q[..., :R], d_cos[:, None], d_sin[:, None]), q[..., R:]], dim=-1) @ had
        d_scores = ref.scores(d_q, r_w, d_keys)

    assert torch.allclose(d_keys, frames.index_to_device(r_keys, R), atol=1e-4), "stored index key: reference, mapped"
    assert torch.allclose(d_q, frames.index_to_device(r_q, R), atol=1e-4)
    # The rotation is orthonormal up to the bf16 rounding of its one entry value, a uniform factor on
    # every score, so the order of the scores (the selection) is the reference's.
    factor = (had[0, 0] ** 2 * cfg.head_dim).item() ** 2
    assert abs(factor - 1) < 5e-3
    assert torch.allclose(d_scores, r_scores * factor, rtol=1e-3, atol=1e-5)
    k_top = min(cfg.topk, T)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    r_sel = r_scores.masked_fill(~causal, float("-inf")).topk(k_top, dim=-1).indices.sort(-1).values
    d_sel = d_scores.masked_fill(~causal, float("-inf")).topk(k_top, dim=-1).indices.sort(-1).values
    rows = torch.arange(T) >= k_top  # rows that see more than top-k keys: the selection is a real choice
    agree = (r_sel[rows] == d_sel[rows]).float().mean().item() if rows.any() else 1.0
    assert agree > 0.999, f"selection agreement {agree}"

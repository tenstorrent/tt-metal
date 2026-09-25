# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: the standalone reference (reference/model.py) against an inline torch golden written
independently here — complex-number RoPE, explicit masked softmax, einsum projections — on a reduced
config with random weights, so the two oracles cannot drift apart."""

import math

import torch

from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_3_5_128b.config import MistralMediumConfig
from models.demos.mistral_medium_3_5_128b.reference.model import (
    ReferenceDecoderLayer,
    build_reference_model,
    random_state_dict,
    rope_cos_sin,
)

REDUCED = dict(hidden_size=512, intermediate_size=1024, num_attention_heads=8, num_key_value_heads=2, vocab_size=1024)


def _inline_yarn_inv_freq(cfg):
    d, base, factor = cfg.head_dim, cfg.rope_theta, cfg.rope_factor
    corr = lambda rot: d * math.log(cfg.original_max_position_embeddings / (rot * 2 * math.pi)) / (2 * math.log(base))
    lo, hi = max(math.floor(corr(cfg.beta_fast)), 0), min(math.ceil(corr(cfg.beta_slow)), d - 1)
    i = torch.arange(d // 2, dtype=torch.float64)
    extrap = base ** (-2 * i / d)
    blend = 1 - ((i - lo) / (hi - lo)).clamp(0, 1)  # 1 -> keep extrapolated, 0 -> interpolated
    return extrap / factor * (1 - blend) + extrap * blend, 0.1 * math.log(factor) + 1.0


def _inline_rope(x, positions, cfg):
    """Rotate pairs (j, j + d/2) by angle pos * inv_freq_j as complex numbers, then scale."""
    inv, scale = _inline_yarn_inv_freq(cfg)
    ang = positions[:, None].double() * inv[None, :]
    rot = torch.polar(torch.full_like(ang, scale), ang).to(torch.complex64)
    half = x.shape[-1] // 2
    z = torch.complex(x[..., :half].float(), x[..., half:].float()) * rot
    return torch.cat([z.real, z.imag], dim=-1).to(x.dtype)


def _inline_layer(x, w, cfg):
    eps = cfg.rms_norm_eps

    def norm(t, g):
        tf = t.float()
        return g * (tf / torch.sqrt((tf * tf).mean(-1, keepdim=True) + eps)).to(t.dtype)

    s, hq, hkv, d = x.shape[1], cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    pos = torch.arange(s)
    h = norm(x, w["input_layernorm.weight"])
    q = torch.einsum("bsh,oh->bso", h, w["self_attn.q_proj.weight"]).view(1, s, hq, d).permute(0, 2, 1, 3)
    k = torch.einsum("bsh,oh->bso", h, w["self_attn.k_proj.weight"]).view(1, s, hkv, d).permute(0, 2, 1, 3)
    v = torch.einsum("bsh,oh->bso", h, w["self_attn.v_proj.weight"]).view(1, s, hkv, d).permute(0, 2, 1, 3)
    q, k = _inline_rope(q, pos, cfg), _inline_rope(k, pos, cfg)
    group = hq // hkv
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float().repeat_interleave(group, 1)) / math.sqrt(d)
    scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), float("-inf"))
    probs = scores.softmax(-1)
    att = torch.einsum("bhqk,bhkd->bhqd", probs, v.float().repeat_interleave(group, 1)).to(x.dtype)
    att = att.permute(0, 2, 1, 3).reshape(1, s, hq * d)
    x = x + torch.einsum("bsi,oi->bso", att, w["self_attn.o_proj.weight"])
    h = norm(x, w["post_attention_layernorm.weight"])
    gate = torch.einsum("bsh,oh->bso", h, w["mlp.gate_proj.weight"])
    up = torch.einsum("bsh,oh->bso", h, w["mlp.up_proj.weight"])
    act = gate * torch.sigmoid(gate.float()).to(gate.dtype) * up
    return x + torch.einsum("bsi,oi->bso", act, w["mlp.down_proj.weight"]), k, v


@torch.no_grad()
def test_decoder_layer_reference_vs_inline_golden():
    cfg = MistralMediumConfig().reduced(num_hidden_layers=1, **REDUCED)
    sd = random_state_dict(cfg, seed=11)
    w = {k[len("layers.0.") :]: v for k, v in sd.items() if k.startswith("layers.0.")}
    layer = ReferenceDecoderLayer(cfg).to(torch.bfloat16).eval()
    layer.load_state_dict(w)

    s = 384
    x = torch.randn(1, s, cfg.hidden_size, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    pos = torch.arange(s)
    cos, sin = rope_cos_sin(cfg, pos)
    out, k, v = layer(x, cos, sin, pos)
    out_g, k_g, v_g = _inline_layer(x, w, cfg)
    for name, a, b in (("out", out, out_g), ("k", k, k_g), ("v", v, v_g)):
        passing, pcc = comp_pcc(b.float(), a.float(), 0.999)
        assert passing, f"{name}: reference vs inline golden {pcc}"


@torch.no_grad()
def test_full_model_reference_vs_inline_golden():
    """M1 widening: the whole model — embedding, all 88 layers, final norm, lm_head — at reduced width.
    Run in fp32 on both sides so 88 layers of rounding do not mask a math difference."""
    cfg = MistralMediumConfig().reduced(**REDUCED)
    assert cfg.num_hidden_layers == 88
    sd = {k: v.float() for k, v in random_state_dict(cfg, seed=13).items()}
    model = build_reference_model(cfg, sd, dtype=torch.float32)
    tokens = torch.randint(0, cfg.vocab_size, (1, 160), generator=torch.Generator().manual_seed(3))
    logits, _, kvs = model(tokens)

    h = sd["embed_tokens.weight"][tokens]
    for i in range(cfg.num_hidden_layers):
        w = {k[len(f"layers.{i}.") :]: v for k, v in sd.items() if k.startswith(f"layers.{i}.")}
        h, k_g, v_g = _inline_layer(h, w, cfg)
        for name, a, b in (("k", kvs[i][0], k_g), ("v", kvs[i][1], v_g)):
            passing, pcc = comp_pcc(b, a, 0.99999)
            assert passing, f"layer {i} {name}: reference vs inline golden {pcc}"
    hf = h / torch.sqrt((h * h).mean(-1, keepdim=True) + cfg.rms_norm_eps) * sd["norm.weight"]
    logits_g = hf @ sd["lm_head.weight"].t()
    passing, pcc = comp_pcc(logits_g, logits, 0.99999)
    assert passing, f"logits: reference vs inline golden {pcc}"

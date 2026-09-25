# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone torch reference for the Mistral-Medium-3.5 language model (Ministral3 architecture).

Torch only: no ttnn, no device code. Computes in bf16 (inputs, weights, cos/sin), the recipe's fixed
convention; ``tests/test_config.py`` pins it against the upstream HF classes.

Provenance (transformers 5.12.1):
  * ``yarn_inv_freq``      <- modeling_rope_utils.py:327-459 ``_compute_yarn_parameters``
  * ``rope_cos_sin``       <- models/ministral3/modeling_ministral3.py:324-336 ``Ministral3RotaryEmbedding.forward``
  * ``rms_norm``           <- modeling_ministral3.py:193-207 ``Ministral3RMSNorm``
  * ``apply_rope``         <- modeling_ministral3.py:35-65 ``rotate_half`` / ``apply_rotary_pos_emb``
  * ``llama4_attn_scale``  <- modeling_ministral3.py:105-108 ``get_llama_4_attn_scale``
  * ``ReferenceAttention`` <- modeling_ministral3.py:111-173 ``Ministral3Attention`` (sdpa path)
  * ``ReferenceMLP``       <- modeling_ministral3.py:176-190 ``Ministral3MLP``
  * ``ReferenceDecoderLayer`` / ``ReferenceModel`` <- modeling_ministral3.py:213-412
"""

import math

import torch
from torch import nn


def yarn_inv_freq(cfg):
    """YaRN inverse frequencies ([head_dim/2] fp32) and the cos/sin attention scaling factor."""
    base = cfg.rope_theta
    dim = cfg.head_dim
    factor = cfg.rope_factor
    orig_max = cfg.original_max_position_embeddings

    def get_mscale(scale, mscale=1.0):
        return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0

    if cfg.mscale and cfg.mscale_all_dim:
        attention_factor = float(get_mscale(factor, cfg.mscale) / get_mscale(factor, cfg.mscale_all_dim))
    else:
        attention_factor = get_mscale(factor)
    beta_fast = cfg.beta_fast or 32
    beta_slow = cfg.beta_slow or 1

    def find_correction_dim(num_rotations):
        return (dim * math.log(orig_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = find_correction_dim(beta_fast), find_correction_dim(beta_slow)
    if cfg.rope_truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    extrapolation_factor = 1 - ramp
    inv_freq = inv_freq_interpolation * (1 - extrapolation_factor) + inv_freq_extrapolation * extrapolation_factor
    return inv_freq, attention_factor


def rope_cos_sin(cfg, positions, dtype=torch.bfloat16):
    """HF-layout ``[S, head_dim]`` cos/sin (``cat(freqs, freqs)``, scaled by attention_factor) in ``dtype``."""
    inv_freq, scaling = yarn_inv_freq(cfg)
    freqs = torch.outer(positions.float(), inv_freq.float())
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * scaling).to(dtype), (emb.sin() * scaling).to(dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x [B, H, S, D]; cos/sin [S, D] (HF half-split convention)."""
    return (x * cos) + (rotate_half(x) * sin)


def rms_norm(x, weight, eps):
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(dtype)


def llama4_attn_scale(positions, beta, orig_max):
    """Per-position query scale [1, 1, S, 1]; exactly 1 when beta == 0 (this checkpoint)."""
    scale = 1 + beta * torch.log(1 + torch.floor(positions.float() / orig_max))
    return scale[None, None, :, None]


def causal_attention(q, k, v, q_offset=0):
    """GQA causal attention. q [B, Hq, Sq, D] at absolute positions q_offset + [0, Sq); k/v [B, Hkv, Sk, D]
    at positions [0, Sk). Softmax scale 1/sqrt(D)."""
    n_rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    sq, sk = q.shape[-2], k.shape[-2]
    if q_offset == 0 and sq == sk:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    q_pos = torch.arange(sq)[:, None] + q_offset
    k_pos = torch.arange(sk)[None, :]
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=k_pos <= q_pos)


class ReferenceAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, positions, past_kv=None):
        """x [B, S, hidden] at ``positions``. ``past_kv``: (k, v) of the preceding prefix (post-RoPE K).
        Returns (out [B, S, hidden], k_rot [B, Hkv, S, D], v [B, Hkv, S, D]) for THIS chunk."""
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q * llama4_attn_scale(
            positions, self.cfg.llama_4_scaling_beta, self.cfg.original_max_position_embeddings
        ).to(q.dtype)
        k_all, v_all, offset = k, v, 0
        if past_kv is not None:
            k_all = torch.cat([past_kv[0], k], dim=-2)
            v_all = torch.cat([past_kv[1], v], dim=-2)
            offset = past_kv[0].shape[-2]
        out = causal_attention(q, k_all, v_all, q_offset=offset)
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out), k, v


class ReferenceMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class ReferenceRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class ReferenceDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = ReferenceAttention(cfg)
        self.mlp = ReferenceMLP(cfg)
        self.input_layernorm = ReferenceRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = ReferenceRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, positions, past_kv=None):
        """Returns (out, k_rot, v) — the layer output and this chunk's cache entries."""
        attn, k, v = self.self_attn(self.input_layernorm(x), cos, sin, positions, past_kv=past_kv)
        h = x + attn
        return h + self.mlp(self.post_attention_layernorm(h)), k, v


class ReferenceModel(nn.Module):
    """Embedding -> N decoder layers -> final norm -> lm_head. HF state-dict names without the
    ``model.language_model.`` prefix (``embed_tokens.weight``, ``layers.{i}.*``, ``norm.weight``,
    ``lm_head.weight``)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([ReferenceDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = ReferenceRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, tokens, start_pos=0, past_kvs=None, hiddens=None, logits_last=None):
        """tokens [B, S] at positions start_pos + [0, S). Returns (logits, final_hidden, [(k, v)] per layer).
        ``hiddens``: optional list, appended with the embedding and every layer's output.
        ``logits_last``: project only the last N positions through the lm_head."""
        positions = torch.arange(tokens.shape[-1]) + start_pos
        cos, sin = rope_cos_sin(self.cfg, positions, dtype=self.embed_tokens.weight.dtype)
        h = self.embed_tokens(tokens)
        if hiddens is not None:
            hiddens.append(h)
        kvs = []
        for i, layer in enumerate(self.layers):
            h, k, v = layer(h, cos, sin, positions, past_kv=None if past_kvs is None else past_kvs[i])
            kvs.append((k, v))
            if hiddens is not None:
                hiddens.append(h)
        h = self.norm(h)
        return self.lm_head(h if logits_last is None else h[:, -logits_last:]), h, kvs


def random_linear(out_features, in_features, generator, dtype=torch.bfloat16):
    """HF ``[out, in]`` weight with std ``in**-0.5``: unit-scale activations, softmax logits of O(1)."""
    return (torch.randn(out_features, in_features, generator=generator) * in_features**-0.5).to(dtype)


def random_state_dict(cfg, seed=0, dtype=torch.bfloat16):
    """Deterministic random weights in ReferenceModel naming (fan-in scaled, norm gains near 1).
    Layer ``i`` uses seed ``seed * 1000 + i``, so a model's layer weights do not depend on its depth."""
    g = torch.Generator().manual_seed(seed)
    sd = {
        "embed_tokens.weight": torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g).to(dtype),
        "norm.weight": (1.0 + 0.1 * torch.randn(cfg.hidden_size, generator=g)).to(dtype),
        "lm_head.weight": random_linear(cfg.vocab_size, cfg.hidden_size, g, dtype),
    }
    for i in range(cfg.num_hidden_layers):
        for k, v in random_layer_state_dict(cfg, seed=seed * 1000 + i, dtype=dtype).items():
            sd[f"layers.{i}.{k}"] = v
    return sd


def random_layer_state_dict(cfg, seed=0, dtype=torch.bfloat16):
    """One decoder layer's random weights (``ReferenceDecoderLayer`` naming) without building a model,
    so full-width layers are cheap to generate."""
    g = torch.Generator().manual_seed(seed)
    h, i = cfg.hidden_size, cfg.intermediate_size
    q, kv = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
    return {
        "input_layernorm.weight": (1.0 + 0.1 * torch.randn(h, generator=g)).to(dtype),
        "self_attn.q_proj.weight": random_linear(q, h, g, dtype),
        "self_attn.k_proj.weight": random_linear(kv, h, g, dtype),
        "self_attn.v_proj.weight": random_linear(kv, h, g, dtype),
        "self_attn.o_proj.weight": random_linear(h, q, g, dtype),
        "post_attention_layernorm.weight": (1.0 + 0.1 * torch.randn(h, generator=g)).to(dtype),
        "mlp.gate_proj.weight": random_linear(i, h, g, dtype),
        "mlp.up_proj.weight": random_linear(i, h, g, dtype),
        "mlp.down_proj.weight": random_linear(h, i, g, dtype),
    }


def build_reference_model(cfg, state_dict, dtype=torch.bfloat16):
    """Reference model holding ``state_dict`` (cast to ``dtype``) without a random init pass."""
    with torch.device("meta"):
        model = ReferenceModel(cfg)
    model.load_state_dict({k: v.to(dtype) for k, v in state_dict.items()}, strict=True, assign=True)
    return model.eval()

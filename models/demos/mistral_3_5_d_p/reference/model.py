# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch (CPU) reference for Mistral-Medium-3.5 prefill — the oracle every PCC test compares against.

Nothing is vendored. Mistral-Medium-3.5's text backbone is HF ``ministral3``, which ships INSIDE
transformers (5.12.1: ``transformers/models/ministral3/modeling_ministral3.py``) and constructs
standalone from a config, so this module imports those classes directly rather than trimming a copy
into the repo — recipe D1 step 2's first branch. That keeps the oracle honest: it is upstream's math,
not our reading of it.

Provenance of what we lean on (transformers 5.12.1, modeling_ministral3.py):
  * ``Ministral3RMSNorm``        L193  — plain RMSNorm, fp32 variance, ``weight * x``
  * ``Ministral3MLP``            L176  — ``down(silu(gate(x)) * up(x))``, no bias
  * ``Ministral3Attention``      L111  — GQA, full rotary, no bias, no QK-norm, no sinks
  * ``Ministral3DecoderLayer``   L213  — pre-norm attention + pre-norm MLP, two residual adds
  * ``Ministral3RotaryEmbedding`` L273 — HF ``ROPE_INIT_FUNCTIONS["yarn"]`` + ``attention_scaling``
  * ``Ministral3Model``          L339  — embedding -> layers -> final norm
  * ``Ministral3ForCausalLM``          — + lm_head

The one Ministral3-only term is the per-position query scale
``1 + llama_4_scaling_beta * log(1 + floor(pos / original_max_position_embeddings))``
(``get_llama_4_attn_scale``, L105). This checkpoint sets ``llama_4_scaling_beta: 0``, so it is
exactly 1.0 at every position; :func:`assert_llama4_scale_is_identity` pins that, and the TT
attention omits the term. A checkpoint that ever sets beta > 0 fails that assert loudly instead of
producing a silently unscaled Q.

Torch only — no ttnn, no device code, no checkpoint. Weights are supplied by the caller so the
reference and the TT module can be driven from IDENTICAL random tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config


# ---------------------------------------------------------------------------------------------
# HF module handles (imported lazily so `import reference.model` stays cheap for the producers)
# ---------------------------------------------------------------------------------------------
def hf_modules():
    """The upstream ministral3 classes this reference is built out of."""
    from transformers.models.ministral3 import modeling_ministral3 as m

    return m


def assert_llama4_scale_is_identity(hf_config) -> None:
    """Fail loudly if the Ministral3 per-position query scale is not the identity.

    The TT attention has no equivalent of ``get_llama_4_attn_scale``; it is correct to omit only
    while ``llama_4_scaling_beta == 0`` (which is what this checkpoint ships). Called by the
    reference builders and by the config test, so a future config that turns the term on cannot
    silently diverge from the device.
    """
    beta = hf_config.rope_parameters.get("llama_4_scaling_beta", 0)
    if beta:
        raise NotImplementedError(
            f"llama_4_scaling_beta={beta} enables the Ministral3 per-position query scale, which the "
            "TT attention does not implement; add it to tt/attention/prefill.py before proceeding."
        )


# ---------------------------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------------------------
def hf_rope_cos_sin(hf_config, seq_len: int, *, offset: int = 0, dtype=torch.float32):
    """HF-convention ``(cos, sin)`` of shape ``[1, seq_len, head_dim]`` for positions
    ``[offset, offset + seq_len)``, exactly as ``Ministral3RotaryEmbedding`` produces them
    (YaRN inv_freq, ``attention_scaling`` folded in, halves concatenated)."""
    m = hf_modules()
    rot = m.Ministral3RotaryEmbedding(config=hf_config)
    pos = torch.arange(offset, offset + seq_len, dtype=torch.long).unsqueeze(0)
    x = torch.zeros(1, seq_len, 1, dtype=dtype)
    cos, sin = rot(x, pos)
    return cos, sin


def causal_mask(seq_len: int, *, offset: int = 0, dtype=torch.float32) -> torch.Tensor:
    """Additive causal mask ``[1, 1, seq_len, offset + seq_len]``: query row ``i`` (global position
    ``offset + i``) may attend keys ``0 .. offset + i``. ``offset`` covers the chunked case where
    the queries are a later chunk and the keys span the whole accumulated prefix."""
    q_pos = torch.arange(offset, offset + seq_len).unsqueeze(-1)
    k_pos = torch.arange(offset + seq_len).unsqueeze(0)
    mask = torch.zeros(seq_len, offset + seq_len, dtype=dtype)
    mask.masked_fill_(k_pos > q_pos, float("-inf"))
    return mask[None, None]


# ---------------------------------------------------------------------------------------------
# Block references — each takes an hf_config + a plain state dict of torch weights
# ---------------------------------------------------------------------------------------------
def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``Ministral3RMSNorm`` on the given weight. Plain RMSNorm — no Gemma ``1 + w`` fold."""
    m = hf_modules()
    norm = m.Ministral3RMSNorm(weight.shape[-1], eps=eps)
    with torch.no_grad():
        norm.weight.copy_(weight.float())
    with torch.no_grad():
        return norm(x.float())


def mlp_reference(x: torch.Tensor, state_dict: dict, hf_config) -> torch.Tensor:
    """``Ministral3MLP``: ``down(silu(gate(x)) * up(x))``. ``state_dict`` holds
    ``{gate,up,down}_proj.weight`` in HF ``[out, in]`` layout."""
    m = hf_modules()
    mlp = m.Ministral3MLP(hf_config).float()
    mlp.load_state_dict({k: v.float() for k, v in state_dict.items()})
    with torch.no_grad():
        return mlp(x.float())


@dataclass
class AttentionRefOut:
    output: torch.Tensor  # [B, S, hidden]
    k_post_rope: torch.Tensor  # [B, num_kv_heads, S, head_dim] — what the KV cache stores
    v: torch.Tensor  # [B, num_kv_heads, S, head_dim]


def attention_reference(
    x: torch.Tensor,
    state_dict: dict,
    hf_config,
    *,
    offset: int = 0,
    past_k: Optional[torch.Tensor] = None,
    past_v: Optional[torch.Tensor] = None,
    cos_sin=None,
) -> AttentionRefOut:
    """``Ministral3Attention`` over a causal prefill chunk, returning the output AND the post-RoPE
    K / raw V the device writes into its KV cache.

    ``offset`` + ``past_k`` / ``past_v`` express the chunked case: the queries are the chunk at
    global positions ``[offset, offset + S)`` and the keys/values are the prefix concatenated with
    this chunk's. ``cos_sin`` lets a test share the exact table with the TT side so the comparison
    measures attention rather than the rope constants.
    """
    m = hf_modules()
    assert_llama4_scale_is_identity(hf_config)
    attn = m.Ministral3Attention(config=hf_config, layer_idx=0).float()
    attn.load_state_dict({k: v.float() for k, v in state_dict.items()})

    x = x.float()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    B, S, _ = x.shape
    head_dim = attn.head_dim
    n_q = hf_config.num_attention_heads
    n_kv = hf_config.num_key_value_heads

    with torch.no_grad():
        q = attn.q_proj(x).view(B, S, n_q, head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(B, S, n_kv, head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(B, S, n_kv, head_dim).transpose(1, 2)

        cos, sin = cos_sin if cos_sin is not None else hf_rope_cos_sin(hf_config, S, offset=offset)
        q, k = m.apply_rotary_pos_emb(q, k, cos, sin)

        k_full = k if past_k is None else torch.cat([past_k.float(), k], dim=-2)
        v_full = v if past_v is None else torch.cat([past_v.float(), v], dim=-2)

        rep = n_q // n_kv
        scores = q @ m.repeat_kv(k_full, rep).transpose(-1, -2) * attn.scaling
        scores = scores + causal_mask(S, offset=k_full.shape[-2] - S)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        out = probs @ m.repeat_kv(v_full, rep)
        out = out.transpose(1, 2).reshape(B, S, n_q * head_dim)
        out = attn.o_proj(out)

    return AttentionRefOut(output=out, k_post_rope=k, v=v)


def decoder_layer_reference(x: torch.Tensor, state_dict: dict, hf_config, *, offset: int = 0):
    """``Ministral3DecoderLayer``: norm -> attention -> residual -> norm -> MLP -> residual.

    ``state_dict`` is the HF layer sub-state (``self_attn.*``, ``mlp.*``, ``input_layernorm.weight``,
    ``post_attention_layernorm.weight``). Returns ``(hidden_states, AttentionRefOut)`` so a test can
    check the composition and the layer's KV in one pass.
    """
    assert_llama4_scale_is_identity(hf_config)
    x = x.float()
    if x.dim() == 2:
        x = x.unsqueeze(0)

    attn_sd = {k[len("self_attn.") :]: v for k, v in state_dict.items() if k.startswith("self_attn.")}
    mlp_sd = {k[len("mlp.") :]: v for k, v in state_dict.items() if k.startswith("mlp.")}

    normed = rms_norm_reference(x, state_dict["input_layernorm.weight"], hf_config.rms_norm_eps)
    attn_out = attention_reference(normed, attn_sd, hf_config, offset=offset)
    hidden = x + attn_out.output
    normed2 = rms_norm_reference(hidden, state_dict["post_attention_layernorm.weight"], hf_config.rms_norm_eps)
    hidden = hidden + mlp_reference(normed2, mlp_sd, hf_config)
    return hidden, attn_out


# ---------------------------------------------------------------------------------------------
# Whole-model reference (M1)
# ---------------------------------------------------------------------------------------------
@dataclass
class ModelRefOut:
    logits: Optional[torch.Tensor]  # [B, S, vocab] (None when lm_head is skipped)
    hidden_states: torch.Tensor  # post-final-norm [B, S, hidden]
    kv: list  # per layer: (k_post_rope, v), each [B, num_kv_heads, S, head_dim]


def build_reference_model(hf_config, state_dict=None, *, seed: Optional[int] = None):
    """Construct ``Ministral3ForCausalLM`` at ``hf_config``.

    ``state_dict`` None + ``seed`` set => HF's own random init under that seed, which is how every
    pre-P1 test gets identical weights on both sides (dump the module's state_dict and feed it to
    the TT model). Returns the eval-mode fp32 model.
    """
    m = hf_modules()
    assert_llama4_scale_is_identity(hf_config)
    if seed is not None:
        torch.manual_seed(seed)
    model = m.Ministral3ForCausalLM(hf_config).float().eval()
    if state_dict is not None:
        model.load_state_dict({k: v.float() for k, v in state_dict.items()}, strict=True)
    return model


def model_reference_forward(model, input_ids: torch.Tensor, *, skip_lm_head: bool = False) -> ModelRefOut:
    """One-shot causal prefill through the whole reference, capturing per-layer post-RoPE K / raw V.

    K/V are captured by hooking each attention's ``k_proj`` output path via a ``DynamicCache``: HF
    writes exactly the post-RoPE K and raw V into it, which is the same pair the device cache holds.
    """
    from transformers.cache_utils import DynamicCache

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model.model(input_ids=input_ids, use_cache=True, past_key_values=cache)
        hidden = out.last_hidden_state
        logits = None if skip_lm_head else model.lm_head(hidden)
    kv = [(cache.layers[i].keys, cache.layers[i].values) for i in range(model.config.num_hidden_layers)]
    return ModelRefOut(logits=logits, hidden_states=hidden, kv=kv)


def bf16_reference_forward(hf_config, state_dict, input_ids: torch.Tensor) -> ModelRefOut:
    """The same forward with bf16 STORAGE — the accuracy class the spec's dataformats put us in.

    Why this exists. The spec fixes ``activations.default = bfloat16``, so the device carries its
    residual stream in bf16 while :func:`model_reference_forward` computes in fp32. Measured on this
    model at 2 layers / 512 tokens, a bf16 CPU forward of the SAME weights lands at PCC 0.968
    against the fp32 one on the final hidden state — i.e. the residual stream is the most
    fp32-sensitive quantity in the model, and a bf16 implementation of any kind deviates there.

    So this is the calibration the whole-model test uses: the device must be at least as faithful to
    fp32 as a bf16 reference is. (It is: 0.989 against 0.968 on the same comparison, and 0.997
    against 0.987 on layer 1's K.) HF still reduces the norms and the softmax in fp32 internally, so
    this is bf16 storage with fp32 reductions — the same shape as the device's arithmetic, which is
    what makes it the right yardstick rather than a weaker oracle.

    This is NOT a substitute for the fp32 oracle. Per-block tests compare against fp32 and clear the
    spec's bar there; this is only for the depth-accumulated residual stream.
    """
    from transformers.cache_utils import DynamicCache

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    model = build_reference_model(hf_config, state_dict=state_dict).to(torch.bfloat16).eval()
    cache = DynamicCache(config=hf_config)
    with torch.no_grad():
        out = model.model(input_ids=input_ids, use_cache=True, past_key_values=cache)
        hidden = out.last_hidden_state.float()
        logits = model.lm_head(out.last_hidden_state).float()
    kv = [(cache.layers[i].keys.float(), cache.layers[i].values.float()) for i in range(hf_config.num_hidden_layers)]
    return ModelRefOut(logits=logits, hidden_states=hidden, kv=kv)


# ---------------------------------------------------------------------------------------------
# Inline torch golden — a SECOND, independent implementation of the same math
# ---------------------------------------------------------------------------------------------
# Written from the architecture description rather than from the HF source, so that
# tests/unit/test_reference_model.py comparing the two catches a misread of either. Deliberately
# plain: no HF classes, no config objects, no caches.
def golden_rms_norm(x, weight, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def golden_silu(x):
    return x * torch.sigmoid(x)


def golden_mlp(x, gate_w, up_w, down_w):
    x = x.float()
    return (golden_silu(x @ gate_w.float().t()) * (x @ up_w.float().t())) @ down_w.float().t()


def golden_yarn_inv_freq(head_dim, theta, factor, orig_max_pos, beta_fast, beta_slow, truncate=True):
    """YaRN inverse frequencies + the mscale that multiplies cos/sin."""

    def correction_dim(rot):
        return head_dim * math.log(orig_max_pos / (rot * 2 * math.pi)) / (2 * math.log(theta))

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, head_dim - 1)
    if low == high:
        high += 0.001

    pos_freqs = theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)
    ramp = ((torch.arange(head_dim // 2).float() - low) / (high - low)).clamp(0, 1)
    extrapolation_factor = 1.0 - ramp
    inv_freq = interpolation * (1.0 - extrapolation_factor) + extrapolation * extrapolation_factor
    mscale = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
    return inv_freq, mscale


def golden_rope_hf(t, cos, sin):
    """HF (concat-halves) rotation of ``t`` [B, H, S, head_dim] by ``cos``/``sin`` [.., S, head_dim]."""
    half = t.shape[-1] // 2
    rotated = torch.cat([-t[..., half:], t[..., :half]], dim=-1)
    return t * cos + rotated * sin


def golden_cos_sin_hf(seq_len, head_dim, *, offset=0, **yarn):
    inv_freq, mscale = golden_yarn_inv_freq(head_dim, **yarn)
    pos = torch.arange(offset, offset + seq_len).float()
    freqs = torch.outer(pos, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1) * mscale
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1) * mscale
    return cos, sin


def golden_attention(x, w, *, n_q, n_kv, head_dim, cos, sin, offset=0, past_k=None, past_v=None):
    """Dense-GQA causal attention. ``w`` holds ``q,k,v,o`` in HF ``[out, in]`` layout (no bias)."""
    x = x.float()
    B, S, _ = x.shape
    q = (x @ w["q"].float().t()).view(B, S, n_q, head_dim).transpose(1, 2)
    k = (x @ w["k"].float().t()).view(B, S, n_kv, head_dim).transpose(1, 2)
    v = (x @ w["v"].float().t()).view(B, S, n_kv, head_dim).transpose(1, 2)
    q, k = golden_rope_hf(q, cos, sin), golden_rope_hf(k, cos, sin)

    k_full = k if past_k is None else torch.cat([past_k.float(), k], dim=-2)
    v_full = v if past_v is None else torch.cat([past_v.float(), v], dim=-2)
    rep = n_q // n_kv
    scores = q @ k_full.repeat_interleave(rep, dim=1).transpose(-1, -2) * head_dim**-0.5
    scores = scores + causal_mask(S, offset=k_full.shape[-2] - S)
    out = torch.softmax(scores, dim=-1) @ v_full.repeat_interleave(rep, dim=1)
    out = out.transpose(1, 2).reshape(B, S, n_q * head_dim)
    return out @ w["o"].float().t(), k, v


def golden_decoder_layer(x, w, *, n_q, n_kv, head_dim, eps, cos, sin, offset=0):
    """One decoder layer of inline golden math: norm -> attn -> add -> norm -> mlp -> add."""
    x = x.float()
    attn_in = golden_rms_norm(x, w["input_layernorm"], eps)
    attn_out, k, v = golden_attention(
        attn_in, w, n_q=n_q, n_kv=n_kv, head_dim=head_dim, cos=cos, sin=sin, offset=offset
    )
    h = x + attn_out
    mlp_in = golden_rms_norm(h, w["post_attention_layernorm"], eps)
    h = h + golden_mlp(mlp_in, w["gate"], w["up"], w["down"])
    return h, k, v


def golden_model(input_ids, w, hf_config, cfg=MistralMedium35Config):
    """Whole-model inline golden: embedding -> N layers -> final norm -> lm_head.

    ``w`` is ``{"embed": [vocab, hidden], "layers": [ {per-layer} ... ], "norm": [hidden],
    "lm_head": [vocab, hidden]}``. Head geometry comes from ``hf_config`` (a reduced config works
    unchanged); the rope constants come from ``cfg``, the transcribed config-constants class.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    seq_len = input_ids.shape[1]
    head_dim = hf_config.head_dim
    n_kv = hf_config.num_key_value_heads
    n_q = hf_config.num_attention_heads
    cos, sin = golden_cos_sin_hf(
        seq_len,
        head_dim,
        theta=cfg.ROPE_THETA,
        factor=cfg.YARN_FACTOR,
        orig_max_pos=cfg.YARN_ORIG_MAX_POS,
        beta_fast=cfg.YARN_BETA_FAST,
        beta_slow=cfg.YARN_BETA_SLOW,
        truncate=cfg.YARN_TRUNCATE,
    )
    h = torch.nn.functional.embedding(input_ids, w["embed"].float())
    kv = []
    for layer_w in w["layers"]:
        h, k, v = golden_decoder_layer(
            h, layer_w, n_q=n_q, n_kv=n_kv, head_dim=head_dim, eps=cfg.RMS_NORM_EPS, cos=cos, sin=sin
        )
        kv.append((k, v))
    h = golden_rms_norm(h, w["norm"], cfg.RMS_NORM_EPS)
    return h @ w["lm_head"].float().t(), h, kv


def random_layer_weights(hf_config, *, seed=0, scale=0.02):
    """Random per-layer weights in the inline-golden naming, shared with the TT side by a test."""
    torch.manual_seed(seed)
    h, i = hf_config.hidden_size, hf_config.intermediate_size
    hd, nq, nkv = hf_config.head_dim, hf_config.num_attention_heads, hf_config.num_key_value_heads
    return {
        "q": torch.randn(nq * hd, h) * scale,
        "k": torch.randn(nkv * hd, h) * scale,
        "v": torch.randn(nkv * hd, h) * scale,
        "o": torch.randn(h, nq * hd) * scale,
        "gate": torch.randn(i, h) * scale,
        "up": torch.randn(i, h) * scale,
        "down": torch.randn(h, i) * scale,
        "input_layernorm": torch.randn(h) * 0.1 + 1.0,
        "post_attention_layernorm": torch.randn(h) * 0.1 + 1.0,
    }


def hf_layer_state_dict(layer_w: dict) -> dict:
    """Translate inline-golden layer weights into the HF layer sub-state naming."""
    return {
        "self_attn.q_proj.weight": layer_w["q"],
        "self_attn.k_proj.weight": layer_w["k"],
        "self_attn.v_proj.weight": layer_w["v"],
        "self_attn.o_proj.weight": layer_w["o"],
        "mlp.gate_proj.weight": layer_w["gate"],
        "mlp.up_proj.weight": layer_w["up"],
        "mlp.down_proj.weight": layer_w["down"],
        "input_layernorm.weight": layer_w["input_layernorm"],
        "post_attention_layernorm.weight": layer_w["post_attention_layernorm"],
    }

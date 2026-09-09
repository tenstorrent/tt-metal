# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-independent torch reference for Llama-3.1-8B-Instruct prefill.

**Purity contract** (same one as `deepseek_v3_d_p/reference/kda/README.md`): this module depends on
PyTorch only. It does not import ttnn, model device code, mesh fixtures, checkpoints or profilers,
and it is the semantic oracle every device PCC test measures against.

## Why the classes are vendored rather than imported

`transformers.models.llama.modeling_llama` does import and construct standalone, and D1's first
choice is to import it. It is vendored anyway, for two reasons:

1. **Hub kernels.** Upstream `LlamaAttention` carries `@use_kernelized_func(apply_rotary_pos_emb)`
   and `LlamaRMSNorm` carries `@use_kernel_forward_from_hub("RMSNorm")`. Those can substitute a
   downloaded kernel for the torch math, which makes the oracle depend on network state.
2. **The 4.x → 5.x rope move.** The checkpoint's config says `transformers 4.42.3`; the installed
   transformers is 5.x, which reads rope settings from `config.rope_parameters` rather than
   `config.rope_scaling`. A reference that goes through the live config plumbing silently loses the
   llama3 scaling on one side of that boundary — and losing it costs nothing at short ISL and
   everything past 8192 tokens, which is the worst possible failure shape for a bring-up.

The live upstream classes remain the **anchor**: `tests/torch_ref/test_llama_reference.py` runs the
vendored math against them (with the hub kernels off) at the real dims, so the two cannot drift.

## Provenance

Transcribed from `transformers` 5.12.1:

| Here | Upstream |
|---|---|
| `rotate_half`, `apply_rotary_pos_emb` | `models/llama/modeling_llama.py` |
| `repeat_kv`, `eager_attention_forward` | `models/llama/modeling_llama.py` |
| `RefRMSNorm` | `LlamaRMSNorm` |
| `RefMLP` | `LlamaMLP` |
| `RefAttention` | `LlamaAttention` |
| `RefDecoderLayer` | `LlamaDecoderLayer` |
| `RefRotaryEmbedding`, `compute_llama3_inv_freq` | `modeling_rope_utils.py:550` `_compute_llama3_parameters` + `LlamaRotaryEmbedding.forward` |

## Numeric convention (recipe §4) — fp16, not bf16

Every reference and golden in this bring-up computes in **fp16** (`torch.float16`), regardless of
the checkpoint's bf16 dtype and of the ttnn dtypes under test: inputs, weights and cos/sin are fp16
and the per-module goldens dumped from here are fp16. This is a fixed convention, not a per-model
choice, and the packages this code was borrowed from do **not** follow it (their casts are
shape-tuned, not structural) — so their casts were replaced rather than carried over.

Two places deliberately keep an internal fp32 accumulation, because that is the upstream math and
the anchor test compares against it:

* `RefRMSNorm.forward` upcasts for the variance and casts the result back (HF does exactly this).
* `eager_attention_forward` takes the softmax in fp32 and casts back to the query dtype.

`compute_llama3_inv_freq` and the cos/sin table are built in fp32 and cast to fp16 at the end: the
inverse frequencies span ~10 orders of magnitude and fp16 cannot hold the smallest of them.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .config import LlamaConfigConstants

REF_DTYPE = torch.float16
"""The bring-up's fixed reference dtype (recipe §4). Not negotiable per model."""


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def compute_llama3_inv_freq(
    head_dim: int,
    rope_theta: float,
    rope_scaling: dict[str, Any],
    device=None,
) -> torch.Tensor:
    """Llama-3.1 inverse frequencies (`rope_type: "llama3"`), in fp32.

    Transcribed from `transformers.modeling_rope_utils._compute_llama3_parameters` (5.12.1, line
    550), reading the scaling parameters from a plain dict instead of a live config so this does not
    depend on whether the installed transformers spells them `rope_scaling` or `rope_parameters`.

    Three frequency bands, split by wavelength against the ORIGINAL 8192-token context:
      * wavelen < high_freq_wavelen (short)  -> unscaled
      * wavelen > low_freq_wavelen  (long)   -> divided by `factor`
      * in between                  (medium) -> smoothly interpolated between the two

    Returns fp32: the smallest inverse frequency here is ~1e-5 * ... and underflows in fp16, so the
    band arithmetic must not be done at the reference dtype even though its consumers are fp16.
    """
    assert rope_scaling["rope_type"] == "llama3", f"not llama3 rope scaling: {rope_scaling}"

    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / head_dim)
    )

    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


class RefRotaryEmbedding(nn.Module):
    """cos/sin tables for a position range, fp32 internally and fp16 out.

    Mirrors `LlamaRotaryEmbedding.forward`: `emb = cat(freqs, freqs)`, which is the **half-split**
    (GPT-NeoX / HF) rotation convention, paired with `rotate_half` below. The original Meta
    checkpoint uses the *interleaved* convention instead. Same weights, different pairing of head
    columns — get it wrong and PCC drops without anything raising, so the device side must be
    checked against THIS convention, not assumed.
    """

    def __init__(self, config: LlamaConfigConstants, device=None):
        super().__init__()
        self.config = config
        inv_freq = compute_llama3_inv_freq(config.head_dim, config.rope_theta, config.rope_scaling, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0  # llama3 rope: unused (upstream returns 1.0)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype = REF_DTYPE):
        """position_ids [batch, seq] -> (cos, sin) each [batch, seq, head_dim] in `dtype`."""
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Upstream `modeling_llama.rotate_half`. Half-split, NOT interleaved."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """Upstream `modeling_llama.apply_rotary_pos_emb`, with the hub-kernel decorator dropped."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class RefRMSNorm(nn.Module):
    """Upstream `LlamaRMSNorm`, hub-kernel decorator dropped.

    **Plain** RMSNorm: `x_normed * weight`. NOT the Gemma `x_normed * (1 + weight)` form that
    minimax_m3's RMSNorm folds in — Llama has no such fold, so the device side must not enable it.
    """

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=REF_DTYPE))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RefMLP(nn.Module):
    """Upstream `LlamaMLP`: `down(silu(gate(x)) * up(x))`, no bias, intermediate 14336.

    Plain SiLU SwiGLU. The clamped "swigluoai" variant (α=1.702, clamp 7.0) that both minimax_m3 and
    gpt_oss_d_p use is a DIFFERENT activation — their MLP *structure* (column-parallel gate/up,
    row-parallel down, CCL tail) ports to this model, their activation math does not.
    """

    def __init__(self, config: LlamaConfigConstants):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        bias = config.mlp_bias
        self.gate_proj = nn.Linear(h, i, bias=bias, dtype=REF_DTYPE)
        self.up_proj = nn.Linear(h, i, bias=bias, dtype=REF_DTYPE)
        self.down_proj = nn.Linear(i, h, bias=bias, dtype=REF_DTYPE)
        assert config.hidden_act == "silu", f"unexpected hidden_act {config.hidden_act}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Upstream `modeling_llama.repeat_kv`: inflate n_kv heads to n_q heads.

    Host-side only. The device path does NOT inflate — `ring_joint_scaled_dot_product_attention`
    consumes grouped V directly with a GQA-causal kernel, and the KV cache stays at n_kv heads.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(query, key, value, attention_mask, scaling: float, num_key_value_groups: int):
    """Upstream `modeling_llama.eager_attention_forward`, minus dropout/module plumbing.

    Softmax in fp32 then back to the query dtype — upstream's own behaviour, kept so the anchor test
    compares like with like.
    """
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def causal_mask(seq_len: int, dtype: torch.dtype = REF_DTYPE, device=None) -> torch.Tensor:
    """Additive causal mask [1, 1, seq_len, seq_len] for a one-shot (no prefix) prefill.

    `finfo.min` rather than `-inf` so the fp32 softmax cannot produce a NaN row.
    """
    m = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    return torch.triu(m, diagonal=1)[None, None, :, :]


class RefAttention(nn.Module):
    """Upstream `LlamaAttention`: QKV proj -> head split -> RoPE -> causal GQA SDPA -> o_proj.

    No QK-norm, no attention sinks, no sliding window — the three features the donor packages carry
    that this model does not have.

    `forward` also returns the post-RoPE K and the raw V, which is what the KV cache stores and what
    the golden trace and every per-layer KV PCC check compare against. Note **K is cached after
    RoPE and V is cached raw** — caching K pre-RoPE would need a re-rotation on every cache read.
    """

    def __init__(self, config: LlamaConfigConstants, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.scaling = config.attn_scale
        bias = config.attention_bias
        h = config.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=bias, dtype=REF_DTYPE)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=bias, dtype=REF_DTYPE)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=bias, dtype=REF_DTYPE)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=bias, dtype=REF_DTYPE)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, return_kv: bool = False):
        """hidden_states [b, s, hidden]; position_embeddings (cos, sin) each [b, s, head_dim].

        -> attn_out [b, s, hidden], or (attn_out, k_rope, v) when `return_kv`, with
        k_rope / v [b, n_kv, s, head_dim].
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if attention_mask is None:
            attention_mask = causal_mask(hidden_states.shape[1], hidden_states.dtype, hidden_states.device)

        attn_output, _ = eager_attention_forward(
            query_states, key_states, value_states, attention_mask, self.scaling, self.num_key_value_groups
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if return_kv:
            return attn_output, key_states, value_states
        return attn_output


class RefDecoderLayer(nn.Module):
    """Upstream `LlamaDecoderLayer`: pre-norm attention + pre-norm MLP, each with a residual add.

    All 32 layers are identical — no hybrid schedule, so no per-layer type dispatch anywhere in
    this bring-up (unlike minimax_m3's dense/sparse split and gpt_oss's sliding/full alternation).
    """

    def __init__(self, config: LlamaConfigConstants, layer_idx: int = 0):
        super().__init__()
        self.self_attn = RefAttention(config, layer_idx)
        self.mlp = RefMLP(config)
        self.input_layernorm = RefRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RefRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, return_kv: bool = False):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn = self.self_attn(normed, position_embeddings, attention_mask, return_kv=return_kv)
        if return_kv:
            attn, k_rope, v = attn
        hidden_states = residual + attn

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        if return_kv:
            return hidden_states, k_rope, v
        return hidden_states


class RefModel(nn.Module):
    """The whole model: embedding -> 32 identical decoder layers -> final norm -> lm_head.

    Written in D1 (rather than left to M1) because the layer stack is what the golden trace
    generator walks, and because there is nothing model-specific left to discover at M1: the layers
    are homogeneous and `tie_word_embeddings` is false, so the LM head is a real separate weight.
    """

    def __init__(self, config: LlamaConfigConstants, num_layers: int | None = None):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers if num_layers is None else num_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=REF_DTYPE)
        self.layers = nn.ModuleList([RefDecoderLayer(config, i) for i in range(self.num_layers)])
        self.norm = RefRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=REF_DTYPE)
        self.rotary_emb = RefRotaryEmbedding(config)

    def forward(self, input_ids: torch.Tensor, return_kv: bool = False):
        """input_ids [b, s] -> logits [b, s, vocab], or (logits, per_layer_kv) when `return_kv`.

        `per_layer_kv` is a list of `(k_rope, v)` in layer order, each [b, n_kv, s, head_dim] — the
        exact tensors the golden trace stores and P1/P2 PCC the device's KV cache against.
        """
        b, s = input_ids.shape
        hidden = self.embed_tokens(input_ids)
        position_ids = torch.arange(s, device=input_ids.device)[None, :].expand(b, -1)
        cos, sin = self.rotary_emb(position_ids, dtype=hidden.dtype)
        mask = causal_mask(s, hidden.dtype, hidden.device)

        per_layer_kv = []
        for layer in self.layers:
            out = layer(hidden, (cos, sin), mask, return_kv=return_kv)
            if return_kv:
                hidden, k_rope, v = out
                per_layer_kv.append((k_rope, v))
            else:
                hidden = out

        logits = self.lm_head(self.norm(hidden))
        if return_kv:
            return logits, per_layer_kv
        return logits

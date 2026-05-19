# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Reusable Qwen3-style TTNN transformer-block building blocks for ACE-Step.

This module previously hosted ``TtQwen3EmbeddingEncoder`` (a hand-rolled TTNN port of
the Qwen3-Embedding-0.6B caption encoder). That encoder has moved to
:mod:`models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_ace_step` which subclasses
the stock :class:`models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm.Qwen3ForEmbedding`
(``tt_transformers``-based).

What's left here are the **transformer-block primitives** that the ACE-Step DiT
condition encoder (:mod:`models.demos.ace_step_v1_5.ttnn_impl.condition_encoder`) reuses
for its Lyric and Timbre encoders — these are *not* Qwen3-Embedding instances; they're
ACE-Step-specific small transformers that happen to share the Qwen3 layer architecture:

- :class:`Qwen3EmbeddingEncoderConfig` — config dataclass
- :class:`TtQwen3EncoderMLP` — Qwen3-style SwiGLU MLP
- :class:`_TtQwen3EncoderLayer` — Qwen3-style attention + MLP layer (with GQA, RoPE,
  q_norm/k_norm, optional sliding-window mask)
- :func:`apply_rope_hf_style`, :func:`rotate_half_ttnn`, :func:`repeat_kv_gqa_ttnn` —
  RoPE + GQA helpers used by ``_TtQwen3EncoderLayer``
"""

from __future__ import annotations

import math

# Bounded LRU cap for the per-prompt causal+padding attention bias cache. Each entry is
# [B,1,S,S] bf16 (~128 KB at S=256), so the default cap of 32 entries keeps the cache at
# ≤ ~4 MB device DRAM. Increase via ACE_STEP_QWEN_BIAS_CACHE_MAX if you batch over many
# distinct prompts.
import os as _os
from dataclasses import dataclass

import ttnn

_QWEN_BIAS_CACHE_MAX = max(1, int(_os.environ.get("ACE_STEP_QWEN_BIAS_CACHE_MAX", "32")))


def _sdpa_head_dim_tile_padding(d_head: int) -> int:
    align = 32
    return (int(d_head) + align - 1) // align * align


def rotate_half_ttnn(x: ttnn.Tensor) -> ttnn.Tensor:
    d = int(x.shape[-1])
    h = d // 2
    b0, b1, b2 = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    x1 = ttnn.slice(x, (0, 0, 0, 0), (b0, b1, b2, h))
    x2 = ttnn.slice(x, (0, 0, 0, h), (b0, b1, b2, d))
    neg_x2 = ttnn.multiply(x2, -1.0)
    ttnn.deallocate(x2)
    return ttnn.concat([neg_x2, x1], dim=-1)


def apply_rope_hf_style(q_bhsd: ttnn.Tensor, k_bhsd: ttnn.Tensor, cos_11sd: ttnn.Tensor, sin_11sd: ttnn.Tensor):
    """RoPE with GQA: expand ``cos``/``sin`` ``[1,1,S,Dh]`` to ``[B,nh,S,Dh]`` for ``q`` and ``[B,nkv,S,Dh]`` for ``k`` (do not use nh for both)."""
    b, hq = int(q_bhsd.shape[0]), int(q_bhsd.shape[1])
    hk = int(k_bhsd.shape[1])
    cos_q = ttnn.repeat(cos_11sd, (b, hq, 1, 1))
    sin_q = ttnn.repeat(sin_11sd, (b, hq, 1, 1))
    cos_k = ttnn.repeat(cos_11sd, (b, hk, 1, 1))
    sin_k = ttnn.repeat(sin_11sd, (b, hk, 1, 1))
    q1 = ttnn.multiply(q_bhsd, cos_q)
    rh = rotate_half_ttnn(q_bhsd)
    q_rot = ttnn.add(q1, ttnn.multiply(rh, sin_q))
    ttnn.deallocate(rh)
    ttnn.deallocate(cos_q)
    ttnn.deallocate(sin_q)
    k1 = ttnn.multiply(k_bhsd, cos_k)
    rhk = rotate_half_ttnn(k_bhsd)
    k_rot = ttnn.add(k1, ttnn.multiply(rhk, sin_k))
    ttnn.deallocate(rhk)
    ttnn.deallocate(cos_k)
    ttnn.deallocate(sin_k)
    return q_rot, k_rot


@dataclass(frozen=True)
class Qwen3EmbeddingEncoderConfig:
    vocab_size: int = 151669
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_seq_len: int = 256


def repeat_kv_gqa_ttnn(x: ttnn.Tensor, n_rep: int) -> ttnn.Tensor:
    """``[B, n_kv, S, D]`` → ``[B, n_kv * n_rep, S, D]`` interleaved per HF ``repeat_kv`` / ``torch.repeat_interleave``."""
    if int(n_rep) == 1:
        return x
    return ttnn.repeat_interleave(x, int(n_rep), dim=1)


class TtQwen3EncoderMLP:
    def __init__(
        self,
        *,
        weights_np: dict,
        base: str,
        device,
        hidden_size,
        intermediate_size,
        dtype,
        mem,
        mapper,
        linear_compute_kernel_config=None,
    ):
        _ = hidden_size, intermediate_size
        self._linear_ck = linear_compute_kernel_config

        def as_w(name: str):
            return ttnn.as_tensor(
                weights_np[f"{base}.{name}.weight"],
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self.w_gate = as_w("gate_proj")
        self.w_up = as_w("up_proj")
        self.w_down = as_w("down_proj")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        lin_kw = {}
        if self._linear_ck is not None:
            lin_kw["compute_kernel_config"] = self._linear_ck
        gate = ttnn.linear(x, self.w_gate, bias=None, transpose_b=True, **lin_kw)
        up = ttnn.linear(x, self.w_up, bias=None, transpose_b=True, **lin_kw)
        gate = ttnn.silu(gate) if hasattr(ttnn, "silu") else ttnn.gelu(gate)
        h = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(h, self.w_down, bias=None, transpose_b=True, **lin_kw)
        ttnn.deallocate(h)
        return out


class _TtQwen3EncoderLayer:
    def __init__(
        self,
        *,
        device,
        weights_np: dict,
        prefix: str,
        cfg: Qwen3EmbeddingEncoderConfig,
        dtype,
        mem,
        mapper,
        sdpa_compute_kernel_config=None,
        sdpa_program_config=None,
        linear_compute_kernel_config=None,
    ):
        self.cfg = cfg
        self.dtype = dtype
        self.mem = mem
        self.eps = float(cfg.rms_norm_eps)
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.dh = cfg.head_dim
        self.scale = 1.0 / math.sqrt(float(self.dh))

        sdpa = getattr(getattr(ttnn, "transformer", None), "scaled_dot_product_attention", None)
        if sdpa is None:
            raise RuntimeError("ttnn.transformer.scaled_dot_product_attention required")
        self._sdpa = sdpa
        self._sdpa_compute_kernel_config = sdpa_compute_kernel_config
        self._sdpa_program_config = sdpa_program_config
        self._linear_ck = linear_compute_kernel_config

        def as_t(suffix: str, *, row_major: bool = False):
            key = f"{prefix}.{suffix}"
            layout = ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT
            return ttnn.as_tensor(
                weights_np[key],
                device=device,
                dtype=dtype,
                layout=layout,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self.input_ln_w = as_t("input_layernorm.weight")
        self.post_ln_w = as_t("post_attention_layernorm.weight")
        self.wq = as_t("self_attn.q_proj.weight")
        self.wk = as_t("self_attn.k_proj.weight")
        self.wv = as_t("self_attn.v_proj.weight")
        self.wo = as_t("self_attn.o_proj.weight")
        self.q_norm_w = as_t("self_attn.q_norm.weight")
        self.k_norm_w = as_t("self_attn.k_norm.weight")
        self.mlp = TtQwen3EncoderMLP(
            weights_np=weights_np,
            base=f"{prefix}.mlp",
            device=device,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            dtype=dtype,
            mem=mem,
            mapper=mapper,
            linear_compute_kernel_config=linear_compute_kernel_config,
        )

    def __call__(self, hidden_b1sh: ttnn.Tensor, cos_11sd: ttnn.Tensor, sin_11sd: ttnn.Tensor, attn_bias_b11ss):
        res = hidden_b1sh
        x = ttnn.to_layout(hidden_b1sh, ttnn.TILE_LAYOUT)
        x = ttnn.rms_norm(x, weight=self.input_ln_w, epsilon=self.eps, memory_config=self.mem)

        b = int(x.shape[0])
        s = int(x.shape[2])
        H, kv_h, Dh = self.nh, self.nkv, self.dh

        lin_kw = {}
        if self._linear_ck is not None:
            lin_kw["compute_kernel_config"] = self._linear_ck
        q = ttnn.linear(x, self.wq, bias=None, transpose_b=True, **lin_kw)
        k = ttnn.linear(x, self.wk, bias=None, transpose_b=True, **lin_kw)
        v = ttnn.linear(x, self.wv, bias=None, transpose_b=True, **lin_kw)

        q = ttnn.reshape(q, (b, 1, s, H, Dh))
        q = ttnn.permute(q, (0, 3, 2, 4, 1))
        q = ttnn.reshape(q, (b, H, s, Dh))
        k = ttnn.reshape(k, (b, 1, s, kv_h, Dh))
        k = ttnn.permute(k, (0, 3, 2, 4, 1))
        k = ttnn.reshape(k, (b, kv_h, s, Dh))
        v = ttnn.reshape(v, (b, 1, s, kv_h, Dh))
        v = ttnn.permute(v, (0, 3, 2, 4, 1))
        v = ttnn.reshape(v, (b, kv_h, s, Dh))

        q = ttnn.rms_norm(q, weight=self.q_norm_w, epsilon=self.eps, memory_config=self.mem)
        k = ttnn.rms_norm(k, weight=self.k_norm_w, epsilon=self.eps, memory_config=self.mem)

        q, k = apply_rope_hf_style(q, k, cos_11sd, sin_11sd)

        if kv_h != H:
            rep = H // kv_h
            k = repeat_kv_gqa_ttnn(k, rep)
            v = repeat_kv_gqa_ttnn(v, rep)

        sdpa_d = _sdpa_head_dim_tile_padding(self.dh)
        if sdpa_d > self.dh:
            pt = sdpa_d - self.dh
            pad4 = ((0, 0), (0, 0), (0, 0), (0, pt))
            q = ttnn.pad(q, padding=pad4, value=0.0)
            k = ttnn.pad(k, padding=pad4, value=0.0)
            v = ttnn.pad(v, padding=pad4, value=0.0)

        mask_repeated = None
        mask_tt = attn_bias_b11ss
        if mask_tt is not None:
            mask_tt = ttnn.repeat(mask_tt, (1, H, 1, 1))
            mask_repeated = mask_tt

        sdpa_kw = dict(attn_mask=mask_tt, is_causal=False, scale=self.scale)
        if self._sdpa_compute_kernel_config is not None:
            sdpa_kw["compute_kernel_config"] = self._sdpa_compute_kernel_config
        if self._sdpa_program_config is not None:
            sdpa_kw["program_config"] = self._sdpa_program_config
        ctx = self._sdpa(q, k, v, **sdpa_kw)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        if mask_repeated is not None:
            ttnn.deallocate(mask_repeated)

        if sdpa_d > self.dh:
            ctx = ttnn.slice(ctx, (0, 0, 0, 0), (b, H, s, self.dh))

        ctx = ttnn.permute(ctx, (0, 2, 1, 3))
        ctx = ttnn.reshape(ctx, (b, 1, s, H * Dh))
        attn_out = ttnn.linear(ctx, self.wo, bias=None, transpose_b=True, **lin_kw)
        ttnn.deallocate(ctx)

        h = ttnn.add(res, attn_out)
        res2 = h
        h2 = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
        h2 = ttnn.rms_norm(h2, weight=self.post_ln_w, epsilon=self.eps, memory_config=self.mem)
        ff = self.mlp(h2)
        return ttnn.add(res2, ff)


__all__ = [
    "Qwen3EmbeddingEncoderConfig",
    "TtQwen3EncoderMLP",
    "apply_rope_hf_style",
    "repeat_kv_gqa_ttnn",
    "rotate_half_ttnn",
]

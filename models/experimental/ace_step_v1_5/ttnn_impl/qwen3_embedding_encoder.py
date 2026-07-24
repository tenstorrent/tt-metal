# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Reusable Qwen3-style TTNN transformer-block building blocks for ACE-Step, plus the full
``TtQwen3EmbeddingEncoder`` model (HF ``Qwen3Model`` forward, no LM head).

The **transformer-block primitives** are reused by the ACE-Step DiT condition encoder
(:mod:`models.experimental.ace_step_v1_5.ttnn_impl.condition_encoder`) for its Lyric and Timbre
encoders — small transformers that share the Qwen3 layer architecture:

- :class:`Qwen3EmbeddingEncoderConfig` — config dataclass
- :class:`TtQwen3EncoderMLP` — Qwen3-style SwiGLU MLP
- :class:`_TtQwen3EncoderLayer` — Qwen3-style attention + MLP layer (with GQA, RoPE,
  q_norm/k_norm, optional sliding-window mask)
- :func:`apply_rope_hf_style`, :func:`rotate_half_ttnn`, :func:`repeat_kv_gqa_ttnn` —
  RoPE + GQA helpers

:class:`TtQwen3EmbeddingEncoder` is the full ``Qwen3Model`` forward (fixed ``max_seq_len``,
RoPE precomputed once from HF ``config.json``).
"""

from __future__ import annotations

import math
import os as _os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import ttnn

from .math_perf_env import (
    ace_step_attn_qo_weight_dtype,
    ace_step_cond_256x1024_width_sharded_memory_config,
    ace_step_cond_linear_program_config,
    ace_step_cond_mlp_gate_up_linear_program_config,
    ace_step_cond_rms_norm_kwargs,
    ace_step_encoder_2d_block_sharded_memory_config,
    ace_step_ensure_dram_activation,
    ace_step_ensure_l1_activation,
    ace_step_ensure_tile_layout,
    ace_step_init_cond_linear_compute_kernel_config,
    ace_step_init_cond_linear_fp32acc_compute_kernel_config,
    ace_step_init_cond_rmsnorm_compute_kernel_config,
    ace_step_init_cond_sdpa_compute_kernel_config,
    ace_step_init_cond_sdpa_program_config,
    ace_step_linear_l1_memory_config,
    ace_step_linear_weight_dtype,
    ace_step_memory_configs_equivalent,
    ace_step_nlp_concat_heads,
    ace_step_permute_kwargs,
    ace_step_reshape_kwargs,
    ace_step_rms_norm_block_sharded,
    ace_step_safe_deallocate,
    ace_step_sdpa_activation_kwargs,
    ace_step_sdpa_mask_memory_config,
    ace_step_try_nlp_qkv_heads_split,
    ace_step_upload_f32_np_as_bf16_tile,
)

# Bounded LRU cap for the per-prompt causal+padding attention bias cache. Each entry is
# [B,1,S,S] bf16 (~128 KB at S=256), so the default cap of 32 entries keeps the cache at
# ≤ ~4 MB device DRAM. Increase via ACE_STEP_QWEN_BIAS_CACHE_MAX if you batch over many
# distinct prompts.
_QWEN_BIAS_CACHE_MAX = max(1, int(_os.environ.get("ACE_STEP_QWEN_BIAS_CACHE_MAX", "32")))


def _sdpa_head_dim_tile_padding(d_head: int) -> int:
    align = 32
    return (int(d_head) + align - 1) // align * align


def load_qwen3_weights_np(safetensors_path: str) -> Dict[str, np.ndarray]:
    import torch
    from safetensors import safe_open

    out: Dict[str, np.ndarray] = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            t = sf.get_tensor(k)
            out[k] = t.detach().to(torch.float32).cpu().numpy()
    return out


def build_hf_rope_cos_sin_np(
    *,
    hf_model_dir: str,
    hidden_size: int,
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """HF-compatible ``cos``/``sin`` shaped ``[1, 1, S, head_dim]`` (float32 host)."""
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    cfg = AutoConfig.from_pretrained(str(hf_model_dir), local_files_only=True)
    cfg.hidden_size = hidden_size
    cfg.head_dim = head_dim
    cfg.max_position_embeddings = max(int(cfg.max_position_embeddings), max_seq_len)
    cfg.rope_theta = rope_theta

    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
    x = torch.zeros(1, max_seq_len, hidden_size, dtype=torch.bfloat16)
    cos, sin = rope(x, position_ids)
    cos_np = cos.float().numpy().reshape(1, 1, max_seq_len, head_dim)
    sin_np = sin.float().numpy().reshape(1, 1, max_seq_len, head_dim)
    return cos_np.astype(np.float32), sin_np.astype(np.float32)


def causal_padding_attn_bias_np(attention_mask_01: np.ndarray, seq_len: int) -> np.ndarray:
    """Match HF ``Qwen3Model`` + SDPA mask: causal (no peeking) and **key** padding only.

    HF ``create_causal_mask`` does not mask entire rows when the query index is padding;
    padded queries may still attend to earlier valid keys. Only **key** positions *j*
    with ``mask[..., j]==0`` are blocked, plus ``j > i`` (causal).

    Returns additive bias ``[B, 1, S, S]`` (float32): 0 keep, large negative masked.
    """
    m = np.asarray(attention_mask_01, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    b, s = m.shape
    if s != seq_len:
        raise ValueError(f"attention mask length {s} != seq_len {seq_len}")
    neg = np.float32(-1.0e9)
    j = np.arange(s, dtype=np.int32)[None, :]
    i = np.arange(s, dtype=np.int32)[:, None]
    causal_ok = (j <= i).astype(np.float32)  # [S, S], query=i, key=j
    k_ok = m[:, None, :]  # [B, 1, S] — validity of key index j only (HF SDPA convention)
    keep = causal_ok[np.newaxis, :, :] * k_ok
    return np.where(keep > 0.5, np.float32(0.0), neg).astype(np.float32)[:, np.newaxis, :, :]


# def padding_attn_bias_np(attention_mask_01: np.ndarray) -> np.ndarray:
#     """1=keep, 0=pad → additive bias ``[B,1,S,S]``."""
#     m = np.asarray(attention_mask_01, dtype=np.float32)
#     if m.ndim == 1:
#         m = m.reshape(1, -1)
#     b, s = m.shape
#     neg = np.float32(-1.0e9)
#     bias = np.zeros((b, 1, s, s), dtype=np.float32)
#     for bi in range(b):
#         row = m[bi][:, None] * m[bi][None, :]
#         bias[bi, 0, :, :] = np.where(row > 0.5, 0.0, neg)
#     return bias


def rotate_half_ttnn(x: ttnn.Tensor, *, eltwise_memory_config=None) -> ttnn.Tensor:
    d = int(x.shape[-1])
    h = d // 2
    b0, b1, b2 = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    x1 = ttnn.slice(x, (0, 0, 0, 0), (b0, b1, b2, h))
    x2 = ttnn.slice(x, (0, 0, 0, h), (b0, b1, b2, d))
    m = eltwise_memory_config
    if m is None:
        neg_x2 = ttnn.multiply(x2, -1.0)
    else:
        neg_x2 = ttnn.multiply(x2, -1.0, memory_config=m)
    ttnn.deallocate(x2)
    return ttnn.concat([neg_x2, x1], dim=-1)


def apply_rope_hf_style(
    q_bhsd: ttnn.Tensor,
    k_bhsd: ttnn.Tensor,
    cos_11sd: ttnn.Tensor,
    sin_11sd: ttnn.Tensor,
    *,
    eltwise_memory_config=None,
):
    """RoPE with GQA using fused ``rotary_embedding_hf`` when available.

    The fused op broadcasts ``cos``/``sin`` ``[1,1,S,Dh]`` across all heads and batch dims,
    eliminating the repeat×4 + rotate_half (slice+neg+concat)×2 + mul×4 + add×2 chain.
    Falls back to the manual element-wise path if the op is not present.
    """
    m = eltwise_memory_config

    hf_rope = getattr(getattr(ttnn, "experimental", None), "rotary_embedding_hf", None)
    if hf_rope is not None:
        kw = {"memory_config": m} if m is not None else {}
        q_rot = hf_rope(q_bhsd, cos_11sd, sin_11sd, is_decode_mode=False, **kw)
        k_rot = hf_rope(k_bhsd, cos_11sd, sin_11sd, is_decode_mode=False, **kw)
        return q_rot, k_rot

    def mul(a, b):
        if m is None:
            return ttnn.multiply(a, b)
        return ttnn.multiply(a, b, memory_config=m)

    def addm(a, b):
        if m is None:
            return ttnn.add(a, b)
        return ttnn.add(a, b, memory_config=m)

    b, hq = int(q_bhsd.shape[0]), int(q_bhsd.shape[1])
    hk = int(k_bhsd.shape[1])
    cos_q = ttnn.repeat(cos_11sd, (b, hq, 1, 1))
    sin_q = ttnn.repeat(sin_11sd, (b, hq, 1, 1))
    cos_k = ttnn.repeat(cos_11sd, (b, hk, 1, 1))
    sin_k = ttnn.repeat(sin_11sd, (b, hk, 1, 1))
    q1 = mul(q_bhsd, cos_q)
    rh = rotate_half_ttnn(q_bhsd, eltwise_memory_config=m)
    q_rot = addm(q1, mul(rh, sin_q))
    ttnn.deallocate(rh)
    ttnn.deallocate(cos_q)
    ttnn.deallocate(sin_q)
    k1 = mul(k_bhsd, cos_k)
    rhk = rotate_half_ttnn(k_bhsd, eltwise_memory_config=m)
    k_rot = addm(k1, mul(rhk, sin_k))
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
    """``[B, n_kv, S, D]`` → ``[B, n_kv * n_rep, S, D]`` interleaved per HF ``repeat_kv`` / ``torch.repeat_interleave``.

    Prefer native GQA in :meth:`_TtQwen3EncoderLayer.__call__` (pass ``[B, kv_h, S, D]`` K/V into SDPA).
    This helper remains for callers that still need explicit head expansion.
    """
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
        activation_l1_memory_config=None,
        linear_output_l1_memory_config=None,
        use_cond_linear_program_config: bool = True,
        mlp_weight_dtype=None,
        mlp_gate_weight_dtype=None,
        mlp_down_weight_dtype=None,
    ):
        self.device = device
        self.mem = mem
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.dtype = dtype
        self._linear_ck = linear_compute_kernel_config
        self._act_l1 = activation_l1_memory_config
        self._linear_out_l1 = linear_output_l1_memory_config
        self._use_cond_linear_pc = bool(use_cond_linear_program_config)
        self._gate_up_pc_cache: dict = {}

        _proj_dtype = ace_step_linear_weight_dtype(ttnn, dtype)
        w_dtype = mlp_weight_dtype if mlp_weight_dtype is not None else _proj_dtype
        _gate_dtype = (
            mlp_gate_weight_dtype if mlp_gate_weight_dtype is not None else ace_step_attn_qo_weight_dtype(ttnn, dtype)
        )
        _down_dtype = (
            mlp_down_weight_dtype if mlp_down_weight_dtype is not None else ace_step_attn_qo_weight_dtype(ttnn, dtype)
        )

        def as_w(name: str, *, dtype: Any | None = None):
            return ttnn.as_tensor(
                weights_np[f"{base}.{name}.weight"],
                device=device,
                dtype=dtype if dtype is not None else w_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        # TP: column-parallel gate/up, row-parallel down + all-reduce. OFF path uses the passed
        # (replicate) mapper unchanged. deg==1 makes np.split/concat the identity of the old layout.
        from models.experimental.ace_step_v1_5.ttnn_impl.tp_config import resolve_tp_config, tp_weight_mesh_mapper

        self._tp = resolve_tp_config(device)
        _deg = self._tp.degree
        _tp_on = self._tp.enabled and _deg > 1
        self._local_intermediate = int(intermediate_size) // _deg if _tp_on else int(intermediate_size)
        _col = tp_weight_mesh_mapper(device, shard_dim=0, cfg=self._tp) if _tp_on else mapper
        _row = tp_weight_mesh_mapper(device, shard_dim=1, cfg=self._tp) if _tp_on else mapper

        def _interleave(mats: list):
            splits = [np.split(m, _deg, axis=0) for m in mats]
            return np.concatenate([splits[j][d] for d in range(_deg) for j in range(len(mats))], axis=0)

        # down_proj: row-parallel (shard input=intermediate on dim 1) under TP, else replicate.
        self.w_down = ttnn.as_tensor(
            weights_np[f"{base}.down_proj.weight"],
            device=device,
            dtype=_down_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=_row,
        )
        # Wide cond MLPs (e.g. timbre 6144×2048 gate/up): L1 linear outputs clash with matmul
        # static CBs (L1 buffer @ 1096832 vs CB end @ 1167872 on Blackhole). Keep DRAM activations.
        self._mlp_keep_dram_activations = int(intermediate_size) >= 4608
        self._fused_gate_up = not self._mlp_keep_dram_activations
        if self._fused_gate_up:
            gate_host = weights_np[f"{base}.gate_proj.weight"]
            up_host = weights_np[f"{base}.up_proj.weight"]
            # Fused: interleave [gate_d, up_d] per chip so a dim-0 shard gives each its half of both.
            gate_up_host = _interleave([gate_host, up_host])
            self.w_gate_up = ttnn.as_tensor(
                gate_up_host,
                device=device,
                dtype=_gate_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=_col,
            )
            self.w_gate = None
            self.w_up = None
        else:
            # Unfused: each of gate/up is column-parallel on its own dim 0 (heads/features contiguous).
            self.w_gate = ttnn.as_tensor(
                weights_np[f"{base}.gate_proj.weight"],
                device=device,
                dtype=_gate_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=_col,
            )
            self.w_up = ttnn.as_tensor(
                weights_np[f"{base}.up_proj.weight"],
                device=device,
                dtype=_gate_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=_col,
            )
            self.w_gate_up = None

    def _l1_activation(self, t: ttnn.Tensor) -> ttnn.Tensor:
        if self._act_l1 is None:
            return t
        return ttnn.to_memory_config(t, self._act_l1)

    def _gate_up_linear_kwargs(self, *, batch_size: int, seq_len: int) -> dict:
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
        pc = None
        if self._use_cond_linear_pc:
            fused = self._fused_gate_up
            key = (int(batch_size), int(seq_len), fused)
            pc = self._gate_up_pc_cache.get(key)
            if pc is None:
                pc = ace_step_cond_mlp_gate_up_linear_program_config(
                    self.device,
                    seq_len=int(seq_len),
                    hidden_size=self.hidden_size,
                    intermediate_size=self._local_intermediate,
                    batch_size=int(batch_size),
                    out_dim=(2 * self._local_intermediate if fused else None),
                )
                if pc is not None:
                    self._gate_up_pc_cache[key] = pc
            if pc is not None:
                kw["program_config"] = pc
        if self._mlp_keep_dram_activations and self.mem is not None:
            kw["memory_config"] = self.mem
        elif self._linear_out_l1 is not None:
            kw["memory_config"] = self._linear_out_l1
        return kw

    def _down_linear_kwargs(self, *, batch_size: int, seq_len: int) -> dict:
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
        pc = None
        if self._use_cond_linear_pc:
            pc = ace_step_cond_linear_program_config(
                self.device,
                seq_len=int(seq_len),
                in_dim=self._local_intermediate,
                out_dim=self.hidden_size,
                batch_size=int(batch_size),
            )
            if pc is not None:
                kw["program_config"] = pc
        if self._mlp_keep_dram_activations and self.mem is not None:
            kw["memory_config"] = self.mem
        elif self._linear_out_l1 is not None:
            kw["memory_config"] = self._linear_out_l1
        return kw

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ace_step_ensure_tile_layout(ttnn, x)
        b_x = int(x.shape[0])
        s = int(x.shape[2])
        if self._mlp_keep_dram_activations:
            if self.mem is not None:
                x = ttnn.to_memory_config(x, self.mem)
        else:
            x = self._l1_activation(x)
        lin_gu = self._gate_up_linear_kwargs(batch_size=b_x, seq_len=s)
        # Use LoFi linear weight dtype (bfloat8_b when available) for MLP matmuls as well.
        _bf8 = getattr(ttnn, "bfloat8_b", None) or getattr(ttnn, "bfloat16", None)
        _tc_kw = {"memory_config": self._linear_out_l1} if self._linear_out_l1 is not None else {}
        x_bf8 = ttnn.typecast(x, dtype=_bf8, **_tc_kw)
        _silu_mc = (self._linear_out_l1 if not self._mlp_keep_dram_activations else None) or getattr(
            ttnn, "DRAM_MEMORY_CONFIG", None
        )
        if self._fused_gate_up:
            gu_bf8 = ttnn.linear(x_bf8, self.w_gate_up, bias=None, transpose_b=True, dtype=_bf8, **lin_gu)
            inter = self._local_intermediate  # per-chip half under TP (== full when off)
            gate = ttnn.slice(gu_bf8, (0, 0, 0, 0), (b_x, 1, s, inter))
            up = ttnn.slice(gu_bf8, (0, 0, 0, inter), (b_x, 1, s, 2 * inter))
            ace_step_safe_deallocate(ttnn, gu_bf8)
        else:
            gate = ttnn.linear(x_bf8, self.w_gate, bias=None, transpose_b=True, dtype=_bf8, **lin_gu)
            up = ttnn.linear(x_bf8, self.w_up, bias=None, transpose_b=True, dtype=_bf8, **lin_gu)
        ace_step_safe_deallocate(ttnn, x_bf8)
        gate = (
            ttnn.silu(gate, memory_config=_silu_mc)
            if hasattr(ttnn, "silu")
            else ttnn.gelu(gate, memory_config=_silu_mc)
        )
        h_bf8 = ttnn.multiply(gate, up, memory_config=_silu_mc, dtype=_bf8)
        if not self._mlp_keep_dram_activations:
            h_bf8 = self._l1_activation(h_bf8)
        lin_down = self._down_linear_kwargs(batch_size=b_x, seq_len=s)
        out_bf8 = ttnn.linear(h_bf8, self.w_down, bias=None, transpose_b=True, dtype=_bf8, **lin_down)
        ace_step_safe_deallocate(ttnn, h_bf8)
        # Row-parallel down_proj: sum the per-chip partial outputs across the TP shards.
        if self._tp.enabled and self._tp.degree > 1:
            from models.experimental.ace_step_v1_5.ttnn_impl.tp_config import tp_all_reduce

            out_bf8 = tp_all_reduce(out_bf8, self.device, cfg=self._tp)
        return ttnn.typecast(out_bf8, dtype=self.dtype, **_tc_kw)


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
        activation_l1_memory_config=None,
        linear_output_l1_memory_config=None,
        use_cond_linear_program_config: bool = True,
        projection_dtype=None,
        attn_qo_dtype=None,
        mlp_weight_dtype=None,
    ):
        self.device = device
        self.cfg = cfg
        self.dtype = dtype
        self._proj_dtype = (
            projection_dtype if projection_dtype is not None else ace_step_linear_weight_dtype(ttnn, dtype)
        )
        self._attn_qo_dtype = attn_qo_dtype if attn_qo_dtype is not None else ace_step_attn_qo_weight_dtype(ttnn, dtype)
        self._mlp_weight_dtype = (
            mlp_weight_dtype if mlp_weight_dtype is not None else ace_step_linear_weight_dtype(ttnn, dtype)
        )
        self.mem = mem
        self.eps = float(cfg.rms_norm_eps)
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.dh = cfg.head_dim
        self.hidden_size = int(cfg.hidden_size)
        self.scale = 1.0 / math.sqrt(float(self.dh))

        sdpa = getattr(getattr(ttnn, "transformer", None), "scaled_dot_product_attention", None)
        if sdpa is None:
            raise RuntimeError("ttnn.transformer.scaled_dot_product_attention required")
        self._sdpa = sdpa
        self._sdpa_compute_kernel_config = sdpa_compute_kernel_config
        self._sdpa_program_config = sdpa_program_config
        self._linear_ck = linear_compute_kernel_config
        self._act_l1 = activation_l1_memory_config
        self._linear_out_l1 = linear_output_l1_memory_config
        self._use_cond_linear_pc = bool(use_cond_linear_program_config)
        # Keys: (batch, seq, in_dim, out_dim) — GQA makes q's N (nh*dh) wider than k/v (nkv*dh).
        self._attn_pc_cache: dict = {}
        self._rms_norm_ck = ace_step_init_cond_rmsnorm_compute_kernel_config(device)

        def as_t(suffix: str, *, row_major: bool = False, proj: bool = False, qo: bool = False):
            key = f"{prefix}.{suffix}"
            layout = ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT
            if qo:
                _d = self._attn_qo_dtype
            elif proj:
                _d = self._proj_dtype
            else:
                _d = dtype
            return ttnn.as_tensor(
                weights_np[key],
                device=device,
                dtype=_d,
                layout=layout,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self.input_ln_w = as_t("input_layernorm.weight")
        self.post_ln_w = as_t("post_attention_layernorm.weight")
        # Fused QKV: concat [q, k, v] projection weights into one [q_dim+2*kv_dim, hidden]
        # tensor. One matmul (instead of separate q + kv) feeds the single-input
        # ``nlp_create_qkv_heads`` kernel — replaces two ShardedToInterleaved + the slower
        # two-tensor head-split with one of each (mirrors tt_transformers Attention prefill).
        # TP: head-parallel attention — Q/K/V column-parallel by head, o_proj row-parallel + all-reduce.
        from models.experimental.ace_step_v1_5.ttnn_impl.tp_config import resolve_tp_config, tp_weight_mesh_mapper

        self._tp = resolve_tp_config(device)
        _deg = self._tp.degree
        _tp_on = self._tp.enabled and _deg > 1
        if _tp_on and (self.nh % _deg != 0 or self.nkv % _deg != 0):
            raise ValueError(f"TP degree {_deg} must divide nh {self.nh} and nkv {self.nkv}")
        self._nh_local = self.nh // _deg if _tp_on else self.nh
        self._nkv_local = self.nkv // _deg if _tp_on else self.nkv
        _col = tp_weight_mesh_mapper(device, shard_dim=0, cfg=self._tp) if _tp_on else mapper
        _row = tp_weight_mesh_mapper(device, shard_dim=1, cfg=self._tp) if _tp_on else mapper

        wq_host = weights_np[f"{prefix}.self_attn.q_proj.weight"]
        wk_host = weights_np[f"{prefix}.self_attn.k_proj.weight"]
        wv_host = weights_np[f"{prefix}.self_attn.v_proj.weight"]
        # Per-chip local q/kv output dims (== full when TP off).
        self._q_dim_o = int(wq_host.shape[0]) // _deg if _tp_on else int(wq_host.shape[0])
        self._kv_dim_o = int(wk_host.shape[0]) // _deg if _tp_on else int(wk_host.shape[0])
        if _tp_on:
            # Interleave [q_d, k_d, v_d] per chip so a dim-0 shard gives each its q+k+v heads.
            gq = np.split(wq_host, _deg, axis=0)
            gk = np.split(wk_host, _deg, axis=0)
            gv = np.split(wv_host, _deg, axis=0)
            wqkv_host = np.concatenate([c for d in range(_deg) for c in (gq[d], gk[d], gv[d])], axis=0)
        else:
            wqkv_host = np.concatenate([wq_host, wk_host, wv_host], axis=0)
        self.wqkv = ttnn.as_tensor(
            wqkv_host,
            device=device,
            dtype=self._attn_qo_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=_col,
        )
        self.wo = ttnn.as_tensor(
            weights_np[f"{prefix}.self_attn.o_proj.weight"],
            device=device,
            dtype=self._attn_qo_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=_row,
        )
        self.q_norm_w = as_t("self_attn.q_norm.weight")
        self.k_norm_w = as_t("self_attn.k_norm.weight")
        _mlp_kw = dict(
            linear_compute_kernel_config=linear_compute_kernel_config,
            activation_l1_memory_config=activation_l1_memory_config,
            linear_output_l1_memory_config=linear_output_l1_memory_config,
            use_cond_linear_program_config=self._use_cond_linear_pc,
        )
        self.mlp = TtQwen3EncoderMLP(
            weights_np=weights_np,
            base=f"{prefix}.mlp",
            device=device,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            dtype=dtype,
            mem=mem,
            mapper=mapper,
            mlp_weight_dtype=self._mlp_weight_dtype,
            mlp_gate_weight_dtype=self._attn_qo_dtype,
            mlp_down_weight_dtype=self._attn_qo_dtype,
            **_mlp_kw,
        )

    def _l1_activation(self, t: ttnn.Tensor) -> ttnn.Tensor:
        if self._act_l1 is None:
            return t
        return ttnn.to_memory_config(t, self._act_l1)

    def _attn_linear_kwargs(
        self,
        *,
        batch_size: int,
        seq_len: int,
        in_dim: int,
        out_dim: int,
        in0_sharded: bool = False,
    ) -> dict:
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
        out_mc = ace_step_encoder_2d_block_sharded_memory_config(
            ttnn,
            self.device,
            seq_len=int(seq_len),
            in_dim=int(in_dim),
            out_dim=int(out_dim),
            batch_size=int(batch_size),
            for_output=True,
        )
        use_out_bs = out_mc is not None
        if out_mc is None:
            out_mc = ace_step_cond_256x1024_width_sharded_memory_config(
                ttnn,
                self.device,
                seq_len=int(seq_len),
                in_dim=int(in_dim),
                out_dim=int(out_dim),
                batch_size=int(batch_size),
                for_output=True,
            )
        pc = None
        if self._use_cond_linear_pc:
            key = (int(batch_size), int(seq_len), int(in_dim), int(out_dim), use_out_bs)
            pc = self._attn_pc_cache.get(key)
            if pc is None:
                pc = ace_step_cond_linear_program_config(
                    self.device,
                    seq_len=int(seq_len),
                    in_dim=int(in_dim),
                    out_dim=int(out_dim),
                    batch_size=int(batch_size),
                    in0_sharded=in0_sharded,
                    out_sharded=use_out_bs,
                )
                if pc is not None:
                    self._attn_pc_cache[key] = pc
            if pc is not None:
                kw["program_config"] = pc
        if self._linear_out_l1 is not None and out_mc is None:
            kw["memory_config"] = self._linear_out_l1
        if out_mc is not None:
            kw["memory_config"] = out_mc
        elif self._linear_out_l1 is None and self.mem is not None:
            # Detokenizer fused batch (n audio codes) has no L1 output config — keep DRAM so
            # post-attn rms_norm does not compile against stray L1 matmul/SDPA buffers.
            kw["memory_config"] = self.mem
        return kw

    def _maybe_shard_attn_in0(self, x: ttnn.Tensor, *, batch_size: int, seq_len: int, in_dim: int, out_dim: int):
        in0_mc = ace_step_encoder_2d_block_sharded_memory_config(
            ttnn,
            self.device,
            seq_len=int(seq_len),
            in_dim=int(in_dim),
            out_dim=int(out_dim),
            batch_size=int(batch_size),
            for_output=False,
        )
        if in0_mc is None:
            in0_mc = ace_step_cond_256x1024_width_sharded_memory_config(
                ttnn,
                self.device,
                seq_len=int(seq_len),
                in_dim=int(in_dim),
                out_dim=int(out_dim),
                batch_size=int(batch_size),
                for_output=False,
            )
        if in0_mc is None:
            return x
        try:
            if ace_step_memory_configs_equivalent(x.memory_config(), in0_mc):
                return x
            return ttnn.to_memory_config(x, in0_mc)
        except Exception:
            return x

    def __call__(self, hidden_b1sh: ttnn.Tensor, cos_11sd: ttnn.Tensor, sin_11sd: ttnn.Tensor, attn_bias_b11ss):
        _l1_mc = self._linear_out_l1 or self.mem
        _act_mc = self._act_l1 if self._act_l1 is not None else _l1_mc
        _rms_kw = ace_step_cond_rms_norm_kwargs(ttnn, _act_mc, device=self.device)
        x = ace_step_ensure_tile_layout(ttnn, hidden_b1sh)
        res = x
        # input_layernorm: BLOCK_SHARDED on 8×4 L1 grid ([1,1,256,K], block_h=2, block_w=K/256tiles/8).
        _bf8 = self._proj_dtype
        _ln_act_dtype = _bf8 if _bf8 != self.dtype else None
        # Block-sharded norm input (I2S unchanged); S2I to L1 interleaved for 1D QKV matmul.
        x = ace_step_rms_norm_block_sharded(
            ttnn,
            x,
            self.input_ln_w,
            self.eps,
            device=self.device,
            l1_mc=_l1_mc,
            compute_kernel_config=self._rms_norm_ck,
            activation_dtype=_ln_act_dtype,
            return_sharded=False,
        )

        b = int(x.shape[0])
        s = int(x.shape[2])
        H, kv_h, Dh = self._nh_local, self._nkv_local, self.dh  # local head counts under TP

        hsz = self.hidden_size
        q_dim_o = self._q_dim_o
        kv_dim_o = self._kv_dim_o
        qkv_dim_o = q_dim_o + 2 * kv_dim_o
        lin_qkv = self._attn_linear_kwargs(
            batch_size=b,
            seq_len=s,
            in_dim=hsz,
            out_dim=qkv_dim_o,
            in0_sharded=False,
        )
        qkv = ttnn.linear(x, self.wqkv, bias=None, transpose_b=True, dtype=_bf8, **lin_qkv)
        ace_step_safe_deallocate(ttnn, x)

        # BFP8 1D-mcast matmul can spill to DRAM despite memory_config=L1; force L1 interleaved
        # before nlp_create_qkv_heads (which does not accept the width-sharded matmul output).
        if _l1_mc is not None:
            qkv = ace_step_ensure_l1_activation(ttnn, qkv, _l1_mc)

        heads = ace_step_try_nlp_qkv_heads_split(
            ttnn,
            q_b1sd=qkv,
            num_heads=H,
            num_kv_heads=kv_h,
            memory_config=_l1_mc,
        )
        if heads is not None:
            q, k, v = heads
            ace_step_safe_deallocate(ttnn, qkv)
        else:
            # Manual fused-QKV split fallback (nlp_create_qkv_heads unavailable): slice the
            # single [B,1,S,q+2kv] output into q/k/v, then reshape + permute to [B,h,S,Dh].
            _sr = ace_step_reshape_kwargs(ttnn)
            _pk = ace_step_permute_kwargs(ttnn)
            q = ttnn.slice(qkv, (0, 0, 0, 0), (b, 1, s, q_dim_o))
            k = ttnn.slice(qkv, (0, 0, 0, q_dim_o), (b, 1, s, q_dim_o + kv_dim_o))
            v = ttnn.slice(qkv, (0, 0, 0, q_dim_o + kv_dim_o), (b, 1, s, qkv_dim_o))
            ace_step_safe_deallocate(ttnn, qkv)
            q = ttnn.reshape(q, (b, s, H, Dh), **_sr)
            k = ttnn.reshape(k, (b, s, kv_h, Dh), **_sr)
            v = ttnn.reshape(v, (b, s, kv_h, Dh), **_sr)
            q = ttnn.permute(q, (0, 2, 1, 3), **_pk)
            k = ttnn.permute(k, (0, 2, 1, 3), **_pk)
            v = ttnn.permute(v, (0, 2, 1, 3), **_pk)
            if _l1_mc is not None:
                q = ace_step_ensure_l1_activation(ttnn, q, _l1_mc)
                k = ace_step_ensure_l1_activation(ttnn, k, _l1_mc)
                v = ace_step_ensure_l1_activation(ttnn, v, _l1_mc)

        q = ttnn.rms_norm(q, weight=self.q_norm_w, epsilon=self.eps, **_rms_kw)
        k = ttnn.rms_norm(k, weight=self.k_norm_w, epsilon=self.eps, **_rms_kw)

        q, k = apply_rope_hf_style(q, k, cos_11sd, sin_11sd, eltwise_memory_config=_l1_mc)

        # Native GQA: SDPA accepts q [B, H, S, D] with k/v [B, kv_h, S, D] when H % kv_h == 0.
        # Avoids repeat_interleave Untilize/Concat/Tilize (~52 μs/layer).

        sdpa_d = _sdpa_head_dim_tile_padding(self.dh)
        if sdpa_d > self.dh:
            pt = sdpa_d - self.dh
            pad4 = ((0, 0), (0, 0), (0, 0), (0, pt))
            q = ttnn.pad(q, padding=pad4, value=0.0)
            k = ttnn.pad(k, padding=pad4, value=0.0)
            v = ttnn.pad(v, padding=pad4, value=0.0)

        # ``attn_bias_b11ss`` is pre-expanded to ``[B, H, S, S]`` once in ``forward()`` (not per layer).
        mask_tt = ace_step_ensure_dram_activation(
            ttnn, attn_bias_b11ss, ace_step_sdpa_mask_memory_config(ttnn) or self.mem
        )

        sdpa_kw = dict(attn_mask=mask_tt, is_causal=False, scale=self.scale)
        if self._sdpa_compute_kernel_config is not None:
            sdpa_kw["compute_kernel_config"] = self._sdpa_compute_kernel_config
        if self._sdpa_program_config is not None:
            sdpa_kw["program_config"] = self._sdpa_program_config
        sdpa_kw.update(ace_step_sdpa_activation_kwargs(ttnn, _act_mc))
        ctx = self._sdpa(q, k, v, **sdpa_kw)

        if sdpa_d > self.dh:
            ctx = ttnn.slice(ctx, (0, 0, 0, 0), (b, H, s, self.dh))

        # [b,H,s,Dh] -> [b,1,s,H*Dh] (permute + reshape view)
        ctx = ace_step_nlp_concat_heads(ttnn, ctx)
        ctx = (
            self._l1_activation(ctx)
            if self._act_l1 is not None
            else ace_step_ensure_dram_activation(ttnn, ctx, self.mem)
        )
        lin_o = self._attn_linear_kwargs(batch_size=b, seq_len=s, in_dim=q_dim_o, out_dim=hsz)
        ctx_o = self._maybe_shard_attn_in0(ctx, batch_size=b, seq_len=s, in_dim=q_dim_o, out_dim=hsz)
        attn_out = ttnn.linear(ctx_o, self.wo, bias=None, transpose_b=True, **lin_o)
        # Row-parallel o_proj: sum per-chip partial outputs across TP shards before the residual add.
        if self._tp.enabled and self._tp.degree > 1:
            from models.experimental.ace_step_v1_5.ttnn_impl.tp_config import tp_all_reduce

            attn_out = tp_all_reduce(attn_out, self.device, cfg=self._tp)
        ace_step_safe_deallocate(ttnn, ctx_o if ctx_o is not ctx else None)
        ttnn.deallocate(ctx)
        if _l1_mc is not None:
            attn_out = ace_step_ensure_l1_activation(ttnn, attn_out, _l1_mc)

        h = ttnn.add(res, attn_out, memory_config=_act_mc)
        res2 = h
        ace_step_safe_deallocate(ttnn, attn_out)
        h2 = ace_step_ensure_tile_layout(ttnn, h)
        if self._act_l1 is None and self.mem is not None:
            h2 = ace_step_ensure_dram_activation(ttnn, h2, self.mem)
        h2 = ace_step_rms_norm_block_sharded(
            ttnn,
            h2,
            self.post_ln_w,
            self.eps,
            device=self.device,
            l1_mc=_l1_mc,
            compute_kernel_config=self._rms_norm_ck,
        )
        ff = self.mlp(h2)
        return ttnn.add(res2, ff, memory_config=_act_mc)


class TtQwen3EmbeddingEncoder:
    """HF ``Qwen3Model`` forward (no LM head): returns last hidden states ``[B,1,S,H]`` TILE."""

    def __init__(
        self,
        *,
        device,
        hf_model_dir: str | Path,
        qwen_safetensors_path: str | Path,
        cfg: Optional[Qwen3EmbeddingEncoderConfig] = None,
        dtype=None,
        projection_dtype=None,
        mlp_weight_dtype=None,
    ) -> None:
        self.device = device
        self.cfg = cfg or Qwen3EmbeddingEncoderConfig()
        self.dtype = dtype or getattr(ttnn, "bfloat16", None)
        if self.dtype is None:
            raise RuntimeError("bfloat16 required")
        _w8 = ace_step_linear_weight_dtype(ttnn, self.dtype)
        self.projection_dtype = projection_dtype if projection_dtype is not None else _w8
        self.attn_qo_dtype = (
            projection_dtype if projection_dtype is not None else ace_step_attn_qo_weight_dtype(ttnn, self.dtype)
        )
        self.mlp_weight_dtype = mlp_weight_dtype if mlp_weight_dtype is not None else _w8
        self.mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        self._attn_bias_cache: dict[tuple, ttnn.Tensor] = {}

        weights_np = load_qwen3_weights_np(str(qwen_safetensors_path))

        cos_np, sin_np = build_hf_rope_cos_sin_np(
            hf_model_dir=str(hf_model_dir),
            hidden_size=self.cfg.hidden_size,
            head_dim=self.cfg.head_dim,
            max_seq_len=self.cfg.max_seq_len,
            rope_theta=self.cfg.rope_theta,
        )
        self.cos_tt = ttnn.as_tensor(
            cos_np,
            device=device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )
        self.sin_tt = ttnn.as_tensor(
            sin_np,
            device=device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )

        # fp32 dest accumulation pairs with the pinned wide-in0_block_w encoder matmul configs
        # (ace_step_encoder_matmul_program_config): keeps their speed while holding PCC.
        linear_compute_kernel_config = ace_step_init_cond_linear_fp32acc_compute_kernel_config(
            device
        ) or ace_step_init_cond_linear_compute_kernel_config(device)
        l1_mc = ace_step_linear_l1_memory_config(ttnn)
        sdpa_compute_kernel_config = ace_step_init_cond_sdpa_compute_kernel_config(device)
        sdpa_program_config = ace_step_init_cond_sdpa_program_config(
            device,
            seq_len=int(self.cfg.max_seq_len),
            num_heads=int(self.cfg.num_attention_heads),
            batch_size=1,
        )

        self.embed_weight = ttnn.as_tensor(
            weights_np["embed_tokens.weight"],
            device=device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )

        _layer_kw = dict(
            linear_compute_kernel_config=linear_compute_kernel_config,
            activation_l1_memory_config=l1_mc,
            linear_output_l1_memory_config=l1_mc,
            sdpa_compute_kernel_config=sdpa_compute_kernel_config,
            sdpa_program_config=sdpa_program_config,
        )
        self.layers = [
            _TtQwen3EncoderLayer(
                device=device,
                weights_np=weights_np,
                prefix=f"layers.{i}",
                cfg=self.cfg,
                dtype=self.dtype,
                mem=self.mem,
                mapper=mapper,
                projection_dtype=self.projection_dtype,
                attn_qo_dtype=self.attn_qo_dtype,
                mlp_weight_dtype=self.mlp_weight_dtype,
                **_layer_kw,
            )
            for i in range(self.cfg.num_hidden_layers)
        ]

        self.final_norm_w = ttnn.as_tensor(
            weights_np["norm.weight"],
            device=device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )

        # Pre-warm default all-valid attention bias (avoids ~2 ms host Tilize gap on first forward).
        _warm_mask = np.ones((1, self.cfg.max_seq_len), dtype=np.float32)
        _warm_bias = causal_padding_attn_bias_np(_warm_mask, self.cfg.max_seq_len)
        _warm_nh = int(self.cfg.num_attention_heads)
        if _warm_nh > 1:
            _warm_bias = np.repeat(_warm_bias, _warm_nh, axis=1)
        _warm_key = (int(self.cfg.max_seq_len), _warm_nh, _warm_mask.tobytes())
        self._attn_bias_cache[_warm_key] = ace_step_upload_f32_np_as_bf16_tile(
            ttnn,
            _warm_bias,
            device=device,
            dtype=self.dtype,
            memory_config=ace_step_sdpa_mask_memory_config(ttnn) or self.mem,
            mesh_mapper=mapper,
        )

    def embed_tokens(self, input_ids: np.ndarray) -> ttnn.Tensor:
        """Token IDs ``[B,S]`` -> embedding hidden states ``[B,S,H]`` on device (ROW_MAJOR).

        Callers (lyric ``embed_tokens`` path) consume via ``to_torch``; keep ROW_MAJOR and defer
        tilize to the consumer. :meth:`forward` reshapes first, then tilizes once in layer 0.
        """
        ids = np.asarray(input_ids, dtype=np.uint32)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        ids_tt = ttnn.as_tensor(
            ids,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )
        l1_mc = ace_step_linear_l1_memory_config(ttnn)
        _emb_mc_kw = {"memory_config": l1_mc} if l1_mc is not None else {}
        h = ttnn.embedding(ids_tt, weight=self.embed_weight, dtype=self.dtype, **_emb_mc_kw)
        ttnn.deallocate(ids_tt)
        # ace_step_ensure_l1_activation is now a no-op (embedding already writes to L1);
        # kept for safety when l1_mc is None.
        return ace_step_ensure_l1_activation(ttnn, h, l1_mc)

    def forward(self, input_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> ttnn.Tensor:
        cfg = self.cfg
        ids = np.asarray(input_ids, dtype=np.uint32)
        b, s = int(ids.shape[0]), int(ids.shape[1])
        if s != cfg.max_seq_len:
            raise ValueError(f"seq_len must be {cfg.max_seq_len}, got {s}")

        mapper = ttnn.ReplicateTensorToMesh(self.device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        _sr = ace_step_reshape_kwargs(ttnn)
        ids_tt = ttnn.as_tensor(
            ids,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )
        l1_mc = ace_step_linear_l1_memory_config(ttnn)
        _emb_mc_kw = {"memory_config": l1_mc} if l1_mc is not None else {}
        h = ttnn.embedding(ids_tt, weight=self.embed_weight, dtype=self.dtype, **_emb_mc_kw)
        ttnn.deallocate(ids_tt)
        # embedding already writes to L1; ace_step_ensure_l1_activation is a no-op when l1_mc matches.
        h = ace_step_ensure_l1_activation(ttnn, h, l1_mc)
        h = ttnn.reshape(h, (b, 1, s, cfg.hidden_size), **_sr)

        attn_bias_tt = None
        m = attention_mask
        if m is None:
            m = np.ones((b, s), dtype=np.float32)
        bias_np = causal_padding_attn_bias_np(np.asarray(m), cfg.max_seq_len)
        nh = int(cfg.num_attention_heads)
        if nh > 1:
            bias_np = np.repeat(bias_np, nh, axis=1)
        mask_key = (int(cfg.max_seq_len), nh, np.asarray(m, dtype=np.float32).tobytes())
        attn_bias_tt = self._attn_bias_cache.get(mask_key)
        if attn_bias_tt is None:
            attn_bias_tt = ace_step_upload_f32_np_as_bf16_tile(
                ttnn,
                bias_np,
                device=self.device,
                dtype=self.dtype,
                memory_config=ace_step_sdpa_mask_memory_config(ttnn) or self.mem,
                mesh_mapper=mapper,
            )
            self._attn_bias_cache[mask_key] = attn_bias_tt

        for layer in self.layers:
            h = layer(h, self.cos_tt, self.sin_tt, attn_bias_tt)

        h = ace_step_ensure_tile_layout(ttnn, h)
        _fn_mc = ace_step_linear_l1_memory_config(ttnn) or self.mem
        h = ace_step_rms_norm_block_sharded(
            ttnn,
            h,
            self.final_norm_w,
            float(cfg.rms_norm_eps),
            device=self.device,
            l1_mc=_fn_mc,
        )
        return h


__all__ = [
    "Qwen3EmbeddingEncoderConfig",
    "TtQwen3EncoderMLP",
    "TtQwen3EmbeddingEncoder",
    "apply_rope_hf_style",
    "build_hf_rope_cos_sin_np",
    "causal_padding_attn_bias_np",
    "load_qwen3_weights_np",
    # "padding_attn_bias_np",
    "repeat_kv_gqa_ttnn",
    "rotate_half_ttnn",
]

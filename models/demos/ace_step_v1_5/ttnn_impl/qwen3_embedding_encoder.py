# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN ``Qwen3Model`` (embedding encoder) for ACE-Step — matches HF ``transformers`` layout.

Fixed ``max_seq_len`` (default 256): RoPE ``cos``/``sin`` are precomputed once from the local HF
``config.json`` + ``Qwen3RotaryEmbedding`` (Transformers, CPU-only at init).

Forward path is TTNN-only (no PyTorch tensors after ``input_ids`` numpy staging).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import ttnn

from .math_perf_env import (
    ace_step_cond_linear_program_config,
    ace_step_cond_mlp_gate_up_linear_program_config,
    ace_step_init_hifi2_linear_compute_kernel_config,
    ace_step_linear_l1_memory_config,
    ace_step_nlp_concat_heads,
    ace_step_permute_kwargs,
    ace_step_reshape_kwargs,
)


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


def padding_attn_bias_np(attention_mask_01: np.ndarray) -> np.ndarray:
    """1=keep, 0=pad → additive bias ``[B,1,S,S]``."""
    m = np.asarray(attention_mask_01, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    b, s = m.shape
    neg = np.float32(-1.0e9)
    bias = np.zeros((b, 1, s, s), dtype=np.float32)
    for bi in range(b):
        row = m[bi][:, None] * m[bi][None, :]
        bias[bi, 0, :, :] = np.where(row > 0.5, 0.0, neg)
    return bias


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
    """RoPE with GQA: expand ``cos``/``sin`` ``[1,1,S,Dh]`` to ``[B,nh,S,Dh]`` for ``q`` and ``[B,nkv,S,Dh]`` for ``k`` (do not use nh for both)."""
    m = eltwise_memory_config

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
        activation_l1_memory_config=None,
        linear_output_l1_memory_config=None,
        use_cond_linear_program_config: bool = True,
    ):
        self.device = device
        self.mem = mem
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self._linear_ck = linear_compute_kernel_config
        self._act_l1 = activation_l1_memory_config
        self._linear_out_l1 = linear_output_l1_memory_config
        self._use_cond_linear_pc = bool(use_cond_linear_program_config)
        self._gate_up_pc_cache: dict = {}

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
        # Wide cond MLPs (e.g. timbre 6144×2048 gate/up): L1 linear outputs clash with matmul
        # static CBs (L1 buffer @ 1096832 vs CB end @ 1167872 on Blackhole). Keep DRAM activations.
        self._mlp_keep_dram_activations = int(intermediate_size) >= 4608

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
            key = (int(batch_size), int(seq_len))
            pc = self._gate_up_pc_cache.get(key)
            if pc is None:
                pc = ace_step_cond_mlp_gate_up_linear_program_config(
                    self.device,
                    seq_len=int(seq_len),
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    batch_size=int(batch_size),
                )
                if pc is not None:
                    self._gate_up_pc_cache[key] = pc
            if pc is not None:
                kw["program_config"] = pc
        if self._linear_out_l1 is not None and not self._mlp_keep_dram_activations:
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
                in_dim=self.intermediate_size,
                out_dim=self.hidden_size,
                batch_size=int(batch_size),
            )
            if pc is not None:
                kw["program_config"] = pc
        if self._linear_out_l1 is not None and not self._mlp_keep_dram_activations:
            kw["memory_config"] = self._linear_out_l1
        return kw

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        b_x = int(x.shape[0])
        s = int(x.shape[2])
        if self._mlp_keep_dram_activations:
            if self.mem is not None:
                x = ttnn.to_memory_config(x, self.mem)
        else:
            x = self._l1_activation(x)
        lin_gu = self._gate_up_linear_kwargs(batch_size=b_x, seq_len=s)
        gate = ttnn.linear(x, self.w_gate, bias=None, transpose_b=True, **lin_gu)
        up = ttnn.linear(x, self.w_up, bias=None, transpose_b=True, **lin_gu)
        _silu_mc = (self._linear_out_l1 if not self._mlp_keep_dram_activations else None) or getattr(
            ttnn, "DRAM_MEMORY_CONFIG", None
        )
        gate = (
            ttnn.silu(gate, memory_config=_silu_mc)
            if hasattr(ttnn, "silu")
            else ttnn.gelu(gate, memory_config=_silu_mc)
        )
        h = ttnn.multiply(gate, up, memory_config=_silu_mc)
        if not self._mlp_keep_dram_activations:
            h = self._l1_activation(h)
        lin_down = self._down_linear_kwargs(batch_size=b_x, seq_len=s)
        return ttnn.linear(h, self.w_down, bias=None, transpose_b=True, **lin_down)


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
    ):
        self.device = device
        self.cfg = cfg
        self.dtype = dtype
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
            **_mlp_kw,
        )

    def _l1_activation(self, t: ttnn.Tensor) -> ttnn.Tensor:
        if self._act_l1 is None:
            return t
        return ttnn.to_memory_config(t, self._act_l1)

    def _attn_linear_kwargs(self, *, batch_size: int, seq_len: int, in_dim: int, out_dim: int) -> dict:
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
        pc = None
        if self._use_cond_linear_pc:
            key = (int(batch_size), int(seq_len), int(in_dim), int(out_dim))
            pc = self._attn_pc_cache.get(key)
            if pc is None:
                pc = ace_step_cond_linear_program_config(
                    self.device,
                    seq_len=int(seq_len),
                    in_dim=int(in_dim),
                    out_dim=int(out_dim),
                    batch_size=int(batch_size),
                )
                if pc is not None:
                    self._attn_pc_cache[key] = pc
            if pc is not None:
                kw["program_config"] = pc
        if self._linear_out_l1 is not None:
            kw["memory_config"] = self._linear_out_l1
        return kw

    def __call__(self, hidden_b1sh: ttnn.Tensor, cos_11sd: ttnn.Tensor, sin_11sd: ttnn.Tensor, attn_bias_b11ss):
        _l1_mc = self._linear_out_l1 or self.mem
        res = hidden_b1sh
        x = ttnn.to_layout(hidden_b1sh, ttnn.TILE_LAYOUT)
        x = ttnn.rms_norm(x, weight=self.input_ln_w, epsilon=self.eps, memory_config=_l1_mc)

        b = int(x.shape[0])
        s = int(x.shape[2])
        H, kv_h, Dh = self.nh, self.nkv, self.dh

        x = self._l1_activation(x)
        hsz = self.hidden_size
        q_dim_o = int(H * Dh)
        kv_dim_o = int(kv_h * Dh)
        lin_q = self._attn_linear_kwargs(batch_size=b, seq_len=s, in_dim=hsz, out_dim=q_dim_o)
        lin_kv = self._attn_linear_kwargs(batch_size=b, seq_len=s, in_dim=hsz, out_dim=kv_dim_o)
        _sr = ace_step_reshape_kwargs(ttnn)
        _pk = ace_step_permute_kwargs(ttnn)
        q = ttnn.linear(x, self.wq, bias=None, transpose_b=True, **lin_q)
        k = ttnn.linear(x, self.wk, bias=None, transpose_b=True, **lin_kv)
        v = ttnn.linear(x, self.wv, bias=None, transpose_b=True, **lin_kv)

        # [B,1,S,H*Dh] -> [B,S,H,Dh] -> [B,H,S,Dh] (4-D path; avoids rank-5 intermediate)
        q = ttnn.reshape(q, (b, s, H, Dh), **_sr)
        k = ttnn.reshape(k, (b, s, kv_h, Dh), **_sr)
        v = ttnn.reshape(v, (b, s, kv_h, Dh), **_sr)
        q = ttnn.permute(q, (0, 2, 1, 3), **_pk)
        k = ttnn.permute(k, (0, 2, 1, 3), **_pk)
        v = ttnn.permute(v, (0, 2, 1, 3), **_pk)

        q = ttnn.rms_norm(q, weight=self.q_norm_w, epsilon=self.eps, memory_config=_l1_mc)
        k = ttnn.rms_norm(k, weight=self.k_norm_w, epsilon=self.eps, memory_config=_l1_mc)

        q, k = apply_rope_hf_style(q, k, cos_11sd, sin_11sd, eltwise_memory_config=_l1_mc)

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

        mask_tt = attn_bias_b11ss
        if mask_tt is not None:
            mask_tt = ttnn.repeat(mask_tt, (1, H, 1, 1))

        sdpa_kw = dict(attn_mask=mask_tt, is_causal=False, scale=self.scale)
        if self._sdpa_compute_kernel_config is not None:
            sdpa_kw["compute_kernel_config"] = self._sdpa_compute_kernel_config
        if self._sdpa_program_config is not None:
            sdpa_kw["program_config"] = self._sdpa_program_config
        ctx = self._sdpa(q, k, v, **sdpa_kw)

        if sdpa_d > self.dh:
            ctx = ttnn.slice(ctx, (0, 0, 0, 0), (b, H, s, self.dh))

        # [b,H,s,Dh] -> [b,1,s,H*Dh] via fused nlp_concat_heads (single kernel vs permute+reshape)
        ctx = ace_step_nlp_concat_heads(ttnn, ctx)
        ctx = self._l1_activation(ctx)
        lin_o = self._attn_linear_kwargs(batch_size=b, seq_len=s, in_dim=q_dim_o, out_dim=hsz)
        attn_out = ttnn.linear(ctx, self.wo, bias=None, transpose_b=True, **lin_o)
        ttnn.deallocate(ctx)

        h = ttnn.add(res, attn_out, memory_config=_l1_mc)
        res2 = h
        h2 = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
        h2 = ttnn.rms_norm(h2, weight=self.post_ln_w, epsilon=self.eps, memory_config=_l1_mc)
        ff = self.mlp(h2)
        return ttnn.add(res2, ff, memory_config=_l1_mc)


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
    ) -> None:
        self.device = device
        self.cfg = cfg or Qwen3EmbeddingEncoderConfig()
        self.dtype = dtype or getattr(ttnn, "bfloat16", None)
        if self.dtype is None:
            raise RuntimeError("bfloat16 required")
        self.mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

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

        init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
        linear_compute_kernel_config = ace_step_init_hifi2_linear_compute_kernel_config(device)
        l1_mc = ace_step_linear_l1_memory_config(ttnn)
        sdpa_compute_kernel_config = None
        if callable(init_ck):
            sdpa_compute_kernel_config = init_ck(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

        sdpa_program_config = None
        if hasattr(device, "compute_with_storage_grid_size") and hasattr(ttnn, "SDPAProgramConfig"):
            sdpa_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                q_chunk_size=32,
                k_chunk_size=256,
                exp_approx_mode=False,
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
                sdpa_compute_kernel_config=sdpa_compute_kernel_config,
                sdpa_program_config=sdpa_program_config,
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

    def embed_tokens(self, input_ids: np.ndarray) -> ttnn.Tensor:
        """Token IDs ``[B,S]`` -> embedding hidden states ``[B,S,H]`` on device."""
        cfg = self.cfg
        ids = np.asarray(input_ids, dtype=np.uint32)
        b, s = int(ids.shape[0]), int(ids.shape[1])
        mapper = ttnn.ReplicateTensorToMesh(self.device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        ids_tt = ttnn.as_tensor(
            ids,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )
        h = ttnn.embedding(ids_tt, weight=self.embed_weight, dtype=self.dtype)
        ttnn.deallocate(ids_tt)
        return h

    def forward(self, input_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> ttnn.Tensor:
        cfg = self.cfg
        ids = np.asarray(input_ids, dtype=np.uint32)
        b, s = int(ids.shape[0]), int(ids.shape[1])
        if s != cfg.max_seq_len:
            raise ValueError(f"seq_len must be {cfg.max_seq_len}, got {s}")

        mapper = ttnn.ReplicateTensorToMesh(self.device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        _sr = ace_step_reshape_kwargs(ttnn)
        h = self.embed_tokens(ids)
        h = ttnn.reshape(h, (b, 1, s, cfg.hidden_size), **_sr)

        attn_bias_tt = None
        m = attention_mask
        if m is None:
            m = np.ones((b, s), dtype=np.float32)
        bias_np = causal_padding_attn_bias_np(np.asarray(m), cfg.max_seq_len)
        attn_bias_tt = ttnn.as_tensor(
            bias_np,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.mem,
            mesh_mapper=mapper,
        )

        for layer in self.layers:
            h = layer(h, self.cos_tt, self.sin_tt, attn_bias_tt)

        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
        _fn_mc = ace_step_linear_l1_memory_config(ttnn) or self.mem
        h = ttnn.rms_norm(h, weight=self.final_norm_w, epsilon=float(cfg.rms_norm_eps), memory_config=_fn_mc)
        return h


__all__ = [
    "Qwen3EmbeddingEncoderConfig",
    "TtQwen3EmbeddingEncoder",
    "load_qwen3_weights_np",
    "build_hf_rope_cos_sin_np",
]

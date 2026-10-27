# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention (HF `EmbeddingGemma2Attention`): bidirectional GQA/MQA, one class for both layer kinds.

    q, k, v = split_heads(x @ [Wq | Wk | Wv])              no biases
    q = RMSNorm_hd(q) * w_q;  k = RMSNorm_hd(k) * w_k;  v = RMSNorm_hd(v)   (v: no scale)
    q, k = rope(q), rope(k)                                  full rotation, HF rotate_half layout
    o = softmax(q k^T * 1.0 + mask) v                        scaling 1.0, no softcap, is_causal False
    out = concat_heads(o) @ Wo

Layer kinds (from config):
    sliding  head_dim 256, 4 Q heads, 2 KV heads (GQA 2), rope theta 1e4, mask = band(|q-k|<=512) + padding
    full     head_dim 512, 4 Q heads, 1 KV head  (MQA),   rope theta 1e6, mask = padding only

Contract
    input   x [B, 1, S, hidden] bf16 TILE DRAM; S a physical bucket (multiple of 32, <= rope table length)
            mask [B, 1, S, S] additive bf16 TILE DRAM (see masks.py)
    output  [B, 1, S, hidden] bf16 TILE DRAM
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import (
    AttentionConfig,
    RMSNormConfig,
    compute_kernel_config_hifi4,
)
from models.demos.blackhole.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.blackhole.embedding_gemma2.tt.rope import RotaryTables
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight, substate


def sdpa_program_config(head_dim: int, seq_len: int) -> ttnn.SDPAProgramConfig:
    """Chunk sizes adapted from models/demos/gemma4/tt/attention/operations.py:prefill_sdpa_program_config.

    head_dim 512 needs the smaller chunks to fit L1; chunks never exceed the (bucketed) seq_len,
    and every bucket is a power of two >= 32, so S % chunk == 0.
    """
    if head_dim >= 512:
        grid, q_chunk, k_chunk = ttnn.CoreCoord(8, 4), 128, 128
    else:
        grid, q_chunk, k_chunk = ttnn.CoreCoord(8, 8), 256, 128
    q_chunk = min(q_chunk, seq_len)
    k_chunk = min(k_chunk, seq_len)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


class Attention(LightweightModule):
    def __init__(
        self,
        config: AttentionConfig,
        wqkv: ttnn.Tensor,
        wo: ttnn.Tensor,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        v_norm: RMSNorm,
        rope: RotaryTables,
    ):
        super().__init__()
        self.config = config
        self.wqkv = wqkv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.v_norm = v_norm
        self.rope = rope
        self.compute_kernel_config = compute_kernel_config_hifi4()
        self.sdpa_compute_kernel_config = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: AttentionConfig,
        device,
        rope: RotaryTables,
        *,
        cache_dir: Path | None = None,
        name: str = "self_attn",
    ) -> "Attention":
        """`state_dict` holds the HF `layers.<i>.self_attn.` sub-state."""
        H = config.hidden_size
        wq, wk, wv, wo = (state_dict[f"{p}_proj.weight"] for p in ("q", "k", "v", "o"))
        expect = {
            "q": (config.q_width, H),
            "k": (config.kv_width, H),
            "v": (config.kv_width, H),
            "o": (H, config.q_width),
        }
        for k_, w in zip("qkvo", (wq, wk, wv, wo)):
            if tuple(w.shape) != expect[k_]:
                raise ValueError(f"{name}.{k_}_proj: expected {expect[k_]}, got {tuple(w.shape)}")
        wqkv = as_weight(
            linear_weight(torch.cat([wq, wk, wv], dim=0)), device, cache_dir=cache_dir, name=f"{name}.wqkv.weight"
        )
        wo_t = as_weight(linear_weight(wo), device, cache_dir=cache_dir, name=f"{name}.o_proj.weight")
        hd_norm = RMSNormConfig(dim=config.head_dim, eps=config.rms_norm_eps)
        q_norm = RMSNorm.from_state_dict(
            substate(state_dict, "q_norm"), hd_norm, device, cache_dir=cache_dir, name=f"{name}.q_norm"
        )
        k_norm = RMSNorm.from_state_dict(
            substate(state_dict, "k_norm"), hd_norm, device, cache_dir=cache_dir, name=f"{name}.k_norm"
        )
        v_norm = RMSNorm(RMSNormConfig(dim=config.head_dim, eps=config.rms_norm_eps, with_scale=False), None)
        return cls(config, wqkv, wo_t, q_norm, k_norm, v_norm, rope)

    def forward(self, x: ttnn.Tensor, mask: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        dram = ttnn.DRAM_MEMORY_CONFIG
        S = x.shape[2]
        xqkv = ttnn.linear(x, self.wqkv, compute_kernel_config=self.compute_kernel_config, memory_config=dram)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            transpose_k_heads=False,
            memory_config=dram,
        )
        ttnn.deallocate(xqkv)

        q_n = self.q_norm(q)
        k_n = self.k_norm(k)
        v_n = self.v_norm(v)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        q_r = self.rope(q_n, cfg.layer_type)
        k_r = self.rope(k_n, cfg.layer_type)
        ttnn.deallocate(q_n)
        ttnn.deallocate(k_n)

        o = ttnn.transformer.scaled_dot_product_attention(
            q_r,
            k_r,
            v_n,
            is_causal=False,
            attn_mask=mask,
            scale=cfg.scaling,
            program_config=sdpa_program_config(cfg.head_dim, S),
            compute_kernel_config=self.sdpa_compute_kernel_config,
            memory_config=dram,
        )
        ttnn.deallocate(q_r)
        ttnn.deallocate(k_r)
        ttnn.deallocate(v_n)

        o_cat = ttnn.experimental.nlp_concat_heads(o, memory_config=dram)
        ttnn.deallocate(o)
        out = ttnn.linear(o_cat, self.wo, compute_kernel_config=self.compute_kernel_config, memory_config=dram)
        ttnn.deallocate(o_cat)
        return out

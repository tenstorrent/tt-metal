# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VisionAttention (HF `Gemma4VisionAttention`, modeling_gemma4.py:903-969): bidirectional MHA.

    q, k, v = split_heads(x @ [Wq | Wk | Wv])                  12 heads x 64, no biases
    q = RMSNorm_64(q) * w_q;  k = RMSNorm_64(k) * w_k;  v = RMSNorm_64(v)   (v: no scale)
    q, k = axial_rope(q), axial_rope(k)                        per-patch (x, y) positions
    o = softmax(q k^T * 1.0 + mask) v                          scaling 1.0, no softcap
    out = concat_heads(o) @ Wo

Contract
    input   x [B, 1, S, 768] bf16 TILE DRAM, S a patch bucket (1280 / 2560)
            mask [B, 1, S, S] additive bf16 TILE DRAM (0 valid key, -1e9 padding key)
            cos, sin [B, 1, S, 64] bf16 TILE DRAM (rope.axial_cos_sin of the patch positions)
    output  [B, 1, S, 768] bf16 TILE DRAM
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import RMSNormConfig, compute_kernel_config_hifi4
from models.demos.blackhole.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.blackhole.embedding_gemma2.tt.vision.config import VisionAttentionConfig
from models.demos.blackhole.embedding_gemma2.tt.vision.rope import AxialRope
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight, substate


def vision_sdpa_program_config(seq_len: int) -> ttnn.SDPAProgramConfig:
    """head_dim 64: 256-row Q chunks / 256-row K chunks on an 8x8 grid (both buckets divide by 256)."""
    chunk = 256 if seq_len % 256 == 0 else 128
    if seq_len % chunk != 0:
        raise ValueError(f"vision seq_len {seq_len} must be a multiple of 128")
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=False,
    )


class VisionAttention(LightweightModule):
    def __init__(
        self,
        config: VisionAttentionConfig,
        wqkv: ttnn.Tensor,
        wo: ttnn.Tensor,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        v_norm: RMSNorm,
        rope: AxialRope,
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
        config: VisionAttentionConfig,
        device,
        rope: AxialRope,
        *,
        cache_dir: Path | None = None,
        name: str = "vision.self_attn",
    ) -> "VisionAttention":
        """`state_dict` holds the HF `encoder.layers.<i>.self_attn.` sub-state (`.linear.` folded)."""
        H, W = config.hidden_size, config.width
        ws = [state_dict[f"{p}_proj.weight"] for p in "qkvo"]
        expect = [(W, H), (W, H), (W, H), (H, W)]
        for p, w, shp in zip("qkvo", ws, expect):
            if tuple(w.shape) != shp:
                raise ValueError(f"{name}.{p}_proj: expected {shp}, got {tuple(w.shape)}")
        wqkv = as_weight(
            linear_weight(torch.cat(ws[:3], dim=0)), device, cache_dir=cache_dir, name=f"{name}.wqkv.weight"
        )
        wo = as_weight(linear_weight(ws[3]), device, cache_dir=cache_dir, name=f"{name}.o_proj.weight")
        hd = RMSNormConfig(dim=config.head_dim, eps=config.rms_norm_eps)
        q_norm = RMSNorm.from_state_dict(
            substate(state_dict, "q_norm"), hd, device, cache_dir=cache_dir, name=f"{name}.q_norm"
        )
        k_norm = RMSNorm.from_state_dict(
            substate(state_dict, "k_norm"), hd, device, cache_dir=cache_dir, name=f"{name}.k_norm"
        )
        v_norm = RMSNorm(RMSNormConfig(dim=config.head_dim, eps=config.rms_norm_eps, with_scale=False), None)
        return cls(config, wqkv, wo, q_norm, k_norm, v_norm, rope)

    def forward(self, x: ttnn.Tensor, mask: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        dram = ttnn.DRAM_MEMORY_CONFIG
        S = x.shape[2]
        xqkv = ttnn.linear(x, self.wqkv, compute_kernel_config=self.compute_kernel_config, memory_config=dram)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=cfg.num_heads, num_kv_heads=cfg.num_heads, transpose_k_heads=False, memory_config=dram
        )
        ttnn.deallocate(xqkv)

        q_n = self.q_norm(q)
        k_n = self.k_norm(k)
        v_n = self.v_norm(v)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        q_r = self.rope(q_n, cos, sin)
        k_r = self.rope(k_n, cos, sin)
        ttnn.deallocate(q_n)
        ttnn.deallocate(k_n)

        o = ttnn.transformer.scaled_dot_product_attention(
            q_r,
            k_r,
            v_n,
            is_causal=False,
            attn_mask=mask,
            scale=cfg.scaling,
            program_config=vision_sdpa_program_config(S),
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

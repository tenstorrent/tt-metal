# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of a single HunyuanImage-3.0 transformer decoder layer.
# Mirrors ref/transformer_layer.py (pre-norm block):
#     residual = x
#     x = input_layernorm(x)
#     x = self_attn(x, rope, mask)
#     x = residual + x
#     residual = x
#     x = post_attention_layernorm(x)
#     x = mlp(x)                       # MoE for layer-0
#     x = residual + x
#
# Composes the verified TT sub-blocks:
#   HunyuanTtRMSNorm   (input / post-attention layernorms)
#   HunyuanTtAttention (self-attn: fused QKV, 2D RoPE, QK-norm, GQA, SDPA, o_proj)
#   HunyuanTtMoE       (gate -> experts -> combine -> shared MLP)

import ttnn
from models.common.lightweightmodule import LightweightModule

from .attention.attention import HunyuanTtAttention
from .attention.rms_norm import HunyuanTtRMSNorm
from .moe.moe import HunyuanTtMoE


class HunyuanTtDecoderLayer(LightweightModule):
    def __init__(
        self,
        device,
        state_dict: dict,
        layer_num: int = 0,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        num_experts: int = 64,
        moe_topk: int = 8,
        use_qk_norm: bool = True,
        use_mixed_mlp_moe: bool = True,
        norm_topk_prob: bool = True,
        rms_norm_eps: float = 1e-6,
        weight_dtype=ttnn.bfloat16,
        stream_experts: bool = True,
    ):
        super().__init__()
        self.device = device
        self.layer_num = layer_num

        prefix = f"model.layers.{layer_num}"
        self.input_layernorm = HunyuanTtRMSNorm(
            device, hidden_size, state_dict, f"{prefix}.input_layernorm", eps=rms_norm_eps
        )
        self.self_attn = HunyuanTtAttention(
            device,
            state_dict,
            layer_num=layer_num,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_qk_norm=use_qk_norm,
            eps=rms_norm_eps,
            weight_dtype=weight_dtype,
        )
        self.post_attention_layernorm = HunyuanTtRMSNorm(
            device, hidden_size, state_dict, f"{prefix}.post_attention_layernorm", eps=rms_norm_eps
        )
        self.mlp = HunyuanTtMoE(
            device,
            hidden_size,
            num_experts,
            moe_topk,
            state_dict,
            f"{prefix}.mlp",
            use_mixed_mlp_moe=use_mixed_mlp_moe,
            norm_topk_prob=norm_topk_prob,
            weight_dtype=weight_dtype,
            stream_experts=stream_experts,
        )

    def forward(self, x, seq_len, image_infos=None, attention_mask=None):
        """
        Args:
            x:              TTNN tensor [B, S, H] in TILE_LAYOUT.
            seq_len:        S — used to build the 2D RoPE cos/sin tables.
            image_infos:    per-batch image span info for 2D RoPE (None => text-only).
            attention_mask: optional additive mask [B,1,S,S]; None => causal SDPA.
        Returns:
            [B, S, H] TTNN tensor.
        """
        # 2D RoPE tables for this sequence.
        cos_tt, sin_tt = self.self_attn.rope.prepare_cos_sin(seq_len, image_infos=image_infos)

        # --- self-attention block ---
        residual = x
        h = self.input_layernorm(x)
        attn_out = self.self_attn(h, cos_tt, sin_tt, attention_mask=attention_mask)
        ttnn.deallocate(h)
        x = ttnn.add(residual, attn_out)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_out)

        # --- MoE block ---
        residual = x
        h = self.post_attention_layernorm(x)
        moe_out = self.mlp(h)
        ttnn.deallocate(h)
        out = ttnn.add(residual, moe_out)
        ttnn.deallocate(residual)
        ttnn.deallocate(moe_out)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)
        return out

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for a single HunyuanImage-3.0 transformer decoder layer.
# Composes the already-verified reference sub-blocks, with the residual wiring
# extracted verbatim from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     HunyuanImage3DecoderLayer.forward  (lines 1406-1463)
#
# Pre-norm transformer block:
#     residual = x
#     x = input_layernorm(x)
#     x = self_attn(x, mask, rope)
#     x = residual + x
#     residual = x
#     x = post_attention_layernorm(x)
#     x = mlp(x)                       # MoE for layer-0 (num_experts > 1)
#     x = residual + x
#
# Sub-blocks (each independently verified bit-exact vs upstream):
#   input_layernorm / post_attention_layernorm -> ref.attention.rms_norm
#   self_attn                                   -> ref.attention.attention
#   mlp (MoE)                                   -> ref.moe.moe

import torch.nn as nn

from .attention.attention import HunyuanImage3SDPAAttention, AttentionConfig
from .attention.rms_norm import HunyuanRMSNorm
from .moe.moe import HunyuanMoE


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_head_dim: int,
        num_experts: int,
        moe_topk: int,
        moe_intermediate_size: int,
        num_shared_expert: int = 1,
        use_mixed_mlp_moe: bool = True,
        hidden_act: str = "silu",
        mlp_bias: bool = False,
        norm_topk_prob: bool = True,
        use_qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        attn_cfg = AttentionConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_key_value_heads=num_key_value_heads,
            use_qk_norm=use_qk_norm,
            use_rotary_pos_emb=True,
            rms_norm_eps=rms_norm_eps,
        )
        self.self_attn = HunyuanImage3SDPAAttention(attn_cfg, layer_idx=layer_idx)

        self.mlp = HunyuanMoE(
            hidden_size,
            moe_intermediate_size,
            num_experts=num_experts,
            moe_topk=moe_topk,
            num_shared_expert=num_shared_expert,
            use_mixed_mlp_moe=use_mixed_mlp_moe,
            hidden_act=hidden_act,
            mlp_bias=mlp_bias,
            norm_topk_prob=norm_topk_prob,
        )

        self.input_layernorm = HunyuanRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = HunyuanRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, custom_pos_emb=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            custom_pos_emb=custom_pos_emb,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

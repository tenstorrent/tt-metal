# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Hybrid decoder block for Qwen3-Coder-Next.

Each decoder block contains:
1. Input RMSNorm
2. Attention (DeltaNet for 3/4 layers, GQA for 1/4 layers)
3. Residual connection
4. Post-attention RMSNorm
5. MoE FFN (512 experts, top-10 routing + shared expert)
6. Residual connection

The attention type is determined by layer_idx:
- layer_idx % full_attention_interval == (full_attention_interval - 1) -> GQA
- Otherwise -> DeltaNet

Reference: DeepSeek V3 decoder at deepseek_v3/tt/decoder_block/moe_decoder_block_2d.py
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.demos.qwen3_coder_next.tt.deltanet_attention import GatedDeltaNetAttention
from models.demos.qwen3_coder_next.tt.gqa_attention import GQAAttention
from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.moe import MoELayer


class RMSNorm(nn.Module):
    """RMSNorm for Qwen3-Coder-Next."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


class HybridDecoderBlock(nn.Module):
    """Hybrid decoder block with DeltaNet or GQA attention + MoE FFN.

    PyTorch reference implementation for correctness validation.
    """

    def __init__(self, config: Qwen3CoderNextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_gqa_layer = config.is_gqa_layer(layer_idx)

        # Norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention: GQA for every 4th layer, DeltaNet otherwise
        if self.is_gqa_layer:
            self.self_attn = GQAAttention(config, layer_idx)
        else:
            self.self_attn = GatedDeltaNetAttention(config, layer_idx)

        # MoE FFN (same for all layers)
        self.mlp = MoELayer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass for hybrid decoder block.

        Args:
            hidden_states: (batch, seq_len, hidden_size).
            cos, sin: RoPE frequencies (only used by GQA layers).
            attention_mask: Causal mask (only used by GQA layers).
            recurrent_state: DeltaNet recurrent state (only used by DeltaNet layers).
            kv_cache: KV cache tuple (only used by GQA layers).

        Returns:
            Tuple of:
                - output: (batch, seq_len, hidden_size)
                - state_dict: Updated states {'recurrent_state', 'kv_cache'}
        """
        residual = hidden_states

        # Pre-attention norm
        hidden_states = self.input_layernorm(hidden_states)

        # Attention
        new_state = {}
        if self.is_gqa_layer:
            attn_output, new_kv_cache = self.self_attn(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            new_state["kv_cache"] = new_kv_cache
        else:
            attn_output, new_recurrent_state, _, _ = self.self_attn(
                hidden_states,
                recurrent_state=recurrent_state,
            )
            new_state["recurrent_state"] = new_recurrent_state

        # Residual connection
        hidden_states = residual + attn_output

        # Post-attention norm + MoE FFN + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_state

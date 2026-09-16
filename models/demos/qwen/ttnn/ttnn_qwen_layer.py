"""
TTNN Qwen2.5 Decoder Layer
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from ..common import Qwen2_5Config
    from .ttnn_qwen_attention import TTNNQwenAttention
    from .ttnn_qwen_mlp import TTNNQwenRMSNorm, TTNNQwenMLP
except ImportError:
    from common import Qwen2_5Config
    from ttnn.ttnn_qwen_attention import TTNNQwenAttention
    from ttnn.ttnn_qwen_mlp import TTNNQwenRMSNorm, TTNNQwenMLP

class TTNNQwenDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5Config):
        super().__init__()
        self.input_layernorm = TTNNQwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TTNNQwenAttention(config)
        self.post_attention_layernorm = TTNNQwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TTNNQwenMLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

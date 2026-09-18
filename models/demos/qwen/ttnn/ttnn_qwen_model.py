"""
TTNN Qwen2.5 Full Backbone and Causal Language Model
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from ..common import Qwen2_5Config
    from .ttnn_qwen_embeddings import TTNNQwenEmbeddings
    from .ttnn_qwen_mlp import TTNNQwenRMSNorm
    from .ttnn_qwen_layer import TTNNQwenDecoderLayer
except ImportError:
    from common import Qwen2_5Config
    from ttnn.ttnn_qwen_embeddings import TTNNQwenEmbeddings
    from ttnn.ttnn_qwen_mlp import TTNNQwenRMSNorm
    from ttnn.ttnn_qwen_layer import TTNNQwenDecoderLayer

class TTNNQwenModel(nn.Module):
    def __init__(self, config: Qwen2_5Config):
        super().__init__()
        self.config = config
        self.embed_tokens = TTNNQwenEmbeddings(config)
        self.layers = nn.ModuleList([TTNNQwenDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = TTNNQwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape

        if attention_mask is None and seq_len > 1:
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device), diagonal=1)
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states

class TTNNQwenForCausalLM(nn.Module):
    def __init__(self, config: Qwen2_5Config):
        super().__init__()
        self.config = config
        self.model = TTNNQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

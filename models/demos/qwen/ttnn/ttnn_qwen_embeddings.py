"""
TTNN Qwen2.5 Embedding Layer
"""

import torch
import torch.nn as nn

try:
    from ..common import Qwen2_5Config
except ImportError:
    from common import Qwen2_5Config

class TTNNQwenEmbeddings(nn.Module):
    def __init__(self, config: Qwen2_5Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.6-27B model configuration for single-device P150a (Blackhole).
"""

from dataclasses import dataclass, field
from pathlib import Path

import ttnn


@dataclass
class Qwen36ModelConfig:
    # Model architecture
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    full_attention_interval: int = 4
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6

    # DeltaNet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # Standard attention
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000000.0

    # FFN
    intermediate_size: int = 17408

    # Weight quantization
    weights_dtype: ttnn.DataType = ttnn.bfloat8_b

    # Inference
    max_batch_size: int = 1
    max_seq_len: int = 8192

    # Paths
    model_name: str = "Qwen/Qwen3.6-27B"
    cache_path: Path = field(default_factory=lambda: Path("/home/yito/ttwork/tt-metal/models/demos/qwen36_27b/weights"))

    @property
    def layer_types(self) -> list[str]:
        types = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.full_attention_interval == 0:
                types.append("full_attention")
            else:
                types.append("linear_attention")
        return types

    @property
    def num_deltanet_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "linear_attention")

    @property
    def num_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def padded_vocab_size(self) -> int:
        tile_size = 32
        return ((self.vocab_size + tile_size - 1) // tile_size) * tile_size

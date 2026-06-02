# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Coder-Next configuration.

Key differences from Qwen3.6-27B:
  - hidden_size: 2048 (vs 5120)
  - num_hidden_layers: 48 (vs 64)
  - vocab_size: 151936 (vs 248320)
  - MoE: 512 experts, top-10 routing, intermediate=512
  - shared_expert_intermediate_size: 512
  - linear_num_value_heads: 32 (vs 48) → head_expand_ratio=2 (vs 3)
  - num_attention_heads: 16 (vs 24)
  - num_key_value_heads: 2 (vs 4)
  - rope_theta: 5_000_000 (vs 1_000_000)

Architecture:
  48 layers, repeating [DeltaNet, DeltaNet, DeltaNet, GQA]
  MoE FFN (512 experts, top-10) + shared expert in every layer
"""

from dataclasses import dataclass


@dataclass
class Qwen3CoderNextConfig:
    # Core
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    full_attention_interval: int = 4
    max_seq_len: int = 262144

    # DeltaNet (linear attention)
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # GQA attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rotary_dim: int = 64
    rope_theta: int = 5000000

    # MoE FFN (replaces dense FFN)
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts: int = 512
    num_experts_per_tok: int = 10

    # Compat
    intermediate_size: int = 5120  # HF compat (not used in MoE path)
    hidden_act: str = "silu"

    # Norm / misc
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936

    # Device-specific
    model_name: str = "Qwen/Qwen3-Coder-Next"
    weights_dtype = None

    @property
    def head_expand_ratio(self):
        return self.linear_num_value_heads // self.linear_num_key_heads  # 2

    @property
    def linear_key_dim(self):
        return self.linear_num_key_heads * self.linear_key_head_dim  # 2048

    @property
    def linear_value_dim(self):
        return self.linear_num_value_heads * self.linear_value_head_dim  # 4096

    @property
    def conv_dim(self):
        return self.linear_key_dim * 2 + self.linear_value_dim  # 2048*2 + 4096 = 8192

    @property
    def num_kv_groups(self):
        return self.num_attention_heads // self.num_key_value_heads  # 8

    @property
    def layer_types(self):
        """3 linear_attention + 1 full_attention repeating pattern."""
        types = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.full_attention_interval == 0:
                types.append("full_attention")
            else:
                types.append("linear_attention")
        return types

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Nano DeepSeek-V3 model package.

Implements the DeepSeek-V3 architecture (671B total / 37B active) at nano scale:
  - Multi-head Latent Attention (MLA) with low-rank Q/KV compression
  - DeepSeek-style Mixture of Experts (MoE) with group routing
  - SwiGLU activation in all feed-forward layers
  - RoPE applied only to a subspace of Q/K heads
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, Embedding, LinearLayer, ModuleList

from .. import RunnerType, memory_efficient_runner
from .transformer import RMSNormLayer, DeepSeekBlock


@dataclass
class DeepSeekConfig:
    """Configuration for nano DeepSeek-V3 model.

    All dimension fields must be divisible by 32 (tile alignment).
    """

    vocab_size: int = 32000
    dim: int = 512
    inter_dim: int = 1536
    moe_inter_dim: int = 256
    n_layers: int = 8
    n_dense_layers: int = 2
    n_heads: int = 8
    # MoE
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 2
    n_limited_groups: int = 1
    score_func: str = "sigmoid"
    route_scale: float = 2.5
    # MLA
    q_lora_rank: int = 256
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    # RoPE
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    # Execution
    runner_type: RunnerType = RunnerType.Default


class DeepSeek(AbstractModuleBase):
    """Nano DeepSeek-V3 transformer model.

    Architecture: token_embed -> N decoder blocks -> final_norm -> lm_head
    First n_dense_layers use dense SwiGLU MLP; remaining layers use MoE.
    """

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.config = config

        padded_vocab = ((config.vocab_size + 31) // 32) * 32
        self.tok_emb = Embedding(padded_vocab, config.dim)

        rope_params = ttml.ops.rope.build_rope_params(config.max_seq_len, config.qk_rope_head_dim, config.rope_theta)

        self.blocks = ModuleList([DeepSeekBlock(layer_id, config, rope_params) for layer_id in range(config.n_layers)])

        self.norm = RMSNormLayer(config.dim)
        self.head = LinearLayer(config.dim, padded_vocab, has_bias=False)

    def forward(
        self,
        tokens: ttml.autograd.Tensor,
        mask: Optional[ttml.autograd.Tensor] = None,
    ) -> ttml.autograd.Tensor:
        """Forward pass.

        Args:
            tokens: Token IDs [B, 1, 1, S]
            mask: Causal attention mask [1, 1, S, S]

        Returns:
            Logits [B, 1, S, vocab_size]
        """
        x = self.tok_emb(tokens)

        for block in self.blocks:
            if self.config.runner_type == RunnerType.MemoryEfficient:
                x = memory_efficient_runner(block, x, mask)
            else:
                x = block(x, mask)

        x = self.norm(x)
        return self.head(x)

    def get_moe_layers(self):
        """Get all MoE layers for external load balance updates."""
        from .moe import MoE

        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]


from .flops import calculate_flops_per_token

__all__ = [
    "DeepSeek",
    "DeepSeekConfig",
    "calculate_flops_per_token",
]

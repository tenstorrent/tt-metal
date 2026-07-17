# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Nano DeepSeek-V3 model package.

Implements the DeepSeek-V3 architecture:
  - Multi-head Latent Attention (MLA) with low-rank Q/KV compression
  - DeepSeek-style Mixture of Experts (MoE) with group routing
  - SwiGLU activation in all feed-forward layers
  - RoPE applied only to a subspace of Q/K heads
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from typing import Literal, Optional

import ttml
from ttml.modules import AbstractModuleBase, ColumnParallelLinear, Embedding, LinearLayer, ModuleList

from .. import RunnerType, memory_efficient_runner
from ..llama.autograd_ops import SliceLastDim
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
    # MoE FFN variant:
    #   dense     — on-device masked experts, reference/debug path (no grouping)
    #   sparse_ep — moe_group/ungroup sparse dispatch with expert-parallel experts
    # sparse_ep partitions the expert list across the "tp" axis (use_tp) or
    # moe_axis_name; with no such axis (single chip) it degenerates to
    # single-device sparse (EP size 1) via SparseMoE.
    moe_type: Literal["dense", "sparse_ep"] = "sparse_ep"
    # MLA (q_lora_rank=0 means direct Q projection without LoRA bottleneck)
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
    # MoE axis. If set and the active mesh has an axis with this name
    # (size > 1), sparse_ep MoE blocks use SparseMoEEP — the routed expert
    # list is partitioned across that axis. Set to None to disable.
    moe_axis_name: str | None = None
    # Full-model TP on the named mesh axis ``"tp"`` (attention, dense MLP,
    # LM head, and sparse MoE). When True, ``SparseMoEEP`` uses ``"tp"``
    # directly; ``moe_axis_name`` is only for MoE-only EP experiments
    # when this flag is False.
    use_tp: bool = False


class DeepSeek(AbstractModuleBase):
    """Nano DeepSeek-V3 transformer model.

    Architecture: token_embed -> N decoder blocks -> final_norm -> lm_head
    First n_dense_layers use dense SwiGLU MLP; remaining layers use MoE.
    """

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_tp:
            tp_size = ttml.mesh().axis_size("tp")
            self._validate_tp(config, tp_size)
            align = lcm(32, tp_size)
            self.padded_vocab_size = ((config.vocab_size + align - 1) // align) * align
            self.head = ColumnParallelLinear(
                config.dim,
                self.padded_vocab_size,
                has_bias=False,
                gather_output=True,
                axis_name="tp",
            )
        else:
            self.padded_vocab_size = ((config.vocab_size + 31) // 32) * 32
            self.head = LinearLayer(config.dim, self.padded_vocab_size, has_bias=False)

        self.tok_emb = Embedding(self.padded_vocab_size, config.dim)

        rope_params = ttml.ops.rope.build_rope_params(config.max_seq_len, config.qk_rope_head_dim, config.rope_theta)

        self.blocks = ModuleList([DeepSeekBlock(layer_id, config, rope_params) for layer_id in range(config.n_layers)])

        self.norm = RMSNormLayer(config.dim)

    @staticmethod
    def _validate_tp(config: DeepSeekConfig, tp_size: int) -> None:
        """Fail fast if the full-model TP size can't evenly shard the model.

        Runs at model build (mesh in scope), not in the config dataclass.
        The MoE routed-expert count is validated in MoE.__init__ where the
        experts are actually EP-sharded.
        """
        # Attention shards heads; dense MLP shards its intermediate dim.
        if config.n_heads % tp_size != 0:
            raise ValueError(f"DeepSeek TP: n_heads ({config.n_heads}) must be divisible by TP size ({tp_size})")
        if config.inter_dim % tp_size != 0:
            raise ValueError(f"DeepSeek TP: inter_dim ({config.inter_dim}) must be divisible by TP size ({tp_size})")
        # MLA parallel linears shard the head-merged feature dims.
        widths = {
            "n_heads * qk_head_dim": config.n_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
            "n_heads * (qk_nope_head_dim + v_head_dim)": config.n_heads * (config.qk_nope_head_dim + config.v_head_dim),
            "n_heads * v_head_dim": config.n_heads * config.v_head_dim,
        }
        for name, width in widths.items():
            if width % tp_size != 0:
                raise ValueError(f"DeepSeek TP: {name}={width} must be divisible by TP size ({tp_size})")

    def forward(
        self,
        tokens: ttml.autograd.Tensor,
        mask: Optional[ttml.autograd.Tensor] = None,
    ) -> ttml.autograd.Tensor:
        """Forward pass.

        Args:
            tokens: Token IDs [B, 1, 1, S]
            mask: Unused. DeepSeek attention is causal-only; the fused SDPA
                generates the causal mask on chip. Kept for the shared
                forward(input, mask) model interface.

        Returns:
            Logits [B, 1, S, vocab_size]
        """

        # `mask` is unused: MLA is causal-only and generates its causal mask on chip
        # (see mla.py). It is still threaded to blocks to keep the shared
        # block(input, mask) / memory_efficient_runner call form intact.
        x = self.tok_emb(tokens)

        for block in self.blocks:
            if self.config.runner_type == RunnerType.MemoryEfficient:
                x = memory_efficient_runner(block, x, mask)
            else:
                x = block(x, mask)

        x = self.norm(x)
        logits = self.head(x)
        if self.padded_vocab_size != self.config.vocab_size:
            logits = SliceLastDim.apply(logits, self.config.vocab_size)
        return logits

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

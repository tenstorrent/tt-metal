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
import os
from math import lcm
from typing import Literal, Optional

import ttnn
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
    #   dense     — on-device masked experts (no token grouping)
    #   sparse    — moe_group/ungroup, single-chip (no MoE parallelism)
    #   sparse_tp — sparse + tensor-parallel experts (shard the intermediate dim)
    #   sparse_ep — sparse + expert-parallel (partition the expert list)
    # sparse_tp/sparse_ep shard across the "tp" axis (use_tp) or moe_axis_name;
    # with no such axis they fall back to plain sparse.
    moe_type: Literal["dense", "sparse", "sparse_tp", "sparse_ep"] = "sparse"
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
    # MoE tensor parallelism. If set and the active mesh has an axis
    # with this name (size > 1), MoE blocks use SparseMoETP — every
    # chip in that axis holds all routed experts with each expert's
    # intermediate dim sharded across the axis. Set to None to disable.
    moe_axis_name: str | None = None
    # Full-model TP on the named mesh axis ``"tp"`` (attention, dense MLP,
    # LM head, and sparse MoE). When True, ``SparseMoETP`` uses ``"tp"``
    # directly; ``moe_axis_name`` is only for MoE-only TP experiments
    # when this flag is False.
    use_tp: bool = False

    def __post_init__(self) -> None:
        if not self.use_tp:
            return
        mesh = ttml.maybe_mesh()
        if mesh is None or not mesh.has_axis("tp"):
            raise ValueError("DeepSeekConfig.use_tp=True requires an active mesh with a 'tp' axis")
        tp_size = mesh.axis_size("tp")
        if self.n_heads % tp_size != 0:
            raise ValueError(
                "DeepSeek TP: n_heads must be divisible by TP size. " f"n_heads={self.n_heads}, tp_size={tp_size}"
            )
        if self.inter_dim % tp_size != 0:
            raise ValueError(
                "DeepSeek TP: inter_dim must be divisible by TP size. " f"inter_dim={self.inter_dim}, tp_size={tp_size}"
            )
        if self.moe_inter_dim % tp_size != 0:
            raise ValueError(
                "DeepSeek TP: moe_inter_dim must be divisible by TP size. "
                f"moe_inter_dim={self.moe_inter_dim}, tp_size={tp_size}"
            )
        # for name, width in (("inter_dim", self.inter_dim), ("moe_inter_dim", self.moe_inter_dim)):
        #     local_width = width // tp_size
        #     if local_width % 32 != 0:
        #         raise ValueError(
        #             f"DeepSeek TP: local {name} shard must be divisible by 32 tiles. "
        #             f"{name}={width}, tp_size={tp_size}, local_shard={local_width}"
        #         )
        qk_out = self.n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        kv_up_out = self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        attn_out = self.n_heads * self.v_head_dim
        for name, width in (
            ("n_heads * qk_head_dim", qk_out),
            ("n_heads * (qk_nope_head_dim + v_head_dim)", kv_up_out),
            ("n_heads * v_head_dim", attn_out),
        ):
            if width % tp_size != 0:
                raise ValueError(
                    f"DeepSeek TP: {name}={width} must be divisible by TP size ({tp_size}) "
                    "(MLA parallel linears shard the head-merged feature dim)."
                )


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

        read_profiler_after_block = os.environ.get("TTML_TRACY_READ_RESULTS_AFTER_BLOCK", "0") == "1"
        profiler_device = ttml.autograd.AutoContext.get_instance().get_device() if read_profiler_after_block else None

        for block in self.blocks:
            if self.config.runner_type == RunnerType.MemoryEfficient:
                x = memory_efficient_runner(block, x, mask)
            else:
                x = block(x, mask)
            if read_profiler_after_block:
                ttnn.ReadDeviceProfiler(profiler_device)

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

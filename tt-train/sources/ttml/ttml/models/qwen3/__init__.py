# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 model package.

Implements the Qwen3 architecture:
  - Separate Q, K, V projections with QK-Norm (RMSNorm per head)
  - SwiGLU MLP
  - RoPE positional encoding
  - Grouped-query attention (GQA)
  - Explicit head_dim (can differ from hidden_size / num_heads)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Embedding, ModuleList, LinearLayer

from .. import RunnerType, WeightTyingType, memory_efficient_runner
from .transformer import Qwen3Block, Qwen3RMSNorm
from .autograd_ops import ConcatLastDim, RMSNormFunction


@dataclass(frozen=True)
class Qwen3RopeScalingConfig:
    scaling_factor: float = 0.0
    high_freq_factor: float = 4.0
    low_freq_factor: float = 1.0
    original_context_length: int = 0


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 256
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_theta: float = 1000000.0
    runner_type: RunnerType = RunnerType.Default
    weight_tying: WeightTyingType = WeightTyingType.Disabled
    rope_scaling: Qwen3RopeScalingConfig = field(default_factory=Qwen3RopeScalingConfig)


def create_qwen3_config_from_hf(
    hf_config,
    max_sequence_length: int,
    runner_type: RunnerType = RunnerType.Default,
) -> Qwen3Config:
    """Create a Qwen3Config from a HuggingFace config object."""
    rope_scaling_factor = 0.0
    rope_original_context_length = 0
    rope_high_freq_factor = 4.0
    rope_low_freq_factor = 1.0
    if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling:
        rs = hf_config.rope_scaling
        rope_scaling_factor = rs.get("factor", 0.0)
        rope_original_context_length = rs.get("original_max_position_embeddings", 0)
        rope_high_freq_factor = rs.get("high_freq_factor", 4.0)
        rope_low_freq_factor = rs.get("low_freq_factor", 1.0)

    return Qwen3Config(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        ),
        max_position_embeddings=max_sequence_length,
        rms_norm_eps=hf_config.rms_norm_eps,
        attention_bias=getattr(hf_config, "attention_bias", True),
        rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
        runner_type=runner_type,
        weight_tying=(
            WeightTyingType.Enabled if getattr(hf_config, "tie_word_embeddings", False) else WeightTyingType.Disabled
        ),
        rope_scaling=Qwen3RopeScalingConfig(
            scaling_factor=rope_scaling_factor,
            original_context_length=rope_original_context_length,
            high_freq_factor=rope_high_freq_factor,
            low_freq_factor=rope_low_freq_factor,
        ),
    )


class Qwen3(AbstractModuleBase):
    """Qwen3 transformer model.

    Architecture: token_embed -> N decoder blocks -> final_norm -> lm_head

    Args:
        config: Model configuration.
        track_memory: When > 0, insert memory snapshots every N layers
            (and after embedding / norm / LM-head).  Default 0 = disabled.
    """

    def __init__(self, config: Qwen3Config, track_memory: int = 0) -> None:
        super().__init__()

        self.config = config
        self.track_memory = track_memory

        self.fc = LinearLayer(
            config.hidden_size,
            config.vocab_size,
            False,
            weight_init=ttml.init.normal(0.0, 0.02),
        )

        vocab_size_divisible_by_32 = (config.vocab_size + 31) // 32 * 32
        self.tok_emb = Embedding(
            vocab_size_divisible_by_32,
            config.hidden_size,
            weight_init=ttml.init.normal(0.0, 0.02),
        )

        if config.weight_tying == ttml.models.WeightTyingType.Enabled:
            self.tok_emb.weight = self.fc.weight

        self.blocks = ModuleList([Qwen3Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.ln_fc = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _snapshot(self, x, fwd_label: str, bwd_label: str):
        if not self.track_memory:
            return x
        from utils.memory import memory_snapshot

        return memory_snapshot(x, fwd_label, bwd_label)

    def forward(
        self,
        input: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        past_key_values=None,
        new_tokens: Optional[int] = None,
        **kwargs,
    ) -> ttml.autograd.Tensor:
        TILE_SIZE = 32
        input_shape = input.shape()
        actual_seq_len = input_shape[-1]
        padded_seq_len = ((actual_seq_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE

        input_padded = input
        if padded_seq_len != actual_seq_len:
            padding = [(0, 0), (0, 0), (0, 0), (0, padded_seq_len - actual_seq_len)]
            input_val_padded = ttnn.pad(input.get_value(), padding=padding, value=0.0)
            input_padded = ttml.autograd.create_tensor(input_val_padded)

        tok_emb_out = self.tok_emb(input_padded)

        out = tok_emb_out
        if padded_seq_len != actual_seq_len:
            slice_start = [0, 0, 0, 0]
            slice_end = [
                tok_emb_out.shape()[0],
                tok_emb_out.shape()[1],
                actual_seq_len,
                tok_emb_out.shape()[3],
            ]
            step = [1, 1, 1, 1]
            out_val = ttnn.slice(tok_emb_out.get_value(), slice_start, slice_end, step)
            out = ttml.autograd.create_tensor(out_val)

        out = self._snapshot(out, "AFTER_EMBEDDING_FWD", "AFTER_EMBEDDING_BWD")

        position_offset = 0
        if past_key_values is not None:
            position_offset = past_key_values.get_seq_length()

        for i, block in enumerate(self.blocks):
            if self.config.runner_type == ttml.models.RunnerType.MemoryEfficient:
                out = memory_efficient_runner(block, out, mask, past_key_values, position_offset)
            elif self.config.runner_type == ttml.models.RunnerType.Default:
                out = block(out, mask, past_key_values, position_offset)
            else:
                raise ValueError("Unknown runner type. Supported runner types ['default', 'memory_efficient']")
            if self.track_memory and (i + 1) % self.track_memory == 0:
                out = self._snapshot(out, f"AFTER_LAYER_{i}_FWD", f"AFTER_LAYER_{i}_BWD")

        out = self.ln_fc(out)
        out = self._snapshot(out, "AFTER_NORM_FWD", "AFTER_NORM_BWD")
        logits = self.fc(out)
        logits = self._snapshot(logits, "AFTER_LM_HEAD_FWD", "AFTER_LM_HEAD_BWD")
        return logits


from .flops import calculate_flops_per_token

__all__ = [
    "ConcatLastDim",
    "Qwen3",
    "Qwen3Block",
    "Qwen3Config",
    "Qwen3RMSNorm",
    "Qwen3RopeScalingConfig",
    "RMSNormFunction",
    "calculate_flops_per_token",
    "create_qwen3_config_from_hf",
]

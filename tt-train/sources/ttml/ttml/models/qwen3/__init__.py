# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 model implementation for ttml.

Architecture: HuggingFace Qwen3 design with:
  - Separate Q, K, V projections (not fused KV)
  - QK-Norm (RMSNorm on Q and K per head_dim)
  - Configurable attention bias
  - SwiGLU MLP
  - RoPE positional encoding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, Embedding, LinearLayer, ModuleList

from .transformer import Qwen3DecoderLayer
from .autograd_ops import Qwen3RMSNorm, memory_snapshot


# =====================================================================
# Configuration
# =====================================================================


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True
    attention_dropout: float = 0.0
    rope_theta: float = 1000000.0
    rope_scaling_factor: float = 0.0
    rope_original_context_length: int = 0
    rope_high_freq_factor: float = 4.0
    rope_low_freq_factor: float = 1.0


def create_qwen3_config_from_hf(hf_config, max_sequence_length: int) -> Qwen3Config:
    """Create Qwen3Config from a HuggingFace config object."""
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
        rope_scaling_factor=rope_scaling_factor,
        rope_original_context_length=rope_original_context_length,
        rope_high_freq_factor=rope_high_freq_factor,
        rope_low_freq_factor=rope_low_freq_factor,
    )


# =====================================================================
# Qwen3Model (backbone)
# =====================================================================


class Qwen3Model(AbstractModuleBase):
    """Qwen3 backbone: embedding + decoder layers + final RMSNorm."""

    def __init__(self, config: Qwen3Config, track_memory: int = 0, use_checkpoint: bool = False):
        super().__init__()
        self.config = config
        self.track_memory = track_memory
        self.use_checkpoint = use_checkpoint

        vocab_size_tiled = ((config.vocab_size + 31) // 32) * 32
        self.embed_tokens = Embedding(
            vocab_size_tiled,
            config.hidden_size,
            weight_init=ttml.init.normal(0.0, 0.02),
        )
        self.layers = ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: ttml.autograd.Tensor,
        attention_mask: Optional[ttml.autograd.Tensor] = None,
        past_key_values=None,
    ) -> ttml.autograd.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        if self.track_memory:
            hidden_states = memory_snapshot(hidden_states, "AFTER_EMBEDDING_FWD", "AFTER_EMBEDDING_BWD")

        position_offset = 0
        if past_key_values is not None:
            position_offset = past_key_values.get_seq_length()

        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                hidden_states = ttml.models.memory_efficient_runner(
                    layer,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    position_offset,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    position_offset=position_offset,
                )
            if self.track_memory and (i + 1) % self.track_memory == 0:
                hidden_states = memory_snapshot(hidden_states, f"AFTER_LAYER_{i}_FWD", f"AFTER_LAYER_{i}_BWD")

        hidden_states = self.norm(hidden_states)
        return hidden_states


# =====================================================================
# Qwen3ForCausalLM
# =====================================================================


class Qwen3ForCausalLM(AbstractModuleBase):
    """Qwen3 for causal language modeling."""

    def __init__(
        self,
        config: Qwen3Config,
        tie_word_embeddings: bool = False,
        track_memory: int = 0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.create_name("Qwen3ForCausalLM")
        self.config = config
        self.tie_word_embeddings = tie_word_embeddings
        self.track_memory = track_memory
        self.model = Qwen3Model(config, track_memory=track_memory, use_checkpoint=use_checkpoint)

        if tie_word_embeddings:
            self.lm_head = None
        else:
            vocab_size_tiled = ((config.vocab_size + 31) // 32) * 32
            self.lm_head = LinearLayer(
                config.hidden_size,
                vocab_size_tiled,
                has_bias=False,
                weight_init=ttml.init.normal(0.0, 0.02),
            )

    def forward(
        self,
        input_ids: ttml.autograd.Tensor,
        attention_mask: Optional[ttml.autograd.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> ttml.autograd.Tensor:
        hidden_states = self.model(input_ids, attention_mask, past_key_values)
        if self.track_memory:
            hidden_states = memory_snapshot(hidden_states, "AFTER_NORM_FWD", "AFTER_NORM_BWD")

        if self.tie_word_embeddings:
            logits = ttml.ops.linear.linear(hidden_states, self.model.embed_tokens.weight.tensor, None)
        else:
            logits = self.lm_head(hidden_states)

        if self.track_memory:
            logits = memory_snapshot(logits, "AFTER_LM_HEAD_FWD", "AFTER_LM_HEAD_BWD")
        return logits


# =====================================================================
# Late imports (these modules may reference types from this file)
# =====================================================================

from .kv_cache import KVCache
from .safetensors_loader import load_from_safetensors

__all__ = [
    "Qwen3Config",
    "Qwen3Model",
    "Qwen3ForCausalLM",
    "create_qwen3_config_from_hf",
    "load_from_safetensors",
    "KVCache",
]

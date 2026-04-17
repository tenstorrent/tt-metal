# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Python Qwen3 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, Embedding, LinearLayer, ModuleList

from .. import RunnerType, WeightTyingType, memory_efficient_runner
from .attention import Qwen3Attention, build_attn_mask
from .rmsnorm import Qwen3RMSNorm, RMSNormFunction
from .transformer import Qwen3DecoderLayer, Qwen3MLP


@dataclass(frozen=True)
class Qwen3RopeScalingConfig:
    scaling_factor: float = 0.0  # 0.0 disables scaling
    high_freq_factor: float = 4.0
    low_freq_factor: float = 1.0
    original_context_length: int = 0  # 0 disables scaling


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True
    attention_dropout: float = 0.0
    rope_theta: float = 1000000.0
    runner_type: RunnerType = RunnerType.Default
    weight_tying: WeightTyingType = WeightTyingType.Disabled
    rope_scaling: Qwen3RopeScalingConfig = field(default_factory=Qwen3RopeScalingConfig)

    def __post_init__(self):
        if self.max_position_embeddings % 32 != 0:
            raise ValueError(
                "max_position_embeddings must be divisible by 32 (tile size); " f"got {self.max_position_embeddings}"
            )
        if self.hidden_size % 32 != 0:
            raise ValueError(f"hidden_size must be divisible by 32 (tile size); got {self.hidden_size}")
        if self.head_dim % 32 != 0:
            raise ValueError(f"head_dim must be divisible by 32 (tile size); got {self.head_dim}")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive; got {self.num_attention_heads}")
        if self.num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive; got {self.num_key_value_heads}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads; "
                f"got num_attention_heads={self.num_attention_heads}, "
                f"num_key_value_heads={self.num_key_value_heads}"
            )


def create_qwen3_config_from_hf(hf_config, max_sequence_length: int) -> Qwen3Config:
    """Build a Qwen3Config from a HuggingFace config object."""
    rope_scaling = Qwen3RopeScalingConfig()
    if getattr(hf_config, "rope_scaling", None):
        rs = hf_config.rope_scaling
        rope_scaling = Qwen3RopeScalingConfig(
            scaling_factor=rs.get("factor", 0.0),
            high_freq_factor=rs.get("high_freq_factor", 4.0),
            low_freq_factor=rs.get("low_freq_factor", 1.0),
            original_context_length=rs.get("original_max_position_embeddings", 0),
        )

    return Qwen3Config(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
        max_position_embeddings=max_sequence_length,
        rms_norm_eps=hf_config.rms_norm_eps,
        attention_bias=getattr(hf_config, "attention_bias", True),
        rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
        rope_scaling=rope_scaling,
        weight_tying=(
            WeightTyingType.Enabled if getattr(hf_config, "tie_word_embeddings", False) else WeightTyingType.Disabled
        ),
    )


def _build_rope_params(config: Qwen3Config) -> ttml.ops.rope.RotaryEmbeddingParams:
    scaling = ttml.ops.rope.RopeScalingParams()
    if config.rope_scaling.scaling_factor != 0.0 and config.rope_scaling.original_context_length != 0:
        scaling.scaling_factor = config.rope_scaling.scaling_factor
        scaling.high_freq_factor = config.rope_scaling.high_freq_factor
        scaling.low_freq_factor = config.rope_scaling.low_freq_factor
        scaling.original_context_length = config.rope_scaling.original_context_length
    return ttml.ops.rope.build_rope_params(
        sequence_length=config.max_position_embeddings,
        head_dim=config.head_dim,
        theta=config.rope_theta,
        rope_scaling_params=scaling,
    )


def _padded_vocab_size(vocab_size: int) -> int:
    return ((vocab_size + 31) // 32) * 32


class Qwen3Model(AbstractModuleBase):
    """Qwen3 backbone: embeddings → decoder stack → final RMSNorm."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config

        rope_params = _build_rope_params(config)

        self.embed_tokens = Embedding(
            _padded_vocab_size(config.vocab_size),
            config.hidden_size,
            weight_init=ttml.init.normal(0.0, 0.02),
        )
        self.layers = ModuleList(
            [
                Qwen3DecoderLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    rope_params=rope_params,
                    attention_bias=config.attention_bias,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: Optional[ttml.models.KvCache] = None,
        new_tokens: Optional[int] = None,
    ) -> ttml.autograd.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(self.layers):
            extra_args = () if kv_cache is None else (kv_cache, layer_idx, new_tokens)
            if self.config.runner_type == RunnerType.MemoryEfficient:
                hidden_states = memory_efficient_runner(layer, hidden_states, mask, *extra_args)
            elif self.config.runner_type == RunnerType.Default:
                hidden_states = layer(hidden_states, mask, *extra_args)
            else:
                raise ValueError(f"Unknown runner_type: {self.config.runner_type}")

        return self.norm(hidden_states)


class Qwen3(AbstractModuleBase):
    """Qwen3 causal-LM: backbone + unembedding head."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config

        self.model = Qwen3Model(config)
        self.lm_head = LinearLayer(
            config.hidden_size,
            _padded_vocab_size(config.vocab_size),
            has_bias=False,
            weight_init=ttml.init.normal(0.0, 0.02),
        )

        if config.weight_tying == WeightTyingType.Enabled:
            self.lm_head.weight = self.model.embed_tokens.weight.tensor

    def forward(
        self,
        input_ids: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: Optional[ttml.models.KvCache] = None,
        new_tokens: Optional[int] = None,
    ) -> ttml.autograd.Tensor:
        hidden_states = self.model(input_ids, mask, kv_cache, new_tokens)
        return self.lm_head(hidden_states)


from .safetensors_loader import export_hf_model, load_from_hf

__all__ = [
    "Qwen3",
    "Qwen3Attention",
    "Qwen3Config",
    "Qwen3DecoderLayer",
    "Qwen3MLP",
    "Qwen3Model",
    "Qwen3RMSNorm",
    "Qwen3RopeScalingConfig",
    "RMSNormFunction",
    "build_attn_mask",
    "create_qwen3_config_from_hf",
    "export_hf_model",
    "load_from_hf",
]

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Llama 3.1 8B model configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for meta-llama/Llama-3.1-8B-Instruct.
"""

from dataclasses import dataclass, field


@dataclass
class RopeScalingConfig:
    """RoPE scaling parameters for Llama 3.1."""

    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192
    rope_type: str = "llama3"


class Llama31ModelConfig:
    """Static model dimension constants (class-level) for the prefill adapter's model_config.
    Accessed as ``ADAPTER.model_config.NUM_KEY_VALUE_HEADS`` etc. by the common runner."""

    NUM_ATTENTION_HEADS = 32
    NUM_KEY_VALUE_HEADS = 8
    HEAD_DIM = 128
    NUM_HIDDEN_LAYERS = 32
    HIDDEN_SIZE = 4096
    INTERMEDIATE_SIZE = 14336
    FABRIC_PAYLOAD_SIZE = 4096


@dataclass
class Llama31Config:
    """Llama 3.1 8B model dimensions (from HF config.json)."""

    # Core dimensions
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    num_hidden_layers: int = 32

    # Normalization
    rms_norm_eps: float = 1e-05

    # RoPE
    rope_theta: float = 500000.0
    rope_scaling: RopeScalingConfig = field(default_factory=RopeScalingConfig)

    # Vocabulary
    vocab_size: int = 128256

    # Activation
    hidden_act: str = "silu"

    # Bias flags
    attention_bias: bool = False
    mlp_bias: bool = False

    # Derived / explicit head dimension
    head_dim: int = 128  # hidden_size // num_attention_heads

    # Special tokens
    bos_token_id: int = 128000

    # Weight tying
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf_config(cls, hf_config) -> "Llama31Config":
        """Construct a Llama31Config from a loaded HuggingFace config object.

        Args:
            hf_config: A HuggingFace PretrainedConfig (e.g. from AutoConfig.from_pretrained)
                or any object with the corresponding attributes / dict keys.

        Returns:
            Llama31Config with values populated from the HF config.
        """

        # Support both attribute access (PretrainedConfig) and dict-like access
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        rope_raw = _get(hf_config, "rope_scaling", {}) or {}
        rope_scaling = RopeScalingConfig(
            factor=_get(rope_raw, "factor", 8.0),
            low_freq_factor=_get(rope_raw, "low_freq_factor", 1.0),
            high_freq_factor=_get(rope_raw, "high_freq_factor", 4.0),
            original_max_position_embeddings=_get(rope_raw, "original_max_position_embeddings", 8192),
            rope_type=_get(rope_raw, "rope_type", "llama3"),
        )

        hidden_size = _get(hf_config, "hidden_size", 4096)
        num_attention_heads = _get(hf_config, "num_attention_heads", 32)
        head_dim = _get(hf_config, "head_dim", hidden_size // num_attention_heads)

        return cls(
            hidden_size=hidden_size,
            intermediate_size=_get(hf_config, "intermediate_size", 14336),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=_get(hf_config, "num_key_value_heads", 8),
            num_hidden_layers=_get(hf_config, "num_hidden_layers", 32),
            rms_norm_eps=_get(hf_config, "rms_norm_eps", 1e-05),
            rope_theta=_get(hf_config, "rope_theta", 500000.0),
            rope_scaling=rope_scaling,
            vocab_size=_get(hf_config, "vocab_size", 128256),
            hidden_act=_get(hf_config, "hidden_act", "silu"),
            attention_bias=_get(hf_config, "attention_bias", False),
            mlp_bias=_get(hf_config, "mlp_bias", False),
            head_dim=head_dim,
            bos_token_id=_get(hf_config, "bos_token_id", 128000),
            tie_word_embeddings=_get(hf_config, "tie_word_embeddings", False),
        )

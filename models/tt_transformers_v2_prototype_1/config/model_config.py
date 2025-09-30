# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model configuration classes for TTTv2"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    """
    Base configuration class for all models.

    Provides common configuration parameters and methods.
    """

    model_type: str
    hidden_size: int
    vocab_size: int
    max_position_embeddings: int = 2048
    dtype: str = "bfloat16"
    tie_word_embeddings: bool = False

    # Device configuration
    num_devices: int = 1
    device_arch: str = "wormhole_b0"

    # Optimization settings
    use_flash_attention: bool = True
    use_fused_ops: bool = True

    # Custom attributes
    custom_attrs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "ModelConfig":
        """
        Load configuration from a pretrained model.

        Args:
            model_name_or_path: Model name or path to config file

        Returns:
            Model configuration instance
        """
        config_path = Path(model_name_or_path)

        if config_path.is_file():
            # Load from file
            return cls.from_file(config_path)
        else:
            # Load from model registry
            return cls.from_model_name(model_name_or_path)

    @classmethod
    def from_file(cls, config_path: Path) -> "ModelConfig":
        """Load configuration from file"""
        if config_path.suffix == ".json":
            with open(config_path) as f:
                config_dict = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        # Determine config class based on model_type
        model_type = config_dict.get("model_type", "transformer")
        config_class = MODEL_CONFIG_REGISTRY.get(model_type, cls)

        return config_class(**config_dict)

    @classmethod
    def from_model_name(cls, model_name: str) -> "ModelConfig":
        """Load configuration from model registry"""
        # This would connect to a model registry
        # For now, return a default config
        raise NotImplementedError(f"Model registry not implemented for: {model_name}")

    def save(self, save_path: Path):
        """Save configuration to file"""
        config_dict = self.to_dict()

        if save_path.suffix == ".json":
            with open(save_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif save_path.suffix in [".yaml", ".yml"]:
            with open(save_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported save format: {save_path.suffix}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_attrs[key] = value

    def validate(self):
        """Validate configuration parameters"""
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_position_embeddings > 0, "max_position_embeddings must be positive"
        assert self.num_devices > 0, "num_devices must be positive"


@dataclass
class TransformerConfig(ModelConfig):
    """
    Configuration for transformer models.

    Extends base configuration with transformer-specific parameters.
    """

    model_type: str = "transformer"

    # Architecture parameters
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    hidden_act: str = "silu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    use_cache: bool = True

    # Attention configuration
    attention_type: str = "standard"  # Options: "standard", "flash", "sliding_window"
    sliding_window_size: Optional[int] = None
    attention_bias: bool = False

    # RoPE configuration
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0

    # Architecture variants
    use_parallel_residual: bool = False
    use_bias: bool = False
    norm_type: str = "rmsnorm"  # Options: "layernorm", "rmsnorm"
    mlp_type: str = "standard"  # Options: "standard", "gated", "moe"

    # MoE configuration (if applicable)
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    expert_routing_type: str = "top_k"

    def __post_init__(self):
        """Post-initialization setup"""
        # Set defaults based on other parameters
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.intermediate_size is None:
            # Default to 4x hidden size
            self.intermediate_size = 4 * self.hidden_size

        # Validate configuration
        self.validate()

    def validate(self):
        """Validate transformer configuration"""
        super().validate()

        assert self.num_hidden_layers > 0, "num_hidden_layers must be positive"
        assert self.num_attention_heads > 0, "num_attention_heads must be positive"
        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        if self.num_key_value_heads is not None:
            assert (
                self.num_attention_heads % self.num_key_value_heads == 0
            ), "num_attention_heads must be divisible by num_key_value_heads"

        if self.mlp_type == "moe":
            assert self.num_experts is not None, "num_experts required for MoE"
            assert self.num_experts_per_tok is not None, "num_experts_per_tok required for MoE"

    def get_layer_config(self, layer_idx: int) -> Dict[str, Any]:
        """
        Get configuration for a specific layer.

        Allows for per-layer configuration overrides.
        """
        layer_config = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "attention_type": self.attention_type,
            "norm_type": self.norm_type,
            "use_parallel_residual": self.use_parallel_residual,
        }

        # Apply layer-specific overrides if any
        layer_overrides = self.custom_attrs.get(f"layer_{layer_idx}", {})
        layer_config.update(layer_overrides)

        return layer_config

    @property
    def head_dim(self) -> int:
        """Calculate head dimension"""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_parameters(self) -> int:
        """Estimate total number of parameters"""
        # Embedding parameters
        embedding_params = self.vocab_size * self.hidden_size

        # Transformer block parameters per layer
        # Attention: Q, K, V, O projections
        attn_params = 4 * self.hidden_size * self.hidden_size

        # MLP parameters
        if self.mlp_type == "gated":
            # 3 matrices: up1, up2, down
            mlp_params = 3 * self.hidden_size * self.intermediate_size
        else:
            # 2 matrices: up, down
            mlp_params = 2 * self.hidden_size * self.intermediate_size

        # Layer norm parameters
        ln_params = 2 * self.hidden_size  # Two layer norms per block

        # Total per layer
        params_per_layer = attn_params + mlp_params + ln_params

        # Total parameters
        total_params = embedding_params + (self.num_hidden_layers * params_per_layer)

        # Output projection (if not tied)
        if not self.tie_word_embeddings:
            total_params += self.vocab_size * self.hidden_size

        return total_params


@dataclass
class VisionTransformerConfig(ModelConfig):
    """Configuration for vision transformer models"""

    model_type: str = "vision_transformer"

    # Vision-specific parameters
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-6

    # Vision embeddings
    use_cls_token: bool = True
    use_position_embeddings: bool = True
    interpolate_position_embeddings: bool = False

    @property
    def num_patches(self) -> int:
        """Calculate number of patches"""
        return (self.image_size // self.patch_size) ** 2

    @property
    def sequence_length(self) -> int:
        """Calculate sequence length including CLS token"""
        seq_len = self.num_patches
        if self.use_cls_token:
            seq_len += 1
        return seq_len


@dataclass
class EncoderDecoderConfig(ModelConfig):
    """Configuration for encoder-decoder models"""

    model_type: str = "encoder_decoder"

    # Encoder configuration
    encoder_config: TransformerConfig = None
    # Decoder configuration
    decoder_config: TransformerConfig = None

    # Cross-attention configuration
    add_cross_attention: bool = True
    tie_encoder_decoder: bool = False

    def __post_init__(self):
        """Initialize encoder and decoder configs"""
        if self.encoder_config is None:
            self.encoder_config = TransformerConfig(
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
            )

        if self.decoder_config is None:
            self.decoder_config = TransformerConfig(
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                add_cross_attention=self.add_cross_attention,
            )


# Model configuration registry
MODEL_CONFIG_REGISTRY = {
    "transformer": TransformerConfig,
    "vision_transformer": VisionTransformerConfig,
    "encoder_decoder": EncoderDecoderConfig,
}


def register_model_config(model_type: str, config_class: type):
    """Register a new model configuration class"""
    MODEL_CONFIG_REGISTRY[model_type] = config_class

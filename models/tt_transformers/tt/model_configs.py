# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Pydantic models for standardizing LLM configuration formats.

This module provides a way to parse different model configuration formats
(Meta, HuggingFace, etc.) into a standardized format that can be consumed
by the rest of the codebase without modification.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, model_validator


class ModelArchitecture(str, Enum):
    """Supported model architectures."""

    LLAMA = "llama"
    QWEN2 = "qwen2"
    DEEPSEEK_V3 = "deepseek_v3"
    MISTRAL = "mistral"


class RopeScalingType(str, Enum):
    """Types of RoPE scaling."""

    # LINEAR = "linear"
    # DYNAMIC = "dynamic"
    YARN = "yarn"
    LLAMA3 = "llama3"


class RopeScaling(BaseModel):
    """RoPE scaling configuration."""

    rope_type: RopeScalingType = Field(exclude=True, description="RoPE scaling type")
    factor: float
    original_max_position_embeddings: int
    # Llama-3.x specific parameters
    low_freq_factor: Optional[float] = None
    high_freq_factor: Optional[float] = None
    # Yarn-specific parameters - we could have a separate class for each type
    beta_fast: Optional[int] = None
    beta_slow: Optional[int] = None
    mscale: Optional[float] = None
    mscale_all_dim: Optional[float] = None


class TTModelConfig(BaseModel):
    """
    Standardized model configuration that all specific formats convert to.
    This is what model_config.py should consume.
    """

    # Core model dimensions
    dim: int = Field(description="Model dimension/hidden size")
    n_layers: int = Field(description="Number of transformer layers")
    n_heads: int = Field(description="Number of attention heads")
    n_kv_heads: int = Field(description="Number of key-value heads")
    vocab_size: int = Field(description="Vocabulary size")
    max_context_len: int = Field(description="Maximum context length")
    head_dim: Optional[int] = Field(None, description="Dimension per attention head")
    padded_vocab_size: Optional[int] = Field(None, description="Padded vocabulary size")

    # MLP configuration
    hidden_dim: Optional[int] = Field(None, description="Hidden dimension for MLP")
    ffn_dim_multiplier: Optional[float] = Field(None, description="FFN dimension multiplier")
    multiple_of: Optional[int] = Field(None, description="FFN dimension must be multiple of this")

    # Normalization
    norm_eps: float = Field(description="RMS norm epsilon")

    # RoPE configuration
    rope_theta: float = Field(description="RoPE theta parameter")
    rope_scaling: Optional[RopeScaling] = Field(None, description="RoPE scaling configuration")

    # Vision model parameters (for multimodal models)
    vision_chunk_size: Optional[int] = Field(-1, description="Vision chunk size")
    vision_max_num_chunks: Optional[int] = Field(4, description="Maximum number of vision chunks")
    vision_num_cross_attention_layers: Optional[int] = Field(-1, description="Number of cross-attention layers")

    # Model metadata
    model_type: Optional[str] = Field(None, description="Model type identifier")
    architecture: Optional[ModelArchitecture] = Field(None, description="Model architecture")

    @model_validator(mode="after")
    def compute_head_dim(self) -> "TTModelConfig":
        """Compute head_dim if not provided."""
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads
        return self

    @model_validator(mode="after")
    def compute_hidden_dim(self) -> "TTModelConfig":
        """Helper function based on logic used in reference model:
        https://github.com/meta-llama/llama-models/blob/e4a6ed52a142bb9b5106dcbf48e41f97f8e7378e/models/llama3/reference_impl/model.py#L227C7-L231C83
        """
        if self.hidden_dim is None:
            if self.ffn_dim_multiplier is not None and self.multiple_of is not None:
                hidden_dim = int(2 * (4 * self.dim) / 3)
                if self.ffn_dim_multiplier is not None:
                    hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
                self.hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)
            else:
                raise ValidationError("hidden_dim is required if ffn_dim_multiplier and multiple_of are not provided")
        return self


class MetaLlamaConfig(BaseModel):
    """Meta format Llama configuration (params.json)."""

    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    ffn_dim_multiplier: float
    multiple_of: int
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    rope_scaling_factor: Optional[float] = None

    def to_standard(self) -> TTModelConfig:
        """Convert to tt format."""
        rope_scaling_obj = None
        if self.rope_scaling_factor:
            rope_scaling_obj = RopeScaling(
                rope_type=RopeScalingType.LLAMA3,
                factor=self.rope_scaling_factor,
                original_max_position_embeddings=8192,
            )
        return TTModelConfig(
            dim=self.dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            vocab_size=self.vocab_size,
            max_context_len=128 * 1024,
            ffn_dim_multiplier=self.ffn_dim_multiplier,
            multiple_of=self.multiple_of,
            norm_eps=self.norm_eps,
            rope_theta=self.rope_theta,
            rope_scaling=rope_scaling_obj,
            architecture=ModelArchitecture.LLAMA,
            model_type="llama",
        )


class HuggingFaceLlamaConfig(BaseModel):
    """HuggingFace format Llama configuration (config.json)."""

    architectures: List[str]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    model_type: str

    # FIXME (Harry): All these optional fields are not used in TTT; we can just not include them here if we want
    # Optional fields
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: Optional[bool] = None
    attention_dropout: Optional[float] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = None
    hidden_act: Optional[str] = None
    initializer_range: Optional[float] = None
    mlp_bias: Optional[bool] = None
    pretraining_tp: Optional[int] = None
    tie_word_embeddings: Optional[bool] = None
    torch_dtype: Optional[str] = None
    transformers_version: Optional[str] = None
    use_cache: Optional[bool] = None

    def to_standard(self) -> TTModelConfig:
        """Convert to tt format."""
        rope_scaling_obj = None
        if self.rope_scaling:
            rope_scaling_obj = RopeScaling(**self.rope_scaling)

        return TTModelConfig(
            dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            hidden_dim=self.intermediate_size,
            norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            rope_scaling=rope_scaling_obj,
            max_context_len=self.max_position_embeddings,
            architecture=ModelArchitecture.LLAMA,
            model_type=self.model_type,
        )


class QwenConfig(BaseModel):
    """Qwen2.5 model configuration."""

    architectures: List[str]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    model_type: str

    # FIXME (Harry): All these optional fields are not used in TTT; we can just not include them here if we want
    # Qwen-specific fields
    attention_dropout: Optional[float] = None
    sliding_window: Optional[int] = None
    max_window_layers: Optional[int] = None
    use_sliding_window: Optional[bool] = None

    # Optional common fields
    rope_scaling: Optional[Dict[str, Any]] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    hidden_act: Optional[str] = None
    initializer_range: Optional[float] = None
    tie_word_embeddings: Optional[bool] = None
    torch_dtype: Optional[str] = None
    transformers_version: Optional[str] = None
    use_cache: Optional[bool] = None

    def to_standard(self) -> TTModelConfig:
        rope_scaling_obj = None
        if self.rope_scaling:
            rope_scaling_obj = RopeScaling(**self.rope_scaling)
        """Convert to tt format."""
        return TTModelConfig(
            dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            hidden_dim=self.intermediate_size,
            norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            max_context_len=self.max_position_embeddings,
            architecture=ModelArchitecture.QWEN2,
            model_type=self.model_type,
        )


def detect_config_format(config_data: Dict[str, Any]) -> str:
    """
    Detect the configuration format based on the structure and fields.

    Args:
        config_data: Raw configuration dictionary

    Returns:
        String identifier for the detected format
    """
    # Check for Meta format (simple structure with 'dim' field)
    if "dim" in config_data and "n_layers" in config_data:
        return "meta_llama"

    # Check for architecture field to determine HF model type
    if "architectures" in config_data:
        architectures = config_data["architectures"]
        if any("Llama" in arch for arch in architectures):
            return "hf_llama"
        elif any("Qwen2" in arch for arch in architectures):
            return "qwen"

    # Check for model_type field
    if "model_type" in config_data:
        model_type = config_data["model_type"]
        if model_type == "llama":
            return "hf_llama"
        elif model_type == "qwen2":
            return "qwen"

    # Default to HF Llama if we can't determine
    return "hf_llama"


def parse_model_config(config_path: Union[str, Path]) -> TTModelConfig:
    """
    Parse a model configuration file and return a standardized configuration.

    Args:
        config_path: Path to the configuration file (config.json or params.json)

    Returns:
        TTModelConfig object with normalized parameters

    Raises:
        ValueError: If the configuration format is not supported
        FileNotFoundError: If the configuration file doesn't exist
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = json.load(f)

    format_type = detect_config_format(config_data)

    if format_type == "meta_llama":
        config = MetaLlamaConfig(**config_data)
    elif format_type == "hf_llama":
        config = HuggingFaceLlamaConfig(**config_data)
    elif format_type == "qwen":
        config = QwenConfig(**config_data)
    else:
        raise ValueError(f"Unsupported configuration format: {format_type}")

    return config.to_standard()


def parse_model_config_from_dict(config_data: Dict[str, Any]) -> TTModelConfig:
    """
    Parse a model configuration from a dictionary and return a standardized configuration.

    Args:
        config_data: Configuration dictionary

    Returns:
        TTModelConfig object with normalized parameters

    Raises:
        ValueError: If the configuration format is not supported
    """
    format_type = detect_config_format(config_data)

    if format_type == "meta_llama":
        config = MetaLlamaConfig(**config_data)
    elif format_type == "hf_llama":
        config = HuggingFaceLlamaConfig(**config_data)
    elif format_type == "qwen":
        config = QwenConfig(**config_data)
    else:
        raise ValueError(f"Unsupported configuration format: {format_type}")

    return config.to_standard()


# Convenience function for backward compatibility
def get_standard_config(checkpoint_dir: Union[str, Path]) -> TTModelConfig:
    """
    Get a standardized configuration from a checkpoint directory.

    This function looks for either config.json (HF format) or params.json (Meta format)
    in the given directory and returns a standardized configuration.

    Args:
        checkpoint_dir: Path to the checkpoint directory

    Returns:
        TTModelConfig object with normalized parameters
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Try HF format first
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        return parse_model_config(config_path)

    # Try Meta format
    params_path = checkpoint_dir / "params.json"
    if params_path.exists():
        return parse_model_config(params_path)

    raise FileNotFoundError(f"No configuration file found in {checkpoint_dir}")

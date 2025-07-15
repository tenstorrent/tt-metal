"""
Pydantic models for standardizing LLM configuration formats.

This module provides a way to parse different model configuration formats
(Meta, HuggingFace, etc.) into a standardized format that can be consumed
by the rest of the codebase without modification.
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import json
from pathlib import Path


class ModelArchitecture(str, Enum):
    """Supported model architectures."""
    LLAMA = "llama"
    QWEN2 = "qwen2"
    DEEPSEEK_V3 = "deepseek_v3"
    MISTRAL = "mistral"


class RopeScalingType(str, Enum):
    """Types of RoPE scaling."""
    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    LLAMA3 = "llama3"


class RopeScaling(BaseModel):
    """RoPE scaling configuration."""
    factor: float
    type: Optional[RopeScalingType] = None
    rope_type: Optional[str] = None
    original_max_position_embeddings: Optional[int] = None
    low_freq_factor: Optional[float] = None
    high_freq_factor: Optional[float] = None
    beta_fast: Optional[int] = None
    beta_slow: Optional[int] = None
    mscale: Optional[float] = None
    mscale_all_dim: Optional[float] = None


class StandardModelConfig(BaseModel):
    """
    Standardized model configuration that all specific formats convert to.
    This is what model_config.py should consume.
    """
    # Core model dimensions
    dim: int = Field(description="Model dimension/hidden size")
    n_layers: int = Field(description="Number of transformer layers")
    n_heads: int = Field(description="Number of attention heads")
    n_kv_heads: int = Field(description="Number of key-value heads")
    head_dim: Optional[int] = Field(None, description="Dimension per attention head")
    vocab_size: int = Field(description="Vocabulary size")
    
    # MLP configuration
    hidden_dim: Optional[int] = Field(None, description="Hidden dimension for MLP")
    intermediate_size: Optional[int] = Field(None, description="Intermediate size for MLP")
    ffn_dim_multiplier: Optional[float] = Field(None, description="FFN dimension multiplier")
    multiple_of: Optional[int] = Field(None, description="FFN dimension must be multiple of this")
    
    # Normalization
    norm_eps: float = Field(description="RMS norm epsilon")
    
    # RoPE configuration
    rope_theta: float = Field(description="RoPE theta parameter")
    rope_scaling: Optional[RopeScaling] = Field(None, description="RoPE scaling configuration")
    rope_scaling_factor: Optional[float] = Field(None, description="Simple RoPE scaling factor")
    use_scaled_rope: Optional[bool] = Field(None, description="Whether to use scaled RoPE")
    max_position_embeddings: Optional[int] = Field(None, description="Maximum sequence length")
    
    # Vision model parameters (for multimodal models)
    vision_chunk_size: Optional[int] = Field(-1, description="Vision chunk size")
    vision_max_num_chunks: Optional[int] = Field(4, description="Maximum number of vision chunks")
    vision_num_cross_attention_layers: Optional[int] = Field(-1, description="Number of cross-attention layers")
    
    # MoE parameters (for mixture of experts models)
    n_routed_experts: Optional[int] = Field(None, description="Number of routed experts")
    n_shared_experts: Optional[int] = Field(None, description="Number of shared experts")
    num_experts_per_tok: Optional[int] = Field(None, description="Number of experts per token")
    moe_intermediate_size: Optional[int] = Field(None, description="MoE intermediate size")
    
    # Model metadata
    model_type: Optional[str] = Field(None, description="Model type identifier")
    architecture: Optional[ModelArchitecture] = Field(None, description="Model architecture")
    
    @validator('head_dim', always=True)
    def compute_head_dim(cls, v, values):
        """Compute head_dim if not provided."""
        if v is None and 'dim' in values and 'n_heads' in values:
            return values['dim'] // values['n_heads']
        return v


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
    use_scaled_rope: Optional[bool] = None
    rope_scaling_factor: Optional[float] = None
    
    def to_standard(self) -> StandardModelConfig:
        """Convert to standard format."""
        return StandardModelConfig(
            dim=self.dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            vocab_size=self.vocab_size,
            ffn_dim_multiplier=self.ffn_dim_multiplier,
            multiple_of=self.multiple_of,
            norm_eps=self.norm_eps,
            rope_theta=self.rope_theta,
            use_scaled_rope=self.use_scaled_rope,
            rope_scaling_factor=self.rope_scaling_factor,
            architecture=ModelArchitecture.LLAMA,
            model_type="llama"
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
    rope_scaling: Optional[Dict[str, Any]] = None
    max_position_embeddings: Optional[int] = None
    model_type: str
    
    # Optional fields
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
    
    def to_standard(self) -> StandardModelConfig:
        """Convert to standard format."""
        rope_scaling_obj = None
        if self.rope_scaling:
            rope_scaling_obj = RopeScaling(**self.rope_scaling)
        
        return StandardModelConfig(
            dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            intermediate_size=self.intermediate_size,
            norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            rope_scaling=rope_scaling_obj,
            max_position_embeddings=self.max_position_embeddings,
            architecture=ModelArchitecture.LLAMA,
            model_type=self.model_type
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
    
    # Qwen-specific fields
    attention_dropout: Optional[float] = None
    sliding_window: Optional[int] = None
    max_window_layers: Optional[int] = None
    use_sliding_window: Optional[bool] = None
    
    # Optional common fields
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    hidden_act: Optional[str] = None
    initializer_range: Optional[float] = None
    tie_word_embeddings: Optional[bool] = None
    torch_dtype: Optional[str] = None
    transformers_version: Optional[str] = None
    use_cache: Optional[bool] = None
    
    def to_standard(self) -> StandardModelConfig:
        """Convert to standard format."""
        return StandardModelConfig(
            dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            intermediate_size=self.intermediate_size,
            norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            max_position_embeddings=self.max_position_embeddings,
            architecture=ModelArchitecture.QWEN2,
            model_type=self.model_type
        )


class DeepSeekV3Config(BaseModel):
    """DeepSeek V3 model configuration with MoE support."""
    architectures: List[str]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]] = None
    max_position_embeddings: int
    model_type: str
    
    # MoE-specific fields
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    moe_layer_freq: Optional[int] = None
    
    # DeepSeek-specific fields
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    first_k_dense_replace: Optional[int] = None
    n_group: Optional[int] = None
    norm_topk_prob: Optional[bool] = None
    routed_scaling_factor: Optional[float] = None
    scoring_func: Optional[str] = None
    topk_group: Optional[int] = None
    topk_method: Optional[str] = None
    
    # Optional common fields
    attention_bias: Optional[bool] = None
    attention_dropout: Optional[float] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    ep_size: Optional[int] = None
    hidden_act: Optional[str] = None
    initializer_range: Optional[float] = None
    tie_word_embeddings: Optional[bool] = None
    torch_dtype: Optional[str] = None
    transformers_version: Optional[str] = None
    use_cache: Optional[bool] = None
    auto_map: Optional[Dict[str, str]] = None
    quantization_config: Optional[Dict[str, Any]] = None
    num_nextn_predict_layers: Optional[int] = None
    
    def to_standard(self) -> StandardModelConfig:
        """Convert to standard format."""
        rope_scaling_obj = None
        if self.rope_scaling:
            rope_scaling_obj = RopeScaling(**self.rope_scaling)
        
        return StandardModelConfig(
            dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            intermediate_size=self.intermediate_size,
            norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            rope_scaling=rope_scaling_obj,
            max_position_embeddings=self.max_position_embeddings,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size,
            architecture=ModelArchitecture.DEEPSEEK_V3,
            model_type=self.model_type
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
        elif any("DeepseekV3" in arch for arch in architectures):
            return "deepseek_v3"
    
    # Check for model_type field
    if "model_type" in config_data:
        model_type = config_data["model_type"]
        if model_type == "llama":
            return "hf_llama"
        elif model_type == "qwen2":
            return "qwen"
        elif model_type == "deepseek_v3":
            return "deepseek_v3"
    
    # Default to HF Llama if we can't determine
    return "hf_llama"


def parse_model_config(config_path: Union[str, Path]) -> StandardModelConfig:
    """
    Parse a model configuration file and return a standardized configuration.
    
    Args:
        config_path: Path to the configuration file (config.json or params.json)
        
    Returns:
        StandardModelConfig object with normalized parameters
        
    Raises:
        ValueError: If the configuration format is not supported
        FileNotFoundError: If the configuration file doesn't exist
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    format_type = detect_config_format(config_data)
    
    if format_type == "meta_llama":
        config = MetaLlamaConfig(**config_data)
    elif format_type == "hf_llama":
        config = HuggingFaceLlamaConfig(**config_data)
    elif format_type == "qwen":
        config = QwenConfig(**config_data)
    elif format_type == "deepseek_v3":
        config = DeepSeekV3Config(**config_data)
    else:
        raise ValueError(f"Unsupported configuration format: {format_type}")
    
    return config.to_standard()


def parse_model_config_from_dict(config_data: Dict[str, Any]) -> StandardModelConfig:
    """
    Parse a model configuration from a dictionary and return a standardized configuration.
    
    Args:
        config_data: Configuration dictionary
        
    Returns:
        StandardModelConfig object with normalized parameters
        
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
    elif format_type == "deepseek_v3":
        config = DeepSeekV3Config(**config_data)
    else:
        raise ValueError(f"Unsupported configuration format: {format_type}")
    
    return config.to_standard()


# Convenience function for backward compatibility
def get_standard_config(checkpoint_dir: Union[str, Path]) -> StandardModelConfig:
    """
    Get a standardized configuration from a checkpoint directory.
    
    This function looks for either config.json (HF format) or params.json (Meta format)
    in the given directory and returns a standardized configuration.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        
    Returns:
        StandardModelConfig object with normalized parameters
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
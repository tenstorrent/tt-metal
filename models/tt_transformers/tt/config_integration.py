"""
Integration helper for using Pydantic model configurations with the existing ModelArgs class.

This module provides helper functions to integrate the new standardized Pydantic models
with the existing model_config.py without breaking changes.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import os

try:
    from .model_configs import (
        parse_model_config_from_dict,
        get_standard_config,
        StandardModelConfig
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("Warning: Pydantic models not available. Falling back to legacy configuration parsing.")


def set_model_params_from_standard_config(model_args, standard_config: "StandardModelConfig"):
    """
    Set ModelArgs parameters from a StandardModelConfig.
    
    This function replaces the logic in _set_params_from_dict with a clean
    mapping from the standardized configuration.
    
    Args:
        model_args: ModelArgs instance to update
        standard_config: StandardModelConfig instance with normalized parameters
    """
    # Core model dimensions
    model_args.dim = standard_config.dim
    model_args.n_heads = standard_config.n_heads
    model_args.n_kv_heads = standard_config.n_kv_heads
    model_args.n_layers = standard_config.n_layers
    model_args.full_model_n_layers = standard_config.n_layers
    model_args.vocab_size = standard_config.vocab_size
    model_args.norm_eps = standard_config.norm_eps
    
    # Head dimension
    if standard_config.head_dim:
        model_args.head_dim = standard_config.head_dim
    else:
        model_args.head_dim = standard_config.dim // standard_config.n_heads
    
    # Max context length
    if standard_config.max_position_embeddings:
        model_args.max_context_len = standard_config.max_position_embeddings
    else:
        # Default for Meta weights
        model_args.max_context_len = 128 * 1024
    
    # MLP configuration
    if standard_config.intermediate_size:
        model_args.hidden_dim = standard_config.intermediate_size
        model_args.ffn_dim_multiplier = None
        model_args.multiple_of = None
    else:
        # Use Meta-style configuration
        model_args.ffn_dim_multiplier = standard_config.ffn_dim_multiplier
        model_args.multiple_of = standard_config.multiple_of
        # Calculate hidden_dim from ffn_dim_multiplier if needed
        if hasattr(model_args, 'calculate_hidden_dim'):
            model_args.hidden_dim = model_args.calculate_hidden_dim(
                model_args.dim, 
                model_args.ffn_dim_multiplier, 
                model_args.multiple_of
            )
    
    # RoPE configuration
    model_args.rope_theta = standard_config.rope_theta
    
    # Handle RoPE scaling
    if standard_config.rope_scaling:
        model_args.rope_scaling_factor = standard_config.rope_scaling.factor
        model_args.orig_context_len = standard_config.rope_scaling.original_max_position_embeddings
    elif standard_config.rope_scaling_factor:
        model_args.rope_scaling_factor = standard_config.rope_scaling_factor
        if standard_config.use_scaled_rope:
            # Set default original context length for Meta weights
            model_args.orig_context_len = 8192
        else:
            model_args.orig_context_len = None
    else:
        model_args.rope_scaling_factor = None
        model_args.orig_context_len = None
    
    # Vision parameters (for multimodal models)
    if hasattr(standard_config, 'vision_chunk_size'):
        model_args.vision_chunk_size = standard_config.vision_chunk_size or -1
        model_args.vision_max_num_chunks = standard_config.vision_max_num_chunks or 4
        model_args.vision_num_cross_attention_layers = standard_config.vision_num_cross_attention_layers or -1
    
    # Set model name and type
    if standard_config.model_type:
        # Update model name based on detected type and architecture
        if not hasattr(model_args, 'model_name') or model_args.model_name == "Unknown":
            model_args.model_name = standard_config.model_type
    
    # Set state dict prefix
    model_args.state_dict_text_prefix = model_args._get_text_prefix()
    
    # Set model-specific parameters
    if hasattr(model_args, '_set_model_specific_params'):
        model_args._set_model_specific_params()


def enhanced_set_params_from_dict(model_args, config: Dict[str, Any], is_hf: bool = False):
    """
    Enhanced version of _set_params_from_dict that uses Pydantic models when available.
    
    This function can be used as a drop-in replacement for the existing _set_params_from_dict
    method in ModelArgs.
    
    Args:
        model_args: ModelArgs instance to update
        config: Configuration dictionary
        is_hf: Whether this is a HuggingFace format configuration
    """
    if PYDANTIC_AVAILABLE:
        try:
            # Use the new Pydantic-based parsing
            standard_config = parse_model_config_from_dict(config)
            set_model_params_from_standard_config(model_args, standard_config)
            return
        except Exception as e:
            print(f"Warning: Failed to parse config with Pydantic models: {e}")
            print("Falling back to legacy parsing...")
    
    # Fallback to the original logic
    _legacy_set_params_from_dict(model_args, config, is_hf)


def _legacy_set_params_from_dict(model_args, config: Dict[str, Any], is_hf: bool = False):
    """
    Legacy implementation of _set_params_from_dict for fallback purposes.
    
    This is the original logic from ModelArgs._set_params_from_dict.
    """
    # Try to get text_config, if it doesn't exist everything is text config
    text_config = config.get("text_config", config)

    # Common params with different names between Meta and HF
    model_args.dim = text_config.get("dim", text_config.get("hidden_size"))
    model_args.n_heads = text_config.get("n_heads", text_config.get("num_attention_heads"))
    model_args.n_kv_heads = text_config.get("n_kv_heads", text_config.get("num_key_value_heads"))
    model_args.n_layers = text_config.get("n_layers", text_config.get("num_hidden_layers"))
    model_args.full_model_n_layers = model_args.n_layers
    model_args.norm_eps = text_config.get("norm_eps", text_config.get("rms_norm_eps"))
    model_args.vocab_size = text_config["vocab_size"]
    model_args.padded_vocab_size = 128 * 1024 if hasattr(model_args, 'is_galaxy') and model_args.is_galaxy else None
    model_args.head_dim = text_config.get("head_dim", model_args.dim // model_args.n_heads)
    
    if is_hf:
        model_args.max_context_len = text_config.get("max_position_embeddings")
    else:
        model_args.max_context_len = 128 * 1024

    # Handle different MLP dimension specifications
    if "intermediate_size" in text_config:
        model_args.hidden_dim = text_config["intermediate_size"]
        model_args.ffn_dim_multiplier = None
        model_args.multiple_of = None
    else:
        model_args.ffn_dim_multiplier = text_config["ffn_dim_multiplier"]
        model_args.multiple_of = text_config["multiple_of"]
        # Calculate hidden_dim using the utility function
        if hasattr(model_args, 'calculate_hidden_dim'):
            model_args.hidden_dim = model_args.calculate_hidden_dim(
                model_args.dim, 
                model_args.ffn_dim_multiplier, 
                model_args.multiple_of
            )

    # Model name handling
    if "_name_or_path" in config:
        if is_hf:
            normalized_path = os.path.normpath(config["_name_or_path"])
            if "snapshots" in normalized_path:
                full_model_name = normalized_path.split(os.path.sep)[-3]
                model_args.model_name = full_model_name.split("--")[-1]
            else:
                model_args.model_name = os.path.basename(normalized_path)
        else:
            model_args.model_name = os.path.basename(config["_name_or_path"])

    # RoPE params
    model_args.rope_theta = text_config.get("rope_theta")
    rope_scaling_params = text_config.get("rope_scaling", None)
    if rope_scaling_params:
        model_args.rope_scaling_factor = rope_scaling_params.get("factor", None)
        model_args.orig_context_len = rope_scaling_params.get("original_max_position_embeddings", model_args.max_context_len)
    else:
        model_args.rope_scaling_factor = None
        model_args.orig_context_len = None


def enhanced_set_model_params(model_args, checkpoint_dir: str):
    """
    Enhanced version of _set_model_params that uses Pydantic models when available.
    
    This function can be used as a drop-in replacement for the existing _set_model_params
    method in ModelArgs.
    
    Args:
        model_args: ModelArgs instance to update
        checkpoint_dir: Path to the checkpoint directory
    """
    if PYDANTIC_AVAILABLE:
        try:
            # Use the new Pydantic-based parsing
            standard_config = get_standard_config(checkpoint_dir)
            set_model_params_from_standard_config(model_args, standard_config)
            return
        except Exception as e:
            print(f"Warning: Failed to parse config with Pydantic models: {e}")
            print("Falling back to legacy parsing...")
    
    # Fallback to the original logic
    _legacy_set_model_params(model_args, checkpoint_dir)


def _legacy_set_model_params(model_args, checkpoint_dir: str):
    """
    Legacy implementation for fallback purposes.
    
    This would be the original _set_model_params logic.
    """
    # Check for HF format
    config_file = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        enhanced_set_params_from_dict(model_args, config, is_hf=True)
        return
    
    # Check for Meta format
    params_file = os.path.join(checkpoint_dir, "params.json")
    if os.path.exists(params_file):
        with open(params_file, "r") as f:
            params = json.load(f)
        enhanced_set_params_from_dict(model_args, params, is_hf=False)
        return
    
    raise FileNotFoundError(f"No configuration file found in {checkpoint_dir}")


# Example usage and migration guide
def example_integration():
    """
    Example showing how to integrate the new Pydantic models with existing code.
    
    To migrate existing ModelArgs usage:
    
    1. Replace _set_params_from_dict calls:
       OLD: self._set_params_from_dict(config, is_hf=True)
       NEW: enhanced_set_params_from_dict(self, config, is_hf=True)
    
    2. Replace _set_model_params calls:
       OLD: self._set_model_params(checkpoint_dir)
       NEW: enhanced_set_model_params(self, checkpoint_dir)
    
    3. For new code, you can use the standardized config directly:
       standard_config = get_standard_config(checkpoint_dir)
       set_model_params_from_standard_config(model_args, standard_config)
    """
    pass 
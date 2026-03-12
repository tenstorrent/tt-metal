# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Centralized model configuration system for ComfyUI Bridge.

Provides a model-agnostic configuration system that allows support for
multiple diffusion models (SDXL, SD3.5, SD1.4, SD1.5, etc.) without hardcoding
channel counts or other model-specific parameters throughout the codebase.

The configuration approach enables:
- Easy addition of new models by adding a dict entry
- Validation of model parameters
- Runtime lookup of model-specific values
- Future extensibility for custom models

Key Functions:
    get_model_config: Retrieve config for a specific model
    validate_config: Validate that a config is complete and correct
    get_latent_channels: Get the number of latent channels for a model
    get_clip_dim: Get the CLIP embedding dimension for a model

Usage Example:
    >>> config = get_model_config("sdxl")
    >>> print(config['latent_channels'])  # 4
    >>> print(config['clip_dim'])  # 2048
    >>> validate_config(config)  # Raises if invalid
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Model configurations - the single source of truth for all model parameters
MODEL_CONFIGS = {
    "sdxl": {
        "name": "Stable Diffusion XL",
        "latent_channels": 4,
        "vae_scale_factor": 8,
        "default_height": 1024,
        "default_width": 1024,
        "cross_attention_dim": 2048,
        "clip_dim": 2048,
        "text_encoder_dim": 768,
        "unet_channels": [320, 640, 1280],
        "num_text_encoders": 2,
        "text_encoders": ["clip_l", "clip_g"],
        "scheduler_config": {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
        },
        "supports_controlnet": True,
        "supports_ip_adapter": True,
        "default_sampler": "euler",
    },
    "sd3.5": {
        "name": "Stable Diffusion 3.5",
        "latent_channels": 16,
        "vae_scale_factor": 8,
        "default_height": 1024,
        "default_width": 1024,
        "cross_attention_dim": 4096,
        "clip_dim": 4096,
        "text_encoder_dim": 768,
        "unet_channels": [384, 768, 1536],
        "num_text_encoders": 3,
        "text_encoders": ["clip_l", "clip_g", "t5"],
        "scheduler_config": {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
        },
        "supports_controlnet": True,
        "supports_ip_adapter": True,
        "default_sampler": "euler",
    },
    "sd1.5": {
        "name": "Stable Diffusion 1.5",
        "latent_channels": 4,
        "vae_scale_factor": 8,
        "default_height": 512,
        "default_width": 512,
        "cross_attention_dim": 768,
        "clip_dim": 768,
        "text_encoder_dim": 768,
        "unet_channels": [320, 640, 1280],
        "num_text_encoders": 1,
        "text_encoders": ["clip"],
        "scheduler_config": {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
        },
        "supports_controlnet": True,
        "supports_ip_adapter": False,
        "default_sampler": "euler",
    },
    "sd1.4": {
        "name": "Stable Diffusion 1.4",
        "latent_channels": 4,
        "vae_scale_factor": 8,
        "default_height": 512,
        "default_width": 512,
        "cross_attention_dim": 768,
        "clip_dim": 768,
        "text_encoder_dim": 768,
        "unet_channels": [320, 640, 1280],
        "num_text_encoders": 1,
        "text_encoders": ["clip"],
        "scheduler_config": {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
        },
        "supports_controlnet": True,
        "supports_ip_adapter": False,
        "default_sampler": "euler",
    },
}

# List of required keys that must be present in every model config
REQUIRED_CONFIG_KEYS = [
    "name",
    "latent_channels",
    "vae_scale_factor",
    "default_height",
    "default_width",
    "cross_attention_dim",
    "clip_dim",
    "text_encoder_dim",
    "unet_channels",
    "num_text_encoders",
    "text_encoders",
    "scheduler_config",
    "supports_controlnet",
    "default_sampler",
]


def get_model_config(model_id: str) -> Dict[str, Any]:
    """
    Retrieve configuration for a specific model.

    Args:
        model_id: Model identifier (e.g., "sdxl", "sd1.5", "sd3.5")

    Returns:
        Configuration dictionary for the model

    Raises:
        ValueError: If model_id not found in configurations
    """
    if model_id not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Model '{model_id}' not configured. " f"Available models: {available}")

    config = MODEL_CONFIGS[model_id].copy()
    logger.debug(f"Loaded configuration for model: {model_id}")
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that a model configuration is complete and correct.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid or incomplete
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")

    # Check all required keys are present
    missing_keys = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing_keys:
        raise ValueError(f"Configuration missing required keys: {missing_keys}")

    # Validate specific fields
    if config["latent_channels"] <= 0:
        raise ValueError("latent_channels must be positive")

    if config["vae_scale_factor"] <= 0:
        raise ValueError("vae_scale_factor must be positive")

    if config["default_height"] <= 0 or config["default_width"] <= 0:
        raise ValueError("default_height and default_width must be positive")

    if config["cross_attention_dim"] <= 0:
        raise ValueError("cross_attention_dim must be positive")

    if config["clip_dim"] <= 0:
        raise ValueError("clip_dim must be positive")

    if config["num_text_encoders"] <= 0:
        raise ValueError("num_text_encoders must be positive")

    if len(config["text_encoders"]) != config["num_text_encoders"]:
        raise ValueError(
            f"text_encoders list length ({len(config['text_encoders'])}) "
            f"must match num_text_encoders ({config['num_text_encoders']})"
        )

    if not isinstance(config["scheduler_config"], dict):
        raise ValueError("scheduler_config must be a dictionary")

    if not isinstance(config["supports_controlnet"], bool):
        raise ValueError("supports_controlnet must be boolean")

    logger.debug(f"Configuration validated: {config['name']}")
    return True


def get_latent_channels(model_id: str) -> int:
    """
    Get the number of latent channels for a model.

    This is a commonly-needed value that's used for tensor shape validation,
    format conversion, and model initialization.

    Args:
        model_id: Model identifier

    Returns:
        Number of latent channels

    Raises:
        ValueError: If model_id not found
    """
    config = get_model_config(model_id)
    return config["latent_channels"]


def get_clip_dim(model_id: str) -> int:
    """
    Get the CLIP embedding dimension for a model.

    Used for text encoding and prompt conditioning.

    Args:
        model_id: Model identifier

    Returns:
        CLIP embedding dimension

    Raises:
        ValueError: If model_id not found
    """
    config = get_model_config(model_id)
    return config["clip_dim"]


def get_cross_attention_dim(model_id: str) -> int:
    """
    Get the cross-attention dimension for a model's UNet.

    Used for conditioning the denoising process.

    Args:
        model_id: Model identifier

    Returns:
        Cross-attention dimension

    Raises:
        ValueError: If model_id not found
    """
    config = get_model_config(model_id)
    return config["cross_attention_dim"]


def get_default_dimensions(model_id: str) -> tuple:
    """
    Get the default height and width for a model.

    Args:
        model_id: Model identifier

    Returns:
        Tuple of (height, width)

    Raises:
        ValueError: If model_id not found
    """
    config = get_model_config(model_id)
    return config["default_height"], config["default_width"]


def get_vae_scale_factor(model_id: str) -> int:
    """
    Get the VAE scale factor for a model.

    The scale factor determines the relationship between image space
    and latent space dimensions. For example, an 8x scale factor means
    a 512x512 image becomes an 64x64 latent.

    Args:
        model_id: Model identifier

    Returns:
        VAE scale factor

    Raises:
        ValueError: If model_id not found
    """
    config = get_model_config(model_id)
    return config["vae_scale_factor"]


def supports_controlnet(model_id: str) -> bool:
    """
    Check if a model supports ControlNet conditioning.

    Args:
        model_id: Model identifier

    Returns:
        True if ControlNet is supported

    Raises:
        ValueError: If model_id not found
    """
    config = get_model_config(model_id)
    return config.get("supports_controlnet", False)


def supports_ip_adapter(model_id: str) -> bool:
    """
    Check if a model supports IP-Adapter.

    Args:
        model_id: Model identifier

    Returns:
        True if IP-Adapter is supported

    Raises:
        ValueError: If model_id not found
    """
    config = get_model_config(model_id)
    return config.get("supports_ip_adapter", False)


def list_available_models() -> List[str]:
    """
    Get list of all available model IDs.

    Returns:
        List of model identifiers
    """
    return list(MODEL_CONFIGS.keys())


def get_text_encoder_config(model_id: str) -> Dict[str, Any]:
    """
    Get text encoder configuration for a model.

    Returns information about the text encoder(s) used by the model,
    including the number of encoders and their identifiers.

    Args:
        model_id: Model identifier

    Returns:
        Dictionary with text encoder information

    Raises:
        ValueError: If model_id not found
    """
    config = get_model_config(model_id)
    return {
        "num_text_encoders": config["num_text_encoders"],
        "text_encoders": config["text_encoders"],
        "clip_dim": config["clip_dim"],
        "text_encoder_dim": config["text_encoder_dim"],
    }


def get_scheduler_config(model_id: str) -> Dict[str, Any]:
    """
    Get scheduler configuration for a model.

    Args:
        model_id: Model identifier

    Returns:
        Scheduler configuration dictionary

    Raises:
        ValueError: If model_id not found
    """
    config = get_model_config(model_id)
    return config["scheduler_config"].copy()

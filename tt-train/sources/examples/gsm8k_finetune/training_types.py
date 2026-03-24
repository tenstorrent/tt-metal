#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Training Type Registry System

This module provides a clean registry system for different training types (SFT, LoRA, etc.).
Each training type defines its own parameter validation, configuration building, and script paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Set
import yaml
from ttml.common.utils import get_tt_metal_runtime_root


@dataclass
class TrainingTypeConfig:
    """Configuration for a specific training type."""

    name: str
    script_path: str  # Path to the training script relative to examples/
    model_configs: Dict[str, str]  # model_id -> config_path mapping
    supported_models: Set[str]  # Set of supported model IDs
    param_validator: Callable[[dict], dict]  # Function to validate/normalize parameters
    config_builder: Callable[[dict, Path], Path]  # Function to build training config YAML


# ── Parameter Validators ──────────────────────────────────────────────────────


def validate_sft_params(params: dict) -> dict:
    """Validate and normalize SFT training parameters.

    Args:
        params: Raw training parameters from API request

    Returns:
        Validated and normalized parameters

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    validated = params.copy()

    # Set defaults for common SFT parameters
    validated.setdefault("batch_size", 32)
    validated.setdefault("validation_batch_size", 4)
    validated.setdefault("max_steps", 60)
    validated.setdefault("gradient_accumulation", 1)
    validated.setdefault("eval_every", 20)

    # Basic validation
    if validated["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if validated["max_steps"] <= 0:
        raise ValueError("max_steps must be positive")

    return validated


def validate_lora_params(params: dict) -> dict:
    """Validate and normalize LoRA training parameters.

    Args:
        params: Raw training parameters from API request

    Returns:
        Validated and normalized parameters

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    validated = params.copy()

    # Set defaults for LoRA parameters
    validated.setdefault("batch_size", 32)
    validated.setdefault("validation_batch_size", 4)
    validated.setdefault("max_steps", 60)
    validated.setdefault("gradient_accumulation", 1)
    validated.setdefault("eval_every", 20)

    # LoRA-specific defaults
    validated.setdefault("lora_rank", 8)
    validated.setdefault("lora_alpha", 16)
    # Map target_modules (OpenAPI spec name) → lora_target_modules (internal name)
    validated["lora_target_modules"] = validated.pop(
        "target_modules", validated.pop("lora_target_modules", ["q_linear", "kv_linear", "out_linear"])
    )
    validated.setdefault("lora_dropout", 0.05)

    # Basic validation
    if validated["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if validated["max_steps"] <= 0:
        raise ValueError("max_steps must be positive")
    if validated["lora_rank"] <= 0:
        raise ValueError("lora_rank must be positive")
    if validated["lora_alpha"] <= 0:
        raise ValueError("lora_alpha must be positive")

    return validated


# ── Configuration Builders ────────────────────────────────────────────────────


def build_sft_config(config: dict, output_dir: Path) -> Path:
    """Build training_overrides.yaml for SFT training.

    Args:
        config: Validated training configuration
        output_dir: Directory to write the config file

    Returns:
        Path to the created config file
    """
    training_config = {
        "batch_size": config.get("batch_size", 32),
        "validation_batch_size": config.get("validation_batch_size", 4),
        "max_steps": config.get("max_steps", 60),
        "gradient_accumulation_steps": config.get("gradient_accumulation", 1),
        "eval_every": config.get("eval_every", 20),
    }

    # Add model_config if specified
    if config.get("model_config"):
        training_config["model_config"] = config["model_config"]
    # Add dataset (HF name or URL) for training script
    if config.get("dataset"):
        training_config["dataset"] = config["dataset"]

    scheduler_config = {
        "warmup_steps": config.get("warmup_steps", 20),
        "hold_steps": config.get("hold_steps", 40),
        "min_lr": config.get("min_lr", 3e-5),
        "max_lr": config.get("max_lr", 1e-4),
    }

    device_config = {
        "enable_ddp": config.get("enable_ddp", False),
        "mesh_shape": config.get("mesh_shape", [1, 1]),
    }

    transformer_config = {
        "max_sequence_length": config.get("max_seq_length", 512),
    }

    config_path = output_dir / "training_overrides.yaml"

    with open(config_path, "w") as f:
        f.write("training_config:\n")
        for key, value in training_config.items():
            f.write(f"  {key}: {value}\n")

        f.write("\ntransformer_config:\n")
        for key, value in transformer_config.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nscheduler_config:\n")
        for key, value in scheduler_config.items():
            if isinstance(value, float) and (value < 0.001 or value > 1000):
                f.write(f"  {key}: {value:.2e}\n")
            else:
                f.write(f"  {key}: {value}\n")

        f.write("\ndevice_config:\n")
        f.write(f"  enable_ddp: {str(device_config['enable_ddp']).lower()}\n")
        mesh_str = yaml.dump(device_config["mesh_shape"], default_flow_style=True).strip()
        f.write(f"  mesh_shape: {mesh_str}\n")

    return config_path


def build_lora_config(config: dict, output_dir: Path) -> Path:
    """Build training_overrides.yaml for LoRA training.

    Args:
        config: Validated training configuration
        output_dir: Directory to write the config file

    Returns:
        Path to the created config file
    """
    training_config = {
        "batch_size": config.get("batch_size", 32),
        "validation_batch_size": config.get("validation_batch_size", 4),
        "max_steps": config.get("max_steps", 60),
        "gradient_accumulation_steps": config.get("gradient_accumulation", 1),
        "eval_every": config.get("eval_every", 20),
    }

    # Add model_config if specified
    if config.get("model_config"):
        training_config["model_config"] = config["model_config"]
    # Add dataset (HF name or URL) for training script
    if config.get("dataset"):
        training_config["dataset"] = config["dataset"]

    scheduler_config = {
        "warmup_steps": config.get("warmup_steps", 20),
        "hold_steps": config.get("hold_steps", 40),
        "min_lr": config.get("min_lr", 3e-5),
        "max_lr": config.get("max_lr", 1e-4),
    }

    device_config = {
        "enable_ddp": config.get("enable_ddp", False),
        "mesh_shape": config.get("mesh_shape", [1, 1]),
    }

    transformer_config = {
        "max_sequence_length": config.get("max_seq_length", 512),
    }

    # LoRA-specific configuration
    lora_config = {
        "rank": config.get("lora_rank", 8),
        "alpha": config.get("lora_alpha", 16),
        "target_modules": config.get("lora_target_modules", ["q_linear", "kv_linear", "out_linear"]),
        "lora_dropout": config.get("lora_dropout", 0.05),
    }

    config_path = output_dir / "training_overrides.yaml"

    with open(config_path, "w") as f:
        f.write("training_config:\n")
        for key, value in training_config.items():
            f.write(f"  {key}: {value}\n")

        f.write("\ntransformer_config:\n")
        for key, value in transformer_config.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nscheduler_config:\n")
        for key, value in scheduler_config.items():
            if isinstance(value, float) and (value < 0.001 or value > 1000):
                f.write(f"  {key}: {value:.2e}\n")
            else:
                f.write(f"  {key}: {value}\n")

        f.write("\ndevice_config:\n")
        f.write(f"  enable_ddp: {str(device_config['enable_ddp']).lower()}\n")
        mesh_str = yaml.dump(device_config["mesh_shape"], default_flow_style=True).strip()
        f.write(f"  mesh_shape: {mesh_str}\n")

        f.write("\nlora_config:\n")
        for key, value in lora_config.items():
            if key == "target_modules":
                # Format target_modules as a YAML list
                modules_str = yaml.dump(value, default_flow_style=True).strip()
                f.write(f"  {key}: {modules_str}\n")
            else:
                f.write(f"  {key}: {value}\n")

    return config_path


# ── Model Configuration Mappings ──────────────────────────────────────────────


def _get_model_to_config_mapping():
    """Get model config mapping with paths relative to tt_metal_runtime_root."""
    tt_train_root = f"{get_tt_metal_runtime_root()}/tt-train"
    return {
        "tinyllama": f"{tt_train_root}/configs/model_configs/tinyllama.yaml",
        "gpt2": f"{tt_train_root}/configs/model_configs/gpt2s.yaml",
        "llama8b": f"{tt_train_root}/configs/model_configs/llama8b.yaml",
    }


# ── Training Type Configurations ──────────────────────────────────────────────


# SFT Training Type Configuration
sft_training_config = TrainingTypeConfig(
    name="sft",
    script_path="gsm8k_finetune/gsm8k_finetune.py",
    model_configs=_get_model_to_config_mapping(),
    supported_models={"tinyllama", "gpt2", "llama8b"},
    param_validator=validate_sft_params,
    config_builder=build_sft_config,
)

# LoRA Training Type Configuration
lora_training_config = TrainingTypeConfig(
    name="lora",
    script_path="lora_llama/train_lora_llama_sft.py",
    model_configs=_get_model_to_config_mapping(),
    supported_models={"tinyllama", "gpt2", "llama8b"},
    param_validator=validate_lora_params,
    config_builder=build_lora_config,
)


# ── Training Types Registry ───────────────────────────────────────────────────


TRAINING_TYPES: Dict[str, TrainingTypeConfig] = {
    "sft": sft_training_config,
    "lora": lora_training_config,
}


# ── Utility Functions ─────────────────────────────────────────────────────────


def get_training_type(trainer_name: str) -> TrainingTypeConfig:
    """Get training type configuration by name.

    Args:
        trainer_name: Name of the trainer (e.g., "sft", "lora")

    Returns:
        TrainingTypeConfig for the specified trainer

    Raises:
        KeyError: If trainer_name is not found in registry
    """
    if trainer_name not in TRAINING_TYPES:
        supported = sorted(TRAINING_TYPES.keys())
        raise KeyError(f"Unsupported trainer: '{trainer_name}'. Supported: {supported}")

    return TRAINING_TYPES[trainer_name]


def get_supported_trainers() -> Set[str]:
    """Get set of all supported trainer names."""
    return set(TRAINING_TYPES.keys())


def get_supported_models(trainer_name: str = None) -> Set[str]:
    """Get set of supported models, optionally filtered by trainer type.

    Args:
        trainer_name: Optional trainer name to filter by

    Returns:
        Set of supported model IDs
    """
    if trainer_name:
        return get_training_type(trainer_name).supported_models

    # Return union of all supported models across all trainers
    all_models = set()
    for config in TRAINING_TYPES.values():
        all_models.update(config.supported_models)
    return all_models

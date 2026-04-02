#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Training Type Registry System

This module provides a clean registry system for different training types (SFT, LoRA, etc.).
Each training type defines its own parameter validation, configuration building, and script paths.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Set
import yaml
from ttml.common.utils import get_tt_metal_runtime_root

# Model-type -> default LoRA target modules.
# Used as fallback when target_modules is not explicitly provided.
DEFAULT_LORA_TARGETS = {
    "gpt2": ["q_linear", "kv_linear", "out_linear"],
    "llama": ["q_linear", "kv_linear", "out_linear"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def resolve_lora_targets(model_type: str) -> list[str]:
    """Return default LoRA target modules for a model type, or raise if unknown."""
    targets = DEFAULT_LORA_TARGETS.get(model_type)
    if not targets:
        raise ValueError(
            f"No default LoRA target_modules for model_type '{model_type}'. "
            f"Specify target_modules in lora_config or add defaults for this model type."
        )
    return targets


_MODEL_ID_TO_TYPE = {
    "tinyllama": "llama",
    "gpt2": "gpt2",
    "llama8b": "llama",
    "llama70b": "llama",
    "llama405b": "llama",
    "qwen3_0_6b": "qwen3",
    "qwen3_1_7b": "qwen3",
}

_MODEL_ID_TO_RELATIVE_CONFIG_PATH = {
    "tinyllama": "model_configs/tinyllama.yaml",
    "gpt2": "model_configs/gpt2s.yaml",
    "llama8b": "model_configs/llama8b.yaml",
    "llama70b": "model_configs/llama70b_tp32.yaml",
    "llama405b": "model_configs/llama405b.yaml",
    "qwen3_0_6b": "model_configs/qwen3_0_6b.yaml",
    "qwen3_1_7b": "model_configs/qwen3_1_7b.yaml",
}


def get_model_type(model_id: str) -> str:
    """Return model type for a model ID."""
    return _MODEL_ID_TO_TYPE[model_id]


def get_model_config_path(model_id: str) -> str:
    """Return config YAML path for a model ID."""
    return os.path.join(get_tt_metal_runtime_root(), "tt-train", "configs", _MODEL_ID_TO_RELATIVE_CONFIG_PATH[model_id])


@dataclass
class TrainingTypeConfig:
    """Configuration for a specific training type."""

    name: str
    script_path: str  # Path to the training script relative to examples/
    supported_model_ids: Set[str]  # Set of supported model IDs
    param_validator: Callable[[dict], dict]  # Function to validate/normalize parameters
    config_builder: Callable[[dict, Path], Path]  # Function to build training config YAML
    # If set, replaces the default "python {script_path}" in the SLURM run step.
    # May reference shell variables available in the SLURM environment (e.g. $TT_METAL_HOME).
    run_command_override: Optional[str] = None

    def __post_init__(self):
        missing_config = self.supported_model_ids - _MODEL_ID_TO_RELATIVE_CONFIG_PATH.keys()
        assert (
            not missing_config
        ), f"{self.name}: model IDs missing from _MODEL_ID_TO_RELATIVE_CONFIG_PATH: {missing_config}"
        missing_type = self.supported_model_ids - _MODEL_ID_TO_TYPE.keys()
        assert not missing_type, f"{self.name}: model IDs missing from _MODEL_ID_TO_TYPE: {missing_type}"


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
    # Normalize key: target_modules (OpenAPI) -> lora_target_modules (internal).
    # Defaults are resolved in build_lora_config where model_type is available.
    if "target_modules" in validated:
        validated["lora_target_modules"] = validated.pop("target_modules")
    validated.setdefault("lora_dropout", 0.05)
    validated.setdefault("use_rslora", False)
    validated.setdefault("is_bias_trainable", False)

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
    if "model_type" not in config:
        raise ValueError("build_lora_config requires 'model_type' in config to resolve default target_modules")
    lora_config = {
        "rank": config.get("lora_rank", 8),
        "alpha": config.get("lora_alpha", 16),
        "target_modules": config.get("lora_target_modules") or resolve_lora_targets(config["model_type"]),
        "lora_dropout": config.get("lora_dropout", 0.05),
        "use_rslora": config.get("use_rslora", False),
        "is_bias_trainable": config.get("is_bias_trainable", False),
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
                modules_str = yaml.dump(value, default_flow_style=True).strip()
                f.write(f"  {key}: {modules_str}\n")
            elif isinstance(value, bool):
                f.write(f"  {key}: {str(value).lower()}\n")
            else:
                f.write(f"  {key}: {value}\n")

    return config_path


# ── Pretrain Parameter Validator / Config Builder ────────────────────────────


def validate_pretrain_params(params: dict) -> dict:
    """Validate Pretrain parameters.

    Pretrain routes to pipeline_parallel_training internally; the training
    script uses its own hardcoded config. batch_size is normalised to 32 so it
    passes the Galaxy divisibility check in slurm_training_service.
    """
    validated = params.copy()
    # Force batch_size=32 — actual training ignores this value
    validated["batch_size"] = 32
    validated.setdefault("max_steps", 60)
    return validated


def build_pretrain_config(config: dict, output_dir: Path) -> Path:
    """Write a placeholder training_overrides.yaml for Pretrain jobs.

    The pipeline_parallel training script uses its own config files, so this
    file is created to satisfy the job_manager interface but is never read.
    """
    config_path = output_dir / "training_overrides.yaml"
    with open(config_path, "w") as f:
        f.write("# Pretrain training uses pipeline_parallel_training configs directly.\n")
        f.write("# This file is not read by the training script.\n")
    return config_path


# ── Training Type Configurations ──────────────────────────────────────────────


# SFT Training Type Configuration
sft_training_config = TrainingTypeConfig(
    name="sft",
    script_path="gsm8k_finetune/gsm8k_finetune.py",
    supported_model_ids={"tinyllama", "gpt2", "qwen3_0_6b", "qwen3_1_7b"},
    param_validator=validate_sft_params,
    config_builder=build_sft_config,
)

# LoRA Training Type Configuration
lora_training_config = TrainingTypeConfig(
    name="lora",
    script_path="gsm8k_finetune/gsm8k_finetune.py",
    supported_model_ids={"tinyllama", "gpt2", "qwen3_0_6b", "qwen3_1_7b"},
    param_validator=validate_lora_params,
    config_builder=build_lora_config,
)


def _build_pretrain_run_command() -> str:
    """Build the SLURM run-step command for Pretrain / pipeline-parallel training."""
    tt_train_root = f"${{TT_METAL_HOME}}/tt-train"
    pp_root = f"{tt_train_root}/sources/examples/python/multihost/pipeline_parallel_training"
    config_file = "training_configs/training_shakespeare_llama70b_pp4_tp32_fabric_galaxy.yaml"
    host_config = "4galaxy_pp4"
    ranks_per_host = "1"
    return (
        f"export TT_TRAIN_OUTPUT_FILE=$(pwd)/output.txt\n"
        f"cd {pp_root}\n"
        f"scontrol show hostnames | python make_rankfile.py -n {ranks_per_host} -o /tmp/rankfile.txt\n"
        f"tt-run --verbose \\\n"
        f'    --mpi-args "--oversubscribe --map-by rankfile:file=/tmp/rankfile.txt" \\\n'
        f"    --rank-binding {pp_root}/configurations/{host_config}/rank_bindings.yaml \\\n"
        f"    python {pp_root}/training.py -c {config_file}"
    )


# Pretrain Training Type Configuration
# Routes all requests to pipeline_parallel_training with llama70b_4stage config.
pretrain_training_config = TrainingTypeConfig(
    name="pretrain",
    script_path="python/multihost/pipeline_parallel_training/training.py",
    supported_model_ids={"llama8b", "llama70b", "llama405b"},
    param_validator=validate_pretrain_params,
    config_builder=build_pretrain_config,
    run_command_override=_build_pretrain_run_command(),
)


# ── Training Types Registry ───────────────────────────────────────────────────


TRAINING_TYPES: Dict[str, TrainingTypeConfig] = {
    "sft": sft_training_config,
    "lora": lora_training_config,
    "pretrain": pretrain_training_config,
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


def get_supported_models(trainer_name: Optional[str] = None) -> Set[str]:
    """Get set of supported models, optionally filtered by trainer type.

    Args:
        trainer_name: Optional trainer name to filter by

    Returns:
        Set of supported model IDs
    """
    if trainer_name:
        return get_training_type(trainer_name).supported_model_ids

    # Return union of all supported models across all trainers
    all_models = set()
    for config in TRAINING_TYPES.values():
        all_models.update(config.supported_model_ids)
    return all_models

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import SavedWeight

# Import TENSOR_CACHE_EXTENSION from config_helpers since it's also used by shard_and_save
from models.demos.deepseek_v3.utils.config_helpers import TENSOR_CACHE_EXTENSION
from models.demos.deepseek_v3.utils.run_config import WeightConfig


# JSON serializer for the weight config
class WeightConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SavedWeight):
            obj = {
                "path": str(obj.path),
                "memory_config": None if obj.memory_config is None else json.loads(obj.memory_config.to_json()),
            }
        return obj


def try_decode_saved_weight(obj: dict[str, Any]) -> Any:
    path_str = obj.get("path", None)
    if not isinstance(path_str, str):
        return obj
    memory_config_dict = obj.get("memory_config", None)
    if not isinstance(memory_config_dict, dict) or not {
        "buffer_type",
        "memory_layout",
        "created_with_nd_shard_spec",
    }.issubset(memory_config_dict.keys()):
        return obj
    return SavedWeight(path=Path(path_str), memory_config=ttnn.MemoryConfig.from_json(json.dumps(memory_config_dict)))


def get_weight_config(
    ModuleClass: type["models.demos.deepseek_v3.utils.abstract_module.AbstractModule"],
    hf_config: PretrainedConfig,
    state_dicts: tuple[dict[str, torch.Tensor] | None, ...] | None = None,
    weight_cache_path: Path | None = None,
    mesh_device: ttnn.Device | None = None,
    force_recalculate: bool = False,
    random_weights: bool = False,
    model_path: str | None = None,
    single_layer: str | None = None,
):
    """
    Get weight configuration, either from cache or by converting weights.

    Args:
        ModuleClass: The module class to convert weights for
        hf_config: HuggingFace model configuration
        state_dicts: Optional pre-loaded state dicts. If None, will be loaded based on random_weights/model_path.
        weight_cache_path: Path to cache weights
        mesh_device: TTNN mesh device
        force_recalculate: Force recalculation even if cached weights exist
        random_weights: If True, generate random weights from reference model
        model_path: Path to HuggingFace model directory (required if random_weights=False and state_dicts=None)
        single_layer: Optional single layer name (used for validation with random weights)

    Returns:
        Weight configuration dictionary
    """
    if weight_cache_path is None:
        raise ValueError("weight_cache_path must be provided")
    if mesh_device is None:
        raise ValueError("mesh_device must be provided")

    weight_cache_path = (
        weight_cache_path
        / f"{hf_config.num_hidden_layers}_layers"
        / f"mesh_{mesh_device.shape[0]}x{mesh_device.shape[1]}"
    )
    config_path = weight_cache_path / "config.json"
    weight_path = weight_cache_path / "weights"
    for _ in range(1):
        if force_recalculate:
            logger.info(f"Forcing recalculating weights")
            break
        if not config_path.exists():
            logger.info(f"Weight configuration file does not exist, forcing recalculating weights")
            break
        with config_path.open() as f:
            weight_config = json.load(f, object_hook=try_decode_saved_weight)
        try:
            validate_weight_config_paths(weight_cache_path, weight_config)
        except ValueError as e:
            logger.warning(f"Cache validation failed, will recalculate weights: {e}")
            break
        logger.info(f"Using weights cached at {weight_cache_path}")
        return normalize_weight_config_paths(weight_cache_path, weight_config)

    # Only prepare state dicts if we need to convert weights
    logger.info(f"Caching weights at {weight_cache_path}")
    if state_dicts is None:
        logger.info(f"State dict was not provided, preparing from random weights or model path")
        from models.demos.deepseek_v3.utils.hf_model_utils import prepare_model_state_dict

        model_state = prepare_model_state_dict(
            hf_config=hf_config,
            random_weights=random_weights,
            model_path=model_path,
            single_layer=single_layer,
        )
        state_dicts = (model_state,)

    # Convert weights to TT tensors-on-disk and build weight_config
    logger.info("Converting weights to TTNN SavedWeight format...")

    weight_config = ModuleClass.convert_weights(hf_config, state_dicts, weight_cache_path, mesh_device)
    breakpoint()

    # Validate the converted weight config
    validate_weight_config_paths(weight_cache_path, weight_config)

    # Save config with relative paths for portability
    with config_path.open("w") as f:
        json.dump(weight_config, f, cls=WeightConfigEncoder)

    # Return normalized config with absolute paths for runtime use
    normalized_config = normalize_weight_config_paths(weight_cache_path, weight_config)
    logger.info("Done converting weights to TTNN SavedWeight format")
    return normalized_config


def validate_weight_config_paths(root_path: Path, weight_config: WeightConfig, path_prefix: str = "") -> None:
    """
    Validate that all SavedWeight paths in the weight config exist and have the correct suffix.

    Args:
        root_path: Base path for resolving relative SavedWeight paths
        weight_config: Weight configuration (dict, list, tuple, or nested structures)
        path_prefix: Prefix for error messages to indicate location in nested structure

    Raises:
        ValueError: If any SavedWeight path is invalid (missing file, wrong suffix, etc.)
    """
    if isinstance(weight_config, dict):
        entries = weight_config.items()
    elif isinstance(weight_config, (list, tuple)):
        entries = enumerate(weight_config)
    else:
        raise ValueError(f"Invalid weight config type: {type(weight_config)}")

    for key, entry in entries:
        if entry is None:
            continue
        current_prefix = f"{path_prefix}.{key}" if path_prefix else str(key)

        if isinstance(entry, SavedWeight):
            # Reject absolute paths - configs should only contain relative paths for portability
            if entry.path.is_absolute():
                raise ValueError(
                    f"SavedWeight at '{current_prefix}' has absolute path '{entry.path}'. "
                    f"Only relative paths are allowed in weight configs for portability."
                )

            # Resolve effective path
            effective_path = root_path / entry.path

            # Validate suffix
            if entry.path.suffix != TENSOR_CACHE_EXTENSION:
                raise ValueError(
                    f"SavedWeight at '{current_prefix}' has invalid suffix '{entry.path.suffix}'. "
                    f"Expected '{TENSOR_CACHE_EXTENSION}'. Path: {entry.path}"
                )

            # Validate file exists
            if not effective_path.exists():
                raise ValueError(
                    f"SavedWeight at '{current_prefix}' references missing file. "
                    f"Resolved path: {effective_path} (original: {entry.path})"
                )
        else:
            # Recursively validate nested structures
            validate_weight_config_paths(root_path, entry, current_prefix)


def normalize_weight_config_paths(root_path: Path, weight_config: WeightConfig) -> WeightConfig:
    """
    Return a new weight config with all relative SavedWeight paths converted to absolute paths.

    Args:
        root_path: Base path for resolving relative SavedWeight paths
        weight_config: Weight configuration (dict, list, tuple, or nested structures)

    Returns:
        New weight config with absolute paths (deep copy, no mutation of input)
    """
    if isinstance(weight_config, dict):
        return {
            key: normalize_weight_config_paths(root_path, value) if value is not None else None
            for key, value in weight_config.items()
        }
    elif isinstance(weight_config, (list, tuple)):
        normalized = [
            normalize_weight_config_paths(root_path, item) if item is not None else None for item in weight_config
        ]
        # Preserve tuple type if input was a tuple
        return tuple(normalized) if isinstance(weight_config, tuple) else normalized
    elif isinstance(weight_config, SavedWeight):
        # Create a new SavedWeight with absolute path
        if weight_config.path.is_absolute():
            normalized_path = weight_config.path
        else:
            normalized_path = root_path / weight_config.path
        return SavedWeight(path=normalized_path, memory_config=weight_config.memory_config)
    else:
        # For other types (None, primitives, etc.), return as-is
        return weight_config


def _normalize_weight_config_paths_inplace(root_path: Path, weight_config: WeightConfig) -> None:
    """
    Internal helper that mutates weight_config in-place (for backward compatibility only).
    New code should use normalize_weight_config_paths instead.
    """
    if isinstance(weight_config, dict):
        entries = weight_config.values()
    elif isinstance(weight_config, (list, tuple)):
        entries = weight_config
    else:
        entries = []

    for entry in entries:
        if entry is None:
            continue
        if isinstance(entry, SavedWeight):
            if not entry.path.is_absolute():
                entry.path = root_path / entry.path
        else:
            _normalize_weight_config_paths_inplace(root_path, entry)

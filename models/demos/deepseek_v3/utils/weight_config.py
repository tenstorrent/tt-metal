# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import SavedWeight
from models.demos.deepseek_v3.utils.config_helpers import TENSOR_CACHE_EXTENSION
from models.demos.deepseek_v3.utils.run_config import WeightConfig

# Key used to store cache-level metadata in config.json.  This entry is
# never a weight and must be skipped by path-validation and normalization.
_META_KEY = "_meta"


@contextmanager
def locked_file(file_path: Path, mode: str = "r", exclusive: bool = False):
    """
    Context manager for file operations with advisory locking.

    Args:
        file_path: Path to the file
        mode: File open mode ('r' for read, 'w' for write, etc.)
        exclusive: If True, use exclusive lock (LOCK_EX) for writes.
                  If False, use shared lock (LOCK_SH) for reads.

    Yields:
        File handle with lock acquired
    """
    # Ensure parent directory exists for write operations
    if mode in ("w", "a", "x") or "+" in mode:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH

    with file_path.open(mode) as f:
        try:
            fcntl.flock(f.fileno(), lock_type)
            yield f
        finally:
            # Lock is automatically released when file is closed
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


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


def _try_load_cached_config(config_path: Path, weight_cache_path: Path, force_recalculate: bool) -> WeightConfig | None:
    """
    Attempt to load weight config from cache.

    Args:
        config_path: Path to the config.json file
        weight_cache_path: Base path for resolving relative weight paths
        force_recalculate: If True, skip cache and return None

    Returns:
        Normalized weight config if cache hit, None if cache miss
    """
    if force_recalculate:
        logger.info("Forcing recalculating weights")
        return None
    if not config_path.exists():
        logger.info("Weight configuration file does not exist, forcing recalculating weights")
        return None

    with locked_file(config_path, "r", exclusive=False) as f:
        weight_config = json.load(f, object_hook=try_decode_saved_weight)

    try:
        validate_weight_config_paths(weight_cache_path, weight_config)
    except ValueError as e:
        logger.warning(f"Cache validation failed, will recalculate weights: {e}")
        return None

    logger.info(f"Using weights cached at {weight_cache_path}")
    normalized = normalize_weight_config_paths(weight_cache_path, weight_config)
    # Strip _meta — it is only stored on disk for discoverability and must not
    # be exposed to callers (it would break create_run_config merging).
    if isinstance(normalized, dict):
        normalized = {k: v for k, v in normalized.items() if k != _META_KEY}
    return normalized


def _load_raw_weight_config(config_path: Path) -> WeightConfig | None:
    """Load a config.json without validating or normalising paths.

    Returns the raw parsed config (SavedWeights with relative paths) or
    ``None`` if the file does not exist.
    """
    if not config_path.exists():
        return None
    with locked_file(config_path, "r", exclusive=False) as f:
        return json.load(f, object_hook=try_decode_saved_weight)


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
    dtype_tag: str | None = None,
):
    """
    Get weight configuration, either from cache or by converting weights.

    When ``dtype_tag`` is provided (non-None) an **accumulating** cache is used.
    The cache lives at ``{weight_cache_path}/mesh_{R}x{C}/{dtype_tag}/`` — there is
    no ``{N}_layers/`` component.  A single ``config.json`` grows over time: if a
    previous run cached 5 layers and the caller now needs 10, the loader reuses the
    existing 5 and converts only the missing 5, then saves a 10-layer config.
    Conversely, if 10 layers are cached and only 5 are needed, the config is sliced
    in-memory and no conversion happens.

    Args:
        ModuleClass: The module class to convert weights for
        hf_config: HuggingFace model configuration
        state_dicts: Optional pre-loaded state dicts. If None, will be loaded based on random_weights/model_path.
        weight_cache_path: Base path to cache weights (before mesh/dtype subdirs are appended)
        mesh_device: TTNN mesh device
        force_recalculate: Force full recalculation even if cached weights exist
        random_weights: If True, generate random weights from reference model
        model_path: Path to HuggingFace model directory (required if random_weights=False and state_dicts=None)
        single_layer: Optional single layer name (used for validation with random weights)
        dtype_tag: String identifying the weight data formats used (auto-derived from
            ``ModuleClass.get_dtype_tag``).  When non-None the accumulating cache is used;
            when None the legacy ``{N}_layers/mesh_{R}x{C}/`` layout is used unchanged.

    Returns:
        Weight configuration dictionary
    """
    if weight_cache_path is None:
        raise ValueError("weight_cache_path must be provided")
    if mesh_device is None:
        raise ValueError("mesh_device must be provided")

    mesh_dir_name = f"mesh_{mesh_device.shape[0]}x{mesh_device.shape[1]}"
    n = hf_config.num_hidden_layers

    if dtype_tag is not None:
        # Accumulating cache layout: no {N}_layers/ component.
        # mesh_cache_path is the root for resolving relative weight paths
        # (config_helpers.py anchors path stripping at "mesh_").
        mesh_cache_path = weight_cache_path / mesh_dir_name
        dtype_cache_path = mesh_cache_path / dtype_tag
        config_path = dtype_cache_path / "config.json"

        # Load whatever is already on disk (raw, unvalidated).
        existing_config = None if force_recalculate else _load_raw_weight_config(config_path)

        if existing_config is not None:
            num_cached = ModuleClass.get_num_cached_layers(existing_config)
            if num_cached >= n:
                # Enough layers cached — validate, slice, and return without any conversion.
                try:
                    validate_weight_config_paths(mesh_cache_path, existing_config)
                except ValueError as e:
                    logger.warning(f"Cache validation failed, will recalculate weights: {e}")
                    existing_config = None
                else:
                    sliced = ModuleClass.slice_weight_config(existing_config, hf_config)
                    if sliced is None:
                        sliced = existing_config
                    logger.info(
                        f"Using {num_cached}-layer cache at {dtype_cache_path} "
                        f"(need {n} layers, dtype_tag={dtype_tag})"
                    )
                    return normalize_weight_config_paths(
                        mesh_cache_path, {k: v for k, v in sliced.items() if k != _META_KEY}
                    )
            else:
                logger.info(
                    f"Cache has {num_cached} layers but {n} are needed — augmenting " f"(dtype_tag={dtype_tag})"
                )
        elif force_recalculate:
            logger.info("Forcing recalculating weights")
        else:
            logger.info(f"No cache found at {config_path}, converting weights")

        # Need to produce (more) weights — prepare state dicts if not supplied.
        if state_dicts is None:
            logger.info("State dict was not provided, preparing from random weights or model path")
            from models.demos.deepseek_v3.utils.hf_model_utils import prepare_model_state_dict

            model_state = prepare_model_state_dict(
                hf_config=hf_config,
                random_weights=random_weights,
                model_path=model_path,
                single_layer=single_layer,
            )
            state_dicts = (model_state,)

        # Augment: reuse existing layers and convert only the missing ones.
        # When existing_config is None this is equivalent to a full conversion.
        logger.info(f"Writing weights to {dtype_cache_path}")
        weight_config = ModuleClass.augment_weight_config(
            hf_config, state_dicts, existing_config, dtype_cache_path, mesh_device
        )

        validate_weight_config_paths(mesh_cache_path, weight_config)

        num_saved = ModuleClass.get_num_cached_layers(weight_config)
        config_to_save = {
            _META_KEY: {
                "num_cached_layers": num_saved,
                "dtype_tag": dtype_tag,
                "mesh": mesh_dir_name,
            },
            **weight_config,
        }
        with locked_file(config_path, "w", exclusive=True) as f:
            json.dump(config_to_save, f, cls=WeightConfigEncoder)

        sliced = ModuleClass.slice_weight_config(weight_config, hf_config)
        if sliced is None:
            sliced = weight_config
        logger.info("Done converting weights to TTNN SavedWeight format")
        return normalize_weight_config_paths(mesh_cache_path, sliced)

    else:
        # Legacy layout: {N}_layers/mesh_{R}x{C}/config.json — exact match only.
        layers_dir_name = f"{n}_layers"
        mesh_cache_path = weight_cache_path / layers_dir_name / mesh_dir_name
        dtype_cache_path = mesh_cache_path
        config_path = mesh_cache_path / "config.json"

        cached_config = _try_load_cached_config(config_path, mesh_cache_path, force_recalculate)
        if cached_config is not None:
            return cached_config

        if state_dicts is None:
            logger.info("State dict was not provided, preparing from random weights or model path")
            from models.demos.deepseek_v3.utils.hf_model_utils import prepare_model_state_dict

            model_state = prepare_model_state_dict(
                hf_config=hf_config,
                random_weights=random_weights,
                model_path=model_path,
                single_layer=single_layer,
            )
            state_dicts = (model_state,)

        logger.info(f"Caching weights at {dtype_cache_path}")
        logger.info("Converting weights to TTNN SavedWeight format...")
        weight_config = ModuleClass.convert_weights(hf_config, state_dicts, dtype_cache_path, mesh_device)

        validate_weight_config_paths(mesh_cache_path, weight_config)

        with locked_file(config_path, "w", exclusive=True) as f:
            json.dump(weight_config, f, cls=WeightConfigEncoder)

        logger.info("Done converting weights to TTNN SavedWeight format")
        return normalize_weight_config_paths(mesh_cache_path, weight_config)


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
        if key == _META_KEY:
            continue
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
            key: value
            if key == _META_KEY
            else (normalize_weight_config_paths(root_path, value) if value is not None else None)
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

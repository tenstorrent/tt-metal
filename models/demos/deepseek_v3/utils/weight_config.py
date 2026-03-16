# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fcntl
import json
import os
import socket
import time
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

_MULTIHOST_CACHE_COORD_DIR = ".multihost_cache"
_MULTIHOST_DECISION_FILE = "decision.json"
_MULTIHOST_GENERATION_LOCK = "generation.lock"
_MULTIHOST_POLL_INTERVAL_SECONDS = 0.1
_MULTIHOST_WAIT_LOG_INTERVAL_SECONDS = 10 * 60
_HOSTNAME = socket.gethostname().split(".", 1)[0]


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


class MissingWeightFileError(ValueError):
    pass


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
    return normalize_weight_config_paths(weight_cache_path, weight_config)


def _load_cached_config_without_visibility_checks(config_path: Path, weight_cache_path: Path) -> WeightConfig:
    _wait_for_path_exists(config_path, description="config.json")

    with locked_file(config_path, "r", exclusive=False) as f:
        weight_config = json.load(f, object_hook=try_decode_saved_weight)

    validate_weight_config_paths(weight_cache_path, weight_config, require_files_exist=False)
    _wait_for_weight_config_paths(weight_cache_path, weight_config, context="rank-0 cache hit")
    logger.info(f"Using weights cached at {weight_cache_path}")
    return normalize_weight_config_paths(weight_cache_path, weight_config)


def _write_json_atomic(path: Path, payload: dict[str, Any], *, encoder: type[json.JSONEncoder] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w") as f:
        json.dump(payload, f, cls=encoder)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _is_multihost_distributed() -> bool:
    if not ttnn.distributed_context_is_initialized():
        return False
    try:
        return int(ttnn.distributed_context_get_size()) > 1
    except Exception:
        return False


def _get_distributed_rank() -> int:
    if not _is_multihost_distributed():
        return 0
    return int(ttnn.distributed_context_get_rank())


def _distributed_barrier() -> None:
    if _is_multihost_distributed():
        ttnn.distributed_context_barrier()


def _maybe_log_wait(last_log_time: float | None, message: str) -> float:
    now = time.monotonic()
    if last_log_time is None or now - last_log_time >= _MULTIHOST_WAIT_LOG_INTERVAL_SECONDS:
        logger.info(message)
        return now
    return last_log_time


def _read_multihost_decision(decision_path: Path) -> str:
    last_log_time = None
    while True:
        try:
            with locked_file(decision_path, "r", exclusive=False) as f:
                payload = json.load(f)
            mode = payload.get("mode")
            if mode not in {"hit", "miss"}:
                raise RuntimeError(f"Invalid multihost cache decision payload in {decision_path}: {payload}")
            return mode
        except FileNotFoundError as e:
            last_log_time = _maybe_log_wait(
                last_log_time,
                f"Still waiting for {decision_path.name} to become visible on host {_HOSTNAME}: {decision_path} ({e})",
            )
        except json.JSONDecodeError as e:
            last_log_time = _maybe_log_wait(
                last_log_time,
                f"Still waiting for {decision_path.name} to become readable on host {_HOSTNAME}: {decision_path} ({e})",
            )
        time.sleep(_MULTIHOST_POLL_INTERVAL_SECONDS)


def _wait_for_path_exists(path: Path, *, description: str) -> None:
    last_log_time = None
    while True:
        if path.exists():
            return
        last_log_time = _maybe_log_wait(
            last_log_time,
            f"Still waiting for {path.name} to become visible on host {_HOSTNAME}: {path} ({description})",
        )
        time.sleep(_MULTIHOST_POLL_INTERVAL_SECONDS)


def _wait_for_weight_config_paths(root_path: Path, weight_config: WeightConfig, *, context: str) -> None:
    last_log_time = None
    while True:
        try:
            validate_weight_config_paths(root_path, weight_config)
            return
        except MissingWeightFileError as e:
            last_log_time = _maybe_log_wait(
                last_log_time,
                f"Still waiting for cached weights to become visible on host {_HOSTNAME}: "
                f"{root_path} during {context} ({e})",
            )
            time.sleep(_MULTIHOST_POLL_INTERVAL_SECONDS)


def _convert_and_cache_weight_config(
    *,
    ModuleClass: type["models.demos.deepseek_v3.utils.abstract_module.AbstractModule"],
    hf_config: PretrainedConfig,
    state_dicts: tuple[dict[str, torch.Tensor] | None, ...] | None,
    weight_cache_path: Path,
    config_path: Path,
    mesh_device: ttnn.Device,
    random_weights: bool,
    model_path: str | None,
    single_layer: str | None,
    require_files_exist: bool,
    write_config: bool,
    wait_for_visibility: bool,
) -> WeightConfig:
    logger.info(f"Caching weights at {weight_cache_path}")
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

    logger.info("Converting weights to TTNN SavedWeight format...")
    weight_config = ModuleClass.convert_weights(hf_config, state_dicts, weight_cache_path, mesh_device)
    validate_weight_config_paths(weight_cache_path, weight_config, require_files_exist=require_files_exist)
    if write_config:
        _write_json_atomic(config_path, weight_config, encoder=WeightConfigEncoder)
    if wait_for_visibility:
        _wait_for_weight_config_paths(
            weight_cache_path, weight_config, context="first-time multihost cache publication"
        )

    logger.info("Done converting weights to TTNN SavedWeight format")
    return normalize_weight_config_paths(weight_cache_path, weight_config)


def _get_weight_config_multihost(
    *,
    ModuleClass: type["models.demos.deepseek_v3.utils.abstract_module.AbstractModule"],
    hf_config: PretrainedConfig,
    state_dicts: tuple[dict[str, torch.Tensor] | None, ...] | None,
    weight_cache_path: Path,
    config_path: Path,
    mesh_device: ttnn.Device,
    force_recalculate: bool,
    random_weights: bool,
    model_path: str | None,
    single_layer: str | None,
) -> WeightConfig:
    coord_dir = weight_cache_path / _MULTIHOST_CACHE_COORD_DIR
    decision_path = coord_dir / _MULTIHOST_DECISION_FILE
    lock_path = coord_dir / _MULTIHOST_GENERATION_LOCK
    rank = _get_distributed_rank()

    if rank == 0:
        with locked_file(lock_path, "a+", exclusive=True):
            cached_config = _try_load_cached_config(config_path, weight_cache_path, force_recalculate)
            decision_mode = "hit" if cached_config is not None else "miss"
            _write_json_atomic(decision_path, {"mode": decision_mode})
            _distributed_barrier()
            try:
                if decision_mode == "hit":
                    return cached_config
                return _convert_and_cache_weight_config(
                    ModuleClass=ModuleClass,
                    hf_config=hf_config,
                    state_dicts=state_dicts,
                    weight_cache_path=weight_cache_path,
                    config_path=config_path,
                    mesh_device=mesh_device,
                    random_weights=random_weights,
                    model_path=model_path,
                    single_layer=single_layer,
                    require_files_exist=True,
                    write_config=True,
                    wait_for_visibility=False,
                )
            finally:
                _distributed_barrier()
                decision_path.unlink(missing_ok=True)

    _distributed_barrier()
    decision_mode = _read_multihost_decision(decision_path)
    try:
        if decision_mode == "hit":
            return _load_cached_config_without_visibility_checks(config_path, weight_cache_path)

        return _convert_and_cache_weight_config(
            ModuleClass=ModuleClass,
            hf_config=hf_config,
            state_dicts=state_dicts,
            weight_cache_path=weight_cache_path,
            config_path=config_path,
            mesh_device=mesh_device,
            random_weights=random_weights,
            model_path=model_path,
            single_layer=single_layer,
            require_files_exist=False,
            write_config=False,
            wait_for_visibility=True,
        )
    finally:
        _distributed_barrier()


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

    weight_cache_path = weight_cache_path.expanduser()
    if not weight_cache_path.is_absolute():
        weight_cache_path = weight_cache_path.resolve()

    weight_cache_path = (
        weight_cache_path
        / f"{hf_config.num_hidden_layers}_layers"
        / f"mesh_{mesh_device.shape[0]}x{mesh_device.shape[1]}"
    )
    config_path = weight_cache_path / "config.json"

    if _is_multihost_distributed():
        return _get_weight_config_multihost(
            ModuleClass=ModuleClass,
            hf_config=hf_config,
            state_dicts=state_dicts,
            weight_cache_path=weight_cache_path,
            config_path=config_path,
            mesh_device=mesh_device,
            force_recalculate=force_recalculate,
            random_weights=random_weights,
            model_path=model_path,
            single_layer=single_layer,
        )

    # Try to load from cache
    cached_config = _try_load_cached_config(config_path, weight_cache_path, force_recalculate)
    if cached_config is not None:
        return cached_config

    return _convert_and_cache_weight_config(
        ModuleClass=ModuleClass,
        hf_config=hf_config,
        state_dicts=state_dicts,
        weight_cache_path=weight_cache_path,
        config_path=config_path,
        mesh_device=mesh_device,
        random_weights=random_weights,
        model_path=model_path,
        single_layer=single_layer,
        require_files_exist=True,
        write_config=True,
        wait_for_visibility=False,
    )


def validate_weight_config_paths(
    root_path: Path, weight_config: WeightConfig, path_prefix: str = "", *, require_files_exist: bool = True
) -> None:
    """
    Validate that all SavedWeight paths in the weight config exist and have the correct suffix.

    Args:
        root_path: Base path for resolving relative SavedWeight paths
        weight_config: Weight configuration (dict, list, tuple, or nested structures)
        path_prefix: Prefix for error messages to indicate location in nested structure
        require_files_exist: If False, validate path structure and suffixes but do not require
            the referenced files to be visible on the current host yet.

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
            if require_files_exist and not effective_path.exists():
                raise MissingWeightFileError(
                    f"SavedWeight at '{current_prefix}' references missing file. "
                    f"Resolved path: {effective_path} (original: {entry.path})"
                )
        else:
            # Recursively validate nested structures
            validate_weight_config_paths(root_path, entry, current_prefix, require_files_exist=require_files_exist)


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

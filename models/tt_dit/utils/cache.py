# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

from ..layers.module import Module

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

CACHE_DICT_FILE = "cache_dict.json"


class MissingCacheError(Exception):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __str__(self) -> str:
        return f"cache does not exist at '{self.path}'"


def config_id(parallel_config):
    config_id = ""
    for n, v in parallel_config._asdict().items():
        if v is not None:
            config_id += f"{''.join([w[0].upper() for w in n.split('_')])}{v.factor}_{v.mesh_axis}_"
    return config_id


def cache_dir_is_set() -> bool:
    return _cache_root() is not None


def load_model(
    tt_model: Module,
    *,
    model_name: str,
    subfolder: str,
    parallel_config: NamedTuple,
    mesh_shape: Sequence[int],
    dtype: str = "bf16",
    is_fsdp: bool = False,
    get_torch_state_dict: Callable[[], dict] | None = None,
    create_cache: bool = True,
) -> None:
    """
    Load model weights from cache or PyTorch state dict.

    Attempts to load from cache first. If the cache does not exist, loads from PyTorch state dict
    (if provided) and optionally creates the cache. Raises `MissingCacheError` if neither is
    available. Finally, any module that needs to be offloaded is taken care of.

    Args:
        `tt_model`: TT model instance to load weights into.
        `model_name`: Model name (e.g., "flux1-dev", "stable-diffusion-3.5").
        `subfolder`: Subfolder within model cache directory (e.g., "transformer", "vae").
        `parallel_config`: Parallelism configuration (tensor/sequence parallel).
        `mesh_shape`: Device mesh shape.
        `dtype`: Data type for cached weights (default: "bf16").
        `is_fsdp`: Whether FSDP is used (default: False).
        `get_torch_state_dict`: Optional callable returning PyTorch state dict. Enables lazy
            evaluation - PyTorch model only loads if the cache does not exist. If `None`, cache
            must exist or `MissingCacheError` is raised.
        `create_cache`: Create cache after loading from PyTorch (default: True).

    Raises:
        `MissingCacheError`: Cache does not exist and `get_torch_state_dict` is `None`.
        `RuntimeError`: `TT_DIT_CACHE_DIR` is not set and `get_torch_state_dict` is `None`.
    """
    _t_load_start = time.perf_counter()
    logger.info(
        f"[TIMING] cache.load_model called: model_name={model_name}, subfolder={subfolder}, mesh_shape={mesh_shape}, dtype={dtype}"
    )

    if tt_model.is_loaded():
        logger.info("[TIMING] Model already loaded, skipping")
        return

    for module in tt_model.unload_set or []:
        logger.info(f"[TIMING] Deallocating weights for unload_set module: {type(module).__name__}")
        _t0 = time.perf_counter()
        module.deallocate_weights()
        logger.info(f"[TIMING] deallocate_weights took {time.perf_counter() - _t0:.2f}s")

    cache_dir = model_cache_dir(
        model_name=model_name,
        subfolder=subfolder,
        parallel_config=parallel_config,
        mesh_shape=mesh_shape,
        dtype=dtype,
        is_fsdp=is_fsdp,
        required=get_torch_state_dict is None,
    )
    logger.info(f"[TIMING] Resolved cache_dir: {cache_dir}")

    if cache_dir is None:
        assert get_torch_state_dict is not None

        logger.info("[TIMING] No cache dir (TT_DIT_CACHE_DIR not set). Loading from PyTorch state dict directly...")
        _t0 = time.perf_counter()
        state_dict = get_torch_state_dict()
        logger.info(
            f"[TIMING] get_torch_state_dict() took {time.perf_counter() - _t0:.2f}s (num keys: {len(state_dict)})"
        )
        _t0 = time.perf_counter()
        tt_model.load_torch_state_dict(state_dict)
        logger.info(f"[TIMING] load_torch_state_dict (no cache) took {time.perf_counter() - _t0:.2f}s")
        logger.info(f"[TIMING] cache.load_model total took {time.perf_counter() - _t_load_start:.2f}s")
        return

    if Path(cache_dir).is_dir():
        logger.info(f"[TIMING] Cache exists at '{cache_dir}'. Loading from cache...")
        _t0 = time.perf_counter()
        tt_model.load(cache_dir)
        logger.info(f"[TIMING] tt_model.load (from cache) took {time.perf_counter() - _t0:.2f}s")
        logger.info(f"[TIMING] cache.load_model total took {time.perf_counter() - _t_load_start:.2f}s")
        return

    if get_torch_state_dict is None:
        raise MissingCacheError(cache_dir)

    logger.info(f"[TIMING] Cache does not exist at '{cache_dir}'. Loading PyTorch state dict and creating cache...")
    _t0 = time.perf_counter()
    state_dict = get_torch_state_dict()
    logger.info(f"[TIMING] get_torch_state_dict() took {time.perf_counter() - _t0:.2f}s (num keys: {len(state_dict)})")

    _t0 = time.perf_counter()
    tt_model.load_torch_state_dict(state_dict, on_host=create_cache)
    logger.info(f"[TIMING] load_torch_state_dict (on_host={create_cache}) took {time.perf_counter() - _t0:.2f}s")

    if create_cache:
        _t0 = time.perf_counter()
        logger.info(f"[TIMING] Writing cache to '{cache_dir}'...")
        tt_model.save(cache_dir)
        logger.info(f"[TIMING] tt_model.save took {time.perf_counter() - _t0:.2f}s")

        _t0 = time.perf_counter()
        tt_model.load(cache_dir)
        logger.info(f"[TIMING] tt_model.load (move to device after cache write) took {time.perf_counter() - _t0:.2f}s")

    logger.info(f"[TIMING] cache.load_model total took {time.perf_counter() - _t_load_start:.2f}s")


def model_cache_dir(
    *,
    model_name: str,
    subfolder: str,
    parallel_config: NamedTuple,
    mesh_shape: Sequence[int],
    dtype: str = "bf16",
    is_fsdp: bool = False,
    required: bool = True,
) -> Path | None:
    cache_dir = _cache_root()
    if cache_dir is None:
        if required:
            msg = "Cache is required. Set the TT_DIT_CACHE_DIR environment variable."
            raise RuntimeError(msg)
        return None

    parallel_key = config_id(parallel_config)
    mesh_key = "x".join(str(x) for x in mesh_shape)

    key = f"{parallel_key}mesh{mesh_key}_{dtype}"
    if is_fsdp:
        key += "_FSDP"

    return Path(cache_dir) / model_name / subfolder / key


def _cache_root() -> str | None:
    return os.environ.get("TT_DIT_CACHE_DIR")

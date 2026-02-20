# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
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
    available.

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
    cache_dir = model_cache_dir(
        model_name=model_name,
        subfolder=subfolder,
        parallel_config=parallel_config,
        mesh_shape=mesh_shape,
        dtype=dtype,
        is_fsdp=is_fsdp,
        required=get_torch_state_dict is None,
    )

    if cache_dir is None:
        assert get_torch_state_dict is not None

        logger.info(
            "Loading transformer weights from PyTorch state dict. "
            "To use caching, set the TT_DIT_CACHE_DIR environment variable."
        )
        tt_model.load_torch_state_dict(get_torch_state_dict())
        return

    if Path(cache_dir).is_dir():
        logger.info(f"loading cache at '{cache_dir}'.")
        tt_model.load(cache_dir)
        return

    if get_torch_state_dict is None:
        raise MissingCacheError(cache_dir)

    logger.info("Cache does not exist. Loading PyTorch state dict.")
    # Create host tensors when creating the cache to circumvent the issue that replicated device
    # tensors lead to redundant copies when saved to disk.
    tt_model.load_torch_state_dict(get_torch_state_dict(), on_host=create_cache)

    if create_cache:
        logger.info(f"Writing cache to '{cache_dir}'.")
        tt_model.save(cache_dir)
        tt_model.load(cache_dir)  # move to device


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

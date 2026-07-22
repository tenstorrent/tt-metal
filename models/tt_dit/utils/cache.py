# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

import ttnn

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
    mesh_device: ttnn.MeshDevice,
    dtype: str = "bf16",
    is_fsdp: bool = False,
    get_torch_state_dict: Callable[[], dict] | None = None,
    create_cache: bool = True,
    post_load_hook: Callable[[Module], None] | None = None,
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
        `mesh_device`: Mesh device used to derive the multi-host ownership cache suffix.
        `dtype`: Data type for cached weights (default: "bf16").
        `is_fsdp`: Whether FSDP is used (default: False).
        `get_torch_state_dict`: Optional callable returning PyTorch state dict. Enables lazy
            evaluation - PyTorch model only loads if the cache does not exist. If `None`, cache
            must exist or `MissingCacheError` is raised.
        `create_cache`: Create cache after loading from PyTorch (default: True).
        `post_load_hook`: Optional callback run on the loaded module before the cache is written
            (and after every cache hit), e.g. a quant typecast, so cached tensorbins carry the
            post-hook weight dtype.

    Raises:
        `MissingCacheError`: Cache does not exist and `get_torch_state_dict` is `None`.
        `RuntimeError`: `TT_DIT_CACHE_DIR` is not set and `get_torch_state_dict` is `None`.
    """
    if tt_model.is_loaded():
        return

    cache_dir = model_cache_dir(
        model_name=model_name,
        subfolder=subfolder,
        parallel_config=parallel_config,
        mesh_shape=mesh_shape,
        mesh_device=mesh_device,
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
        if post_load_hook is not None:
            post_load_hook(tt_model)
        ttnn.distributed_context_barrier()
        return

    if _cache_is_complete(cache_dir):
        logger.info(f"loading cache at '{cache_dir}'.")
        tt_model.load(cache_dir)
        if post_load_hook is not None:
            post_load_hook(tt_model)
        ttnn.distributed_context_barrier()
        return

    if get_torch_state_dict is None:
        raise MissingCacheError(cache_dir)

    logger.info("Cache does not exist. Loading PyTorch state dict.")
    tt_model.load_torch_state_dict(get_torch_state_dict())

    # Run the hook (e.g. quant typecast) BEFORE save, so the cache holds the post-hook weight
    # dtype; otherwise a later reload reads stale-dtype tensorbins into a hook-mutated module and
    # the strict dtype check fails.
    if post_load_hook is not None:
        post_load_hook(tt_model)

    # If distributed, ensure that all processes have completed the check whether cache_dir exists,
    # before any rank might proceed to create that dir to save.
    ttnn.distributed_context_barrier()

    if create_cache:
        logger.info(f"Writing cache to '{cache_dir}'.")
        tt_model.save(cache_dir)
        _mark_cache_complete(cache_dir)


def model_cache_dir(
    *,
    model_name: str,
    subfolder: str,
    parallel_config: NamedTuple,
    mesh_shape: Sequence[int],
    mesh_device: ttnn.MeshDevice,
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

    path = Path(cache_dir) / model_name / subfolder / key

    ownership_suffix = _cache_ownership_suffix(mesh_device)
    if ownership_suffix:
        path = path / ownership_suffix

    return path


def _cache_ownership_suffix(mesh_device: ttnn.MeshDevice) -> str:
    """Multi-host cache dir suffix keyed by local mesh-coordinate ownership.

    Single-host / no distributed context: empty (same unsuffixed path as before).
    Multi-host: ``host_coords_r{r0}-{r1}_c{c0}-{c1}`` for the local coord bounding box.
    """
    if _distributed_world_size() <= 1:
        return ""

    view = mesh_device.get_view()
    rows = []
    cols = []
    for coord in ttnn.MeshCoordinateRange(view.shape()):
        if view.is_local(coord):
            rows.append(int(coord[0]))
            cols.append(int(coord[1]))
    return f"host_coords_r{min(rows)}-{max(rows)}_c{min(cols)}-{max(cols)}"


def _cache_is_complete(cache_dir: str | Path) -> bool:
    return (Path(cache_dir) / CACHE_DICT_FILE).is_file()


def _mark_cache_complete(cache_dir: str | Path) -> None:
    (Path(cache_dir) / CACHE_DICT_FILE).touch()


def _distributed_world_size() -> int:
    if not ttnn.distributed_context_is_initialized():
        return 1
    return int(ttnn.distributed_context_world_size())


def _cache_root() -> str | None:
    return os.environ.get("TT_DIT_CACHE_DIR")

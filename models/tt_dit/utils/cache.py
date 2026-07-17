# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

import ttnn

from ..layers.module import Module
from . import walltime

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

CACHE_DICT_FILE = "cache_dict.json"

# Manifest schema. Bump to make every content-keyed cache miss and rebuild.
CACHE_FORMAT = 1

# Bump when weight-prep code changes the VALUES it writes without changing any parameter's
# shape or dtype — e.g. GemmaFeatureExtractor's D-major→layer-major column permutation.
# Shape, dtype, layout and sharding changes need no bump: `module_signature` already sees them.
CACHE_VERSION = 1


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


def source_id(path: str | Path) -> str:
    """Content identity of a source checkpoint, cheap enough to recompute every process start.

    A HuggingFace hub blob is stored under its own git-lfs sha256, so for the common case the
    resolved filename *is* an exact content hash and costs one `readlink` — no read of the 46GB
    file. Anything else falls back to size+mtime, which catches a re-download or a swapped file
    but not an in-place rewrite that preserves both.
    """
    p = Path(path).resolve()
    st = p.stat()
    if p.parent.name == "blobs" and len(p.name) == 64 and all(c in "0123456789abcdef" for c in p.name):
        return f"sha256:{p.name}"
    return f"stat:{st.st_size}:{st.st_mtime_ns}"


def module_signature(module: Module) -> str:
    """Hash the structure of every parameter the cache will hold.

    Read off the built-but-unloaded module, so it costs no weight I/O. It covers exactly what
    determines a tensorbin's bytes — name, shape, dtype, layout, mesh sharding — which makes it
    the authority on weight dtype: a quant preset that only retunes compute (fidelity, SDPA
    inputs) leaves every `Parameter.dtype` alone and so keeps its cache, while one that restores
    a weight gets a new key. Value-only prep changes are invisible here; `CACHE_VERSION` covers
    those.
    """
    parts = [
        f"{name}|{p.total_shape}|{p.dtype}|{p.layout}|{p.mesh_axes}"
        for name, p in sorted(_walk_parameters(module), key=lambda kv: kv[0])
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def content_key(module: Module, sources: Sequence[str | Path] = ()) -> str:
    """Cache-key component binding an artifact to the weights and the code that produced it.

    Invalidates when the source checkpoint changes (`source_id`), when the module's weight
    structure or dtype changes (`module_signature`), or when weight-prep code changes values
    under a fixed structure (`CACHE_VERSION`). Every input is O(1) in the weight bytes.
    """
    h = hashlib.sha256(f"fmt{CACHE_FORMAT}|v{CACHE_VERSION}".encode())
    for s in sources:
        h.update(f"|src:{source_id(s)}".encode())
    h.update(f"|sig:{module_signature(module)}".encode())
    return h.hexdigest()[:12]


def _walk_parameters(module: Module, prefix: str = ""):
    for name, child in module.named_children():
        yield from _walk_parameters(child, f"{prefix}{name}.")
    for name, parameter in module.named_parameters():
        yield f"{prefix}{name}", parameter


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
    sources: Sequence[str | Path] = (),
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
        `sources`: Checkpoint files the weights are prepared from. Supplying them keys the cache
            on their content (see `content_key`) and turns on manifest verification, so an
            artifact built from other weights, other prep code or a torn write cannot be served.
            Omitting them keeps the legacy content-blind key and its existing on-disk caches.
        `get_torch_state_dict`: Optional callable returning PyTorch state dict. Enables lazy
            evaluation - PyTorch model only loads if the cache does not exist. If `None`, cache
            must exist or `MissingCacheError` is raised.
        `create_cache`: Create cache after loading from PyTorch (default: True).

    Raises:
        `MissingCacheError`: Cache does not exist and `get_torch_state_dict` is `None`.
        `RuntimeError`: `TT_DIT_CACHE_DIR` is not set and `get_torch_state_dict` is `None`.
    """
    if tt_model.is_loaded():
        return

    key = content_key(tt_model, sources) if sources else None

    key_args = {
        "model_name": model_name,
        "subfolder": subfolder,
        "parallel_config": parallel_config,
        "mesh_shape": mesh_shape,
        "mesh_device": mesh_device,
        "dtype": dtype,
        "is_fsdp": is_fsdp,
    }
    cache_dir = model_cache_dir(**key_args, content=key, required=get_torch_state_dict is None)

    if cache_dir is None:
        assert get_torch_state_dict is not None

        logger.info(
            "Loading transformer weights from PyTorch state dict. "
            "To use caching, set the TT_DIT_CACHE_DIR environment variable."
        )
        with walltime.timed("weight_load", f"{model_name}/{subfolder}", cached=False):
            tt_model.load_torch_state_dict(get_torch_state_dict())
        if post_load_hook is not None:
            post_load_hook(tt_model)
        ttnn.distributed_context_barrier()
        return

    if _cache_is_complete(cache_dir, tt_model, key):
        logger.info(f"loading cache at '{cache_dir}'.")
        with walltime.timed("weight_load", f"{model_name}/{subfolder}", cached=True):
            tt_model.load(cache_dir)
        if post_load_hook is not None:
            post_load_hook(tt_model)
        ttnn.distributed_context_barrier()
        return

    if get_torch_state_dict is None:
        raise MissingCacheError(cache_dir)

    logger.info("Cache does not exist. Loading PyTorch state dict.")
    with walltime.timed("weight_load", f"{model_name}/{subfolder}", cached=False):
        tt_model.load_torch_state_dict(get_torch_state_dict())

    # Hook (e.g. quant typecast) must run BEFORE save so the cache holds the post-hook
    # weight dtype; otherwise a dynamic-load reload reads stale-dtype tensorbins into a
    # hook-mutated module and the dtype check fails.
    if post_load_hook is not None:
        post_load_hook(tt_model)

    # If distributed, ensure that all processes have completed the check whether cache_dir exists,
    # before any rank might proceed to create that dir to save.
    ttnn.distributed_context_barrier()

    if create_cache:
        if key is not None:
            _reclaim_superseded(model_cache_dir(**key_args, content=None, required=False))
        logger.info(f"Writing cache to '{cache_dir}'.")
        _publish_cache(tt_model, cache_dir, key)


def model_cache_dir(
    *,
    model_name: str,
    subfolder: str,
    parallel_config: NamedTuple,
    mesh_shape: Sequence[int],
    mesh_device: ttnn.MeshDevice,
    dtype: str = "bf16",
    is_fsdp: bool = False,
    content: str | None = None,
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
    # Content lives in the path, not beside it, so weights that disagree with the key land in
    # different directories: a stale artifact misses and rebuilds instead of being read back
    # under a key that no longer describes it.
    if content is not None:
        key += f"_c{content}"

    path = Path(cache_dir) / model_name / subfolder / key

    ownership_suffix = _cache_ownership_suffix(mesh_device)
    if ownership_suffix:
        path = path / ownership_suffix

    return path


def _cache_is_complete(cache_dir: str | Path, tt_model: Module, key: str | None) -> bool:
    """Whether `cache_dir` holds an artifact that is safe to serve.

    Without a content key the caller has opted out of verification and only the completion
    marker is checked, which keeps every pre-existing cache loadable. With one, the manifest
    must agree with the module we are about to load into — a disagreement is reported as a miss
    so the weights are rebuilt, never quietly read back as something they are not.
    """
    marker = Path(cache_dir) / CACHE_DICT_FILE
    if not marker.is_file():
        return False
    if key is None:
        return True

    try:
        manifest = json.loads(marker.read_text())
    except (OSError, ValueError) as err:
        logger.warning(f"cache at '{cache_dir}' has an unreadable manifest ({err}); rebuilding.")
        return False

    if manifest.get("format") != CACHE_FORMAT or manifest.get("content_key") != key:
        logger.warning(f"cache at '{cache_dir}' was built for another key; rebuilding.")
        return False

    recorded = manifest.get("parameters", {})
    if {name: dict(spec, bytes=None) for name, spec in recorded.items()} != _structure(tt_model):
        logger.warning(f"cache at '{cache_dir}' does not describe this module; rebuilding.")
        return False

    for name, spec in recorded.items():
        path = Path(cache_dir) / f"{name}.tensorbin"
        if not path.is_file() or path.stat().st_size != spec["bytes"]:
            logger.warning(f"cache at '{cache_dir}' is missing or truncated at '{name}'; rebuilding.")
            return False

    return True


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


def _structure(tt_model: Module) -> dict:
    """What each tensorbin must contain, as recorded in the manifest (byte count filled at save)."""
    return {
        name: {"shape": list(p.total_shape), "dtype": str(p.dtype), "layout": str(p.layout), "bytes": None}
        for name, p in _walk_parameters(tt_model)
    }


def _reclaim_superseded(legacy_dir: Path | None) -> None:
    """Drop the content-blind directory that a content key replaces, before rebuilding over it.

    Once a module keys on content, nothing constructs its old path again, so whatever sits there
    is unreachable — and for the 22B transformer it is 37GB. Freeing it first makes the one-time
    migration disk-neutral; leaving it would need room for two copies of a checkpoint that only
    just fits once. It is not a fallback: an unverifiable artifact is exactly what is being
    retired, so there is nothing to keep.
    """
    if legacy_dir is None or not legacy_dir.is_dir():
        return
    logger.info(f"Reclaiming superseded content-blind cache at '{legacy_dir}'.")
    shutil.rmtree(legacy_dir, ignore_errors=True)


def _publish_cache(tt_model: Module, cache_dir: str | Path, key: str | None) -> None:
    """Write the cache to a scratch directory and move it into place in one step.

    The tensorbins and their completion marker become visible together or not at all, so a build
    that dies partway, or two processes racing to fill the same key, can never leave a
    half-written directory that later reads as complete.
    """
    cache_dir = Path(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.staging-", dir=cache_dir.parent))

    try:
        tt_model.save(staging)

        manifest = {"format": CACHE_FORMAT}
        if key is not None:
            parameters = _structure(tt_model)
            for name, spec in parameters.items():
                spec["bytes"] = (staging / f"{name}.tensorbin").stat().st_size
            manifest |= {"content_key": key, "parameters": parameters}
        (staging / CACHE_DICT_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True))

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(staging, cache_dir)
    except OSError:
        # A racing process may have published this exact key first. Its content key matches ours,
        # so its bytes are ours; adopt them rather than fight over the directory.
        if _cache_is_complete(cache_dir, tt_model, key):
            logger.info(f"cache at '{cache_dir}' was published concurrently; adopting it.")
        else:
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _distributed_world_size() -> int:
    if not ttnn.distributed_context_is_initialized():
        return 1
    return int(ttnn.distributed_context_world_size())


def _cache_root() -> str | None:
    return os.environ.get("TT_DIT_CACHE_DIR")

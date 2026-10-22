# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# TT_CACHE_PATH integration for the HunyuanImage-3.0 transformer backbone.
#
# Mirrors the tt_dit DiT stack caching model: on first load, each weight tensor is
# converted from PyTorch -> TTNN (with the correct mesh sharding) and written as a
# ``.tensorbin`` flatbuffer under a deterministic directory keyed by model variant,
# mesh shape, and parallelism. Subsequent runs load the pre-tilized tensors directly
# from disk, skipping the expensive host-side reorder / shard / tilize path.

from __future__ import annotations

import errno
import os
from pathlib import Path

import ttnn

_RO_ERRNOS = (errno.EROFS, errno.EACCES, errno.EPERM)


def cache_root() -> Path | None:
    """Return ``TT_CACHE_PATH`` when set, else ``None`` (caching disabled)."""
    root = os.environ.get("TT_CACHE_PATH")
    return Path(root) if root else None


def cache_dir_is_set() -> bool:
    return cache_root() is not None


def _writable_cache_mirror(preferred: Path) -> Path:
    """Writable mirror for a preferred cache dir on read-only mounts (CI MLPerf :ro)."""
    root = Path(os.environ.get("TT_METAL_HOME") or os.environ.get("HOME") or "/tmp")
    tt_root = cache_root()
    if tt_root is not None:
        try:
            return root / "generated" / "hunyuan_tt_cache" / preferred.relative_to(tt_root)
        except ValueError:
            pass
    suffix = Path(*preferred.parts[-3:]) if len(preferred.parts) >= 3 else Path(preferred.name)
    return root / "generated" / "hunyuan_tt_cache" / suffix


def ensure_cache_dir(path: Path | str | None) -> Path | None:
    """Return ``path``, creating it when possible.

    CI mounts ``/mnt/MLPerf/huggingface`` read-only. ``Path.mkdir`` raises
    ``OSError: [Errno 30] Read-only file system`` when the cache subdir is
    missing. Reuse an existing dir; otherwise mirror under
    ``$TT_METAL_HOME/generated/hunyuan_tt_cache/...`` so cold builds can write.
    """
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        return path
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as e:
        if e.errno not in _RO_ERRNOS:
            raise
        if path.is_dir():
            return path
        alt = _writable_cache_mirror(path)
        if os.environ.get("HY_VERBOSE", "1") != "0":
            print(
                f"[cache] TT cache dir not writable ({path}): {e}; using {alt}",
                flush=True,
            )
        alt.mkdir(parents=True, exist_ok=True)
        return alt


def _dtype_key(dtype: ttnn.DataType) -> str:
    return dtype.name if hasattr(dtype, "name") else str(dtype)


def _bf16_layers_key(bf16_layers: set[int] | frozenset[int] | None) -> str:
    if not bf16_layers:
        return "bf16_none"
    parts = []
    for i in sorted(bf16_layers):
        if parts and parts[-1][1] == i - 1:
            start, end = parts[-1][0], i
            parts[-1] = (start, end)
        else:
            parts.append((i, i))
    return "bf16_" + "_".join(f"{a}" if a == b else f"{a}-{b}" for a, b in parts)


def transformer_cache_dir(
    *,
    model_name: str,
    mesh_shape: tuple[int, ...],
    tp_axis: int,
    tp_factor: int,
    sp_axis: int,
    sp_factor: int,
    weight_dtype: ttnn.DataType,
    num_layers: int,
    bf16_layers: set[int] | frozenset[int] | None = None,
) -> Path | None:
    """
    Resolve the on-disk cache directory for the resident transformer stack.

    Layout (same convention as tt_dit):
        ``$TT_CACHE_PATH/<model_name>/transformer/<parallel_mesh_dtype_key>/``

    Individual weight files are named after their checkpoint keys, e.g.
    ``model.layers.0.self_attn.qkv_proj.weight_dtype_BFLOAT8_B_layout_TILE.tensorbin``.
    """
    root = cache_root()
    if root is None:
        return None

    mesh_key = "x".join(str(x) for x in mesh_shape)
    parallel = f"SP{sp_factor}a{sp_axis}_TP{tp_factor}a{tp_axis}"
    key = f"{parallel}_mesh{mesh_key}_L{num_layers}_{_dtype_key(weight_dtype)}" f"_{_bf16_layers_key(bf16_layers)}"
    return root / model_name / "transformer" / key


def resolve_transformer_cache(
    *,
    model_name: str,
    device,
    tp_axis: int,
    tp_factor: int,
    sp_axis: int,
    sp_factor: int,
    weight_dtype: ttnn.DataType,
    num_layers: int,
    bf16_layers: set[int] | frozenset[int] | None = None,
    weight_cache_path: Path | str | None = None,
) -> Path | None:
    """
    Return the cache directory to pass as ``weight_cache_path`` to ``HunyuanTtModel``.

    Explicit ``weight_cache_path`` wins; otherwise derive from ``TT_CACHE_PATH``.
    """
    if weight_cache_path is not None:
        return Path(weight_cache_path)
    return transformer_cache_dir(
        model_name=model_name,
        mesh_shape=tuple(device.shape),
        tp_axis=tp_axis,
        tp_factor=tp_factor,
        sp_axis=sp_axis,
        sp_factor=sp_factor,
        weight_dtype=weight_dtype,
        num_layers=num_layers,
        bf16_layers=bf16_layers,
    )


def cache_file(weight_cache_path: Path | None, key: str) -> str | None:
    """Build a ``cache_file_name`` argument for ``ttnn.as_tensor``."""
    if weight_cache_path is None:
        return None
    return str(weight_cache_path / key)


def cached_tensor_path(
    weight_cache_path: Path | str | None,
    key: str,
    *,
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> Path | None:
    """On-disk path for a tilized cache entry (matches ``ttnn.as_tensor`` naming)."""
    base = cache_file(weight_cache_path, key)
    if base is None:
        return None
    dtype_name = dtype.name if hasattr(dtype, "name") else str(dtype)
    layout_name = layout.name if hasattr(layout, "name") else str(layout)
    return Path(f"{base}_dtype_{dtype_name}_layout_{layout_name}.tensorbin")


def cache_file_exists(
    weight_cache_path: Path | str | None,
    key: str,
    *,
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> bool:
    """True when the tilized cache file for ``key`` is already on disk."""
    path = cached_tensor_path(weight_cache_path, key, dtype=dtype, layout=layout)
    return path is not None and path.is_file()

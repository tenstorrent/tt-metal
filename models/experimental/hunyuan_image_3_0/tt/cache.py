# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# TT_DIT_CACHE_DIR integration for the HunyuanImage-3.0 transformer backbone.
#
# Mirrors the tt_dit DiT stack caching model: on first load, each weight tensor is
# converted from PyTorch -> TTNN (with the correct mesh sharding) and written as a
# ``.tensorbin`` flatbuffer under a deterministic directory keyed by model variant,
# mesh shape, and parallelism. Subsequent runs load the pre-tilized tensors directly
# from disk, skipping the expensive host-side reorder / shard / tilize path.

from __future__ import annotations

import os
from pathlib import Path

import ttnn


def cache_root() -> Path | None:
    """Return ``TT_DIT_CACHE_DIR`` when set, else ``None`` (caching disabled)."""
    root = os.environ.get("TT_DIT_CACHE_DIR")
    return Path(root) if root else None


def cache_dir_is_set() -> bool:
    return cache_root() is not None


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
        ``$TT_DIT_CACHE_DIR/<model_name>/transformer/<parallel_mesh_dtype_key>/``

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

    Explicit ``weight_cache_path`` wins; otherwise derive from ``TT_DIT_CACHE_DIR``.
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

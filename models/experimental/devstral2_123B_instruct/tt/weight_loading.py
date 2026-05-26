# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host → device weight upload for Devstral-2 / Ministral3.

Tests load bf16 CPU tensors via ``tests/_devstral_weights``; each ``Tt*`` module uploads through the
helpers below (TP shard or replicate, tile layout materialized at upload). Matmul weights use DRAM +
TILE; embeddings use DRAM + ROW_MAJOR; RoPE tables use L1 + TILE.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch
import ttnn

from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args, torch_default_dtype_for

PathLike = Union[str, Path]

DEVSTRAL2_LARGE_REPO_ID = "mistralai/Devstral-2-123B-Instruct-2512"

# Default on-disk cache for tiled device weights (override root with ``TT_CACHE_PATH``).
_DEFAULT_CACHE_ROOT = Path("generated/ttnn/devstral2_123B_instruct/weight_cache")

_DTYPE_CACHE_DIR = {
    ttnn.bfloat16: "tensor_cache_bf16",
    ttnn.bfloat8_b: "tensor_cache_bfp8",
}

# Memory configs used at upload (match first consumer where possible).
MATMUL_WEIGHT_MEM_CONFIG = ttnn.DRAM_MEMORY_CONFIG
NORM_WEIGHT_MEM_CONFIG = ttnn.DRAM_MEMORY_CONFIG
EMBED_WEIGHT_MEM_CONFIG = ttnn.DRAM_MEMORY_CONFIG
ROPE_TABLE_MEM_CONFIG = ttnn.L1_MEMORY_CONFIG
KV_CACHE_MEM_CONFIG = ttnn.DRAM_MEMORY_CONFIG


def resolve_weight_cache_path(
    weight_cache_path: Optional[PathLike],
    args: Devstral2Args,
    *,
    num_layers: Optional[int] = None,
) -> Optional[str]:
    """Return the directory for ``ttnn.as_tensor`` weight caches.

    - ``None`` (default): use the standard cache path under ``generated/`` or ``TT_CACHE_PATH``.
    - explicit path: use as-is.
    - Set env ``DEVSTRAL2_DISABLE_WEIGHT_CACHE=1`` to disable caching (upload + tilize every run).
    """
    if os.getenv("DEVSTRAL2_DISABLE_WEIGHT_CACHE", "").lower() in ("1", "true", "yes"):
        return None
    if weight_cache_path is not None:
        path = Path(weight_cache_path)
    else:
        root = os.getenv("TT_CACHE_PATH")
        base = Path(root) / "devstral2_123B_instruct" if root else _DEFAULT_CACHE_ROOT
        dtype_dir = _DTYPE_CACHE_DIR.get(args.weight_dtype, f"dtype_{args.weight_dtype}")
        mesh_tag = f"mesh_{args.mesh_shape[0]}x{args.mesh_shape[1]}"
        n_layers = num_layers if num_layers is not None else args.num_hidden_layers
        path = base / DEVSTRAL2_LARGE_REPO_ID / dtype_dir / mesh_tag / f"layers_{n_layers}" / f"seq_{args.max_seq_len}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def weight_cache_file(weight_cache_path: Optional[PathLike], state_dict_key: str) -> Optional[str]:
    """Flatbuffer cache path for ``ttnn.as_tensor``, or ``None`` to skip caching."""
    if weight_cache_path is None:
        return None
    safe = state_dict_key.replace(".", "_")
    return str(Path(weight_cache_path) / safe)


def materialize_tile_layout(tensor: ttnn.Tensor) -> ttnn.Tensor:
    """Force tile layout on device once at upload (avoids lazy tilize on first matmul)."""
    return ttnn.to_layout(tensor, layout=ttnn.TILE_LAYOUT)


def _mesh_mapper_col_parallel(mesh_device, args: Devstral2Args) -> ttnn.TensorToMesh:
    dims = (None, -1) if args.cluster_axis == 1 else (-1, None)
    return ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=args.mesh_shape)


def _mesh_mapper_row_parallel(mesh_device, args: Devstral2Args) -> ttnn.TensorToMesh:
    dims = (None, -2) if args.cluster_axis == 1 else (-2, None)
    return ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=args.mesh_shape)


def upload_matmul_weight(
    weight_hf: torch.Tensor,
    mesh_device,
    args: Devstral2Args,
    *,
    dtype: ttnn.DataType,
    shard_dim: Optional[int],
    weight_cache_path: Optional[PathLike] = None,
    cache_key: str,
) -> ttnn.Tensor:
    """Upload HF linear weight ``(out, in)`` as TT ``(in, out)`` with TILE in DRAM.

    ``shard_dim`` is in TT ``(in, out)`` orientation: ``-1`` column-parallel, ``-2`` row-parallel,
    ``None`` replicate.
    """
    torch_dtype = torch_default_dtype_for(dtype)
    w = weight_hf.to(torch_dtype).T.contiguous()
    if shard_dim is None:
        mapper: Optional[ttnn.TensorToMesh] = ttnn.ReplicateTensorToMesh(mesh_device)
    elif shard_dim == -1:
        mapper = _mesh_mapper_col_parallel(mesh_device, args)
    elif shard_dim == -2:
        mapper = _mesh_mapper_row_parallel(mesh_device, args)
    else:
        raise ValueError(f"Unsupported shard_dim {shard_dim}")

    tt = ttnn.as_tensor(
        w,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=MATMUL_WEIGHT_MEM_CONFIG,
        mesh_mapper=mapper,
        cache_file_name=weight_cache_file(weight_cache_path, cache_key),
    )
    return materialize_tile_layout(tt)


def upload_replicated_tile(
    tensor: torch.Tensor,
    mesh_device,
    *,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
    weight_cache_path: Optional[PathLike] = None,
    cache_key: str,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> ttnn.Tensor:
    """Replicated upload (norm gamma, RoPE tables, trans_mat)."""
    torch_dtype = torch_default_dtype_for(dtype)
    host = tensor.to(torch_dtype).contiguous()
    tt = ttnn.as_tensor(
        host,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=weight_cache_file(weight_cache_path, cache_key),
    )
    if layout == ttnn.TILE_LAYOUT:
        return materialize_tile_layout(tt)
    return tt


def upload_embedding_weight(
    weight_hf: torch.Tensor,
    mesh_device,
    *,
    dtype: ttnn.DataType,
    weight_cache_path: Optional[PathLike] = None,
    cache_key: str = "model_embed_tokens_weight",
) -> ttnn.Tensor:
    """``(vocab, hidden)`` embedding table — ROW_MAJOR in DRAM (``ttnn.embedding`` expectation)."""
    torch_dtype = torch_default_dtype_for(dtype)
    w = weight_hf.to(torch_dtype).contiguous()
    return ttnn.as_tensor(
        w,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=EMBED_WEIGHT_MEM_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=weight_cache_file(weight_cache_path, cache_key),
    )


def upload_kv_cache_buffer(
    shape: tuple[int, ...],
    mesh_device,
    *,
    dtype: ttnn.DataType,
    weight_cache_path: Optional[PathLike] = None,
    cache_key: str,
) -> ttnn.Tensor:
    """Zero-init KV cache in DRAM TILE (materialized at upload)."""
    torch_dtype = torch_default_dtype_for(dtype)
    zeros = torch.zeros(shape, dtype=torch_dtype)
    tt = ttnn.as_tensor(
        zeros,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=KV_CACHE_MEM_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=weight_cache_file(weight_cache_path, cache_key),
    )
    return materialize_tile_layout(tt)


def upload_paged_kv_cache_buffer(
    *,
    num_total_blocks: int,
    n_kv_heads: int,
    block_size: int,
    head_dim: int,
    mesh_device,
    dtype: ttnn.DataType,
    weight_cache_path: Optional[PathLike] = None,
    cache_key: str,
) -> ttnn.Tensor:
    """Zero-init **paged** KV cache in DRAM TILE.

    Layout: ``[num_total_blocks, n_kv_heads, block_size, head_dim]``. Each block stores
    ``block_size`` consecutive KV positions for one logical user; the user-to-block mapping
    is the page_table uploaded by :func:`upload_page_table`.
    """
    torch_dtype = torch_default_dtype_for(dtype)
    zeros = torch.zeros((num_total_blocks, n_kv_heads, block_size, head_dim), dtype=torch_dtype)
    tt = ttnn.as_tensor(
        zeros,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=KV_CACHE_MEM_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=weight_cache_file(weight_cache_path, cache_key),
    )
    return materialize_tile_layout(tt)


def pad_page_table_cols_to_multiple_of_8(
    table: torch.Tensor,
    *,
    pad_value: int = 0,
) -> torch.Tensor:
    """Pad page-table columns so flexible chunked SDPA stick size is a multiple of 32 bytes.

    Chunked SDPA with ``chunk_start_idx_tensor`` uses ``cols * sizeof(int32)`` as the stick size;
    hardware requires that to be divisible by 32, i.e. column count must be a multiple of 8.
    """
    if table.ndim != 2:
        raise ValueError(f"expected 2D page table, got shape={tuple(table.shape)}")
    cols = table.shape[1]
    cols_padded = ((cols + 7) // 8) * 8
    if cols_padded == cols:
        return table
    padded = torch.full((table.shape[0], cols_padded), pad_value, dtype=table.dtype)
    padded[:, :cols] = table
    return padded


def upload_page_table(
    *,
    batch_size: int,
    num_blocks_per_user: int,
    mesh_device,
    weight_cache_path: Optional[PathLike] = None,
    cache_key: str = "kv_page_table_pad8",
) -> ttnn.Tensor:
    """Upload a contiguous page table ``[batch, num_blocks_per_user]`` (int32, DRAM ROW_MAJOR).

    Each entry is the physical block id a logical (user, block_position) maps to. For the simple
    single-user demo with ``batch=1``, ``num_blocks_per_user=12``::

        page_table = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

    Multi-user case lays each user's blocks contiguously::

        page_table = [[0..11], [12..23], [24..35], ...]

    Column count is padded up to a multiple of 8 when needed (flexible chunked SDPA alignment).
    """
    # Caller should pass ``kv_page_table_blocks_per_user`` (already rounded up to a multiple of 8).
    block_ids = torch.arange(batch_size * num_blocks_per_user, dtype=torch.int32)
    page_table = block_ids.reshape(batch_size, num_blocks_per_user).contiguous()
    page_table = pad_page_table_cols_to_multiple_of_8(page_table)  # no-op when already aligned
    return ttnn.as_tensor(
        page_table,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=weight_cache_file(weight_cache_path, cache_key),
    )

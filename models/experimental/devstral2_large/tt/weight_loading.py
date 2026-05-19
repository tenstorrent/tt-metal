# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host → device weight upload for Devstral-2 / Ministral3.

Weight flow (tests / inference setup)
------------------------------------
1. **Download** — ``tests/_devstral_weights.load_hf_tensors_for_keys`` pulls safetensor shards from
   the Hub, dequantizes FP8 weights to **bf16 CPU** tensors keyed like HF (``model.layers.0...``).
2. **HF reference** — the same dict is copied into ``Ministral3Model`` for PCC (host only).
3. **TT upload** — each ``Tt*`` module reads from ``state_dict``, transposes for ``ttnn.linear`` /
   ``ttnn.embedding``, and uploads via :func:`upload_*` helpers below.
4. **Mesh mapping** — ``ShardTensor2dMesh`` for TP (column / row parallel) or
   ``ReplicateTensorToMesh`` for replicated norms / RoPE tables.
5. **Tile materialization** — :func:`materialize_tile_layout` runs ``ttnn.to_layout(TILE)`` once at
   upload so matmul/norm do not enqueue ``TilizeDeviceOperation`` on the first forward pass.

Matmul weights use **DRAM + TILE**. Embedding lookup weights stay **DRAM + ROW_MAJOR** (see
``ttnn.embedding``). RoPE tables use **L1 + TILE** because prefill slices and decode RoPE read them
from L1.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch
import ttnn

from models.experimental.devstral2_large.tt.model_args import Devstral2Args, torch_default_dtype_for

PathLike = Union[str, Path]

DEVSTRAL2_LARGE_REPO_ID = "mistralai/Devstral-2-123B-Instruct-2512"

# Default on-disk cache for tiled device weights (override root with ``TT_CACHE_PATH``).
_DEFAULT_CACHE_ROOT = Path("generated/ttnn/devstral2_large/weight_cache")

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
        base = Path(root) / "devstral2_large" if root else _DEFAULT_CACHE_ROOT
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

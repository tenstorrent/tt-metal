# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh-aware tensor upload helpers and ``MESH_DEVICE`` topology for Voxtral TT modules."""

from __future__ import annotations

import os
from typing import Any

import torch
import ttnn

# ``MESH_DEVICE`` names → compute mesh (rows, cols). Matches tt_transformers / vLLM conventions.
_MESH_DEVICE_TO_COMPUTE_SHAPE: dict[str, tuple[int, int]] = {
    "P100": (1, 1),
    "P150": (1, 1),
    "P150X4": (1, 4),
    "P150X8": (1, 8),
}


def voxtral_mesh_device_compute_shape(mesh_device: str | None = None) -> tuple[int, int]:
    """Resolve Voxtral compute mesh shape from ``MESH_DEVICE`` (default ``(1, 1)`` when unset).

    ``P150`` → 1×1 single-device compute; ``P150x4`` → 1×4 tensor-parallel text on QB2.
    Safe to call before opening a device.
    """
    raw = mesh_device if mesh_device is not None else os.getenv("MESH_DEVICE", "").strip()
    if not raw:
        return (1, 1)
    key = raw.upper()
    shape = _MESH_DEVICE_TO_COMPUTE_SHAPE.get(key)
    if shape is not None:
        return shape
    if "X" in key:
        base, _, count_str = key.rpartition("X")
        if base and count_str.isdigit():
            count = int(count_str)
            return (1, 1) if count == 1 else (1, count)
    raise ValueError(
        f"Unsupported MESH_DEVICE={raw!r} for Voxtral. " "Use P150 (1×1) or P150x4 (1×4 tensor-parallel text on QB2)."
    )


def voxtral_is_multi_device_mesh(mesh_device: Any) -> bool:
    """True when ``mesh_device`` is a mesh with more than one rank."""
    if hasattr(mesh_device, "get_num_devices"):
        return int(mesh_device.get_num_devices()) > 1
    return False


def voxtral_replicate_mesh_mapper(mesh_device: Any):
    """``ReplicateTensorToMesh`` for multi-device meshes; ``None`` on single device."""
    if voxtral_is_multi_device_mesh(mesh_device):
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return None


def voxtral_tp_shard_last_dim_mapper(mesh_device: Any, cluster_shape: tuple[int, int] | Any):
    """Column-shard the last dim across the mesh (tensor-parallel text activations)."""
    if voxtral_is_multi_device_mesh(mesh_device):
        return ttnn.ShardTensor2dMesh(
            mesh_device=mesh_device,
            dims=(None, -1),
            mesh_shape=cluster_shape,
        )
    return None


def voxtral_tp_shard_dim3_mapper(mesh_device: Any, cluster_shape: tuple[int, int] | Any):
    """Column-shard dim 3 (matches ``tok_embeddings`` / ``Embedding`` weight layout)."""
    if voxtral_is_multi_device_mesh(mesh_device):
        return ttnn.ShardTensor2dMesh(
            mesh_device=mesh_device,
            dims=(None, 3),
            mesh_shape=cluster_shape,
        )
    return None


def voxtral_device_embed_dim(full_dim: int, num_devices: int) -> int:
    """Hidden width stored on each device (full ``dim`` when replicated, ``dim // N`` when TP)."""
    if num_devices > 1:
        return int(full_dim // num_devices)
    return int(full_dim)


def voxtral_to_torch_replicated(tensor: ttnn.Tensor) -> torch.Tensor:
    """Host readback from mesh rank 0 (replicated tensors are identical on every device)."""
    dev_tensors = ttnn.get_device_tensors(tensor)
    return ttnn.to_torch(dev_tensors[0] if len(dev_tensors) > 1 else tensor)


def voxtral_from_torch(
    tensor: torch.Tensor,
    mesh_device: Any,
    *,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
    layout: ttnn.Layout | None = None,
) -> ttnn.Tensor:
    """Upload host torch → TTNN, replicating across mesh ranks when needed."""
    kwargs: dict[str, Any] = {
        "device": mesh_device,
        "dtype": dtype,
        "memory_config": memory_config,
        "mesh_mapper": voxtral_replicate_mesh_mapper(mesh_device),
    }
    if layout is not None:
        kwargs["layout"] = layout
    return ttnn.from_torch(tensor, **kwargs)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device weight helpers: mesh mappers for a 1xN mesh and cached ``ttnn.as_tensor`` with dtype/TP-tagged names."""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn


def tp_of(mesh_device) -> int:
    return tuple(mesh_device.shape)[1] if hasattr(mesh_device, "shape") else 1


def shard_mapper(mesh_device, dim: int | None):
    """Shard along ``dim`` across the mesh columns (TP axis); ``None`` replicates."""
    if mesh_device.get_num_devices() == 1:
        return None
    if dim is None:
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.ShardTensor2dMesh(mesh_device, dims=(None, dim), mesh_shape=tuple(mesh_device.shape))


def dtype_tag(dtype) -> str:
    return {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4", ttnn.float32: "f32"}.get(
        dtype, str(dtype)
    )


def as_device_tensor(
    mesh_device,
    host: torch.Tensor | None,
    *,
    name: str,
    dtype,
    shard_dim: int | None,
    cache_path: Path | None,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Upload (or load from the tensorbin cache) one weight. ``host`` may be None only when the cache is complete."""
    cache_file = None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / f"{name}.tp{tp_of(mesh_device)}.{dtype_tag(dtype)}"
    if host is None:
        if cache_file is None:
            raise ValueError(f"{name}: no host tensor and no cache path")
        return ttnn.load_tensor(
            Path(f"{cache_file}_dtype_{dtype.name}_layout_{layout.name}.tensorbin"), device=mesh_device
        )
    return ttnn.as_tensor(
        host.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=shard_mapper(mesh_device, shard_dim),
        cache_file_name=cache_file,
    )


def linear_weight(w: torch.Tensor) -> torch.Tensor:
    """torch ``Linear.weight`` [out, in] -> matmul layout [in, out] (4D for TILE safety)."""
    return w.transpose(-2, -1).contiguous()

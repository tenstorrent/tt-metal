# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from models.demos.dots_ocr.tt._ttnn_import import get_ttnn


def build_rope_cos_sin_1d_ttnn(
    *,
    mesh_device,
    max_grid_size: int,
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build 1D RoPE cos/sin tables on device using TTNN ops, then return them as torch tensors.

    Returns:
      cos_1d: [max_grid_size, head_dim/2]
      sin_1d: [max_grid_size, head_dim/2]

    Note: the gather/indexing to construct per-token 2D vision RoPE is still done on host.
    This helper ensures the expensive trig table generation is driven by TTNN.
    """
    ttnn = get_ttnn()
    if ttnn is None or mesh_device is None:
        raise RuntimeError("build_rope_cos_sin_1d_ttnn requires ttnn and mesh_device")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    half = head_dim // 2
    # inv_freq[j] = 1 / theta^(2j/head_dim)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))  # [half]

    # positions: [max_grid, 1], inv_freq: [1, half] -> freqs: [max_grid, half]
    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
    pos = ttnn.arange(
        0, max_grid_size, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT, memory_config=mem
    )
    pos = ttnn.reshape(pos, (1, 1, max_grid_size, 1))
    inv = ttnn.from_torch(
        inv_freq.to(torch.bfloat16).reshape(1, 1, 1, half),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem,
        mesh_mapper=mapper,
    )
    freqs = ttnn.matmul(pos, inv, memory_config=mem)
    cos = ttnn.cos(freqs, memory_config=mem)
    sin = ttnn.sin(freqs, memory_config=mem)

    cos_t = ttnn.to_torch(cos).reshape(max_grid_size, half)
    sin_t = ttnn.to_torch(sin).reshape(max_grid_size, half)
    return cos_t, sin_t


__all__ = ["build_rope_cos_sin_1d_ttnn"]

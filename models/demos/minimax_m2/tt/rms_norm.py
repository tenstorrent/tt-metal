# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm for MiniMax-M2.5 — thin wrapper around ttnn.rms_norm.

Weight reshape: [H] → [1, 1, H//TILE_SIZE, TILE_SIZE] for ttnn.rms_norm.
Supports both single device and MeshDevice (with optional mesh_mapper).
"""

import torch

import ttnn


class TtRMSNorm:
    """RMSNorm backed by ttnn.rms_norm. Stateless forward (no state between calls)."""

    def __init__(self, device, weight: torch.Tensor, eps: float = 1e-6, mesh_mapper=None, cache_path=None):
        """
        Args:
            device:      TTNN device or MeshDevice
            weight:      1-D torch tensor of shape [normalized_shape]
            eps:         epsilon for numerical stability
            mesh_mapper: optional ttnn mesh mapper for multi-device (e.g. ReplicateTensorToMesh)
                         When None and device is a MeshDevice, defaults to Replicate.
            cache_path:  Path for weight caching (speeds up subsequent runs)
        """
        self.eps = eps

        # Determine mesh mapper: replicate norm weights across all devices by default
        if mesh_mapper is None and isinstance(device, ttnn.MeshDevice):
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)

        # Reshape weight to [1, 1, H//TILE_SIZE, TILE_SIZE] as required by ttnn.rms_norm
        w = weight.reshape(1, 1, -1, ttnn.TILE_SIZE).to(torch.bfloat16)
        self.tt_weight = ttnn.as_tensor(
            w,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
            cache_file_name=cache_path,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.eps)

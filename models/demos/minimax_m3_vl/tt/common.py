# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Shared host-side helpers for the MiniMax-M3-VL ttnn modules.

Small utilities that every submodule needs: the mesh mapper, the
torch->ttnn Linear weight/bias converters, the head-dim tile rounding, and
the standard compute-kernel config. Centralised here so the modules don't
each copy them.
"""
from __future__ import annotations

import torch

import ttnn


def next_tile_multiple(x: int, tile: int = ttnn.TILE_SIZE) -> int:
    """Round up to the next multiple of the tile size (32)."""
    return ((x + tile - 1) // tile) * tile


def mesh_mapper(device):
    """Replicate-to-mesh mapper for a MeshDevice, else None (single device)."""
    return ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None


def mesh_composer(device, dim: int = 0):
    """Concat-from-mesh composer for a MeshDevice, else None."""
    return ttnn.ConcatMeshToTensor(device, dim=dim) if isinstance(device, ttnn.MeshDevice) else None


def as_linear_weight(mesh_device, torch_w: torch.Tensor, dtype) -> ttnn.Tensor:
    """torch nn.Linear weight [out, in] -> ttnn tile tensor [in, out] (operand-b convention)."""
    w = torch_w.detach().to(torch.bfloat16).transpose(-2, -1).contiguous()
    return ttnn.as_tensor(
        w,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper(mesh_device),
    )


def as_linear_bias(mesh_device, torch_b: torch.Tensor, dtype) -> ttnn.Tensor:
    """torch bias [out] -> ttnn tile tensor [1, 1, 1, out], replicated."""
    b = torch_b.detach().to(torch.bfloat16).view(1, 1, 1, -1).contiguous()
    return ttnn.as_tensor(
        b,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper(mesh_device),
    )


def hifi4_compute_config(fp32_dest_acc: bool = True):
    """Standard HiFi4 compute-kernel config. Matmul-bearing ops use fp32_dest_acc=True;
    LayerNorm uses False (matches the original per-module settings)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=False,
    )

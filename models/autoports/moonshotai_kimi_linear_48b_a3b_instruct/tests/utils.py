# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

import ttnn


def pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    g, a = golden.float().flatten(), actual.float().flatten()
    assert g.numel() == a.numel(), (golden.shape, actual.shape)
    if torch.allclose(g, a):
        return 1.0
    g = g - g.mean()
    a = a - a.mean()
    denom = g.norm() * a.norm()
    return float((g @ a) / denom) if denom > 0 else float("nan")


def assert_pcc(golden, actual, threshold: float, name: str = "") -> float:
    p = pcc(golden, actual)
    print(f"[pcc] {name}: {p:.6f} (need >= {threshold})")
    assert p >= threshold, f"{name}: PCC {p:.6f} < {threshold}"
    return p


def replicated(
    mesh_device, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh_device, memory_config=memory_config, mesh_mapper=mapper
    )


def first_shard(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def gather_dim(t: ttnn.Tensor, mesh_device, dim: int) -> torch.Tensor:
    if mesh_device.get_num_devices() == 1:
        return ttnn.to_torch(t)
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))


def gather_conv_carry(conv: ttnn.Tensor, tp: int, q_dim: int, k_dim: int, v_dim: int) -> torch.Tensor:
    """Per-device carries [1,3,q_loc+k_loc+v_loc] (channels grouped by rank) -> global [1,3,q+k+v]."""
    shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(conv)]
    if len(shards) == 1:
        return shards[0]
    ql, kl, vl = q_dim // tp, k_dim // tp, v_dim // tp
    q = torch.cat([s[..., :ql] for s in shards], -1)
    k = torch.cat([s[..., ql : ql + kl] for s in shards], -1)
    v = torch.cat([s[..., ql + kl : ql + kl + vl] for s in shards], -1)
    return torch.cat([q, k, v], -1)

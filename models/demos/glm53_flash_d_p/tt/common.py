# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared device helpers for GLM-5.3-Flash: compute config, replicated upload / readback (harness boundary only)."""

import torch

import ttnn


def hifi4_config(fp32_acc: bool = True):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=False
    )


def replicate(mesh, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    """Host tensor -> replicated device tensor (load time or the harness boundary, never inside a forward)."""
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def replicated_to_host(t: ttnn.Tensor) -> torch.Tensor:
    """Replicated device tensor -> host (chip 0's copy; harness boundary only)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


# ---- residual layout (GLM_RESIDUAL_LAYOUT): "split" (default) or "replicated"
# split: chip d = 2 r + c (row-major over the 2x2 mesh) holds rows [d S/4, (d + 1) S/4) of every per-token tensor
# between the attention / FFN modules (the indexer / MLA query split); replicated: every chip holds all S rows.
LAYOUTS = ("split", "replicated")
MC = ttnn.DRAM_MEMORY_CONFIG


def residual_layout() -> str:
    import os

    mode = os.environ.get("GLM_RESIDUAL_LAYOUT", "split")
    assert mode in LAYOUTS, f"GLM_RESIDUAL_LAYOUT={mode!r}, want one of {LAYOUTS}"
    return mode


def local_rows(t: ttnn.Tensor, dim: int = -2) -> ttnn.Tensor:
    """Replicated rows -> this chip's split quarter (no CCL): mesh_partition on axis 0, then axis 1."""
    a = ttnn.mesh_partition(t, dim=dim, cluster_axis=0, memory_config=MC)
    b = ttnn.mesh_partition(a, dim=dim, cluster_axis=1, memory_config=MC)
    ttnn.deallocate(a)
    return b


def gather_half(t: ttnn.Tensor) -> ttnn.Tensor:
    """Split quarter -> mesh row r's half [r S/2, (r + 1) S/2) on both of its chips (all_gather on axis 1)."""
    return ttnn.all_gather(t, dim=-2, cluster_axis=1, memory_config=MC)


def gather_rows(t: ttnn.Tensor) -> ttnn.Tensor:
    """Split quarter -> all S rows on every chip (all_gather on axis 1, then axis 0)."""
    h = gather_half(t)
    out = ttnn.all_gather(h, dim=-2, cluster_axis=0, memory_config=MC)
    ttnn.deallocate(h)
    return out


def scatter_rows(t: ttnn.Tensor) -> ttnn.Tensor:
    """Per-chip partial sums over all S rows -> the split quarter of the 4-chip sum (reduce_scatter on axis 0, then
    axis 1): half the bytes of an all_reduce, and no slice afterwards."""
    a = ttnn.reduce_scatter(t, dim=-2, cluster_axis=0, memory_config=MC)
    b = ttnn.reduce_scatter(a, dim=-2, cluster_axis=1, memory_config=MC)
    ttnn.deallocate(a)
    return b


def split_to_host(t: ttnn.Tensor) -> torch.Tensor:
    """Split device tensor -> host rows in order (chip d's quarter is rows d S/4 ..; harness boundary only)."""
    return torch.cat([ttnn.to_torch(p) for p in ttnn.get_device_tensors(t)], dim=-2)


def split_from_host(mesh, t: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Host [1, 1, S, W] -> the split layout (harness boundary only)."""
    s, w = t.shape[-2], t.shape[-1]
    rows = tuple(mesh.shape)
    return ttnn.from_torch(
        t.reshape(rows[0], rows[1], s // (rows[0] * rows[1]), w).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=MC,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=rows),
    )

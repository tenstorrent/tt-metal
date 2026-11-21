# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The residual layout on the 8x4 mesh and its host boundary.

Axis 0 (rows) = SP: each mesh row holds S/8 consecutive rows of the chunk. Axis 1 (columns) = TP: each chip holds
H/4 hidden columns. A per-chip tensor is [1, 1, S/8, H/4], bf16, TILE, DRAM interleaved.

These helpers are the test harness's host boundary only; no module forward calls them.
"""

import torch

import ttnn

SP_AXIS, TP_AXIS = 0, 1


def residual_to_device(x: torch.Tensor, mesh, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """[S, H] host -> [1, 1, S/8, H/4] per chip (rows over axis 0, columns over axis 1)."""
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape[-2:]),
        device=mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 3)),
    )


def residual_to_host(t: ttnn.Tensor, mesh) -> torch.Tensor:
    """Inverse of residual_to_device: [S, H] fp32."""
    out = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 3)))
    return out.reshape(out.shape[-2:]).float()


def router_to_device(r: torch.Tensor, mesh) -> ttnn.Tensor:
    """Inverse of router_to_host: dense routing [S, E] host -> [1, 1, S/8, E] fp32 per chip (rows over axis 0,
    replicated over axis 1), as TtRouter writes it."""
    return ttnn.from_torch(
        r.float().reshape(1, 1, *r.shape[-2:]),
        device=mesh,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None)),
    )


def router_to_host(t: ttnn.Tensor, mesh) -> torch.Tensor:
    """The router's dense weights, rows over axis 0 and replicated over axis 1: [S, E] fp32 from TP column 0."""
    out = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 0)))
    return out[0].reshape(-1, out.shape[-1]).float()


def topk_to_host(t: ttnn.Tensor, mesh) -> torch.Tensor:
    """The indexer's top-k, chip (r, c) holding query rows (4 r + c) * S/32 + [0, S/32): [S, k] int64, pads -1."""
    out = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(0, 1)))
    out = out.reshape(-1, out.shape[-1]).to(torch.int64) & 0xFFFFFFFF
    return torch.where(out >= 0xFFFFFFFF, torch.full_like(out, -1), out)


def index_cache_load(mod, keys: torch.Tensor | None, length: int, mesh, slot: int = 0) -> None:
    """Writes the index keys of positions [0, length) (a golden prefix, [>= length, D] host) into the module's served
    cache, one chunk at a time through the module's own cache write, so they land where the module writes its own."""
    if not length:
        return
    assert keys is not None and keys.shape[0] >= length, "an index-key prefix is required past position 0"
    assert length % mod.chunk == 0, f"prefix {length} is not whole chunks of {mod.chunk}"
    for a in range(0, length, mod.chunk):
        k = ttnn.from_torch(
            keys[a : a + mod.chunk].float().reshape(1, 1, mod.chunk, -1),
            device=mesh,
            dtype=mod.cache.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None)),
        )
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            mod.cache,
            k,
            slot_idx=slot,
            layer_idx=mod.cache_layer,
            num_layers=mod.cache_layers,
            kv_actual_global=a,
            cluster_axis=SP_AXIS,
        )
        ttnn.deallocate(k)

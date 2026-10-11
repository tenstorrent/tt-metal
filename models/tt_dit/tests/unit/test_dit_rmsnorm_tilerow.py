# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""dit_fused_distributed_rmsnorm with `affine_tile_row_map`: mapped weight/bias tables must match the per-token tables
bit for bit, also on a program-cache hit that binds a different map buffer."""

from __future__ import annotations

import pytest
import torch

import ttnn

from ...parallel.manager import CCLManager
from ...utils.tensor import bf16_tensor, from_torch
from ...utils.test import ring_params

TILE = 32
EPS = 1e-6


def _expand(table: torch.Tensor, tile_map: torch.Tensor) -> torch.Tensor:
    """Per-token [1, 1, rows, dim] table whose tile row r is tile row `tile_map[r]` of `table` [1, 1, K * 32, dim]."""
    tiles = table.reshape(-1, TILE, table.shape[-1])
    return tiles[tile_map].reshape(1, 1, tile_map.numel() * TILE, table.shape[-1])


def _reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return normed * weight.float() + bias.float()


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param((4, 8), ring_params, id="galaxy_tp4")],
    indirect=["mesh_device", "device_params"],
)
def test_affine_tile_row_map(mesh_device):
    torch.manual_seed(0)
    tp_axis = 0
    tp = tuple(mesh_device.shape)[tp_axis]
    rows, dim, table_tiles = 2048, 256 * tp, 6
    num_tile_rows = rows // TILE

    x = torch.randn((1, 1, rows, dim), dtype=torch.bfloat16)
    table_w = torch.randn((1, 1, table_tiles * TILE, dim), dtype=torch.bfloat16)
    table_b = torch.randn((1, 1, table_tiles * TILE, dim), dtype=torch.bfloat16)
    maps = [torch.randint(0, table_tiles, (num_tile_rows,)), torch.randint(0, table_tiles, (num_tile_rows,))]
    assert not torch.equal(maps[0], maps[1])

    x_tt = bf16_tensor(x, device=mesh_device, mesh_axis=tp_axis, shard_dim=-1)
    table_w_tt = bf16_tensor(table_w, device=mesh_device, mesh_axis=tp_axis, shard_dim=-1)
    table_b_tt = bf16_tensor(table_b, device=mesh_device, mesh_axis=tp_axis, shard_dim=-1)
    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    sem = ccl.get_ag_ping_pong_semaphore(tp_axis)

    def run(weight, bias, tile_map=None):
        pob = ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
            x_tt, tp_axis, mesh_device, num_heads_per_device=1, per_head_norm=False, num_links=2, weight=weight
        )
        extra = {} if tile_map is None else {"affine_tile_row_map": tile_map}
        out = ttnn.experimental.dit_fused_distributed_rmsnorm(
            x_tt,
            tp_axis,
            mesh_device,
            sem,
            topology=ttnn.Topology.Ring,
            epsilon=EPS,
            weight=weight,
            bias=bias,
            persistent_output_buffer=pob,
            num_preferred_links=2,
            **extra,
        )
        ttnn.synchronize_device(mesh_device)
        full = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[-1, 0], mesh_shape=tuple(mesh_device.shape))
        )
        for replica in range(1, full.shape[0]):
            assert torch.equal(full[replica], full[0]), f"replica {replica} differs from replica 0"
        return full[:1]

    def per_token_and_mapped(index: int):
        tile_map = maps[index]
        weight = _expand(table_w, tile_map)
        bias = _expand(table_b, tile_map)
        w_tt = bf16_tensor(weight, device=mesh_device, mesh_axis=tp_axis, shard_dim=-1)
        b_tt = bf16_tensor(bias, device=mesh_device, mesh_axis=tp_axis, shard_dim=-1)
        map_tt = from_torch(
            tile_map.to(torch.int32).reshape(1, 1, 1, num_tile_rows),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
        )
        expected = _reference(x, weight, bias)
        per_token = run(w_tt, b_tt)
        mapped = run(table_w_tt, table_b_tt, map_tt)
        assert _pcc(expected, per_token.float()) >= 0.999
        assert torch.equal(mapped, per_token), f"map {index}: mapped weight/bias differ from the per-token tables"
        return mapped

    mesh_device.enable_program_cache()
    per_token_and_mapped(0)
    entries = mesh_device.num_program_cache_entries()
    for index in (1, 0, 1):
        per_token_and_mapped(index)
        assert mesh_device.num_program_cache_entries() == entries

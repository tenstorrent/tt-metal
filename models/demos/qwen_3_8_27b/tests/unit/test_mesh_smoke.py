# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh prerequisite: the 8x4 mesh opens, MeshConfig + CCLManager construct, and all-gather /
all-reduce agree with torch on both mesh axes."""

import torch

import ttnn
from models.demos.qwen_3_8_27b.tt.mesh import fabric_name


def test_mesh_smoke(mesh, mesh_config, ccl_manager):
    sp, tp = mesh_config.sp, mesh_config.tp
    torch.manual_seed(0)
    x = torch.randn(sp, 1, 64, 128 * tp)
    tt_x = ttnn.from_torch(
        x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_config.shard(0, 3)
    )

    # all-gather over TP (dim 3) then SP (dim 0): every device should then hold the full x
    g = mesh_config.all_gather_tp(tt_x, dim=3)
    g = mesh_config.all_gather(g, dim=0, axis=mesh_config.sp_axis)
    for dev_t in ttnn.get_device_tensors(g):
        assert torch.allclose(ttnn.to_torch(dev_t).float(), x.bfloat16().float())

    # all-reduce over TP: sum of the tp column shards
    r = mesh_config.all_reduce_tp(tt_x)
    got = ttnn.to_torch(r, mesh_composer=mesh_config.compose(0, 3))  # [sp, 1, 64, 128*tp] (tp copies)
    shards = x.bfloat16().float().reshape(sp, 1, 64, tp, 128)
    want = shards.sum(dim=3)
    for c in range(tp):
        assert torch.allclose(got[..., c * 128 : (c + 1) * 128].float(), want, atol=0.1, rtol=0.02)

    # all-reduce over SP (axis 0)
    r2 = ttnn.all_reduce(tt_x, cluster_axis=0, topology=mesh_config.topology, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got2 = ttnn.to_torch(r2, mesh_composer=mesh_config.compose(0, 3))
    want2 = x.bfloat16().float().sum(dim=0, keepdim=True)
    for row in range(sp):
        assert torch.allclose(got2[row : row + 1].float(), want2, atol=0.2, rtol=0.02)
    print(f"mesh smoke ok on fabric={fabric_name()}")

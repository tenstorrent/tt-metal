# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Column-split Engram: forward_v2_own (kv matmul over the group + all_to_all) vs the gather path of prefill_model (all_gather x, 8 forward_v2, reduce_scatter)."""
import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards

T = 32


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_engram_own(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    chain = DSV41DecodeChain(md, users_per_row=4, max_comp=256)
    mc, cc = chain.mesh_config, chain.ccl
    eng = DSV41DeviceEngram(md, 1, _Shards(), mesh_config=mc, ccl=cc)
    D, kin = 5120, eng.kin
    torch.manual_seed(0)
    xh = torch.randn(cols * T, 1, 4, D) * 0.5
    rh = torch.randn(1, 1, cols * T, kin).to(torch.bfloat16)
    x = ttnn.from_torch(
        xh,
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 0), mesh_shape=(rows, cols)),
    )
    rg = ttnn.from_torch(
        rh,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    sl = lambda c: ttnn.slice(rg, [0, 0, c * T, 0], [1, 1, (c + 1) * T, kin])

    def old():
        xg = mc.allgather(x, cc, axis=1, dim=0)
        outs = [eng.forward_v2(ttnn.slice(xg, [j * T, 0, 0, 0], [(j + 1) * T, 1, 4, D]), sl(j)) for j in range(cols)]
        cat8 = ttnn.concat(outs, dim=0)
        rs8 = ttnn.experimental.reduce_scatter_minimal_async(
            cat8,
            dim=0,
            multi_device_global_semaphore=cc.get_rs_ping_pong_semaphore(),
            num_links=cc.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=cc.topology,
            cluster_axis=1,
            barrier_semaphore=cc.get_barrier_semaphore(),
        )
        return ttnn.multiply(rs8, 1.0 / cols)

    def new():
        return eng.forward_v2_own(x, rg)

    for it in range(2):
        o_old, o_new = old(), new()
        ttnn.synchronize_device(md)
        devs_o = ttnn.get_device_tensors(o_old)
        devs_n = ttnn.get_device_tensors(o_new)
        res = []
        for r in range(rows):
            for c in range(cols):
                a = ttnn.to_torch(devs_o[r * cols + c]).float()
                b = ttnn.to_torch(devs_n[r * cols + c]).float()
                res.append((r, c, pcc(a, b), float((a - b).abs().max())))
        worst = min(res, key=lambda t: t[2])
        print(
            f"ENGRAM_OWN iter {it}: min pcc {worst[2]:.6f} at (row {worst[0]}, col {worst[1]}); max abs diff {max(t[3] for t in res):.5f}; col0 {res[0][2]:.6f} col3 {res[3][2]:.6f}",
            flush=True,
        )

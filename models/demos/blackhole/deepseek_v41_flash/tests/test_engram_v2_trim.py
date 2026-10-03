# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""forward_v2 with rows [1,1,T,Kin] (M=T matmul) vs forward (rows [T,1,1,Kin]): PCC and chain timing."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tests.test_engram_v2 import PARAMS, D, chain_ms
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram, _MappedTable
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_engram_v2_trim(mesh_device):
    md = mesh_device
    rows_, cols_ = tuple(md.shape)
    sh = _Shards()
    cfg, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    eng = DSV41DeviceEngram(md, 1, sh, mesh_config=cfg, ccl=ccl)
    table = _MappedTable(sh, 1)
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows_, cols_))
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard
    )
    torch.manual_seed(0)
    nrows = table.w.shape[0]
    for T in (4, 8, 16, 32):
        worst = 1.0
        for trial in range(4):
            xs = torch.randn(rows_ * T, 1, 4, D) * (torch.exp(torch.randn(rows_ * T, 1, 4, 1)) if trial % 2 else 1)
            rw = (
                table.rows(torch.randint(0, nrows, (rows_ * T, eng.kin // 256))).reshape(rows_ * T, 1, 1, eng.kin)
                if trial >= 2
                else torch.randn(rows_ * T, 1, 1, eng.kin).to(torch.bfloat16)
            )
            x = up(xs)
            r_old = up(rw, ttnn.bfloat16)
            r_new = (
                up(rw.reshape(rows_, T, eng.kin).reshape(rows_, 1, T, eng.kin), ttnn.bfloat16)
                if False
                else ttnn.from_torch(
                    rw.reshape(rows_ * T, 1, 1, eng.kin)
                    .reshape(rows_, 1, T, eng.kin)
                    .repeat(1, 1, 1, 1)
                    .reshape(rows_, 1, T, eng.kin),
                    device=md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=shard,
                )
            )
            ref = ttnn.to_torch(ttnn.get_device_tensors(eng.forward(x, r_old))[0]).float()
            new = ttnn.to_torch(ttnn.get_device_tensors(eng.forward_v2(x, r_new))[0]).float()
            worst = min(worst, R.pcc(ref, new))
        print(f"TRIM T={T:2d} PCC vs forward (rows [1,1,T,Kin]) worst {worst:.9f}", flush=True)
        assert worst > 0.9999
        x, r_old = up(torch.randn(rows_ * T, 1, 4, D)), up(torch.randn(rows_ * T, 1, 1, eng.kin), ttnn.bfloat16)
        r_new = ttnn.from_torch(
            torch.randn(rows_, 1, T, eng.kin),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        print(
            f"TRIMT T={T:2d} v2 legacy rows [T,1,1,K] {chain_ms(md, lambda: eng.forward_v2(x, r_old)) * 1e3:8.1f} us",
            flush=True,
        )
        print(
            f"TRIMT T={T:2d} v2 trimmed rows [1,1,T,K] {chain_ms(md, lambda: eng.forward_v2(x, r_new)) * 1e3:8.1f} us",
            flush=True,
        )

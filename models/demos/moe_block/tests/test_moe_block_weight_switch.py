# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MoEBlock with a layer's tensors after another layer's (a model's every MoE layer on one block): program-cache hits
with x / head weights / expert weights at new addresses (allocated after the programs were built, as a model's are)
must give the output of the same values at the build's addresses. GLM-5.3 dims, 8x4, random weights; the combine
snapped to 2^-14 so the outputs compare bit for bit."""

import os

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.moe_block.tests.test_moe_block_smoke import CHUNK, C, R, _flat_weights_module, _glm53
from models.demos.moe_block.tt.moe_block import MoEBlock, MoEBlockWeights

MOCK = "blackhole_galaxy" in os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        (
            (R, C),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
                # (the combine's L1 hand-off: a GLM row (2 x 6144 B) and its trailer in one packet)
                "fabric_router_config": create_fabric_router_config(12352),
                "l1_small_size": 5888,  # (the block's persistent words and semaphores)
            },
        )
    ],
    indirect=True,
)
def test_moe_block_weight_switch(mesh_device):
    mesh = mesh_device
    if MOCK:
        pytest.skip("needs a Galaxy")
    mb = ttnn.experimental.moe_block
    fw = _flat_weights_module()
    dims = _glm53()
    E, H, I, Is, N = dims.num_experts, dims.emb_dim, dims.routed_expert_hidden_dim, 2048, R * C
    EPC = E // N
    dtypes = mb.MoEDtypes()
    res = mb.MoEBlockResources(
        mesh,
        dims,
        mb.MoEBlockConfig(chunk_tokens=CHUNK, num_links=2, combine_snap=2.0**-14, persist_l1_small=True),
        dtypes,
    )
    try:
        block = MoEBlock(mesh, res, dtypes=dtypes, accumulators=11, activation=fw.ACTS["silu"])
        kw = block.expert_kw

        def head(seed):
            g = torch.Generator().manual_seed(seed)
            h = lambda *s, sc=0.02: ttnn.from_torch(torch.randn(*s, generator=g) * sc, dtype=ttnn.float32)
            return mb.prepare_fused_head_weights(
                h(Is, H), h(Is, H), h(H, Is), None, h(E, H, sc=0.05), h(E, sc=0.01),
                resources=res, dtypes=dtypes, accumulators=block.accumulators,
            )  # fmt: skip

        def experts(seed):
            g = torch.Generator().manual_seed(100 + seed)
            r = lambda *s: torch.randn(*s, generator=g) * 0.02  # noqa: E731
            one = [(r(H, I), r(H, I), r(I, H)) for _ in range(EPC)]
            return fw.FlatRoutedExpert(
                mesh, [one] * N, m=kw["max_tokens_per_expert"], H=H, I=I,
                gids=[list(range(c * EPC, (c + 1) * EPC)) for c in range(N)], n_global=E, act="silu", pin=1,
                layout=kw["layout"],
            )  # fmt: skip

        def x_of(seed):
            g = torch.Generator().manual_seed(1000 + seed)
            return ttnn.from_torch(
                torch.randn(1, 1, CHUNK, H, generator=g) * 0.5,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
            )

        comp = ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(R, C), dims=(2, 3))

        def run(x, h, e):
            y = block(x, MoEBlockWeights(head=h, expert_ids=e.gidx, gate_up=e.w_gu, down=e.w_d))
            out = ttnn.to_torch(y, mesh_composer=comp).float()
            ttnn.deallocate(y)
            return out

        x0, h0, e0 = x_of(0), head(0), experts(0)
        x1, h1, e1 = x_of(1), head(1), experts(1)
        want0, want1 = run(x0, h0, e0), run(x1, h1, e1)  # (the build, then a hit with every tensor moved)
        bad = []

        def check(name, got, want):
            diff = int((got != want).sum())
            print(f"MOE_WEIGHT_SWITCH {name}: {diff} values differ")
            if diff:
                bad.append(name)

        check("back to layer 0's tensors", run(x0, h0, e0), want0)
        # (a model prepares each layer's weights, and gets each layer's x, after the programs were built - and frees the
        # x a layer had, whose memory the next tensors reuse: the same values at new addresses, the build's tensors
        # freed and overwritten, must give the same output)
        x0b, h0b, e0b = x_of(0), head(0), experts(0)
        ttnn.deallocate(x0)
        for t in (h0.w, h0.w_router, h0.router_bias, h0.router_index):
            ttnn.deallocate(t)
        for t in (e0.gidx, e0.w_gu, e0.w_d):
            ttnn.deallocate(t)
        junk = (x_of(5), head(5), experts(5))  # noqa: F841 (in the freed memory, other values)
        check("new x, head and expert tensors, the build's freed and overwritten", run(x0b, h0b, e0b), want0)
        check("another layer's tensors after", run(x1, h1, e1), want1)
        check("and back", run(x0b, h0b, e0b), want0)
        x0, h0, e0 = x0b, h0b, e0b
        # (between two MoE layers the model's programs leave their data in every free L1 byte - the CB region and the
        # arena's place: fill all of it with noise, then the same call again)
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        free = int(ttnn.get_memory_view(mesh, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank)
        rows = (free // 2048 - 1) * 32  # (bf16 tiles a core)
        ncores = grid.x * grid.y
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, (rows, 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
        noise = ttnn.from_torch(
            torch.randn(ncores * rows, 32) * 1e3,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(noise)
        check(f"after noise in all free L1 ({rows * 64 >> 10} KB a core)", run(x0, h0, e0), want0)
        check("then once more", run(x0, h0, e0), want0)
        # (what the model's attention adds between two MoE layers, one at a time)
        try:
            g = ttnn.all_gather(x0, 3, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(g)
            check("after an all_gather over TP", run(x0, h0, e0), want0)
        except Exception as e:  # noqa: BLE001 (a probe: report and go on)
            print(f"MOE_WEIGHT_SWITCH all_gather probe failed: {e}"[:300])
        sem = ttnn.create_global_semaphore(mesh, cores, 0)
        ttnn.synchronize_device(mesh)
        del sem
        check("after a global semaphore", run(x0, h0, e0), want0)
        rows_sd = 2
        gather = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, rows_sd - 1))])
        topk = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, rows_sd), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        mid = mesh.create_sub_device_manager([ttnn.SubDevice([topk]), ttnn.SubDevice([gather])], 0)
        ttnn.synchronize_device(mesh)
        check("after creating a sub-device manager", run(x0, h0, e0), want0)
        mesh.load_sub_device_manager(mid)
        mesh.clear_loaded_sub_device_manager()
        ttnn.synchronize_device(mesh)
        check("after loading and clearing it", run(x0, h0, e0), want0)
        mesh.remove_sub_device_manager(mid)
        check("after removing it", run(x0, h0, e0), want0)
        assert not bad, f"a program-cache hit with tensors at new addresses differs: {bad}"
        for e in (e0, e1):
            e.close()
    finally:
        res.release()

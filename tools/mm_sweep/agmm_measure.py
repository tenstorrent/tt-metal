# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Task 4 measurement harness for the fused all_gather_regime_a_matmul_async op.
#
# Runs ONE mode per subprocess (the streaming-vs-full-gather choice is a build-time env flag in create_at, so
# the two program variants must not share a program cache — hence separate processes):
#   stream      : the production progressive-gather path (matmul overlaps the gather).
#   fullgather  : same binary, TT_AGMM_FULL_GATHER=1 -> reader waits for the whole gather before any matmul
#                 (the no-overlap A/B baseline).
#   unfused     : ttnn.all_gather(in0, dim=K) then ttnn.experimental.regime_a_matmul (T_ag + T_mm reference).
#
# Timing: warm up, then time N synchronized launches on the mesh CQ (host wall / N). Semaphores are reset each
# launch (real usage). The stream-vs-fullgather DELTA isolates the overlap benefit; unfused is the AG+MM ref.
# NOTE: host wall is a coarse proxy; per-RISC device-profiler spans (Task 4 full) are a follow-up. This harness
# establishes the overlap A/B and absolute wall numbers with median + spread over relaunches.

import json
import statistics
import sys
import time

import torch
import ttnn
from models.common.utility_functions import comp_pcc

FABRIC = {"stream": ttnn.FabricConfig.FABRIC_1D_RING, "fullgather": ttnn.FabricConfig.FABRIC_1D_RING,
          "unfused": ttnn.FabricConfig.FABRIC_1D_RING}
TOPO = ttnn.Topology.Ring


def run(mode, D, mesh_y, mesh_x, M, K, N, iters=30, warmup=5):
    md = a = b = out = sems = None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    full = ttnn.open_mesh_device(ttnn.MeshShape(mesh_y, mesh_x))
    try:
        md = full.create_submesh(ttnn.MeshShape(1, D))
        grid = md.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        md.load_sub_device_manager(md.create_sub_device_manager([ttnn.SubDevice([crs])], 0))
        md.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

        torch.manual_seed(0)
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        ref = (t0.float() @ t1.float())[0, 0]
        shard = ttnn.create_mesh_mapper(
            md, ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], ttnn.MeshShape(1, D)))
        repl = ttnn.create_mesh_mapper(
            md, ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, D)))
        a = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=md, dtype=ttnn.bfloat16, mesh_mapper=shard)
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, md)
        b = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=md, dtype=ttnn.bfloat16, memory_config=wcfg,
                            mesh_mapper=repl)
        cfg = ttnn.RegimeAMatmulConfig(k_slices=3, n_slices=1, m_slices=1, k_block_tiles=4, n_subblock_tiles=6)
        sems = [ttnn.create_global_semaphore(md, crs, 0) for _ in range(D + 1)]

        def one():
            if mode == "unfused":
                gathered = ttnn.all_gather(a, dim=3, subdevice_id=ttnn.SubDeviceId(0))
                return ttnn.experimental.regime_a_matmul(gathered, b, config=cfg)
            for s in sems:
                ttnn.reset_global_semaphore_value(s, 0)
            return ttnn.experimental.all_gather_regime_a_matmul_async(
                a, b, config=cfg, cluster_axis=1, topology=TOPO, num_links=1, num_workers_per_link=1,
                multi_device_global_semaphore=sems)

        out = one()
        ttnn.synchronize_device(md)
        per = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
        _, pcc = comp_pcc(ref, per[0].float().reshape(ref.shape), 0.999)

        for _ in range(warmup):
            one()
        ttnn.synchronize_device(md)
        walls = []
        for _ in range(iters):
            t = time.perf_counter()
            one()
            ttnn.synchronize_device(md)
            walls.append((time.perf_counter() - t) * 1e6)
        walls.sort()
        print("MEASJSON " + json.dumps({
            "mode": mode, "D": D, "shape": [M, K, N], "pcc": float(pcc),
            "median_us": statistics.median(walls), "min_us": walls[0], "max_us": walls[-1],
            "p10_us": walls[max(0, len(walls) // 10)], "iters": iters}))
    finally:
        a = b = out = md = sems = None
        ttnn.close_mesh_device(full)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    mode = sys.argv[1]
    D = int(sys.argv[2])
    my, mx = (int(sys.argv[3]), int(sys.argv[4]))
    M, K, N = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    run(mode, D, my, mx, M, K, N)

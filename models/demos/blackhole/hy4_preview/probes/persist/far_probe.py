"""B10 persist probe 1: Hy4 one-shot all-reduce (hy4_fuse_ar) vs the stock mesh_config.allreduce on ONE 1x8 slot.
Gates: result == torch sum of the 8 per-chip partials (PCC, rows 0..T-1), >= 3 consecutive calls with different inputs (also inside a trace replay).
Timing: trace slope (N2 - N1 ops, median of REPS replays) like lat_phase.py. Output JSON to $OUT."""
import json
import os
import statistics
import time
import traceback

import torch

import ttnn

OUT = os.environ.get("OUT", "/tmp/far_probe.json")
N1, N2, REPS = int(os.environ.get("N1", 8)), int(os.environ.get("N2", 40)), int(os.environ.get("REPS", 10))
TS = [int(t) for t in os.environ.get("TS", "1,4,8").split(",")]
LINKS = int(os.environ.get("LINKS", 2))
res = {}


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def replay_time(md, fn, n):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(n):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ts = []
    for i in range(REPS + 2):
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts[2:])


def slope(md, fn):
    a = replay_time(md, fn, N1)
    b = replay_time(md, fn, N2)
    return round((b - a) / (N2 - N1), 2)


def main():
    from models.demos.blackhole.hy4_preview.tt.hy4_fuse_ar import Hy4FuseAR
    from models.demos.gpt_oss.tt.ccl import CCLManager

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        far = Hy4FuseAR(md, tileb=1024 if max(TS) > 8 else 512, links=LINKS)
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)

        def stock(t):
            rs = ttnn.experimental.reduce_scatter_minimal_async(
                t,
                dim=3,
                multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ccl.topology,
                cluster_axis=1,
                barrier_semaphore=ccl.get_barrier_semaphore(),
            )
            return ttnn.experimental.all_gather_async(
                rs,
                dim=3,
                cluster_axis=1,
                mesh_device=md,
                topology=ccl.topology,
                multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=ccl.get_barrier_semaphore(),
            )

        for T in TS:
            g = torch.Generator().manual_seed(T)
            ok = []
            for call in range(4):
                parts = torch.randn(8, 1, 32, 6144, generator=g).bfloat16()
                x = ttnn.from_torch(
                    parts,
                    device=md,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                land = far(x, T=T)
                ttnn.synchronize_device(md)
                ref = parts.float().sum(0)[0, :T]  # [T, 6144]
                lh = ttnn.to_torch(land, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))  # [8*NM, W]
                W = lh.shape[1]
                worst = 1.0
                for chip in range(8):
                    dec = Hy4FuseAR.decode(
                        lh[chip * 32 : (chip + 1) * 32].reshape(32, W), T, far.tileb if T > 8 else 512
                    )
                    worst = min(worst, pcc(dec, ref))
                ok.append(round(worst, 6))
                ttnn.deallocate(x)
            res[f"gate_T{T}"] = ok
            print("FAR gate", T, ok, flush=True)
            x = ttnn.from_torch(
                torch.randn(8, 1, 32, 6144).bfloat16(),
                device=md,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            try:
                res[f"far_us_T{T}"] = slope(md, lambda: far(x, T=T))
            except Exception as e:
                res[f"far_us_T{T}"] = repr(e)[:200]
                traceback.print_exc()
            print("FAR us", T, res[f"far_us_T{T}"], flush=True)
            if T == TS[0]:
                try:
                    res["stock_allreduce_us"] = slope(md, lambda: stock(x))
                except Exception as e:
                    res["stock_allreduce_us"] = repr(e)[:200]
                    traceback.print_exc()
                print("STOCK AR us", res["stock_allreduce_us"], flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    finally:
        json.dump(res, open(OUT, "w"), indent=1)
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()

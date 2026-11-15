"""B9 lat probe 1: dependent-phase cost on ONE 1x8 slot (column axis, 8 chips). Trace replay slope method:
per-op time = (t_replay(N2 ops) - t_replay(N1 ops)) / (N2-N1), median of REPS replays, wall clock incl. sync. Ops are enqueued back to back in one trace
(so the number = op latency + op-to-op gap in a trace, as in the model's decode trace). Output JSON to $OUT. Each case in try/except."""
import json
import os
import statistics
import time
import traceback

import torch

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager

OUT = os.environ.get("OUT", "/tmp/lat_phase.json")
N1, N2, REPS = int(os.environ.get("N1", 8)), int(os.environ.get("N2", 40)), int(os.environ.get("REPS", 10))
CASES = os.environ.get("CASES", "all")
res = {}


def rep(md, x):
    return ttnn.from_torch(
        x,
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def replay_time(md, fn, n):
    fn()
    ttnn.synchronize_device(md)  # compile/warm
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
    return statistics.median(ts[2:]), min(ts[2:])


def case(md, name, fn):
    if CASES != "all" and not any(c in name for c in CASES.split(",")):
        return
    try:
        a, amin = replay_time(md, fn, N1)
        b, bmin = replay_time(md, fn, N2)
        res[name] = dict(
            us_per_op=round((b - a) / (N2 - N1), 2),
            us_per_op_min=round((bmin - amin) / (N2 - N1), 2),
            fixed_us=round(a - N1 * (b - a) / (N2 - N1), 1),
            t_n1=round(a, 1),
            t_n2=round(b, 1),
        )
    except Exception as e:
        res[name] = dict(error=repr(e)[:300])
        traceback.print_exc()
    print("LAT", name, json.dumps(res[name]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        ri = lambda *s: torch.randn(*s).bfloat16()
        for nl in (1, 2):
            for shp in ((8, 8, 32, 32), (8, 8, 32, 544)):
                t = rep(md, ri(*shp))
                case(
                    md,
                    f"C_a2a_{'x'.join(map(str, shp))}_nl{nl}",
                    lambda t=t, nl=nl: ttnn.experimental.all_to_all_async_generic(
                        t, in_dim=1, out_dim=0, num_links=nl, topology=ttnn.Topology.Ring, cluster_axis=1
                    ),
                )
        # all_reduce_async, L1 width sharded (DSV4 microbench recipe), [32,6144] on 20 cores
        M, N, RING = 32, 6144, 8
        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        ncores = int(os.environ.get("ARC", 24))
        npc = N // ncores
        if N % ncores == 0 and npc % 32 == 0:
            crs = ttnn.num_cores_to_corerangeset(ncores, ttnn.CoreCoord(8, 8), row_wise=True)
            in_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, [M, npc], ttnn.ShardOrientation.ROW_MAJOR),
            )
            mid_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, [M, npc * RING], ttnn.ShardOrientation.ROW_MAJOR),
            )
            for links in (1, 2):
                try:
                    xin = ttnn.from_torch(
                        ri(1, 1, M, N),
                        device=md,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=in_mc,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                    )
                    mid = ttnn.from_torch(
                        torch.zeros(1, 1, M, N * RING),
                        device=md,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=mid_mc,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                    )
                    sems = [ttnn.create_global_semaphore(md, all_cores, 0) for _ in range(2)]
                    idx = [0]

                    def run(xin=xin, mid=mid, sems=sems, idx=idx, links=links):
                        i = idx[0]
                        idx[0] ^= 1
                        return ttnn.experimental.all_reduce_async(
                            xin,
                            mid,
                            cluster_axis=1,
                            mesh_device=md,
                            multi_device_global_semaphore=sems[i],
                            memory_config=in_mc,
                            dtype=ttnn.bfloat16,
                            topology=ttnn.Topology.Ring,
                            num_links=links,
                        )

                    case(md, f"C_all_reduce_async_l1sharded_32x6144_c{ncores}_nl{links}", run)
                except Exception as e:
                    res[f"C_all_reduce_async_nl{links}"] = dict(error=repr(e)[:300])
                    print("LAT AR error", repr(e)[:300], flush=True)
        # fused matmul + reduce-scatter vs separate
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        xk = rep(md, ri(1, 1, 32, 768))
        wk = rep(md, ri(1, 1, 768, 6144))

        def sep():
            p = ttnn.matmul(xk, wk)
            return ttnn.experimental.reduce_scatter_minimal_async(
                p,
                dim=3,
                multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ccl.topology,
                cluster_axis=1,
                barrier_semaphore=ccl.get_barrier_semaphore(),
            )

        case(md, "F_matmul_then_rs_32x768x6144_nl2", sep)
        try:
            ib = rep(md, torch.zeros(1, 1, 32, 6144).bfloat16())
            ob = rep(md, torch.zeros(1, 1, 32, 768).bfloat16())

            def fused():
                return ttnn.experimental.matmul_reduce_scatter_async(
                    xk,
                    wk,
                    persistent_intermediate_buffer=ib,
                    persistent_output_buffer=ob,
                    dim=3,
                    multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
                    reduce_scatter_core_grid_offset=(0, 8),
                    barrier_semaphore=ccl.get_barrier_semaphore(),
                    num_links=1,
                    memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ccl.topology,
                    subdevice_id=None,
                    memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                )

            case(md, "F_matmul_rs_fused_32x768x6144_nl1", fused)
        except Exception as e:
            res["F_fused"] = dict(error=repr(e)[:300])
            print("LAT F error", repr(e)[:300], flush=True)
    finally:
        json.dump(res, open(OUT, "w"), indent=1)
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()

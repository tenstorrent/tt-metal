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
        # ---- A. op chain / launch
        for w in (32, 1024, 6144):
            x = rep(md, ri(1, 1, 32, w))
            case(md, f"A_add_dep_w{w}", (lambda x=x: ttnn.add(x, x)))
        x = rep(md, ri(1, 1, 32, 6144))
        w_ = rep(md, ri(1, 1, 6144, 256))
        case(md, "A_matmul_32x6144x256", lambda: ttnn.matmul(x, w_))
        xs = rep(md, ri(1, 1, 32, 32))
        case(md, "A_relu_1tile", lambda: ttnn.relu(xs))
        # ---- C. column-axis CCLs
        for nl in (1, 2):
            ccl = CCLManager(md, num_links=nl, topology=ttnn.Topology.Ring)

            def ag(t, dim, nl=nl, ccl=ccl):
                return ttnn.experimental.all_gather_async(
                    t,
                    dim=dim,
                    cluster_axis=1,
                    mesh_device=md,
                    topology=ccl.topology,
                    multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                    num_links=nl,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    barrier_semaphore=ccl.get_barrier_semaphore(),
                )

            for w in (
                32,
                64,
                256,
                768,
            ):  # per-chip [32,w] tile rows; w=64 -> 4 KB/chip tile pair; w=768 = 6144/8 (hidden AG)
                t = rep(md, ri(1, 1, 32, w))
                case(md, f"C_ag_dim3_tile_w{w}_nl{nl}", lambda t=t, ag=ag: ag(t, 3))
            t = rep(md, ri(1, 8, 4, 576))
            t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
            case(md, f"C_ag_dim2_rm_8x4x576_nl{nl}", lambda t=t, ag=ag: ag(t, 2))
            for T in (32,):
                t = rep(md, ri(1, 1, T, 6144))

                def rs(t=t, nl=nl, ccl=ccl):
                    return ttnn.experimental.reduce_scatter_minimal_async(
                        t,
                        dim=3,
                        multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
                        num_links=nl,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        topology=ccl.topology,
                        cluster_axis=1,
                        barrier_semaphore=ccl.get_barrier_semaphore(),
                    )

                case(md, f"C_rs_6144_tile32_nl{nl}", rs)
                case(md, f"C_allreduce_rs+ag_6144_tile32_nl{nl}", lambda rs=rs, ag=ag: ag(rs(), 3))
            for shp in ((1, 8, 32, 64), (1, 8, 32, 512)):
                t = rep(md, ri(*shp))
                case(
                    md,
                    f"C_a2a_{'x'.join(map(str, shp))}_nl{nl}",
                    lambda t=t, nl=nl: ttnn.experimental.all_to_all_async_generic(
                        t, in_dim=1, out_dim=0, num_links=nl, topology=ttnn.Topology.Ring, cluster_axis=1
                    ),
                )
        # ---- fused CCL+op variants (best effort)
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        try:
            xa = rep(md, ri(1, 1, 32, 768))
            wm = rep(md, ri(1, 1, 6144, 256))

            def agmm():
                return ttnn.experimental.all_gather_matmul_async(
                    xa,
                    wm,
                    dim=3,
                    multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                    cluster_axis=1,
                    mesh_device=md,
                    topology=ccl.topology,
                    num_links=2,
                    memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                    memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                )

            case(md, "F_ag_matmul_async_32x6144x256_nl2", agmm)
        except Exception as e:
            res["F_ag_matmul_async"] = dict(error=repr(e)[:300])
            print("LAT F error", repr(e)[:300], flush=True)
    finally:
        json.dump(res, open(OUT, "w"), indent=1)
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()

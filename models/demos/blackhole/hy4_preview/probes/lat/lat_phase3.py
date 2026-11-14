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
        x = rep(md, ri(1, 1, 32, 1024))
        xs = rep(md, ri(1, 1, 32, 32))
        w = rep(md, torch.ones(1, 1, 1, 1024).bfloat16())
        x8 = rep(md, ri(1, 1, 8, 1024))
        x16 = ttnn.typecast(x, ttnn.bfloat16)
        # A2: glue ops of the decode layer (dependent-free, back to back in a trace: launch + gap + tiny work)
        case(md, "A2_typecast_bf16_to_bf8_32x1024", lambda: ttnn.typecast(x, ttnn.bfloat8_b))
        case(md, "A2_slice_32x1024_to_32x512", lambda: ttnn.slice(x, [0, 0, 0, 0], [1, 1, 32, 512]))
        case(md, "A2_reshape_view", lambda: ttnn.reshape(x, [1, 1, 1024, 32]))
        case(md, "A2_to_layout_tile_to_rm", lambda: ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT))
        case(
            md,
            "A2_rms_norm_32x1024",
            lambda: ttnn.rms_norm(x, weight=ttnn.to_layout(w, ttnn.TILE_LAYOUT) if False else None, epsilon=1e-6),
        )
        case(md, "A2_multiply_32x1024", lambda: ttnn.multiply(x, x))
        case(md, "A2_silu_32x1024", lambda: ttnn.silu(x))
        case(md, "A2_concat_2x32x1024", lambda: ttnn.concat([x, x], dim=3))
        case(md, "A2_sum_dim3_32x1024", lambda: ttnn.sum(x, dim=3, keepdim=True))
        case(md, "A2_clone_32x1024", lambda: ttnn.clone(x))
        case(md, "A2_to_l1_32x1024", lambda: ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG))
        # core-range launch: eltwise on L1 height-sharded tensors over 1 / 8 / 64 cores
        for nc in (1, 8, 64):
            crs = ttnn.num_cores_to_corerangeset(nc, ttnn.CoreCoord(8, 8), row_wise=True)
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, [32, 64 * 32 // nc if nc <= 64 else 32], ttnn.ShardOrientation.ROW_MAJOR),
            )
            try:
                t = ttnn.from_torch(
                    ri(1, 1, 32, 64 * 32),
                    device=md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mc,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                )
                case(md, f"A3_add_l1_sharded_{nc}cores", lambda t=t, mc=mc: ttnn.add(t, t, memory_config=mc))
            except Exception as e:
                res[f"A3_{nc}"] = dict(error=repr(e)[:200])
        # repeatability of the AG numbers (fresh process, other order)
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        for nl in (2, 1):
            ccl = CCLManager(md, num_links=nl, topology=ttnn.Topology.Ring)
            for wd in (768, 256, 64, 32):
                t = rep(md, ri(1, 1, 32, wd))
                case(
                    md,
                    f"R_ag_dim3_tile_w{wd}_nl{nl}",
                    lambda t=t, nl=nl, ccl=ccl: ttnn.experimental.all_gather_async(
                        t,
                        dim=3,
                        cluster_axis=1,
                        mesh_device=md,
                        topology=ccl.topology,
                        multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                        num_links=nl,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        barrier_semaphore=ccl.get_barrier_semaphore(),
                    ),
                )
    finally:
        json.dump(res, open(OUT, "w"), indent=1)
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()

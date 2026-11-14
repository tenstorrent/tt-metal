"""verify-rows: per-stage trace timing of the decode select pipeline (select_v2) at S query rows, local n rows per column. Env: NL (local rows, default 131072 = 1M ctx), SS, B/KB/KLOC/SPLIT."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_idx_select as SEL
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.gpt_oss.tt.ccl import CCLManager

K = 2048


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def traced_us(md, fn, n=7, reps=3):
    def body():
        for _ in range(reps):
            o = fn()
            for t in o if isinstance(o, (tuple, list)) else [o]:
                if isinstance(t, ttnn.Tensor):
                    ttnn.deallocate(t)

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / reps * 1e6)
    ttnn.release_trace(md, tid)
    return round(statistics.median(ts), 1)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=400_000_000)
    try:
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        n = int(os.environ.get("NL", "131072"))
        B = int(os.environ.get("B", "16"))
        kb = int(os.environ.get("KB", "512"))
        kloc = int(os.environ.get("KLOC", "512"))
        split = int(os.environ.get("SPLIT", "4"))
        colid = upload(md, HP.colid_host(1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, shard_dim=1, row_dim=0)

        def ag(t):
            return ttnn.experimental.all_gather_async(
                t,
                dim=3,
                cluster_axis=1,
                mesh_device=md,
                topology=ttnn.Topology.Ring,
                multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=ccl.get_barrier_semaphore(),
            )

        for S in [int(x) for x in os.environ.get("SS", "1,2,4,8").split(",")]:
            rows = ttnn.from_torch(
                torch.randn(1, 1, S, n).bfloat16(),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            m = n // B
            r = dict(S=S, n=n, B=B, kb=kb, kloc=kloc, split=split)
            blk = ttnn.reshape(rows, [1, 1, S * B, m])
            r["a_topk_blk"] = traced_us(md, lambda: SEL.topk_ids(blk, kb, m))
            bids = SEL.topk_ids(blk, kb, m)
            cand = lambda: SEL.cs_cand2(
                ttnn.reshape(rows, [S * B, 1, 1, m]), ttnn.reshape(bids, [S * B, 1, 1, kb]), colid, 1, m, B, split
            )
            outs = {}
            for rm in ("0", "1"):
                os.environ["HY4_CS_ROW"] = rm
                r["b_cand2_row%s" % rm] = traced_us(md, cand)
                v_, g_ = cand()
                outs[rm] = (
                    ttnn.to_torch(v_, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)),
                    ttnn.to_torch(g_, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)),
                )
            r["cand2_identical"] = bool(
                torch.equal(outs["0"][0], outs["1"][0]) and torch.equal(outs["0"][1], outs["1"][1])
            )
            os.environ["HY4_CS_ROW"] = "1"
            bvals, bgids = SEL.cs_cand2(
                ttnn.reshape(rows, [S * B, 1, 1, m]), ttnn.reshape(bids, [S * B, 1, 1, kb]), colid, 1, m, B, split
            )
            cvals = ttnn.reshape(bvals, [1, 1, S, B * kb])
            r["c_topk_cand"] = traced_us(md, lambda: SEL.topk_ids(cvals, kloc))
            pos = SEL.topk_ids(cvals, kloc)
            args = (
                ttnn.reshape(pos, [S, 1, 1, kloc]),
                ttnn.reshape(bgids, [S, 1, 1, B * kb]),
                ttnn.reshape(bvals, [S, 1, 1, B * kb]),
                split,
            )
            r["d_gather_loc"] = traced_us(md, lambda: SEL.cs_gather(*args))
            lids, lvals = SEL.cs_gather(*args)
            r["e_ag_vals"] = traced_us(md, lambda: ag(lvals))
            r["e_ag_ids"] = traced_us(md, lambda: ag(lids))
            av, ai = ag(lvals), ag(lids)
            W = int(av.shape[-1])
            avr = ttnn.reshape(av, [1, 1, S, W])
            r["f_topk_final"] = traced_us(md, lambda: SEL.topk_ids(avr, K))
            pos2 = ttnn.reshape(SEL.topk_ids(avr, K), [S, 1, 1, K])
            r["g_gather_final"] = traced_us(md, lambda: SEL.cs_gather(pos2, ai, None, split))
            for rm in ("0", "1"):
                os.environ["HY4_CS_ROW"] = rm
                r["total_row%s" % rm] = traced_us(
                    md, lambda: SEL.select_v2(ag, colid, rows, n, B, kb, kloc, K, split, False, None)
                )
            R(**r)
            ttnn.deallocate(rows)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()

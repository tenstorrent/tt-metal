# SPDX-License-Identifier: Apache-2.0
"""tp32-prefill2 whole-mesh block test: Hy4MoeTp32PrefillBlock (router -> AG rows -> plan -> K1b/K2b -> RS cols -> RS rows) + shared expert vs the
real layer-1 MoE output (acts_l0_l1.pt) as oracle; per-stage eager timing and a traced device-only time (fixed plan).
Env: HY4_LAYER=1 N=512 (tokens per row, 4 rows = 4 users) PM_REPS=2 TIME=1 TRACE=1"""
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, open_mesh, p, pcc
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp24 import Tp24Experts
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32_prefill_block import Hy4MoeTp32PrefillBlock
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_router import Hy4Gate
from models.demos.blackhole.hy4_preview.tt.hy4_tp32_prefill import PlanTensors
from models.demos.blackhole.hy4_preview.tt.hy4_tp32pf_plan import build_plan
from models.demos.gpt_oss.tt.ccl import CCLManager

LAYER = int(os.environ.get("HY4_LAYER", "1"))
N = int(os.environ.get("N", "512"))
REPS = int(os.environ.get("PM_REPS", "2"))
HID = 6144
ROWS, COLS = 4, 8

acts = torch.load(ACTS + "acts_l0_l1.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]])
Y = torch.cat([y.reshape(-1, HID) for h, y, r in acts["moe"]])
md = open_mesh(900_000_000)
try:
    W = load_moe_layer(LAYER, load_experts=False)
    p("layer weights loaded", list(W.keys())[:8])
    gate = Hy4Gate(md, W["gate_weight"], W["gate_bias"])
    shared_w = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
    shared = D.make_shared(md, *shared_w, decode=False)
    tpe = Tp24Experts(md, layer=LAYER)
    p("tp32 weights from cache:", tpe.from_cache)
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    blk = Hy4MoeTp32PrefillBlock(md, ccl, gate, tpe, N)
    for rep in range(REPS):
        idx = torch.stack([(torch.arange(N) + 80 * r + 17 * rep) % X.shape[0] for r in range(ROWS)])
        x = X[idx.flatten()].reshape(ROWS, N, HID).to(torch.bfloat16)
        xd = ttnn.from_torch(
            x.reshape(ROWS, 1, N, HID),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS)),
        )
        t0 = time.time()
        y2 = blk.forward(xd)  # [1,1,N,D/8] per chip
        ttnn.synchronize_device(md)
        p("rep", rep, "forward eager (incl host plan) s", round(time.time() - t0, 2))
        sh = D.allreduce_cols(shared.forward(xd))
        ttnn.synchronize_device(md)
        full = torch.zeros(ROWS, N, HID)
        devs = ttnn.get_device_tensors(y2)
        for r in range(ROWS):
            for c in range(COLS):
                full[r, :, c * 768 : (c + 1) * 768] = ttnn.to_torch(devs[r * COLS + c]).float().reshape(N, 768)
        shf = torch.stack(
            [ttnn.to_torch(ttnn.get_device_tensors(sh)[r * COLS]).float().reshape(N, HID) for r in range(ROWS)]
        )
        ref = Y[idx.flatten()].reshape(ROWS, N, HID)
        pr = [pcc(full[r] + shf[r], ref[r]) for r in range(ROWS)]
        p(
            "rep",
            rep,
            "PCC routed+shared vs oracle per row",
            [round(v, 5) for v in pr],
            "routed-only",
            [round(pcc(full[r], ref[r]), 4) for r in range(ROWS)],
            "norms",
            [float(full[r].norm()) for r in range(ROWS)],
            [float(ref[r].norm()) for r in range(ROWS)],
        )
        ids, scs = blk.last
        if os.environ.get("TIME", "1") == "1":
            # eager per-stage times with the plan kept fixed
            pl = build_plan(ids, scs, nc=blk.nc)
            pt = PlanTensors(md, pl, mapper=blk.repl)
            act, out = blk.ex.alloc(blk.tt)
            h_rm = ttnn.to_layout(xd, ttnn.ROW_MAJOR_LAYOUT)

            def stage(fn, n=5):
                ts = []
                for _ in range(n):
                    ttnn.synchronize_device(md)
                    t0 = time.perf_counter()
                    r_ = fn()
                    ttnn.synchronize_device(md)
                    ts.append((time.perf_counter() - t0) * 1e6)
                    if r_ is not None:
                        ttnn.deallocate(r_)
                return statistics.median(ts)

            x_all = blk._ag_rows(h_rm, 2)
            res = {}
            res["ag_x_us"] = stage(lambda: blk._ag_rows(h_rm, 2))
            res["k1_us"] = stage(lambda: blk.ex.k1(x_all, pt, act))
            res["k2x6_us"] = stage(lambda: [blk.ex.k2(S, pt, act, out, blk.tt) for S in range(6)] and None)
            res["rs_cols_us"] = stage(lambda: blk._rs(out, 3, 1, ttnn.Topology.Ring))
            y_ = blk._rs(out, 3, 1, ttnn.Topology.Ring)
            res["rs_rows_us"] = stage(lambda: blk._rs(y_, 2, 0, ttnn.Topology.Linear))
            p(
                "rep",
                rep,
                "stage us (eager, synced)",
                {k: round(v, 1) for k, v in res.items()},
                "sum",
                round(sum(res.values()), 1),
            )
            if os.environ.get("TRACE", "1") == "1":

                def body():
                    xa = blk._ag_rows(h_rm, 2)
                    blk.ex(xa, pt, act, out, blk.tt)
                    a_ = blk._rs(out, 3, 1, ttnn.Topology.Ring)
                    b_ = blk._rs(a_, 2, 0, ttnn.Topology.Linear)
                    ttnn.deallocate(xa)
                    ttnn.deallocate(a_)
                    ttnn.deallocate(b_)

                body()
                ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                body()
                ttnn.end_trace_capture(md, tid, cq_id=0)
                ttnn.synchronize_device(md)
                ts = []
                for _ in range(6):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(md)
                    ts.append((time.perf_counter() - t0) * 1e6)
                ttnn.release_trace(md, tid)
                p(
                    "rep",
                    rep,
                    "TRACED device-only us per layer-chunk (AG x + K1 + K2x6 + RS cols + RS rows, no router/shared)",
                    round(statistics.median(ts), 1),
                )
finally:
    close_mesh(md)

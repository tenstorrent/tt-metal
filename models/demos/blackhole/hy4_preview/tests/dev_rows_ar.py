# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rows-phase one-shot reduce (Hy4RowsReduce): correctness over >= 5 varying-input calls + traced us. Mesh 1x8 (single slot, groups of 4 chips of the row; RP_MESH=1x8) or 4x8/4x2 (axis0).
Env: RP_T (1), RP_H (2), RP_NRP (16), RP_CALLS (6), RP_TRACE (1), OUT json."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import p, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_rows_ar import D, Hy4RowsReduce

T = int(os.environ.get("RP_T", "1"))
H = int(os.environ.get("RP_H", "2"))
NRP = int(os.environ.get("RP_NRP", "16"))
CALLS = int(os.environ.get("RP_CALLS", "6"))
OUT = os.environ.get("OUT", "/tmp/rows_ar.json")
shape = tuple(int(v) for v in os.environ.get("RP_MESH", "1x8").split("x"))
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if shape == (1, 8) else ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(*shape), l1_small_size=16384, trace_region_size=50_000_000)
rows, cols = shape
Tt = 4 * T
res = {"T": T, "H": H, "NRP": NRP, "pcc": [], "maxerr": []}
try:
    rr = Hy4RowsReduce(md, T=T, h=H, nrp=NRP)
    p("built", rr.mode, "Q", rr.Q, "chunk", rr.chunk)
    g = torch.Generator().manual_seed(7)

    def up(P):
        return ttnn.from_torch(
            P,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols)),
        )

    def groups():  # list of lists of (row, col) in position order
        if rr.mode == "axis0":
            return [[(r, c) for r in range(4)] for c in range(cols)]
        return [[(0, 4 * g + k) for k in range(4)] for g in range(cols // 4)]

    for call in range(CALLS):
        P = torch.randn(rows, cols, Tt, D, generator=g).to(torch.bfloat16)
        if os.environ.get("RP_ONLY") == "1" and call < 4:
            m = torch.zeros(rows, cols, 1, 1)
            for grp in groups():
                m[grp[call][0], grp[call][1]] = 1
            P = (P.float() * m).to(torch.bfloat16)
        part = up(P)
        o = rr(part)
        ttnn.synchronize_device(md)
        for grp in groups():
            tot = sum(P[r, c].float() for (r, c) in grp)
            for pos, (r, c) in enumerate(grp):
                got = ttnn.to_torch(ttnn.get_device_tensors(o)[r * cols + c])[0, 0].float()
                exp = tot[pos * T : (pos + 1) * T]
                a, e = got[:T], exp
                rest = float(got[T:].abs().max())
                if pos == 1 and c == 1 and os.environ.get("RP_ONLY") == "1":
                    p(
                        "ONLY",
                        call,
                        "err per tile",
                        [
                            round(float(v), 2)
                            for v in (got[:T] - tot[pos * T : (pos + 1) * T]).abs().reshape(T, 192, 32)[0].amax(1)[:24]
                        ],
                    )
                if call == 0 and os.environ.get("RP_DBG", "1") == "1":
                    G = {k: pcc(got[:T], tot[b * T : (b + 1) * T]) for b in range(4) for k in [("tot", b)]}
                    O = {
                        ("own", k, b): pcc(got[:T], P[g[0], g[1]].float()[b * T : (b + 1) * T])
                        for k, g in enumerate(grp)
                        for b in range(4)
                    }
                    best = sorted(list(G.items()) + list(O.items()), key=lambda kv: -kv[1])[:3]
                    if pos == 1 and c in (1, 5) and os.environ.get("RP_ONLY") == "1":
                        p(
                            "ONLY",
                            call,
                            "err per tile",
                            [
                                round(float(v), 2)
                                for v in (got[:T] - tot[pos * T : (pos + 1) * T])
                                .abs()
                                .reshape(T, 192, 32)[0]
                                .amax(1)[:24]
                            ],
                        )
                    if False:
                        err = (got[:T] - tot[pos * T : (pos + 1) * T]).abs().reshape(T, 192, 32).amax((0, 2))
                        p(
                            "DBG tile-err per reducer chunk",
                            [round(float(err[i * rr.Q : (i + 1) * rr.Q].max()), 2) for i in range(NRP)],
                        )
                        e2 = (got[:T] - tot[pos * T : (pos + 1) * T]).abs().reshape(T, 192, 32)[0]
                        p("DBG err by col-in-tile", [round(float(v), 2) for v in e2.mean(0)])
                        p(
                            "DBG err per tile",
                            [
                                round(float(v), 2)
                                for v in (got[:T] - tot[pos * T : (pos + 1) * T])
                                .abs()
                                .reshape(T, 192, 32)[0]
                                .amax(1)[:48]
                            ],
                        )
                        p("DBG got/exp tile0", got[0, :32].tolist(), tot[pos * T, :32].tolist())
                        # subsets
                        import itertools

                        for k in range(1, 5):
                            for sub in itertools.combinations(range(4), k):
                                s_ = sum(P[grp[j][0], grp[j][1]].float() for j in sub)[pos * T : (pos + 1) * T]
                                v = pcc(got[:T], s_)
                                if v > 0.95:
                                    p("DBG subset match", sub, v)
                    p(
                        "DBG pos",
                        pos,
                        (r, c),
                        "best",
                        [(k, round(v, 4)) for k, v in best],
                        "got[:4,:4]",
                        got[0, :4].tolist(),
                    )
                res["pcc"].append(pcc(a, e))
                res["maxerr"].append(float((a - e).abs().max()))
                if rest != 0.0:
                    p("WARN rows>=T nonzero", rest)
        p("call", call, "pcc min", min(res["pcc"][-cols * 4 :]), "maxerr", max(res["maxerr"][-cols * 4 :]))
        ttnn.deallocate(part)
    if os.environ.get("RP_PROF") == "1":
        ttnn.ReadDeviceProfiler(md)
    res["pcc_min"] = min(res["pcc"])
    res["pcc"] = None
    if os.environ.get("RP_TRACE", "1") == "1":
        part = up(torch.randn(rows, cols, Tt, D, generator=g).to(torch.bfloat16))

        def timed_nosync(n):
            for _ in range(2):
                for _ in range(n):
                    rr(part)
                ttnn.synchronize_device(md)
            ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            for _ in range(n):
                rr(part)
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ts = []
            for _ in range(15):
                ttnn.synchronize_device(md)
                t0 = time.perf_counter()
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                ts.append((time.perf_counter() - t0) * 1e6)
            ttnn.release_trace(md, tid)
            ts.sort()
            return ts[len(ts) // 2]

        if shape != (1, 8):

            def tr(fn, n):
                for _ in range(2):
                    for _ in range(n):
                        fn()
                    ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                for _ in range(n):
                    fn()
                ttnn.end_trace_capture(md, tid, cq_id=0)
                ts = []
                for _ in range(15):
                    ttnn.synchronize_device(md)
                    t0 = time.perf_counter()
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(md)
                    ts.append((time.perf_counter() - t0) * 1e6)
                ttnn.release_trace(md, tid)
                ts.sort()
                return ts[len(ts) // 2]

            small = ttnn.from_torch(
                torch.randn(1, 1, 32, 64).to(torch.bfloat16),
                device=md,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
            res["op_floor_add_us"] = round(
                (tr(lambda: ttnn.add(small, small), 20) - tr(lambda: ttnn.add(small, small), 4)) / 16, 1
            )
            p("op floor (tiny ttnn.add) us/op", res["op_floor_add_us"])
            n2, n10 = timed_nosync(2), timed_nosync(10)
            res["rr_pingpong_us"] = round((n10 - n2) / 8, 1)
            p("TRACE pingpong back-to-back rr: 2 calls", n2, "10 calls", n10, "=> us/call", res["rr_pingpong_us"])
            raise SystemExit
        from models.demos.gpt_oss.tt.ccl import CCLManager

        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        tok = ttnn.from_torch(
            torch.randn(cols, 1, 1, 64).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols))
            if rows > 1
            else ttnn.ShardTensorToMesh(md, dim=0),
        )

        def sync_ag():
            return ttnn.experimental.all_gather_async(
                tok,
                dim=0,
                cluster_axis=1,
                mesh_device=md,
                topology=ttnn.Topology.Ring,
                multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                num_links=ccl.num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=ccl.get_barrier_semaphore(),
            )

        def timed(n, with_rr=True):
            for _ in range(2):  # warm: compile + program cache before the capture (no writes allowed during capture)
                for _ in range(n):
                    if with_rr:
                        rr(part)
                    sync_ag()
                ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            for _ in range(n):
                if with_rr:
                    rr(part)
                sync_ag()
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ts = []
            for _ in range(15):
                ttnn.synchronize_device(md)
                t0 = time.perf_counter()
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                ts.append((time.perf_counter() - t0) * 1e6)
            ttnn.release_trace(md, tid)
            ts.sort()
            return ts[len(ts) // 2]

        if os.environ.get("RP_PROF") == "2":
            for _ in range(3):
                rr(part)
            ttnn.synchronize_device(md)
            ttnn.ReadDeviceProfiler(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            for _ in range(4):
                rr(part)
            ttnn.end_trace_capture(md, tid, cq_id=0)
            for _ in range(3):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
            ttnn.ReadDeviceProfiler(md)
            raise SystemExit
        n2, n10 = timed_nosync(2), timed_nosync(10)
        res["rr_pingpong_us"] = round((n10 - n2) / 8, 1)
        p("TRACE pingpong back-to-back rr: 2 calls", n2, "10 calls", n10, "=> us/call", res["rr_pingpong_us"])
        a1, a5 = timed(1, False), timed(5, False)
        t1, t5 = timed(1), timed(5)
        res["ag_only_slope"] = round((a5 - a1) / 4, 1)
        res["rr_plus_ag_slope"] = round((t5 - t1) / 4, 1)
        res["rr_us"] = round(res["rr_plus_ag_slope"] - res["ag_only_slope"], 1)
        p("TRACE ag-only", a1, a5, "rr+ag", t1, t5, "=> rr us/call", res["rr_us"])
finally:
    json.dump(res, open(OUT, "w"))
    p("RESULT", json.dumps(res))
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

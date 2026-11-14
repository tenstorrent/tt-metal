"""B10 persist probe 2: boundary (b) = one-shot all-reduce of the o-path partial landing in the fused iHC pre kernel (expand + mixes + collapse + norm), 1x8 slot.
Stage 2a: two programs (hy4_fuse_ar, then Hy4IHCFused.pre(land=...)): y never touches DRAM.  Reference path = stock reduce_scatter_minimal_async + all_gather_async -> y tensor ->
Hy4IHCFused.pre(y=...).  Real layer-1 iHC streams (ACTS ihc_l1.pt), random per-chip partials.  Gates: h / x_new / pp PCC vs the reference path >= 0.9999 and vs a torch fp32 sum
for >= 3 consecutive calls with different data (eager and in a replayed trace).  Timing: trace slope us per (AR + iHC call), both paths.  Output JSON to $OUT."""
import json
import os
import statistics
import time
import traceback

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, pcc, rep
from models.demos.blackhole.hy4_preview.tt import hy4_fuse_ar as FAR
from models.demos.blackhole.hy4_preview.tt.hy4_fuse_ar import Hy4FuseAR
from models.demos.blackhole.hy4_preview.tt.hy4_ihc_fused import R0, Hy4IHCFused
from models.demos.gpt_oss.tt.ccl import CCLManager

OUT = os.environ.get("OUT", "/tmp/ihc_far_probe.json")
N1, N2, REPS = int(os.environ.get("N1", 8)), int(os.environ.get("N2", 40)), int(os.environ.get("REPS", 10))
TS = [int(t) for t in os.environ.get("TS", "1,4").split(",")]
HID = 6144
res = {}


def replay_time(md, fn, n):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        for _ in range(n):
            fn()
    finally:
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


def xy(name):
    v = os.environ.get(name)
    return [tuple(int(a) for a in c.split(",")) for c in v.split(";")] if v else None


def cm(name):
    v = xy(name)
    if v is None:
        return None
    keys = [("fwd", 0), ("fwd", 1), ("bwd", 0), ("bwd", 1)]
    return dict(zip(keys, v))


def main():
    R = torch.load(ACTS + "ihc_l1.pt")
    cat = lambda k: torch.cat(
        [t.reshape(-1, *t.shape[2:]) if t.dim() > 3 else t.reshape(-1, t.shape[-1]) for t in R[k]]
    )
    X4 = cat("x4_in")
    hp, nw = R["hc_attn"], R["norm_in"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=200_000_000)
    d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    try:
        fu = Hy4IHCFused(
            md, hp["hc_fn"], hp["hc_base"], hp["hc_scale"], dim=HID, hc_eps=R["cfg"]["hc_eps"], norm_eps=1e-5
        )
        if os.environ.get("CCL_FIRST", "0") == "1":
            ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        far = Hy4FuseAR(
            md,
            tileb=1024 if max(TS) > 8 else 512,
            links=int(os.environ.get("LINKS", 2)),
            ihc_row0=R0,
            nred=int(os.environ.get("NRED", 4)),
            recv_small=os.environ.get("RSMALL", "0") == "1",
            red=xy("RED"),
            comm=cm("COMM"),
        )
        if os.environ.get("CCL_FIRST", "0") != "1":
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
            w4 = fu.norm_weight(nw, T)
            g = torch.Generator().manual_seed(100 + T)
            ok = []
            for call in range(4):
                idx = torch.randperm(X4.shape[0], generator=g)[:T]
                xin = X4[idx].reshape(1, 1, 4 * T, HID)
                parts = (torch.randn(8, 1, 32, HID, generator=g) * 0.05).bfloat16()
                part = ttnn.from_torch(
                    parts,
                    device=md,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                x_a, x_b = rep(md, xin, dtype=ttnn.float32), rep(md, xin, dtype=ttnn.float32)
                x_p = rep(md, xin, dtype=ttnn.float32)
                _, _, pp0 = fu.pre(x_p, w4)  # pp_prev: the mixes of the previous sub-block
                # reference path: stock AR -> y tensor -> pre(y=...)
                y = stock(part)
                h_r, _, pp_r, xn_r = fu.pre(x_a, w4, y=y, pp_prev=pp0, x_out=rep(md, xin, dtype=ttnn.float32))
                # new path: one-shot AR -> landing -> pre(land=...)
                far(part, T=T)
                h_n, _, pp_n, xn_n = fu.pre(
                    x_b,
                    w4,
                    land=(far.out.buffer_address(), FAR.tile_bytes(T)),
                    pp_prev=pp0,
                    x_out=rep(md, xin, dtype=ttnn.float32),
                )
                # stage 2b: AR + iHC in ONE mesh program
                x_c = rep(md, xin, dtype=ttnn.float32)
                h_f, _, pp_f, xn_f = fu.pre(
                    x_c, w4, fuse=(far, part), pp_prev=pp0, x_out=rep(md, xin, dtype=ttnn.float32)
                )
                ttnn.synchronize_device(md)
                ysum = parts.float().sum(0)[0, :T]
                row0 = dict(stock_vs_torch=round(pcc(d0(y).reshape(32, HID).float()[:T], ysum), 6))
                lh = ttnn.to_torch(far.out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
                row0["far_vs_torch"] = round(
                    pcc(Hy4FuseAR.decode(lh[0:32].reshape(32, -1), T, FAR.tile_bytes(T)), ysum), 6
                )
                print("IFAR direct", T, call, row0, flush=True)
                post = d0(pp0).reshape(32, 32).float()[:T, 4:8]
                xn_t = xin.reshape(T, 4, HID).float() + post[:, :, None] * ysum[:, None, :]
                got = d0(xn_n).reshape(T, 4, HID).float()
                ref = d0(xn_r).reshape(T, 4, HID).float()
                row = dict(
                    h_vs_ref=round(pcc(d0(h_n).reshape(T, HID).float(), d0(h_r).reshape(T, HID).float()), 6),
                    xnew_vs_ref=round(pcc(got, ref), 6),
                    xnew_vs_torch=round(pcc(got, xn_t), 6),
                    ref_vs_torch=round(pcc(ref, xn_t), 6),
                    pp=round(
                        pcc(d0(pp_n).reshape(32, 32).float()[:T, :8], d0(pp_r).reshape(32, 32).float()[:T, :8]), 6
                    ),
                )
                row["fused_h_vs_ref"] = round(pcc(d0(h_f).reshape(T, HID).float(), d0(h_r).reshape(T, HID).float()), 6)
                row["fused_xnew_vs_ref"] = round(pcc(d0(xn_f).reshape(T, 4, HID).float(), ref), 6)
                row["fused_xnew_vs_torch"] = round(pcc(d0(xn_f).reshape(T, 4, HID).float(), xn_t), 6)
                ok.append(row)
                print("IFAR gate", T, call, row, flush=True)
            res[f"gate_T{T}"] = ok
            # timing
            part = ttnn.from_torch(
                (torch.randn(8, 1, 32, HID) * 0.05).bfloat16(),
                device=md,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            xs = rep(md, X4[:T].reshape(1, 1, 4 * T, HID), dtype=ttnn.float32)
            _, _, ppT = fu.pre(rep(md, X4[:T].reshape(1, 1, 4 * T, HID), dtype=ttnn.float32), w4)
            land = (far.out.buffer_address(), FAR.tile_bytes(T))

            def ref_chain():
                yy = stock(part)
                return fu.pre(xs, w4, y=yy, pp_prev=ppT, x_out=xs)

            def new_chain():
                far(part, T=T)
                return fu.pre(xs, w4, land=land, pp_prev=ppT, x_out=xs)

            def fused_chain():
                return fu.pre(xs, w4, fuse=(far, part), pp_prev=ppT, x_out=xs)

            for name, fn in (
                ("ihc_only_dram_y", lambda: fu.pre(xs, w4, y=part, pp_prev=ppT, x_out=xs)),
                ("far_only", lambda: far(part, T=T)),
                ("stock_ar_only", lambda: stock(part)),
                ("ihc_only_land", lambda: fu.pre(xs, w4, land=land, pp_prev=ppT, x_out=xs)),
                ("ref_chain_stockAR+ihc", ref_chain),
                ("new_chain_far+ihc", new_chain),
                ("fused_chain_far+ihc_1prog", fused_chain),
            ):
                try:
                    res[f"{name}_us_T{T}"] = slope(md, fn)
                except Exception as e:
                    res[f"{name}_us_T{T}"] = repr(e)[:200]
                    traceback.print_exc()
                print("IFAR us", T, name, res[f"{name}_us_T{T}"], flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
    finally:
        json.dump(res, open(OUT, "w"), indent=1)
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()

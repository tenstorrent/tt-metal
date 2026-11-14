"""B10 persist layer gate (1x8 slot): the attention half of a REAL decode layer with the all-reduce fused into the next iHC pre (HY4_FUSE_AR path), on layer-attn goldens (real activations, real layer weights).
Per position P: context = unfused prefill of the golden activations handed to the pool; then
 (a) base  = decode_step_pool (stock lean all-reduce) -> y_base;  PCC vs the oracle golden out[P]
 (b) fused = decode_step_pool with attn.far_defer (FusedPartial) -> Hy4FuseAR landing; PCC of the landed rows vs y_base on all 8 chips (gate >= 0.9999), >= 3 consecutive calls with the same + different P
 (c) chain: Hy4IHCFused.pre(y=y_base) vs pre(land) on real iHC streams: h / x_new PCC (gate >= 0.9999)
 (d) us per call (4 calls in one trace, median of 9 replays): [layer + stock AR + iHC pre] vs [layer deferred + one-shot AR + iHC pre(land)]
Env: LAYER (1 full | 2 shared), POSITIONS (512,2560), PAGES, BUCKET, OPT. Prints GATE json lines."""
import json
import os
import statistics
import time

import torch

import ttnn

_WNAME = os.environ.pop("WDT", "bfp8")
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS
from models.demos.blackhole.hy4_preview.tests.run_attn_device import load_sd, pcc
from models.demos.blackhole.hy4_preview.tt import hy4_fuse_ar as FAR
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt
from models.demos.blackhole.hy4_preview.tt.hy4_common import to_host, upload
from models.demos.blackhole.hy4_preview.tt.hy4_ihc_fused import Hy4IHCFused
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import Hy4AttentionPool, PoolCtx
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

GOLD = os.environ.get("GOLD", "/gold/golden")
CKPT = os.environ.get("HY4_CKPT", "/ckpt")
L = int(os.environ.get("LAYER", "2"))
POS = [int(x) for x in os.environ.get("POSITIONS", "512,2560").split(",")]
PAGES = int(os.environ.get("PAGES", "8"))
BUCKET = int(os.environ.get("BUCKET", "4096"))
CHUNK = 512
HID = 6144
WDT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[_WNAME]
os.environ["HY4_FUSE_AR"] = "1"


def G(**kw):
    print("GATE " + json.dumps(kw), flush=True)


def traced(md, fn, calls=4, n=9):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        for _ in range(calls):
            fn()
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts) / calls


def main():
    cfg = Hy4Config.from_checkpoint(CKPT)
    g = torch.load(f"{GOLD}/L{L}.pt")
    conv = g["conv"]
    R = torch.load(ACTS + "ihc_l1.pt")
    cat = lambda k: torch.cat(
        [t.reshape(-1, *t.shape[2:]) if t.dim() > 3 else t.reshape(-1, t.shape[-1]) for t in R[k]]
    )
    X4 = cat("x4_in")
    hp, nw = R["hc_attn"], R["norm_in"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=400_000_000)
    d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean

        ccl_lean.make_lean(mc)
        ccl_lean.make_lean_sems(ccl)
        attn = Hy4AttentionPool(
            md,
            mc,
            ccl,
            cfg,
            load_sd(L),
            L,
            kv_fmt="scaled_fp8",
            rope_interleave=conv["rope_interleave"],
            idx_rope_interleave=conv["idx_rope_interleave"],
            wdtype=WDT,
        )
        shared = not attn.has_indexer
        ctx = PoolCtx(md, mc, ccl, PAGES, rows=1, S=1, mode="dense", bucket_tokens=BUCKET)
        kvp = ttnn.empty(
            [1, 1, PAGES * 64, 656],
            dtype=ttnn.fp8_e4m3,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kp = (
            None
            if shared
            else ttnn.empty(
                [PAGES, 1, 64, 128],
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        attn.init_pool(ctx, kvp, kp)
        opt = Opt.parse(os.environ.get("OPT", ""))
        far = FAR.shared(md)  # AFTER the CCLManager and the pool
        fu = Hy4IHCFused(
            md, hp["hc_fn"], hp["hc_base"], hp["hc_scale"], dim=HID, hc_eps=R["cfg"]["hc_eps"], norm_eps=1e-5
        )
        w4 = fu.norm_weight(nw, 1)
        rep = lambda t, dt=ttnn.float32: ttnn.from_torch(
            t,
            device=md,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        built = 0
        for P in sorted(POS):
            while built < P:
                p0 = built
                h = upload(md, g["h"][p0 : p0 + CHUNK].reshape(1, 1, CHUNK, -1).to(torch.bfloat16))
                ids_in = (
                    upload(
                        md,
                        g["idx"][p0 : p0 + CHUNK].to(torch.int32).reshape(1, 1, CHUNK, -1),
                        ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
                    if shared
                    else None
                )
                attn.prefill_chunk(h, p0, ids_in)
                built += CHUNK
            ttnn.synchronize_device(md)
            h = upload(md, g["h"][P].reshape(1, 1, 1, -1).to(torch.bfloat16))
            ids_in = (
                upload(md, g["idx"][P].to(torch.int32).reshape(1, 1, 1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                if shared
                else None
            )
            ref_gold = g["out"][P]
            npg = built // CHUNK
            HP.slice_copy(attn.cache.rows, kvp, ctx.colid, 64, HP.ROWB, npg)
            if not shared:
                kb = ttnn.typecast(attn.indexer.keys, ttnn.bfloat8_b)
                HP.slice_copy(kb, ttnn.reshape(kp, [1, 1, PAGES * 64, 128]), ctx.colid, 8, HP.KEYTILE, npg)
            ttnn.synchronize_device(md)
            snap = (attn.indexer.keys if attn.indexer else None, attn.indexer.length if attn.indexer else 0)
            ctx.mode = "dense" if P + 1 <= cfg.index_topk else "topk"
            ctx.set_step([P])
            tabs = attn.rope_tabs_opt(P, 1)
            if shared:
                gids = upload(md, g["idx"][P].to(torch.int32).reshape(1, 1, 1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                ctx.ids8[attn.spec.indexer_source] = HP.id_filter(gids, ctx.colinfo, 1)
            dense8 = lambda: ctx.dense_ids8() if ctx.mode == "dense" else None

            def layer(defer):
                attn.far_defer = defer
                try:
                    return attn.decode_step_pool(h, opt, tabs, dense8())
                finally:
                    attn.far_defer = False

            y_base = layer(False)
            ttnn.synchronize_device(md)
            yb = to_host(y_base, md).reshape(-1).float()
            rec = dict(layer=L, pos=P, mode=ctx.mode, pcc_base_vs_gold=round(pcc(yb, ref_gold), 6))
            # ---- (b) fused partial -> one-shot AR, 3 consecutive calls (the last two with the buffers reused)
            worst = []
            for call in range(3):
                fp = layer(True)
                assert isinstance(fp, FAR.FusedPartial), type(fp)
                far(fp.part, T=1)
                ttnn.synchronize_device(md)
                lh = ttnn.to_torch(far.out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
                w = 1.0
                for chip in range(8):
                    dec = FAR.Hy4FuseAR.decode(lh[chip * 32 : (chip + 1) * 32].reshape(32, -1), 1, 512)
                    w = min(w, pcc(dec.reshape(-1), yb))
                worst.append(round(w, 6))
                ttnn.deallocate(fp.part)
            rec["pcc_far_landing_vs_base_min_chip"] = worst
            # ---- (c) the chain with the iHC pre
            xin = X4[torch.randperm(X4.shape[0], generator=torch.Generator().manual_seed(P))[:1]].reshape(1, 1, 4, HID)
            _, _, pp0 = fu.pre(rep(xin), w4)
            h_r, _, pp_r, xn_r = fu.pre(rep(xin), w4, y=y_base, pp_prev=pp0, x_out=rep(xin))
            fp = layer(True)
            h_n, _, pp_n, xn_n = FAR.pre_with(fu, far, rep(xin), w4, fp, pp0, rep(xin))
            ttnn.synchronize_device(md)
            rec["pcc_ihc_h_far_vs_stock"] = round(pcc(d0(h_n).reshape(-1).float(), d0(h_r).reshape(-1).float()), 6)
            rec["pcc_ihc_xnew_far_vs_stock"] = round(pcc(d0(xn_n).reshape(-1).float(), d0(xn_r).reshape(-1).float()), 6)
            # ---- (d) timing
            xs = rep(xin)

            def base_chain():
                y = layer(False)
                return fu.pre(xs, w4, y=y, pp_prev=pp0, x_out=xs)

            def far_chain():
                fp = layer(True)
                return FAR.pre_with(fu, far, xs, w4, fp, pp0, xs)

            if os.environ.get("TIMING", "1") == "1":
                rec["us_base_chain_layer+AR+ihc"] = round(traced(md, base_chain), 1)
                rec["us_far_chain_layer+AR+ihc"] = round(traced(md, far_chain), 1)
                rec["us_base_chain_2"] = round(traced(md, base_chain), 1)
                rec["us_far_chain_2"] = round(traced(md, far_chain), 1)
            G(**rec)
            if attn.indexer:
                attn.indexer.keys, attn.indexer.length = snap
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

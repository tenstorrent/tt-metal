# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""layer-other phase 2 gate: Hy4AttentionOpt decode with flag stacks vs (a) the unfused device path (Hy4Attention.decode_step, same weights/CCL/state) and (b) the oracle goldens
(layer-attn goldens, real activations), + us/layer before/after. One 1x8 slot. The context (KV rows + indexer keys of positions < P) is built by the unfused prefill path.
Env: LAYER (1 full | 2 shared), POSITIONS (512,2560), WDT (bf16|bfp8|bfp4), CONFIGS (';'-separated Opt tags, default the stack list), RING/LEAN (1), TIMING (1), S2 (1: S=2 front/back gate + timing).
Prints GATE json lines."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.run_attn_device import load_sd, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Hy4AttentionOpt, Opt
from models.demos.blackhole.hy4_preview.tt.hy4_common import to_host, upload
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

GOLD = os.environ.get("GOLD", "/mnt/tt-data/ssinghal/hy4-logs/layer-attn/golden")
CKPT = os.environ.get("HY4_CKPT", "/mnt/tt-data/Hy4-Preview")
L = int(os.environ.get("LAYER", "1"))
WDT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get("WDT", "bf16")]
POS = [int(x) for x in os.environ.get("POSITIONS", "512,2560").split(",")]
RING = os.environ.get("RING", "1") == "1"
LEAN = os.environ.get("LEAN", "1") == "1"
DEFAULT_CFGS = "base;tune;tune,l1;tune,rmssh;tune,rope1;tune,fuseproj;all"
CFGS = [Opt.parse("" if c == "base" else c) for c in os.environ.get("CONFIGS", DEFAULT_CFGS).split(";")]
CHUNK = 512
_DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
WDTYPES = {
    kv.split("=")[0]: _DT[kv.split("=")[1]] for kv in os.environ.get("WDTYPES", "").split(",") if kv
}  # per-tensor override (abf4), e.g. gate=bfp4,q_a=bfp4,q_b=bfp4


def G(**kw):
    print("GATE " + json.dumps(kw), flush=True)


def traced(md, fn, n=7, reps=1):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / reps * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


def snap(a):
    return (a.cache.rows, a.cache.length, a.indexer.keys if a.indexer else None, a.indexer.length if a.indexer else 0)


def restore(a, s):
    a.cache.rows, a.cache.length = s[0], s[1]
    if a.indexer:
        a.indexer.keys, a.indexer.length = s[2], s[3]


def main():
    cfg = Hy4Config.from_checkpoint(CKPT)
    g = torch.load(f"{GOLD}/L{L}.pt")
    conv = g["conv"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if RING else ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=400_000_000)
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2 if RING else 1, topology=ttnn.Topology.Ring if RING else ttnn.Topology.Linear)
        if LEAN:
            from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean

            ccl_lean.make_lean(mc)
            ccl_lean.make_lean_sems(ccl)
        attn = Hy4AttentionOpt(
            md,
            mc,
            ccl,
            cfg,
            load_sd(L),
            L,
            kv_fmt="bf16",
            rope_interleave=conv["rope_interleave"],
            idx_rope_interleave=conv["idx_rope_interleave"],
            wdtype=WDT,
            wdtypes=WDTYPES or None,
        )
        shared = not attn.has_indexer
        built = 0
        allpcc = []
        for P in sorted(POS):
            while built < P:
                p0 = built
                h = upload(md, g["h"][p0 : p0 + CHUNK].reshape(1, 1, CHUNK, -1).to(torch.bfloat16))
                ids_in = None
                if shared:
                    ids_in = upload(
                        md,
                        g["idx"][p0 : p0 + CHUNK].to(torch.int32).reshape(1, 1, CHUNK, -1),
                        ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT,
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
            s0 = snap(attn)
            ref_gold = g["out"][P]
            attn.taps = None
            out0, _ = attn.decode_step(h, P, ids_in)
            ttnn.synchronize_device(md)
            base = to_host(out0, md).reshape(-1).float()
            restore(attn, s0)
            G(layer=L, pos=P, cfg="unfused decode_step", pcc_gold=round(pcc(base, ref_gold), 6))
            tabs_base = attn.rope_tabs(P, 1)
            dids = attn.dense_ids(P) if (attn.has_indexer and P + 1 <= cfg.index_topk) else None
            t_base = (
                traced(md, lambda: attn.decode_step(h, P, ids_in, append=False, tabs=tabs_base, dense_ids=dids))
                if os.environ.get("TIMING", "1") == "1"
                else None
            )
            G(layer=L, pos=P, cfg="unfused decode_step", us_layer_incl_stock_sparse=t_base)
            for o in CFGS:
                tabs = attn.rope_tabs_opt(P, 1)
                try:
                    out1, ids = attn.decode_step_opt(h, P, ids_in, o, tabs=tabs)
                    ttnn.synchronize_device(md)
                    got = to_host(out1, md).reshape(-1).float()
                    restore(attn, s0)
                    rec = dict(
                        layer=L,
                        pos=P,
                        cfg=o.tag(),
                        wdt=os.environ.get("WDT", "bf16"),
                        pcc_vs_unfused=round(pcc(got, base), 6),
                        pcc_vs_gold=round(pcc(got, ref_gold), 6),
                    )
                    if os.environ.get("TIMING", "1") == "1":
                        dids = attn.dense_ids(P) if (attn.has_indexer and P + 1 <= cfg.index_topk) else None
                        rec["us_layer_incl_stock_sparse"] = round(
                            traced(
                                md,
                                lambda: attn.decode_step_opt(h, P, ids_in, o, append=False, tabs=tabs, dense_ids=dids),
                            ),
                            1,
                        )
                        rec["unfused_us"] = round(t_base, 1)
                    allpcc.append((rec["pcc_vs_unfused"], rec["pcc_vs_gold"]))
                    G(**rec)
                except Exception as e:
                    import traceback

                    restore(attn, s0)
                    G(layer=L, pos=P, cfg=o.tag(), error=repr(e)[:300], tb=traceback.format_exc()[-800:])
        # ---- S = 1 / 2 rows: front + back only (stub o_lat), unfused vs opt, + oracle for qa/lat/k_pe, timing without scan/sparse
        if os.environ.get("S2", "1") == "1":
            P = POS[-1]
            for S in (1, 2):
                hS = upload(md, g["h"][P : P + S].reshape(1, 1, S, -1).to(torch.bfloat16))
                olat = upload(md, torch.randn(1, 8, S, 512) * 0.1)
                base_o = Opt()
                tabs0 = attn.rope_tabs_opt(P, S)

                def fb(o, tabs):
                    d = attn.front(hS, o, tabs)
                    out = attn.back(o, ttnn.clone(olat), d["g"])
                    extra = []
                    if attn.has_indexer:
                        extra = [attn.idx_q(o, d, tabs)]
                    return d, out, extra

                d0, out0, ex0 = fb(base_o, tabs0)
                ttnn.synchronize_device(md)
                ref = {k: to_host(d0[k], md).reshape(S, -1).float() for k in ("qa", "lat", "k_pe", "qfull")}
                refo = to_host(out0, md).reshape(S, -1).float()
                G(
                    S=S,
                    cfg="front/back unfused",
                    pcc_qa_gold=round(pcc(ref["qa"], g["qa"][P : P + S]), 6),
                    pcc_lat_gold=round(pcc(ref["lat"], g["lat"][P : P + S]), 6),
                    pcc_kpe_gold=round(pcc(ref["k_pe"], g["k_pe"][P : P + S]), 6),
                )
                tb = traced(md, lambda: fb(base_o, tabs0))
                G(S=S, cfg="front/back unfused", us=round(tb, 1))
                for o in CFGS[1:]:
                    tabs = attn.rope_tabs_opt(P, S)
                    d1, out1, ex1 = fb(o, tabs)
                    ttnn.synchronize_device(md)
                    pr = {k: round(pcc(to_host(d1[k], md).reshape(S, -1).float(), ref[k]), 6) for k in ref}
                    po = round(pcc(to_host(out1, md).reshape(S, -1).float(), refo), 6)
                    pqa_g = round(pcc(to_host(d1["qa"], md).reshape(S, -1).float(), g["qa"][P : P + S]), 6)
                    t1 = traced(md, lambda: fb(o, tabs))
                    G(
                        S=S,
                        cfg=o.tag(),
                        pcc_vs_unfused=pr,
                        pcc_back_out_vs_unfused=po,
                        pcc_qa_gold=pqa_g,
                        us=round(t1, 1),
                        unfused_us=round(tb, 1),
                    )
        G(
            summary="done",
            min_pcc_vs_unfused=min((a for a, b in allpcc), default=None),
            min_pcc_vs_gold=min((b for a, b in allpcc), default=None),
        )
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

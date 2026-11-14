# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""integ step-1 layer gate (1x8 slot): the POOL decode step of one layer (Hy4AttentionPool.decode_step_pool) vs (a) the unfused device decode (Hy4Attention.decode_step on the replicated SCALED_FP8 cache,
same weights) and (b) the oracle golden out[P]. Context = the unfused prefill of the layer-attn goldens (real activations), handed off to the pool by slice_copy.
Env: LAYER (1 full | 2 shared), POSITIONS (512,2560), OPT (Opt tag, default base), PAGES (8), BUCKET (4096), TIMING (1). Prints GATE json lines."""
import json
import os
import statistics
import time

import torch

import ttnn

_WNAME = os.environ.pop("WDT", "bfp8")
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.run_attn_device import idx_setmatch, load_sd, pcc
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt
from models.demos.blackhole.hy4_preview.tt.hy4_common import to_host, upload
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import Hy4AttentionPool, PoolCtx
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

GOLD = os.environ.get("GOLD", "/gold/golden")
CKPT = os.environ.get("HY4_CKPT", "/ckpt")
L = int(os.environ.get("LAYER", "1"))
POS = [int(x) for x in os.environ.get("POSITIONS", "512,2560").split(",")]
PAGES = int(os.environ.get("PAGES", "8"))
BUCKET = int(os.environ.get("BUCKET", "4096"))
CHUNK = 512
WDT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[_WNAME]


def G(**kw):
    print("GATE " + json.dumps(kw), flush=True)


def traced(md, fn, n=7):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


def main():
    cfg = Hy4Config.from_checkpoint(CKPT)
    g = torch.load(f"{GOLD}/L{L}.pt")
    conv = g["conv"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=400_000_000)
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean

        ccl_lean.make_lean(mc)
        ccl_lean.make_lean_sems(ccl)
        attn = Hy4AttentionPool(
            md, mc, ccl, cfg, load_sd(L), L, kv_fmt="scaled_fp8",
            rope_interleave=conv["rope_interleave"], idx_rope_interleave=conv["idx_rope_interleave"], wdtype=WDT,
        )
        shared = not attn.has_indexer
        ctx = PoolCtx(md, mc, ccl, PAGES, rows=1, S=1, mode="dense", bucket_tokens=BUCKET)
        kvp = ttnn.empty([1, 1, PAGES * 64, 656], dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kp = None if shared else ttnn.empty([PAGES, 1, 64, 128], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.init_pool(ctx, kvp, kp)
        opt = Opt.parse(os.environ.get("OPT", ""))
        built = 0
        for P in sorted(POS):
            while built < P:
                p0 = built
                h = upload(md, g["h"][p0 : p0 + CHUNK].reshape(1, 1, CHUNK, -1).to(torch.bfloat16))
                ids_in = None
                if shared:
                    ids_in = upload(md, g["idx"][p0 : p0 + CHUNK].to(torch.int32).reshape(1, 1, CHUNK, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                attn.prefill_chunk(h, p0, ids_in)
                built += CHUNK
            ttnn.synchronize_device(md)
            h = upload(md, g["h"][P].reshape(1, 1, 1, -1).to(torch.bfloat16))
            ids_in = upload(md, g["idx"][P].to(torch.int32).reshape(1, 1, 1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if shared else None
            ref_gold = g["out"][P]
            # hand-off of the context (positions < built, whole pages) to the pool BEFORE the unfused decode appends its row
            npg = built // CHUNK
            HP.slice_copy(attn.cache.rows, kvp, ctx.colid, 64, HP.ROWB, npg)
            if not shared:
                kb = ttnn.typecast(attn.indexer.keys, ttnn.bfloat8_b)
                HP.slice_copy(kb, ttnn.reshape(kp, [1, 1, PAGES * 64, 128]), ctx.colid, 8, HP.KEYTILE, npg)
            ttnn.synchronize_device(md)
            snap = (attn.cache.rows, attn.cache.length, attn.indexer.keys if attn.indexer else None, attn.indexer.length if attn.indexer else 0)
            out0, ids0 = attn.decode_step(h, P, ids_in)
            ttnn.synchronize_device(md)
            base = to_host(out0, md).reshape(-1).float()
            G(layer=L, pos=P, cfg="unfused decode_step (stock sparse_sdpa, SCALED_FP8)", pcc_gold=round(pcc(base, ref_gold), 6))
            attn.cache.rows, attn.cache.length = snap[0], snap[1]
            if attn.indexer:
                attn.indexer.keys, attn.indexer.length = snap[2], snap[3]
            # ---- pool decode
            ctx.mode = "dense" if P + 1 <= cfg.index_topk else "topk"
            ctx.set_step([P])
            tabs = attn.rope_tabs_opt(P, 1)
            if shared:  # IndexCache: the source layer's ids come from the golden (global natural ids -> local per column by the filter)
                gids = upload(md, g["idx"][P].to(torch.int32).reshape(1, 1, 1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                ctx.ids8[attn.spec.indexer_source] = HP.id_filter(gids, ctx.colinfo, 1)
            dense8 = ctx.dense_ids8() if ctx.mode == "dense" else None
            out1 = attn.decode_step_pool(h, opt, tabs, dense8)
            ttnn.synchronize_device(md)
            got = to_host(out1, md).reshape(-1).float()
            rec = dict(layer=L, pos=P, cfg="pool " + ctx.mode + " " + opt.tag(), pcc_vs_unfused=round(pcc(got, base), 6), pcc_vs_gold=round(pcc(got, ref_gold), 6))
            if not shared and ctx.mode == "topk":
                gi = ctx.ids8[L]
                loc = ttnn.to_torch(ttnn.get_device_tensors(gi)[0]).reshape(8, 2048).to(torch.int64) & 0xFFFFFFFF
                # reconstruct the global set from every column's local ids
                allg = set()
                for c in range(8):
                    li = ttnn.to_torch(ttnn.get_device_tensors(gi)[c]).reshape(8, 2048).to(torch.int64)[0] & 0xFFFFFFFF
                    li = li[li != HP.SENT]
                    pg, sl = li // 64, li % 64
                    allg |= set((pg * 512 + c * 64 + sl).tolist())
                ref = set(g["idx"][P][g["idx"][P] >= 0].tolist())
                rec["idx_overlap_vs_gold"] = round(len(allg & ref) / max(1, len(ref)), 5)
                rec["n_ids"] = len(allg)
            if os.environ.get("TIMING", "1") == "1":
                ctx.set_step([P])
                rec["us_pool_layer"] = round(traced(md, lambda: attn.decode_step_pool(h, opt, tabs, ctx.dense_ids8() if ctx.mode == "dense" else None)), 1)
            G(**rec)
            if attn.indexer:
                attn.indexer.keys, attn.indexer.length = snap[2], snap[3]
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

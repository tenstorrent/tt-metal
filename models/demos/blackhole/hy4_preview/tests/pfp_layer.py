"""pool-prefill S1c: direct-to-pool prefill attention of ONE layer on a 1x8 slot vs (a) the oracle golden (layer-attn goldens: 6 chunks of 512, top-k active at chunks 4, 5), (b) the replicated-cache
Hy4Attention path on the same input, (c) write-path exactness (pool rows == the replicated path's packed rows / key tiles). Env: LAYER (1|2), GOLD, HY4_PF_OPT, WDT (bf16|bfp8), CHUNKS, TIMING."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.run_attn_device import idx_setmatch, load_sd, pcc
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
from models.demos.blackhole.hy4_preview.tt.hy4_common import to_host, upload
from models.demos.blackhole.hy4_preview.tt.hy4_pf_attention import Hy4AttentionPF
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

L = int(os.environ.get("LAYER", "1"))
GOLD = os.environ.get("GOLD", "/gold/golden")
CKPT = "/ckpt"
NCH = int(os.environ.get("CHUNKS", "6"))
CH = 512
PFR = int(os.environ.get("PF_ROWS", "512"))  # rows per attention call (512 | 1024)
WDT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("WDT", "bf16")]
R = lambda **kw: print("RES " + json.dumps(kw), flush=True)


def glob_ids(loc_cols, cols=8):
    """local ids of the 8 columns [8][n,2048] (0xFFFFFFFF = none) -> global natural ids [n,2048] (-1 = none), as sets per row."""
    n = loc_cols[0].shape[0]
    out = torch.full((n, cols * 2048), -1, dtype=torch.int64)
    for c in range(cols):
        l = loc_cols[c].to(torch.int64)
        ok = (l >= 0) & (l < 2**31)
        g = (l // 64) * 512 + 64 * c + (l % 64)
        out[:, c * 2048 : (c + 1) * 2048] = torch.where(ok, g, torch.full_like(g, -1))
    return out


def main():
    cfg = Hy4Config.from_checkpoint(CKPT)
    g = torch.load(f"{GOLD}/L{L}.pt")
    conv = g["conv"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=200_000_000)
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        sd = load_sd(L)
        kw = dict(
            kv_fmt="scaled_fp8",
            rope_interleave=conv["rope_interleave"],
            idx_rope_interleave=conv["idx_rope_interleave"],
            wdtype=WDT,
        )
        pf = Hy4AttentionPF(md, mc, ccl, cfg, sd, L, **kw)
        rep = Hy4Attention(md, mc, ccl, cfg, sd, L, **kw)
        P = 8
        ctx = PoolCtx(md, mc, ccl, P, rows=1, S=1, mode="topk", bucket_tokens=P * 512)
        kv = ttnn.empty(
            [1, 1, P * 64, 656],
            dtype=ttnn.fp8_e4m3,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kp = (
            ttnn.empty(
                [P, 1, 64, 128],
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if pf.has_indexer
            else None
        )
        pf.init_pool(ctx, kv, kp)
        h_all, out_all, idx_gold = g["h"], g["out"], g["idx"]
        shared = not pf.has_indexer
        res = []
        for i in range((NCH * CH) // PFR):
            p0 = i * PFR
            h = upload(md, h_all[p0 : p0 + PFR].reshape(1, 1, PFR, -1).to(torch.bfloat16))
            ids_in = None
            if shared:
                ix = idx_gold[p0 : p0 + PFR].to(torch.int32).reshape(1, 1, PFR, -1)
                ids_in = upload(md, ix, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            t = time.perf_counter()
            out, ids = pf.prefill_chunk(h, p0, ids_in, ids_are_global=shared)
            ttnn.synchronize_device(md)
            dt = time.perf_counter() - t
            o = to_host(out, md).reshape(PFR, -1).float()
            ref = out_all[p0 : p0 + PFR]
            # replicated path on the same input
            out_r, ids_r = rep.prefill_chunk(h, p0, ids_in)
            ttnn.synchronize_device(md)
            o_r = to_host(out_r, md).reshape(PFR, -1).float()
            r = {
                "layer": L,
                "chunk": i,
                "pcc_oracle": round(pcc(o, ref), 6),
                "pcc_vs_replicated": round(pcc(o, o_r), 6),
                "rep_vs_oracle": round(pcc(o_r, ref), 6),
                "eager_ms": round(dt * 1e3, 1),
            }
            if not shared:
                loc = [to_host(ids, md, col=c).reshape(PFR, -1) for c in range(8)]
                gi = glob_ids(loc)
                ov, same = idx_setmatch(gi, idx_gold[p0 : p0 + PFR])
                r["idx_overlap_gold"] = round(ov, 4)
            R(**r)
            res.append(r)
        # ---- write-path exactness: pool rows vs the replicated packed rows (bytes)
        npg = NCH
        f2b = lambda t, c: torch.nan_to_num(
            ttnn.to_torch(ttnn.get_device_tensors(ttnn.typecast(t, ttnn.bfloat16))[c]).float(), nan=777.0
        )
        rows_rep = f2b(rep.cache.rows, 0).reshape(NCH * CH, -1)
        pool_bf = [f2b(kv, c).reshape(P * 64, -1) for c in range(8)]
        ok_rows, nbad = True, 0
        for c in range(8):
            for pg in range(npg):
                if not torch.equal(
                    pool_bf[c][pg * 64 : pg * 64 + 64], rows_rep[pg * 512 + 64 * c : pg * 512 + 64 * c + 64]
                ):
                    nbad += 1
                    ok_rows = False
        R(test="write_path_kv_rows", pool_equals_replicated_packed=ok_rows, bad_slices=nbad, total_slices=8 * npg)
        if pf.has_indexer:
            kb_rep = (
                ttnn.to_torch(ttnn.typecast(rep.indexer.keys, ttnn.bfloat8_b)).reshape(NCH * CH, 128).float()
                if False
                else None
            )
            keys_rep = (
                ttnn.to_torch(ttnn.get_device_tensors(ttnn.typecast(rep.indexer.keys, ttnn.bfloat8_b))[0])
                .reshape(NCH * CH, 128)
                .float()
            )
            nbadk = 0
            for c in range(8):
                kpool = ttnn.to_torch(ttnn.get_device_tensors(kp)[c]).reshape(P * 64, 128).float()
                for pg in range(npg):
                    if not torch.equal(
                        kpool[pg * 64 : pg * 64 + 64], keys_rep[pg * 512 + 64 * c : pg * 512 + 64 * c + 64]
                    ):
                        nbadk += 1
            R(test="write_path_key_rows", bad_slices=nbadk, total_slices=8 * npg)
        # ---- repeated calls: DRAM leak + determinism (same chunk 12 times, page rewrite is idempotent)
        if os.environ.get("REPEAT", "0") == "1":
            from models.demos.blackhole.hy4_preview.tt.hy4_chain import dram_view

            i = NCH * CH // PFR - 1
            p0 = i * PFR
            h = upload(md, h_all[p0 : p0 + PFR].reshape(1, 1, PFR, -1).to(torch.bfloat16))
            ids_in = (
                upload(
                    md,
                    idx_gold[p0 : p0 + PFR].to(torch.int32).reshape(1, 1, PFR, -1),
                    ttnn.uint32,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                if shared
                else None
            )
            first, allocs, pccs = None, [], []
            for k in range(12):
                out, ids = pf.prefill_chunk(h, p0, ids_in, ids_are_global=shared)
                o = to_host(out, md).reshape(PFR, -1).float()
                if first is None:
                    first = o
                pccs.append(round(pcc(o, first), 7))
                for t_ in (out,) + (() if shared else (ids,)):
                    ttnn.deallocate(t_)
                ttnn.synchronize_device(md)
                allocs.append(dram_view(md)["alloc_per_bank"])
            R(
                test="repeat_calls",
                pcc_vs_first=pccs,
                alloc_per_bank=allocs,
                leak_bytes_per_bank_after_warmup=allocs[-1] - allocs[2],
            )
        # ---- warm eager time of the last chunk (page rewrite is idempotent)
        if os.environ.get("TIMING", "1") == "1":
            i = NCH - 1
            p0 = i * CH
            h = upload(md, h_all[p0 : p0 + CH].reshape(1, 1, CH, -1).to(torch.bfloat16))
            ids_in = (
                upload(
                    md, idx_gold[p0 : p0 + CH].to(torch.int32).reshape(1, 1, CH, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                )
                if shared
                else None
            )
            ts, tr = [], []
            for _ in range(3):
                ttnn.synchronize_device(md)
                t = time.perf_counter()
                out, ids = pf.prefill_chunk(h, p0, ids_in, ids_are_global=shared)
                ttnn.synchronize_device(md)
                ts.append(time.perf_counter() - t)
                ttnn.deallocate(out)
                t = time.perf_counter()
                out, ids2 = rep.prefill_chunk(h, p0, ids_in, append=False)
                ttnn.synchronize_device(md)
                tr.append(time.perf_counter() - t)
                ttnn.deallocate(out)
            R(test="eager_ms_last_chunk", pool_path=round(min(ts) * 1e3, 1), replicated_path=round(min(tr) * 1e3, 1))
        # ---- traced device time per layer-chunk at depth (synthetic random pool, one full or shared layer)
        depths = [int(x) for x in os.environ.get("DEPTHS", "").split(",") if x]
        if depths:
            from models.demos.blackhole.hy4_preview.tests.dev_common import traced_us

            dp = ttnn.experimental.deepseek_prefill
            Pmax = max(depths) + 1
            gen = torch.Generator().manual_seed(5)
            rp = lambda t, dt, lay: ttnn.from_torch(
                t,
                device=md,
                dtype=dt,
                layout=lay,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ctx2 = PoolCtx(md, mc, ccl, Pmax, rows=1, S=1, mode="topk", bucket_tokens=Pmax * 512)
            kv2 = ttnn.empty(
                [1, 1, Pmax * 64, 656],
                dtype=ttnn.fp8_e4m3,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kp2 = ttnn.empty(
                [Pmax, 1, 64, 128],
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            CHF = 8192
            for c0 in range(0, Pmax * 512, CHF):
                n_ = min(CHF, Pmax * 512 - c0)
                lat = rp(torch.randn(1, 1, n_, 512, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                rope = rp(torch.randn(1, 1, n_, 64, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                f8, sc = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
                nat = dp.pack_scaled_fp8_kv_cache(f8, sc, rope)
                kd = rp(torch.randn(1, 1, n_, 128, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                kb = ttnn.typecast(kd, ttnn.bfloat8_b)
                HP.slice_copy(nat, kv2, ctx2.colid, 64, HP.ROWB, n_ // 512, dst_page0=c0 // 512)
                HP.slice_copy(
                    kb,
                    ttnn.reshape(kp2, [1, 1, Pmax * 64, 128]),
                    ctx2.colid,
                    8,
                    HP.KEYTILE,
                    n_ // 512,
                    dst_page0=c0 // 512,
                )
                ttnn.synchronize_device(md)
                for t_ in (lat, rope, f8, sc, nat, kd, kb):
                    ttnn.deallocate(t_)
            pf.init_pool(ctx2, kv2, kp2)
            h = upload(md, h_all[5 * CH : 6 * CH].reshape(1, 1, CH, -1).to(torch.bfloat16))
            ids_src = None
            for pg in depths:
                start = pg * 512
                tabs = pf.rope_tabs(start, CH)
                if shared:
                    ids_src = upload(
                        md,
                        idx_gold[5 * CH : 6 * CH].to(torch.int32).reshape(1, 1, CH, -1),
                        ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
                fn = lambda: pf.prefill_chunk(h, start, ids_src, tabs=tabs, ids_are_global=shared)
                us = traced_us(md, fn, n=3, rep_=2)
                R(test="traced_us_pool_prefill_layer", layer=L, page=pg, ctx_tokens=start + 512, us=round(us, 1))
        R(
            summary=dict(
                layer=L,
                min_pcc_oracle=min(r["pcc_oracle"] for r in res),
                min_pcc_vs_rep=min(r["pcc_vs_replicated"] for r in res),
            )
        )
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

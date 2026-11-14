# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""integ step-1 long-context block gate (1x8 slot, synthetic fill): the decode indexer + attention chain of ONE full layer on the sequence-sharded pool at 2k .. 1M tokens, through the integrated
glue (slice_copy fill, row_write append, paged_update_cache key append, decode score kernel with kv_len from a tensor, select_v2, id_filter, split kernel, XMerge) with a planted NEEDLE:
a context of N random keys / latents, one needle token whose indexer key and attention latent are aligned with the (random) indexer query and the attention queries. PASS = the needle id is in
the selected top-2048 and the merged attention output equals the needle's latent (PCC), per head. Also traced us per stage. FILL = SYNTHETIC (random rows, written chunk-wise by slice_copy).
Env: CTXS (default 2048,16384,65536,262144,1048576), TIMING (1), NEEDLE_FRAC (0.31)."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import pcc
from models.demos.blackhole.hy4_preview.tt import hy4_idx_score as IS
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx, _pad_rows, q_headrows, select_global
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CTXS = [int(x) for x in os.environ.get("CTXS", "2048,16384,65536,262144,1048576").split(",")]
NEEDLE_FRAC = float(os.environ.get("NEEDLE_FRAC", "0.31"))
SCALE = 256**-0.5
CH = 65536


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def rep(md, t, dt, lay):
    return ttnn.from_torch(t, device=md, dtype=dt, layout=lay, mesh_mapper=ttnn.ReplicateTensorToMesh(md), memory_config=ttnn.DRAM_MEMORY_CONFIG)


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
    return statistics.median(ts)


def run_ctx(md, mc, ccl, N, gen, xm=None):
    dp = ttnn.experimental.deepseek_prefill
    pages = N // 512
    pos = N - 1
    pn = int(N * NEEDLE_FRAC) // 64 * 64 + 17  # needle token position (not on a slice edge)
    ctx = PoolCtx(md, mc, ccl, pages, rows=1, S=1, mode="topk", bucket_tokens=N, xm=xm)
    run_ctx.xm = ctx.xm
    kv = ttnn.empty([1, 1, pages * 64, 656], dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kp = ttnn.empty([pages, 1, 64, 128], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kview = ttnn.reshape(kp, [1, 1, pages * 64, 128])
    # queries and the needle
    q_idx = torch.randn(32, 128, generator=gen)
    wts = torch.full((32,), 1.0 / 32)
    k_needle_idx = 6.0 * q_idx.mean(0)
    lat_needle = torch.randn(512, generator=gen) * 1.5
    rope_needle = torch.randn(64, generator=gen)
    kfull = torch.cat([lat_needle, rope_needle])
    qa = kfull * (25.0 / (kfull.pow(2).sum() * SCALE))  # needle logit 25 (scaled domain)
    # ---- fill (synthetic, chunk-wise)
    t0 = time.perf_counter()
    for c0 in range(0, N, CH):
        n = min(CH, N - c0)
        lat = torch.randn(n, 512, generator=gen)
        rope = torch.randn(n, 64, generator=gen)
        keys = torch.randn(n, 128, generator=gen)
        if c0 <= pn < c0 + n:
            lat[pn - c0], rope[pn - c0], keys[pn - c0] = lat_needle, rope_needle, k_needle_idx
        latd = rep(md, lat.reshape(1, 1, n, 512).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        roped = rep(md, rope.reshape(1, 1, n, 64).bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        f8, sc = dp.per_token_cast_to_fp8(latd, round_scale_to_power_of_two=True)
        nat = dp.pack_scaled_fp8_kv_cache(f8, sc, roped)
        HP.slice_copy(nat, kv, ctx.colid, 64, HP.ROWB, n // 512, dst_page0=c0 // 512)
        kd = rep(md, keys.reshape(1, 1, n, 128).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        kb = ttnn.typecast(kd, ttnn.bfloat8_b)
        HP.slice_copy(kb, kview, ctx.colid, 8, HP.KEYTILE, n // 512, dst_page0=c0 // 512)
        ttnn.synchronize_device(md)
        for t in (latd, roped, f8, sc, nat, kd, kb):
            ttnn.deallocate(t)
    t_fill = time.perf_counter() - t0
    # ---- decode inputs for position pos: the appended token (random row + key) via the decode write paths
    ctx.set_step([pos])
    new_lat = torch.randn(1, 1, 32, 512, generator=gen).bfloat16()
    new_rope = torch.randn(1, 1, 32, 64, generator=gen).bfloat16()
    f8, sc = dp.per_token_cast_to_fp8(rep(md, new_lat, ttnn.bfloat16, ttnn.TILE_LAYOUT), round_scale_to_power_of_two=True)
    packed = dp.pack_scaled_fp8_kv_cache(f8, sc, rep(md, new_rope, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT))
    HP.row_write(packed, ctx.wr, kv, n=1)
    new_key = torch.zeros(1, 1, 32, 128).bfloat16()
    new_key[0, 0, 0] = torch.randn(128, generator=gen).bfloat16()
    kd = ttnn.to_memory_config(rep(md, new_key, ttnn.bfloat16, ttnn.TILE_LAYOUT), ctx.kcfg_key)
    ttnn.experimental.paged_update_cache(kp, kd, update_idxs_tensor=ctx.upd, page_table=ctx.pt)
    # indexer query / weights (stock layouts)
    qh = rep(md, q_idx.reshape(1, 1, 32, 128).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)  # [S=1,1,32,128] heads = tile rows
    wp = torch.zeros(1, 1, 32, 32)
    wp[0, 0, 0] = wts
    w32 = rep(md, wp.bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    qfull = rep(md, qa.reshape(1, 1, 1, 576).repeat(1, 8, 1, 1).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    sink = upload(md, torch.full((1, 1, 32, 8), -30.0).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard_dim=3)
    ttnn.synchronize_device(md)

    def chain():
        gids = select_global(ctx, qh, w32, kview)
        idx8 = HP.id_filter(gids, ctx.colinfo, 1)
        ttnn.deallocate(gids)
        o = ctx.attend(qfull, kv, idx8, sink)
        return o, idx8

    # un-batched: ids and output
    gids = select_global(ctx, qh, w32, kview)
    idx8 = HP.id_filter(gids, ctx.colinfo, 1)
    ttnn.synchronize_device(md)
    gh = ttnn.to_torch(ttnn.get_device_tensors(gids)[0]).reshape(-1).to(torch.int64) & 0xFFFFFFFF
    got_needle = bool((gh == pn).any())
    nvalid = int((gh != HP.SENT).sum())
    o = ctx.attend(qfull, kv, idx8, sink)
    ttnn.synchronize_device(md)
    oh = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(8, 512)
    exp = lat_needle.bfloat16().float()
    pccs = [round(pcc(oh[h], exp), 5) for h in range(8)]
    # local ids of all columns hold the needle exactly once (owner column)
    own = 0
    for c in range(8):
        li = ttnn.to_torch(ttnn.get_device_tensors(idx8)[c]).reshape(8, 2048).to(torch.int64)[0] & 0xFFFFFFFF
        pg, cc, loc = HP.locate(pn)
        own += int((li == int(loc)).any()) if c == int(cc) else 0
    r = dict(event="needle", ctx=N, pages=pages, needle_pos=pn, needle_in_topk=got_needle, valid_ids=nvalid, owner_has_local_id=bool(own), o_pcc_min=min(pccs), o_pcc_mean=round(sum(pccs) / 8, 5), fill_s=round(t_fill, 1), fill="SYNTHETIC")
    if os.environ.get("TIMING", "1") == "1":
        r["us_score"] = round(traced_us(md, lambda: IS.idx_score(qh, kview, w32, 0, 1, kvt=ctx.kvcnt, n_rows=ctx.n_local)), 1)
        r["us_chain_sel"] = round(traced_us(md, lambda: select_global(ctx, qh, w32, kview)), 1)
        r["us_filter"] = round(traced_us(md, lambda: HP.id_filter(gids, ctx.colinfo, 1)), 1)
        r["us_attend"] = round(traced_us(md, lambda: ctx.attend(qfull, kv, idx8, sink)), 1)
        r["us_full_chain"] = round(traced_us(md, chain), 1)
    R(**r)
    for t in (kv, kp, kview, o, idx8, gids):
        ttnn.deallocate(t)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=400_000_000)
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        gen = torch.Generator().manual_seed(7)
        for N in CTXS:
            try:
                run_ctx(md, mc, ccl, N, gen, getattr(run_ctx, "xm", None))
            except Exception as e:  # noqa: BLE001
                import traceback

                R(event="error", ctx=N, err=repr(e)[:500], tb=traceback.format_exc()[-1200:])
                raise
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

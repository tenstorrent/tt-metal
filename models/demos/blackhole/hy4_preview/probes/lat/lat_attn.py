"""B9 lat probe 3: decode indexer+attention chain of ONE full layer vs query rows per user S (verify rows) and context (2k/64k/1M), SCALED_FP8 pool, 1x8 slot.
Derived from tests/integ_block_scale.py (S=1 only there). SYNTHETIC fill (one random 64k chunk repeated), timing only (no needle). Env: CTXS, SS (default 1..7), HY4_FUSE, HY4_XM_NSPLIT."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_idx_score as IS
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx, select_global
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CTXS = [int(x) for x in os.environ.get("CTXS", "2048,65536,1048576").split(",")]
SS = [int(x) for x in os.environ.get("SS", "1,2,3,4,5,6,7").split(",")]
SCALE = 256**-0.5
CH = 65536


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def rep(md, t, dt, lay):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=lay,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


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


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=400_000_000)
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        gen = torch.Generator().manual_seed(7)
        dp = ttnn.experimental.deepseek_prefill
        for N in CTXS:
            pages = N // 512
            kv = ttnn.empty(
                [1, 1, pages * 64, 656],
                dtype=ttnn.fp8_e4m3,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kp = ttnn.empty(
                [pages, 1, 64, 128],
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kview = ttnn.reshape(kp, [1, 1, pages * 64, 128])
            ctx0 = PoolCtx(md, mc, ccl, pages, rows=1, S=1, mode="topk", bucket_tokens=N, fuse="sparse")
            nchunk = min(CH, N)
            lat = torch.randn(nchunk, 512, generator=gen)
            rope = torch.randn(nchunk, 64, generator=gen)
            keys = torch.randn(nchunk, 128, generator=gen)
            latd = rep(md, lat.reshape(1, 1, nchunk, 512).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            roped = rep(md, rope.reshape(1, 1, nchunk, 64).bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            f8, sc = dp.per_token_cast_to_fp8(latd, round_scale_to_power_of_two=True)
            nat = dp.pack_scaled_fp8_kv_cache(f8, sc, roped)
            kd = rep(md, keys.reshape(1, 1, nchunk, 128).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            kb = ttnn.typecast(kd, ttnn.bfloat8_b)
            for c0 in range(0, N, nchunk):
                n = min(nchunk, N - c0)
                HP.slice_copy(nat, kv, ctx0.colid, 64, HP.ROWB, n // 512, dst_page0=c0 // 512)
                HP.slice_copy(kb, kview, ctx0.colid, 8, HP.KEYTILE, n // 512, dst_page0=c0 // 512)
                ttnn.synchronize_device(md)
            R(event="filled", ctx=N)
            xm = None
            for S in SS:
                try:
                    ctx = PoolCtx(md, mc, ccl, pages, rows=1, S=S, mode="topk", bucket_tokens=N, xm=None)
                    if (
                        os.environ.get("SPLIT_AUTO", "0") == "1" and S > 3
                    ):  # probe-only: the stock (G, kc) choice breaks for S>=4 (8S rows -> G=3 -> 3 chunks per sender); widen kc so each sender holds <= 2 chunks
                        G = max(1, 120 // (8 * S))
                        kc = 128
                        while -(-(2048 // kc) // G) > 2:
                            kc *= 2
                        ctx.G, ctx.kc = G, kc
                    ctx.set_step([N - S])  # S verify rows end at position N-1
                    qh = rep(md, torch.randn(S, 1, 32, 128).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                    wp = torch.zeros(1, 1, 32, 32)
                    wp[0, 0, :S] = 1.0 / 32
                    w32 = rep(md, wp.bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                    qfull = rep(md, torch.randn(1, 8, S, 576).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                    sink = upload(
                        md, torch.full((1, S, 32, 8), -30.0).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard_dim=3
                    )
                    ttnn.synchronize_device(md)
                    gids = select_global(ctx, qh, w32, kview)
                    idx8 = HP.id_filter(gids, ctx.colinfo, S, perrow=ctx.perrow)
                    ttnn.synchronize_device(md)

                    def chain():
                        g = select_global(ctx, qh, w32, kview)
                        i8 = HP.id_filter(g, ctx.colinfo, S, perrow=ctx.perrow)
                        ttnn.deallocate(g)
                        o = ctx.attend(qfull, kv, i8, sink)
                        ttnn.deallocate(i8)
                        return o

                    r = dict(
                        event="timing",
                        ctx=N,
                        S=S,
                        G=ctx.G,
                        kc=ctx.kc,
                        fuse=os.environ.get("HY4_FUSE", ""),
                        xm_nsplit=os.environ.get("HY4_XM_NSPLIT", ""),
                    )
                    r["us_score"] = round(
                        traced_us(
                            md,
                            lambda: IS.idx_score(
                                qh, kview, w32, 0, S, kvt=ctx.kvcnt, n_rows=ctx.n_local, kvt_rows=ctx.perrow
                            ),
                        ),
                        1,
                    )
                    r["us_select"] = round(traced_us(md, lambda: select_global(ctx, qh, w32, kview)), 1)
                    r["us_filter"] = round(
                        traced_us(md, lambda: HP.id_filter(gids, ctx.colinfo, S, perrow=ctx.perrow)), 1
                    )
                    r["us_attend"] = round(traced_us(md, lambda: ctx.attend(qfull, kv, idx8, sink)), 1)
                    r["us_chain"] = round(traced_us(md, chain), 1)
                    R(**r)
                    for t in (gids, idx8):
                        ttnn.deallocate(t)
                except Exception as e:
                    import traceback

                    R(event="error", ctx=N, S=S, err=repr(e)[:400], tb=traceback.format_exc()[-600:])
            for t in (kv, kp, kview):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()

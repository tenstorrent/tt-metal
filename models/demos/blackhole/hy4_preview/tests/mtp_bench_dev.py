"""mtp-device S1c: traced cost of the MTP draft step on a 1x8 slot WITHOUT the routed experts (moe_fn='skip': shared expert only; routed MoE T=4/8/16 cost is added from the layer-moe numbers).
One process = one S (XMerge is per process). Per ctx: synthetic MTP pool fill (random SCALED_FP8 rows / bfp8 keys), eager warm step, trace of (a) the MTP layer, (b) layer + lm_head + greedy sampling.
Env: S, CTXS (e.g. 2048,65536,1048576), HY4_FUSE=sparse,xchg HY4_XM_NSPLIT=8 HY4_OPT=all. Prints RES lines (us per call, median of 10)."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt
from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID, Hy4Chain, rms_exact
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_mtp import Hy4MTP, greedy_tokens
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def traced(md, fn, n=10):
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
    return round(statistics.median(ts), 1), round(min(ts), 1)


def synth_fill(md, ctx, kvp, kp, tokens, gen):
    dp = ttnn.experimental.deepseek_prefill
    rep = lambda t, dt, lay: ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=lay,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    chunk = 65536
    for c0 in range(0, tokens, chunk):
        n = min(chunk, tokens - c0)
        lat = rep(torch.randn(1, 1, n, 512, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        rope = rep(torch.randn(1, 1, n, 64, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        f8, sc = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
        nat = dp.pack_scaled_fp8_kv_cache(f8, sc, rope)
        kd = rep(torch.randn(1, 1, n, 128, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        kb = ttnn.typecast(kd, ttnn.bfloat8_b)
        HP.slice_copy(nat, kvp, ctx.colid, 64, HP.ROWB, n // 512, dst_page0=c0 // 512)
        HP.slice_copy(
            kb, ttnn.reshape(kp, [1, 1, ctx.pages * 64, 128]), ctx.colid, 8, HP.KEYTILE, n // 512, dst_page0=c0 // 512
        )
        ttnn.synchronize_device(md)
        for t in (lat, rope, f8, sc, nat, kd, kb):
            ttnn.deallocate(t)


def main():
    S = int(os.environ.get("S", "1"))
    ctxs = [int(x) for x in os.environ.get("CTXS", "2048,65536").split(",")]
    full, md = open_row()
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean

        ccl_lean.make_lean(mc)
        ccl_lean.make_lean_sems(ccl)
        ch = Hy4Chain(md, mc, ccl, os.environ["HY4_CKPT"])
        ch.build_head()
        ch.add_layers(78, 79)
        ly = ch.mtp_layers[0]
        opt = Opt.from_env()
        mtp = Hy4MTP(ch, ly, ch.rd.layer_sd(-1), opt, moe_fn="skip")
        xm = None
        gen = torch.Generator().manual_seed(3)
        for N in ctxs:
            pages = N // 512 + 1
            bucket = (N // 256 + 1) * 256
            bucket = min(bucket, pages * 512)
            mode = "dense" if N + S <= 2048 else "topk"
            ctx = PoolCtx(md, mc, ccl, pages, rows=1, S=S, mode=mode, bucket_tokens=bucket, xm=xm)
            xm = ctx.xm
            kvp = ttnn.empty(
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
            ly.attn.init_pool(ctx, kvp, kp)
            synth_fill(md, ctx, kvp, kp, (N // 512) * 512, gen)
            P0 = N
            tok = upload(md, torch.randint(0, 100000, (1, S)).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            h = upload(md, torch.randn(1, 1, S, HID).bfloat16())
            ctx.set_step([P0])
            tabs = ly.attn.rope_tabs_opt(P0, S)

            def layer_only():
                d8 = ctx.dense_ids8() if ctx.mode == "dense" else None
                o = mtp.forward(tok, h, tabs, d8)
                ttnn.deallocate(o)

            def layer_head():
                d8 = ctx.dense_ids8() if ctx.mode == "dense" else None
                o = mtp.forward(tok, h, tabs, d8)
                lg = ttnn.linear(o, ch.lm_head, compute_kernel_config=ch.ckc_head, dtype=ttnn.float32)
                ttnn.deallocate(o)
                t = greedy_tokens(ch, lg)
                ttnn.deallocate(lg)
                ttnn.deallocate(t)

            def part_pre():
                e = mtp.embed(tok)
                x = mtp.pre(e, h)
                ttnn.deallocate(e)
                ttnn.deallocate(x)

            def part_attn():
                d8 = ctx.dense_ids8() if ctx.mode == "dense" else None
                h1 = ttnn.clone(h)
                a_ = ly.attn.decode_step_pool(h1, opt, tabs, d8)
                ttnn.deallocate(h1)
                ttnn.deallocate(a_)

            def part_mlp():
                y = mtp.mlp(h)
                ttnn.deallocate(y)

            def part_norm():
                x32 = ttnn.typecast(h, ttnn.float32)
                for _ in range(3):
                    o_ = rms_exact(x32, mtp.w_fin, mtp.eps)
                    ttnn.deallocate(o_)
                ttnn.deallocate(x32)

            if os.environ.get("PARTS", "1") == "1":
                res(
                    event="parts",
                    S=S,
                    ctx=N,
                    us_pre=traced(md, part_pre)[0],
                    us_attn=traced(md, part_attn)[0],
                    us_mlp_shared_only=traced(md, part_mlp)[0],
                    us_3rms=traced(md, part_norm)[0],
                )
            a = traced(md, layer_only)
            b = traced(md, layer_head)
            res(
                event="mtp_step",
                S=S,
                ctx=N,
                mode=ctx.mode,
                us_layer_noroutedmoe=a[0],
                us_layer_min=a[1],
                us_layer_head_sample=b[0],
                us_lhs_min=b[1],
                G=ctx.G,
                kc=ctx.kc,
                xm=ctx.xm is not None,
            )
            for t in (kvp, kp, tok, h):
                ttnn.deallocate(t)
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()

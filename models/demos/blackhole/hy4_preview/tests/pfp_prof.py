"""pool-prefill: per-stage traced us of Hy4AttentionPF at depth (layer 1 real weights, synthetic random pool). Env: LAYER, PAGE (chunk page, default 255), HY4_PF_OPT."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.dev_common import traced_us
from models.demos.blackhole.hy4_preview.tests.run_attn_device import load_sd
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt import hy4_pool_prefill as PP
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_pf_attention import Hy4AttentionPF
from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import plinear
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

L = int(os.environ.get("LAYER", "1"))
PAGE = int(os.environ.get("PAGE", "255"))
R = lambda **kw: print("RES " + json.dumps(kw), flush=True)


def main():
    cfg = Hy4Config.from_checkpoint("/ckpt")
    g = torch.load(f"{os.environ.get('GOLD', '/gold/golden')}/L{L}.pt")
    conv = g["conv"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=300_000_000)
    try:
        mc, ccl = mesh_1x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        pf = Hy4AttentionPF(
            md,
            mc,
            ccl,
            cfg,
            load_sd(L),
            L,
            kv_fmt="scaled_fp8",
            rope_interleave=conv["rope_interleave"],
            idx_rope_interleave=conv["idx_rope_interleave"],
            wdtype=ttnn.bfloat16,
        )
        Pm = PAGE + 2
        ctx = PoolCtx(md, mc, ccl, Pm, rows=1, S=1, mode="topk", bucket_tokens=Pm * 512)
        dp = ttnn.experimental.deepseek_prefill
        gen = torch.Generator().manual_seed(5)
        rp = lambda t, dt, lay: ttnn.from_torch(
            t,
            device=md,
            dtype=dt,
            layout=lay,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kv = ttnn.empty(
            [1, 1, Pm * 64, 656],
            dtype=ttnn.fp8_e4m3,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kp = ttnn.empty(
            [Pm, 1, 64, 128],
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        CHF = 8192
        for c0 in range(0, Pm * 512, CHF):
            n_ = min(CHF, Pm * 512 - c0)
            lat = rp(torch.randn(1, 1, n_, 512, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            rope = rp(torch.randn(1, 1, n_, 64, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            f8, sc = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
            nat = dp.pack_scaled_fp8_kv_cache(f8, sc, rope)
            kd = rp(torch.randn(1, 1, n_, 128, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            kb = ttnn.typecast(kd, ttnn.bfloat8_b)
            HP.slice_copy(nat, kv, ctx.colid, 64, HP.ROWB, n_ // 512, dst_page0=c0 // 512)
            HP.slice_copy(
                kb, ttnn.reshape(kp, [1, 1, Pm * 64, 128]), ctx.colid, 8, HP.KEYTILE, n_ // 512, dst_page0=c0 // 512
            )
            ttnn.synchronize_device(md)
            for t_ in (lat, rope, f8, sc, nat, kd, kb):
                ttnn.deallocate(t_)
        pf.init_pool(ctx, kv, kp)
        pf._pf_init()
        h = upload(md, g["h"][5 * 512 : 6 * 512].reshape(1, 1, 512, -1).to(torch.bfloat16))
        start = PAGE * 512
        cos, sin, icos, isin = pf.rope_tabs(start, 512)
        ix = pf.indexer
        T = lambda name, fn, **kw: R(stage=name, page=PAGE, us=round(traced_us(md, fn, n=3, rep_=2), 1), **kw)
        # shared inputs
        qa = upload(md, torch.randn(1, 1, 512, 2048).bfloat16())
        lat = upload(md, torch.randn(1, 1, 512, 512).bfloat16())
        k_pe = upload(md, torch.randn(1, 1, 512, 64).bfloat16())
        T("write_kv", lambda: (pf._pf_write_kv(lat, k_pe, PAGE), None)[1])
        T("write_key", lambda: (pf._pf_write_key(h, icos, isin, PAGE), None)[1])
        T("idx_select_total", lambda: pf._pf_select(h, qa, start, PAGE, icos, isin))
        ids = pf._pf_select(h, qa, start, PAGE, icos, isin)
        # indexer parts
        n_local = 64 * (PAGE + 1)

        def idx_q():
            q = plinear(pf.pf, qa, ix.w_q, pf.ckc_pf)
            q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                q, num_heads=ix.nh, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            q = ix._rot(q, icos, isin)
            w = plinear(pf.pf, h, ix.w_p, pf.ckc_pf)
            return [q, w]

        T("idx_q_w", idx_q)
        qw = idx_q()
        ksl = ttnn.slice(pf.key_view, [0, 0, 0, 0], [1, 1, n_local, 128])
        T("idx_key_slice", lambda: ttnn.slice(pf.key_view, [0, 0, 0, 0], [1, 1, n_local, 128]))
        sc_f = lambda: ttnn.experimental.indexer_score_dsa(
            qw[0],
            ksl,
            qw[1],
            chunk_start_idx=n_local - 32,
            program_config=pf.pf_score_cfg,
            compute_kernel_config=pf.pf_score_ckc,
            kv_len=n_local,
        )
        T("idx_score", sc_f)
        score = sc_f()
        T("idx_stair", lambda: (PP.pf_stair(score, ctx.colid), None)[1])
        from models.demos.blackhole.hy4_preview.tt.hy4_idx_select import select_v2

        kl = min(512, n_local)
        T(
            "idx_select_v2",
            lambda: select_v2(
                ctx.ag, ctx.colid, ttnn.reshape(score, [1, 1, 512, n_local]), n_local, 1, kl, kl, 2048, 1, False, None
            ),
        )
        gids = select_v2(
            ctx.ag, ctx.colid, ttnn.reshape(score, [1, 1, 512, n_local]), n_local, 1, kl, kl, 2048, 1, False, None
        )
        T("idx_id_filter", lambda: PP.pf_id_filter(gids, ctx.colid))
        # attend parts
        qfull = upload(md, torch.randn(1, 8, 512, 576).bfloat16())
        T("attend_total", lambda: pf._pf_attend(qfull, ids))
        T("ag_q", lambda: pf._ag(qfull, 1))
        qg = pf._ag(qfull, 1)
        T("q_to_rm", lambda: ttnn.to_layout(qg, ttnn.ROW_MAJOR_LAYOUT))
        q_rm = ttnn.to_layout(qg, ttnn.ROW_MAJOR_LAYOUT)
        T("sdpa_pf_kernel", lambda: PP.sparse_sdpa_pf(q_rm, kv, ids, pf.scale, 128, pf.ckc_sdpa))
        out, stats = PP.sparse_sdpa_pf(q_rm, kv, ids, pf.scale, 128, pf.ckc_sdpa)
        T("merge_scale_ops", lambda: pf._pf_merge_scale(stats))
        s_t = pf._pf_merge_scale(stats)
        T("o_tilize", lambda: ttnn.to_layout(out, ttnn.TILE_LAYOUT))
        o_t = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        T("o_mul_scale", lambda: ttnn.multiply(o_t, s_t))
        T("o_mul_scale_bfp8", lambda: ttnn.multiply(o_t, s_t, dtype=ttnn.bfloat8_b))
        o_s = ttnn.multiply(o_t, s_t)
        T("rs_heads", lambda: pf._rs(o_s, 1))
        # whole layer
        tabs = (cos, sin, icos, isin)
        T("layer_total", lambda: pf.prefill_chunk(h, start, None, tabs=tabs))
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

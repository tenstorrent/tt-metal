"""pool-prefill S1d: block-level NEEDLE through the prefill chunk path (indexer on the pool + multi-row split kernel + pre-scaled reduce-scatter merge) at long context, ONE layer on a 1x8 slot,
synthetic pool fill (random latents / keys, one planted needle): the chunk = the last page of the context, its 512 query rows all aim at the needle. PASS = the needle id is in every row's top-2048
and the merged o_lat equals the needle latent (PCC per head, all 512 rows). Env: CTXS (default 65536,262144,1048576), LAYER (1; real sink / weights of that layer), NEEDLE_FRAC (0.31)."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.dev_common import pcc
from models.demos.blackhole.hy4_preview.tests.run_attn_device import load_sd
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_pf_attention import Hy4AttentionPF
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

L = int(os.environ.get("LAYER", "1"))
CTXS = [int(x) for x in os.environ.get("CTXS", "65536,262144,1048576").split(",")]
NF = float(os.environ.get("NEEDLE_FRAC", "0.31"))
SCALE = 256**-0.5
R = lambda **kw: print("RES " + json.dumps(kw), flush=True)


def main():
    cfg = Hy4Config.from_checkpoint("/ckpt")
    g = torch.load(f"{os.environ.get('GOLD', '/gold/golden')}/L{L}.pt")
    conv = g["conv"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
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
        pf._pf_init()
        dp = ttnn.experimental.deepseek_prefill
        rp = lambda t, dt, lay: ttnn.from_torch(
            t,
            device=md,
            dtype=dt,
            layout=lay,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for N in CTXS:
            gen = torch.Generator().manual_seed(N)
            Pn = N // 512
            Pm = Pn + 1
            ctx = PoolCtx(md, mc, ccl, Pm, rows=1, S=1, mode="topk", bucket_tokens=Pm * 512)
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
            pf.init_pool(ctx, kv, kp)
            pn = int(N * NF) // 64 * 64 + 17
            q_idx = torch.randn(32, 128, generator=gen)
            k_needle_idx = 6.0 * q_idx.mean(0)
            lat_needle = torch.randn(512, generator=gen) * 1.5
            rope_needle = torch.randn(64, generator=gen)
            kfull = torch.cat([lat_needle, rope_needle])
            qa = kfull * (float(os.environ.get("NEEDLE_LOGIT", "25")) / (kfull.pow(2).sum() * SCALE))
            CHF = 65536
            t0 = time.perf_counter()
            for c0 in range(0, Pn * 512, CHF):
                n_ = min(CHF, Pn * 512 - c0)
                lat = torch.randn(n_, 512, generator=gen)
                rope = torch.randn(n_, 64, generator=gen)
                keys = torch.randn(n_, 128, generator=gen)
                if c0 <= pn < c0 + n_:
                    lat[pn - c0], rope[pn - c0], keys[pn - c0] = lat_needle, rope_needle, k_needle_idx
                latd = rp(lat.reshape(1, 1, n_, 512).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                roped = rp(rope.reshape(1, 1, n_, 64).bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                f8, sc = dp.per_token_cast_to_fp8(latd, round_scale_to_power_of_two=True)
                nat = dp.pack_scaled_fp8_kv_cache(f8, sc, roped)
                HP.slice_copy(nat, kv, ctx.colid, 64, HP.ROWB, n_ // 512, dst_page0=c0 // 512)
                kd = rp(keys.reshape(1, 1, n_, 128).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                kb = ttnn.typecast(kd, ttnn.bfloat8_b)
                HP.slice_copy(kb, pf.key_view, ctx.colid, 8, HP.KEYTILE, n_ // 512, dst_page0=c0 // 512)
                ttnn.synchronize_device(md)
                for t in (latd, roped, f8, sc, nat, kd, kb):
                    ttnn.deallocate(t)
            t_fill = time.perf_counter() - t0
            # the chunk = last page (its own rows are random too); all 512 query rows aim at the needle
            p = Pn - 1
            qd = rp(q_idx.reshape(1, 32, 1, 128).repeat(1, 1, 512, 1).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            wd = rp(torch.full((1, 1, 512, 32), 1.0 / 32).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            t0 = time.perf_counter()
            loc, gids = pf._pf_select_core(qd, wd, p, return_global=True)
            ttnn.synchronize_device(md)
            t_sel = time.perf_counter() - t0
            gi = ttnn.to_torch(ttnn.get_device_tensors(gids)[0]).reshape(512, -1).to(torch.int64)
            has = int(((gi == pn).any(dim=1)).sum().item())
            nvalid = int(((gi >= 0) & (gi < 2**31)).sum(dim=1).min().item())
            qf = rp(qa.reshape(1, 1, 1, 576).repeat(1, 8, 512, 1).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            if os.environ.get("DBG", "0") == "1":
                from models.demos.blackhole.hy4_preview.tt import hy4_pool_prefill as PP

                qg = pf._ag(qf, 1)
                q_rm = ttnn.to_layout(qg, ttnn.ROW_MAJOR_LAYOUT)
                out, st = PP.sparse_sdpa_pf(q_rm, kv, loc, SCALE, 128, pf.ckc_sdpa)
                ttnn.synchronize_device(md)
                s_t = pf._pf_merge_scale(st)
                for c in (0, 5):
                    sv = ttnn.to_torch(ttnn.get_device_tensors(s_t)[c]).reshape(64, 512).float()
                    R(
                        dbg_scale_col=c,
                        row0_heads=[round(float(x), 4) for x in sv[:, 0][:16]],
                        nan=int(torch.isnan(sv).sum()),
                        mx=float(sv.max()),
                    )
                for c in range(8):
                    o_ = ttnn.to_torch(ttnn.get_device_tensors(out)[c]).reshape(64, 512, 512).float()
                    s_ = ttnn.to_torch(ttnn.get_device_tensors(st)[c]).reshape(512, 128).float()
                    ref_ = lat_needle.bfloat16().float()
                    if c == 5:
                        R(
                            dbg_stats_col5_h0_15=dict(
                                m=[round(float(x) * SCALE, 2) for x in s_[0, :16]],
                                l=[round(float(x), 3) for x in s_[0, 64:80]],
                            )
                        )
                    R(
                        dbg_col=c,
                        o_pcc_head0_row0=round(pcc(o_[0, 0], ref_), 5),
                        m_scaled_row0_h0=float(s_[0, 0] * SCALE),
                        l_row0_h0=float(s_[0, 64]),
                        nan_o=int(torch.isnan(o_).sum()),
                        inf_o=int(torch.isinf(o_).sum()),
                        nan_s=int(torch.isnan(s_).sum()),
                    )
            t0 = time.perf_counter()
            o_lat = pf._pf_attend(qf, loc)
            ttnn.synchronize_device(md)
            t_att = time.perf_counter() - t0
            ok = True
            pcs = []
            for c in (0, 3, 7):
                o = ttnn.to_torch(ttnn.get_device_tensors(o_lat)[c]).reshape(8, 512, 512).float()
                ref = lat_needle.bfloat16().float()[None, None, :].expand(8, 512, 512)
                pcs.append(pcc(o, ref))
                ok &= min(pcc(o[h, r], ref[h, r]) for h in range(0, 8, 3) for r in range(0, 512, 97)) > 0.99
            R(
                test="needle_prefill_chunk",
                ctx=N,
                pages=Pn,
                chunk_page=p,
                needle_pos=pn,
                rows_with_needle_in_topk=has,
                rows=512,
                min_valid_ids=nvalid,
                pcc_o_vs_needle=[round(x, 5) for x in pcs],
                per_row_ok=ok,
                fill_s=round(t_fill, 1),
                select_ms_eager=round(t_sel * 1e3, 1),
                attend_ms_eager=round(t_att * 1e3, 1),
                PASS=bool(has == 512 and ok and min(pcs) > 0.995),
            )
            for t in (loc, gids, o_lat, qd, wd, qf, kv, kp):
                ttnn.deallocate(t)
            ctx = None
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

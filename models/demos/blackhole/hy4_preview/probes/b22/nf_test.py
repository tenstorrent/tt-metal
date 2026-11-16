"""B22: fused_norms (q_a rms + kv latent rms + k_pe split, ONE program) vs torch and vs the stock chain (to_memory_config -> sharded rms_norm -> to_memory_config; slice -> rms_norm; slice): PCC / max error over varying inputs and S, plus traced timing."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import pcc
from models.demos.blackhole.hy4_preview.tt import hy4_fuse_attn as FA
from models.demos.blackhole.hy4_preview.tt.hy4_common import compute_cfg

NCALL = int(os.environ.get("NCALL", "6"))
EPS = 1e-5


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def ref_rms(x, w):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def main():
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        g = torch.Generator().manual_seed(11)
        rep = ttnn.ReplicateTensorToMesh(md)
        L1 = ttnn.L1_MEMORY_CONFIG
        up = lambda t, ly=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(
            t, device=md, dtype=ttnn.bfloat16, layout=ly, mesh_mapper=rep, memory_config=mc
        )
        d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])
        ckc = compute_cfg(md, "hifi4", True)
        SMC = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(y=2, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        SPC = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 2), subblock_w=4, block_h=1, block_w=4, inplace=False
        )
        allok = True
        for S in [int(v) for v in os.environ.get("SS", "1,2,4,8").split(",")]:
            for it in range(NCALL):
                mag = 10.0 ** float(torch.randint(-2, 3, (1,), generator=g))
                xq = torch.randn(1, 1, S, 2048, generator=g) * mag
                xq[..., 5] *= 30.0  # outlier columns
                xk = torch.randn(1, 1, S, 576, generator=g) * mag
                wq = (1 + 0.3 * torch.randn(1, 1, 64, 32, generator=g)).bfloat16()
                wk = (1 + 0.3 * torch.randn(1, 1, 16, 32, generator=g)).bfloat16()
                qa_g, kv = up(xq, mc=L1), up(xk, mc=L1)
                qw, kw = up(wq, ttnn.ROW_MAJOR_LAYOUT), up(wk, ttnn.ROW_MAJOR_LAYOUT)
                qa, lat, kpe = FA.fused_norms(qa_g, kv, qw, kw, EPS, L1)
                ttnn.synchronize_device(md)
                a, b, c = d0(qa)[..., :S, :], d0(lat)[..., :S, :], d0(kpe)[..., :S, :]
                xqb, xkb = xq.bfloat16(), xk.bfloat16()
                ra, rb = ref_rms(xqb, wq.reshape(1, 1, 1, 2048)), ref_rms(xkb[..., :512], wk.reshape(1, 1, 1, 512))
                # stock
                sq = ttnn.rms_norm(qa_g, weight=qw, epsilon=EPS, compute_kernel_config=ckc)
                sl = ttnn.rms_norm(
                    ttnn.slice(kv, [0, 0, 0, 0], [1, 1, S, 512]), weight=kw, epsilon=EPS, compute_kernel_config=ckc
                )
                sa, sb = d0(sq)[..., :S, :], d0(sl)[..., :S, :]
                xs_ = ttnn.to_memory_config(qa_g, SMC)
                ys_ = ttnn.rms_norm(
                    xs_, weight=qw, epsilon=EPS, program_config=SPC, memory_config=SMC, compute_kernel_config=ckc
                )
                rsh = d0(ttnn.to_memory_config(ys_, L1))[..., :S, :]  # the model's stock path (HY4_OPT rmssh)
                esh = (rsh.float() - ra).abs().max().item() / ra.abs().max().item()
                ndiff_sh = int((rsh != a).sum())
                ndiff_st = int((sa != a).sum())
                ea = (a.float() - ra).abs().max().item() / ra.abs().max().item()
                eb = (b.float() - rb).abs().max().item() / rb.abs().max().item()
                esa = (sa.float() - ra).abs().max().item() / ra.abs().max().item()
                esb = (sb.float() - rb).abs().max().item() / rb.abs().max().item()
                kpe_eq = bool(torch.equal(c, xkb[..., 512:]))
                pa, pb = pcc(a, ra), pcc(b, rb)
                ok = pa > 0.9999 and pb > 0.9999 and kpe_eq and ea < 2.5 * esa + 1e-3 and eb < 2.5 * esb + 1e-3
                allok &= ok
                R(
                    event="call",
                    S=S,
                    it=it,
                    mag=mag,
                    pcc_q=pa,
                    pcc_kv=pb,
                    relmax_q=ea,
                    relmax_kv=eb,
                    stock_relmax_q=esa,
                    rmssh_relmax_q=esh,
                    ndiff_vs_rmssh=ndiff_sh,
                    ndiff_vs_stock=ndiff_st,
                    nel=int(a.numel()),
                    stock_relmax_kv=esb,
                    kpe_equal=kpe_eq,
                    ok=ok,
                )
        R(event="summary", all_ok=allok)
        if os.environ.get("TIME", "0") == "1":
            S = 1
            qa_g, kv = up(torch.randn(1, 1, S, 2048), mc=L1), up(torch.randn(1, 1, S, 576), mc=L1)
            qw, kw = up(torch.ones(1, 1, 64, 32), ttnn.ROW_MAJOR_LAYOUT), up(
                torch.ones(1, 1, 16, 32), ttnn.ROW_MAJOR_LAYOUT
            )
            smc = ttnn.create_sharded_memory_config(
                shape=(32, 128),
                core_grid=ttnn.CoreGrid(y=2, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            spc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 2), subblock_w=4, block_h=1, block_w=4, inplace=False
            )

            def stock():
                xs = ttnn.to_memory_config(qa_g, smc)
                y = ttnn.rms_norm(
                    xs, weight=qw, epsilon=EPS, program_config=spc, memory_config=smc, compute_kernel_config=ckc
                )
                q = ttnn.to_memory_config(y, L1)
                s1 = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, S, 512])
                lat = ttnn.rms_norm(s1, weight=kw, epsilon=EPS, compute_kernel_config=ckc)
                kp = ttnn.slice(kv, [0, 0, 0, 512], [1, 1, S, 576])
                for t in (xs, y, q, s1, lat, kp):
                    ttnn.deallocate(t)

            def fused():
                o = FA.fused_norms(qa_g, kv, qw, kw, EPS, L1)
                for t in o:
                    ttnn.deallocate(t)

            NI = 40
            for nm, fn in (("stock", stock), ("fused", fused)):
                for _ in range(2):
                    fn()
                ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                for _ in range(NI):
                    fn()
                ttnn.end_trace_capture(md, tid, cq_id=0)
                ttnn.synchronize_device(md)
                ts = []
                for _ in range(6):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                    ts.append((time.perf_counter() - t0) / NI * 1e6)
                R(event="time", which=nm, us_per_call=min(ts), all=ts)
    finally:
        ttnn.close_mesh_device(md)


main()

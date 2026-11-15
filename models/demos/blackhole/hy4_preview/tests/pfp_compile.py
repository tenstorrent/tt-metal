"""pool-prefill: do consecutive chunk pages recompile? eager time of PF layer 1 over pages 8..23 (random h) with a FRESH kernel cache (run with CACHE=<new>): compile stalls show as seconds per call."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.run_attn_device import load_sd
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_pf_attention import Hy4AttentionPF
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

R = lambda **kw: print("RES " + json.dumps(kw), flush=True)


def main():
    cfg = Hy4Config.from_checkpoint("/ckpt")
    g = torch.load("/gold/golden/L1.pt")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        mc, ccl = mesh_1x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        a = Hy4AttentionPF(
            md,
            mc,
            ccl,
            cfg,
            load_sd(1),
            1,
            kv_fmt="scaled_fp8",
            rope_interleave=g["conv"]["rope_interleave"],
            idx_rope_interleave=g["conv"]["idx_rope_interleave"],
            wdtype=ttnn.bfloat16,
        )
        P0 = int(os.environ.get("P0", "8"))
        NPG = int(os.environ.get("NPG", "16"))
        P = P0 + NPG + 2
        ctx = PoolCtx(md, mc, ccl, P, rows=1, S=1, mode="topk", bucket_tokens=P * 512)
        e = lambda shape, dt, lay: ttnn.empty(
            shape, dtype=dt, layout=lay, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        kv = e([1, 1, P * 64, 656], ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT)
        kp = e([P, 1, 64, 128], ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        # random fill of the pool so nothing is NaN
        dp = ttnn.experimental.deepseek_prefill
        from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP

        rp = lambda t, dt, lay: ttnn.from_torch(
            t,
            device=md,
            dtype=dt,
            layout=lay,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        n_ = P * 512
        lat = rp(torch.randn(1, 1, n_, 512).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        rope = rp(torch.randn(1, 1, n_, 64).bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        f8, sc = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
        nat = dp.pack_scaled_fp8_kv_cache(f8, sc, rope)
        kd = rp(torch.randn(1, 1, n_, 128).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        kb = ttnn.typecast(kd, ttnn.bfloat8_b)
        HP.slice_copy(nat, kv, ctx.colid, 64, HP.ROWB, P, dst_page0=0)
        HP.slice_copy(kb, ttnn.reshape(kp, [1, 1, P * 64, 128]), ctx.colid, 8, HP.KEYTILE, P, dst_page0=0)
        ttnn.synchronize_device(md)
        a.init_pool(ctx, kv, kp)
        h = upload(md, g["h"][5 * 512 : 6 * 512].reshape(1, 1, 512, -1).to(torch.bfloat16))
        import glob

        def kcount():
            d = {}
            for k in glob.glob("/kcache/tt-metal-cache*/kernels/*"):
                d[os.path.basename(k)] = len(os.listdir(k))
            return d

        prev = kcount()
        ts = []
        for p in range(P0, P0 + NPG):
            t = time.perf_counter()
            out, ids = a.prefill_chunk(h, p * 512, None)
            ttnn.synchronize_device(md)
            ts.append(round((time.perf_counter() - t) * 1e3, 1))
            ttnn.deallocate(out)
            ttnn.deallocate(ids)
            cur = kcount()
            R(page=p, new_variants={k: cur[k] - prev.get(k, 0) for k in cur if cur[k] != prev.get(k, 0)})
            prev = cur
        R(test="eager_ms_per_page", pages=list(range(P0, P0 + NPG)), ms=ts)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()

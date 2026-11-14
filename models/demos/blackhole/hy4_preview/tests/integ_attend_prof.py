"""integ: prefix timing (traced, us) of PoolCtx.attend stages on a 1x8 slot at 64k tokens: q->RM, q all-gather, split kernel, XMerge, post (RM slice / permute / tilize)."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.integ_block_scale import rep, traced_us
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import SCALE, PoolCtx
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

N = int(os.environ.get("CTX", "65536"))


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=400_000_000)
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        pages = N // 512
        ctx = PoolCtx(md, mc, ccl, pages, rows=1, S=1, mode="topk", bucket_tokens=N)
        kv = rep(md, torch.randint(0, 120, (1, 1, pages * 64, 656), dtype=torch.int32).to(torch.uint8), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT) if False else ttnn.empty([1, 1, pages * 64, 656], dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ids = torch.randint(0, pages * 64, (8, 1, 1, 2048), dtype=torch.int32)
        idx8 = upload(md, ids.reshape(1, 1, 8, 2048), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        qfull = rep(md, torch.randn(1, 8, 1, 576).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        sink = upload(md, torch.full((1, 1, 32, 8), -30.0).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard_dim=3)
        out = {}

        def s1():
            return ttnn.to_layout(qfull, ttnn.ROW_MAJOR_LAYOUT)

        def s2():
            return ctx.gather_q(ttnn.to_layout(qfull, ttnn.ROW_MAJOR_LAYOUT))

        def s3():
            qg = s2()
            return ctx.mod.sparse_sdpa_split(qg, kv, idx8, 512, SCALE, ctx.kc, ctx.G, ctx.ckc, None, emit_stats=3, owner_s=1)

        def s4():
            ctx.xm(s3(), sink)  # returns the persistent xm.out: not handed to the tracer (it would deallocate it)
            return None

        def s5():
            return ctx.attend(qfull, kv, idx8, sink)

        for name, f in (("q_to_rm", s1), ("+gather", s2), ("+split", s3), ("+xmerge", s4), ("+post=attend", s5)):
            out[name] = round(traced_us(md, f), 1)
        print("RES " + json.dumps(dict(ctx=N, **out)), flush=True)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

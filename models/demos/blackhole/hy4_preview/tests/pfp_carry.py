"""pool-prefill: IndexCache carry through LOCAL ids (the chain's contract): layer 1 (full indexer) -> layer 2 (shared) with ids_in = the local ids returned by layer 1, golden prompt, 6 chunks, 1x8 slot.
Reports layer-2 PCC vs the oracle golden (which used the oracle's own ids) and vs the replicated path fed the same-source global ids."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.run_attn_device import load_sd, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_common import to_host, upload
from models.demos.blackhole.hy4_preview.tt.hy4_pf_attention import Hy4AttentionPF
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

R = lambda **kw: print("RES " + json.dumps(kw), flush=True)
GOLD = "/gold/golden"
CH = int(os.environ.get("PF_ROWS", "512"))
WDT = ttnn.bfloat16


def main():
    cfg = Hy4Config.from_checkpoint("/ckpt")
    g1, g2 = torch.load(f"{GOLD}/L1.pt"), torch.load(f"{GOLD}/L2.pt")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        mc, ccl = mesh_1x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        mk = lambda L, g: Hy4AttentionPF(
            md,
            mc,
            ccl,
            cfg,
            load_sd(L),
            L,
            kv_fmt="scaled_fp8",
            rope_interleave=g["conv"]["rope_interleave"],
            idx_rope_interleave=g["conv"]["idx_rope_interleave"],
            wdtype=WDT,
        )
        a1, a2 = mk(1, g1), mk(2, g2)
        P = 8
        ctx = PoolCtx(md, mc, ccl, P, rows=1, S=1, mode="topk", bucket_tokens=P * 512)
        e = lambda shape, dt, lay: ttnn.empty(
            shape, dtype=dt, layout=lay, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        a1.init_pool(
            ctx,
            e([1, 1, P * 64, 656], ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT),
            e([P, 1, 64, 128], ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        )
        a2.init_pool(ctx, e([1, 1, P * 64, 656], ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT), None)
        ids_by_layer = {}
        res = []
        for i in range(3072 // CH):
            p0 = i * CH
            h1 = upload(md, g1["h"][p0 : p0 + CH].reshape(1, 1, CH, -1).to(torch.bfloat16))
            h2 = upload(md, g2["h"][p0 : p0 + CH].reshape(1, 1, CH, -1).to(torch.bfloat16))
            if 1 in ids_by_layer:
                ttnn.deallocate(ids_by_layer[1])
            o1, ids1 = a1.prefill_chunk(h1, p0, None)
            ids_by_layer[1] = ids1  # what Hy4ChainLayer does
            o2, _ = a2.prefill_chunk(h2, p0, ids_by_layer[1])  # local ids carried
            ttnn.synchronize_device(md)
            r = dict(
                chunk=i,
                pcc_L1_oracle=round(pcc(to_host(o1, md).reshape(CH, -1).float(), g1["out"][p0 : p0 + CH]), 6),
                pcc_L2_oracle=round(pcc(to_host(o2, md).reshape(CH, -1).float(), g2["out"][p0 : p0 + CH]), 6),
            )
            R(**r)
            res.append(r)
        R(summary=dict(min_L1=min(r["pcc_L1_oracle"] for r in res), min_L2=min(r["pcc_L2_oracle"] for r in res)))
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

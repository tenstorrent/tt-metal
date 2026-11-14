"""mtp-device S1m: the MTP block on the WHOLE 4x8 mesh (real routed MoE decode block, bfp4 experts from the layer_78 cache) vs the oracle golden: one user per mesh row, S rows per user per call
(T = 4 S global MoE rows: S = 1 / 2 / 4 -> T = 4 / 8 / 16), positions p0.. from 0 on 4 prompts at a time. >= 2 calls per config, repeated calls with different rows (>= 5 for the MoE gate).
Env: GOLDEN, PROMPTS (4 prompt indices, one per row), SS, PAGES, MAXCALLS, HY4_WEIGHT_CACHE (dir holding layer_78), HY4_FUSE etc. Prints RES lines."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt
from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID, Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_mtp import Hy4MTP
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

R = 4


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def rows_tabs(attn, pos, S):
    """rope tables of R users (positions pos[r] .. pos[r]+S-1), sharded over the mesh rows; the 6-tuple of Hy4Decoder._attn / Hy4AttentionPool.decode_step_pool."""
    p = torch.tensor([[int(x) + j for j in range(S)] for x in pos])
    c, s = attn.rope.tables(p)  # [R,1,S,64] row-sharded
    first = attn.indexer.idx_rope_first
    ones = upload(attn.md, torch.ones(R, 1, S, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
    zeros = upload(attn.md, torch.zeros(R, 1, S, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
    c128 = ttnn.concat([ones, c] if first else [c, ones], dim=3)
    s128 = ttnn.concat([s, zeros] if first else [zeros, s], dim=3)
    return c, s, c, s, c128, s128


def main():
    gold = torch.load(os.environ["GOLDEN"])
    prompts = [int(x) for x in os.environ.get("PROMPTS", "0,1,2,3").split(",")]
    assert len(prompts) == R
    SS = [int(x) for x in os.environ.get("SS", "1,4").split(",")]
    PAGES = int(os.environ.get("PAGES", "2"))
    maxcalls = int(os.environ.get("MAXCALLS", "1000"))
    md = open_mesh(400_000_000)
    try:
        mc = mesh_4x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean

        ccl_lean.make_lean(mc)
        ccl_lean.make_lean_sems(ccl)
        t0 = time.time()
        ch = Hy4Chain(md, mc, ccl, os.environ["HY4_CKPT"])
        ch.build_head()
        ch.add_layers(78, 79)
        ly = ch.mtp_layers[0]
        sd = ch.rd.layer_sd(-1)
        opt = Opt.from_env()
        res(event="built", sec=round(time.time() - t0, 1), dram=dram_view(md), opt=opt.tag())
        mesh_map = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
        for S in SS:
            ctx = PoolCtx(md, mc, ccl, PAGES, rows=R, S=S, mode="dense", bucket_tokens=PAGES * 512)
            blk = ly.blk if S == 1 else ly.blk.for_batch(S)
            ly.blk_S = blk
            orig = ly.blk
            ly.blk = blk
            mtp = Hy4MTP(ch, ly, sd, opt, moe_fn=None)
            kvp = ttnn.empty(
                [1, 1, PAGES * 64, 656],
                dtype=ttnn.fp8_e4m3,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kp = ttnn.empty(
                [PAGES, 1, 64, 128],
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ly.attn.init_pool(ctx, kvp, kp)
            n = min(gold[p]["ids"].shape[0] - 1 for p in prompts)
            outs = [[] for _ in range(R)]
            ncalls = 0
            for call, p0 in enumerate(range(0, n - S + 1, S)):
                if call >= maxcalls:
                    break
                tok = torch.stack([gold[p]["ids"][p0 + 1 : p0 + S + 1].to(torch.int32) for p in prompts]).reshape(R, S)
                hh = torch.stack([gold[p]["H"][p0 : p0 + S] for p in prompts]).reshape(R, 1, S, HID).bfloat16()
                tt_tok = ttnn.from_torch(
                    tok,
                    device=md,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_map,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                tt_h = upload(md, hh, ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
                ctx.mode = "dense"
                ctx.set_step([p0] * R)
                tabs = rows_tabs(ly.attn, [p0] * R, S)
                out = mtp.forward(tt_tok, tt_h, tabs, ctx.dense_ids8())
                ttnn.synchronize_device(md)
                devs = ttnn.get_device_tensors(out)
                c = int(md.shape[1])
                for r in range(R):
                    outs[r].append(ttnn.to_torch(devs[r * c]).reshape(S, HID).float())
                ncalls += 1
                for t in (tt_tok, tt_h, out):
                    ttnn.deallocate(t)
            m = ncalls * S
            for r, p in enumerate(prompts):
                dev = torch.cat(outs[r])
                g = gold[p]
                per_row = [pcc(dev[i], g["out"][i]) for i in range(m)]
                res(
                    event="block_mesh",
                    S=S,
                    T=R * S,
                    row=r,
                    prompt=p,
                    rows=m,
                    calls=ncalls,
                    pcc_out=round(pcc(dev, g["out"][:m]), 5),
                    row_min_no0=round(min(per_row[1:]) if m > 1 else per_row[0], 4),
                )
            ly.blk = orig
            ttnn.deallocate(kvp)
            ttnn.deallocate(kp)
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()

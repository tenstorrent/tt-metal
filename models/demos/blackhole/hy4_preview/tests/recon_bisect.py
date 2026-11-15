# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""reconcile task: in-context sub-block bisect by CUMULATIVE ABLATION inside the real traced pool decode (4x8 mesh, B=4, real weights, layers 1..NLMAX-1 MoE layers).
A variant (a, m) runs, in every selected layer, attention stages 1..a and MLP stages 1..m, a stubbed stage returns its input h (zero ops), so the trace of variant v minus the trace of
variant v-1 is the in-context cost of the stage. Selection sel = layers [1..n]; per-layer cost = (T(sel8) - T(sel4)) / 4 (layers 5-8: one full-indexer layer + three shared ones) and
T(sel4)-T(none) / 4 as a cross-check; T(none) = embedding + head only.
Attention stages: 0 iHC only | 1 front (projections, q/kv norm, rope, q build, qa allgather) | 2 + KV append (cast/pack/row_write) | 3 + indexer key append + select + id_filter (full layers; topk mode)
                  | 4 + attend (q gather, split kernel, xmerge, glue) | 5 + back (value proj, o_proj, all-reduce)
MLP stages: 0 none | 1 + shared expert | 2 + router (gate.forward) | 3 + moe decode (dispatch, experts, combine, reduce-scatter; forced routing) | 4 + to DRAM + all-gather to full width.
Env: NLMAX (9), MODE (dense|topk), CTX (2048 or 65536 / 1048576), VARIANTS ('0:0,1:0,2:0,3:0,4:0,5:0,5:1,5:2,5:3,5:4'), SELS ('0,4,8'), REPS."""
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import DENSE
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tests.recon_slope import res, trace_ms
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_chain import CHUNK, HID, NSTREAM, Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CKPT = os.environ.get("HY4_CKPT", "/ckpt")
OUT = os.environ.get("OUT", "/out")
NLMAX = int(os.environ.get("NLMAX", "9"))
MODE = os.environ.get("MODE", "dense")
CTX = int(os.environ.get("CTX", "2048"))
VARS = [
    tuple(int(y) for y in x.split(":"))
    for x in os.environ.get("VARIANTS", "0:0,1:0,2:0,3:0,4:0,5:0,5:1,5:2,5:3,5:4").split(",")
]
SELS = [int(x) for x in os.environ.get("SELS", "0,4,8").split(",")]
PAD = 120002


def free(d):
    for v in d.values():
        try:
            ttnn.deallocate(v)
        except Exception:  # noqa: BLE001
            pass


def attn_cut(dec, ly, h, a):
    """mirror of Hy4AttentionPool.decode_step_pool up to stage a; returns the attention output (stage 5) or None."""
    if a < 1:
        return None  # stage 0: attention fully stubbed (BUG FIX 2026-11-14: b1 ran front even at a=0, so b1's V0 == V1)
    at, ctx, o = ly.attn, dec.ctx, dec.opt
    tabs = (dec.cos, dec.sin, dec.cos, dec.sin, dec.c128, dec.s128)
    d = at.front(h, o, tabs)
    if a < 2:
        free(d)
        return None
    at.pool_append_kv(d["lat"], d["k_pe"])
    for t in (d["lat"], d["k_pe"]):
        ttnn.deallocate(t)
    if a < 3:
        free(d)
        return None
    if at.has_indexer:
        at.pool_append_key(d["ik"], o, tabs)
        if ctx.mode == "dense":
            idx8 = dec._dense8
        else:
            gids = at.pool_select(d, o, tabs)
            idx8 = HP.id_filter(gids, ctx.colinfo, ctx.S, out=at.idx8_buf)
            ttnn.deallocate(gids)
        ctx.ids8[at.layer] = idx8
    else:
        idx8 = dec._dense8 if ctx.mode == "dense" else ctx.ids8[at.spec.indexer_source]
    if a < 4:
        free(d)
        return None
    o_lat = ctx.attend(d["qfull"], at.kv_pool, idx8, at.sink_pool)
    ttnn.deallocate(d["qfull"])
    if a < 5:
        ttnn.deallocate(o_lat)
        free(d)
        return None
    out = at.back(o, o_lat, d["g"])
    ttnn.deallocate(o_lat)
    return out


def mlp_cut(dec, i, ly, h, m):
    ch = dec.ch
    if ly.spec.mlp == DENSE or m == 0:
        return None if m == 0 else dec._mlp(i, ly, h)
    part = dec.mlp[i].forward(h)
    if m < 2:
        ttnn.deallocate(part)
        return None
    routing = ly.blk.gate.forward(h)
    if m < 3:
        for t in routing:
            ttnn.deallocate(t)
        ttnn.deallocate(part)
        return None
    tok = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
    tok = ttnn.reshape(tok, [1, 1, 1, HID])
    out = ly.blk.forward(h, tok, forced_routing=routing, fold=part)
    ttnn.deallocate(part)
    if m < 4:
        ttnn.deallocate(out)
        return None
    out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
    y = ch.mc.allgather(out, ch.ccl, axis=1, dim=3)
    ttnn.deallocate(out)
    return y


def make_step(dec, full, sel, a, m):
    ch = dec.ch

    def step():
        dec._dense8 = dec.ctx.dense_ids8() if dec.mode == "dense" else None
        e = ttnn.embedding(ttnn.reshape(dec.tok, [1, 1]), ch.embed, layout=ttnn.TILE_LAYOUT)
        e = ttnn.reshape(e, [1, 1, 1, HID // 8])
        e = ch.mc.allgather(e, ch.ccl, axis=1, dim=3)
        e32 = ttnn.typecast(e, ttnn.float32)
        ttnn.deallocate(e)
        x4 = ttnn.concat([e32] * NSTREAM, dim=1)
        ttnn.deallocate(e32)
        for i in sel:
            ly = full[i]
            pre, post = ly.ihc_a.pre_post(x4)
            raw = ly.ihc_a.collapse(x4, pre)
            ttnn.deallocate(pre)
            h = ly._norm(raw, ly.w_in)
            ttnn.deallocate(raw)
            att = attn_cut(dec, ly, h, a)
            if att is None:
                att = h
            else:
                ttnn.deallocate(h)
            x4 = ly.ihc_a.expand(x4, att, post)
            if att is not h:
                ttnn.deallocate(att)
            pre, post = ly.ihc_m.pre_post(x4)
            raw = ly.ihc_m.collapse(x4, pre)
            ttnn.deallocate(pre)
            h = ly._norm(raw, ly.w_post)
            ttnn.deallocate(raw)
            y = mlp_cut(dec, i, ly, h, m)
            if y is None:
                y = h
            else:
                ttnn.deallocate(h)
            x4 = ly.ihc_m.expand(x4, y, post)
        hn, logits = ch.head(x4)
        ttnn.deallocate(hn)
        ttnn.deallocate(x4)
        return logits

    return step


def capture(md, dec, step):
    dec.release()
    lg = step()
    ttnn.synchronize_device(md)
    ttnn.deallocate(lg)
    dec.trace = ttnn.begin_trace_capture(md, cq_id=0)
    dec.out = step()
    ttnn.end_trace_capture(md, dec.trace, cq_id=0)
    ttnn.synchronize_device(md)


def main():
    prompts = torch.load(os.environ.get("PROMPTS", f"{OUT}/decode_prompts.pt"))
    lens = [len(p) for p in prompts]
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "700e6"))))
    try:
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as _pd
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest

        links = hy4_fabric_selftest(md)
        _pd._POOL_AG_LINKS = links.pool_ag
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=CKPT)
        ch.add_layers(0, NLMAX, log=lambda s: None)
        ch.build_head()
        pages = max(CTX // 512, 8)
        dec = Hy4PoolDecoder(ch, pages, mode="dense")
        full = list(ch.layers)
        if MODE == "dense":
            ids = torch.full((4, CHUNK), PAD, dtype=torch.long)
            for r in range(4):
                ids[r, : lens[r]] = torch.tensor(prompts[r])
            x4, _ = ch.prefill(ids)
            ttnn.synchronize_device(md)
            ttnn.deallocate(x4)
            dec.handoff(CHUNK)
            ch.reset()
            pos = list(lens)
        else:
            dec.fill_synthetic(CTX // 512 * 512, torch.Generator().manual_seed(11))
            dec.mode = "topk"
            dec.ctx.set_bucket(CTX, "topk")
            pos = [CTX - 1] * 4
        dec.set_inputs([1000, 2000, 3000, 4000], pos)
        res(event="ready", mode=MODE, ctx=CTX, **dram_view(md))
        for nsel in SELS:
            sel = list(range(1, nsel + 1))
            for a, m in VARS if nsel else [(5, 4)]:
                step = make_step(dec, full, sel, a, m)
                t0 = time.time()
                capture(md, dec, step)
                med, lo, hi = trace_ms(md, dec)
                res(
                    event="cut",
                    mode=MODE,
                    ctx=CTX,
                    nsel=nsel,
                    a=a,
                    m=m,
                    trace_ms_med=med,
                    min=lo,
                    max=hi,
                    capture_s=round(time.time() - t0, 1),
                )
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()

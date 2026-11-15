# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mtp-device S3/S4 on the full 78-layer chain, 4x8 mesh (WHOLE MESH; HY4_POOL=1 HY4_KV_FMT=scaled_fp8 HY4_IHC_FUSED=1 HY4_FUSE=sparse,xchg set by the runner): greedy speculative decode vs plain decode.
Env: K (0 = plain S=1 decode with top-1/top-2 gaps, 1..3 = spec rounds), NL (78), MTP (1), NSTEPS (tokens per user, 64), PAGES (8), PROMPTS (pt: 4 token lists), TAGNAME, OUT, HY4_CKPT, PF_ROWS (512).
Stage per K>0: chain prefill (4 users x PF_ROWS) -> hand-off -> MTP prefill rows through the S-row decode-form MTP block (positions 0..L-2 from the prefill hidden rows) -> seed round (tail only, m forced 0:
pair (h_{L-1}, x_L) at position L-1 gives the first k drafts) -> trace capture of ONE round -> rounds until NSTEPS tokens per user -> tokens, m histogram, tokens / round, trace-only ms / round.
Output JSON: OUT/spec_tokens_<TAG>.json (+ gaps for K=0). Compare with ``tests/spec_compare.py``."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tests.mtp_block_mesh import rows_tabs
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt
from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID, Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_mtp import Hy4MTP, greedy_tokens
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.blackhole.hy4_preview.tt.hy4_spec_trace import Hy4SpecRunner
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CKPT = os.environ.get("HY4_CKPT", "/ckpt")
OUT = os.environ.get("OUT", "/out")
K = int(os.environ.get("K", "3"))
NL = int(os.environ.get("NL", "78"))
NSTEPS = int(os.environ.get("NSTEPS", "64"))
PAGES = int(os.environ.get("PAGES", "8"))
TAG = os.environ.get("TAGNAME", f"k{K}")
PF_ROWS = int(os.environ.get("PF_ROWS", "512"))  # chunk rows per user
PAD = 120002
R = 4


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    prompts = torch.load(os.environ.get("PROMPTS", "/out/decode_prompts.pt"))
    lens = [len(p) for p in prompts]
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "900e6"))))
    try:
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as _pd
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest

        links = hy4_fabric_selftest(md)
        _pd._POOL_AG_LINKS = links.pool_ag
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=CKPT)
        t0 = time.time()
        ch.add_layers(0, NL, log=lambda s: None)
        ch.add_layers(78, 79, log=lambda s: None)
        ch.build_head()
        res(event="built", sec=round(time.time() - t0, 1), dram=dram_view(md), K=K)
        S = K + 1 if K > 0 else 1
        MODE = os.environ.get("MODE", "dense")  # dense: <= 2048 keys; topk: long context (pool top-2048 selection)
        dec = Hy4PoolDecoder(ch, PAGES, mode=MODE, S=S, bucket_tokens=(PAGES * 512 if MODE == "topk" else None))
        nch = (
            max(lens) + PF_ROWS - 1
        ) // PF_ROWS  # prefill chunks of PF_ROWS rows per user (dense mode: prompt + generation <= 2048 keys)
        ids = torch.full((R, nch * PF_ROWS), PAD, dtype=torch.long)
        for r in range(R):
            ids[r, : lens[r]] = torch.tensor(prompts[r])

        def prefill_all():
            """chunked chain prefill -> (first tokens, per-chunk post-norm hidden rows); hand-off to the pool."""
            ibl, hns, first = {}, [], [None] * R
            for c in range(nch):
                x4, ibl = ch.prefill(ids[:, c * PF_ROWS : (c + 1) * PF_ROWS], start=c * PF_ROWS, ids_by_layer=ibl)
                hn_, lg_ = ch.head(x4)
                ttnn.synchronize_device(md)
                ttnn.deallocate(x4)
                hns.append(hn_)
                lgt = (
                    ttnn.to_torch(
                        lg_, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape))
                    )
                    .reshape(R, PF_ROWS, -1)
                    .float()
                )
                ttnn.deallocate(lg_)
                for r in range(R):
                    if (lens[r] - 1) // PF_ROWS == c:
                        first[r] = int(lgt[r, (lens[r] - 1) % PF_ROWS].argmax())
            dec.handoff(nch * PF_ROWS)
            ch.reset()
            return first, hns

        first, hns = prefill_all()
        res(event="prefill", first=first, lens=lens, chunks=nch)
        if K == 0:
            toks = [[t] for t in first]
            pos = list(lens)
            gaps = [[] for _ in range(R)]
            dec.set_inputs([t[-1] for t in toks], pos)
            dec.capture()
            for s in range(NSTEPS):
                dec.set_inputs([t[-1] for t in toks], pos)
                nxt, lgs = dec.read_argmax(dec.run())
                for r in range(R):
                    top2 = torch.topk(lgs[r], 2).values
                    gaps[r].append(round(float(top2[0] - top2[1]), 5))
                    toks[r].append(nxt[r])
                    pos[r] += 1
            json.dump(
                {"prompts": prompts, "generated": toks, "gaps": gaps, "lens": lens},
                open(f"{OUT}/spec_tokens_{TAG}.json", "w"),
            )
            res(event="plain_done", steps=NSTEPS)
            return
        # ---- spec
        ly = ch.mtp_layers[0]
        mtp = Hy4MTP(ch, ly, ch.rd.layer_sd(-1), Opt.from_env(), moe_fn=None)
        gr = lambda lg_: greedy_tokens(ch, lg_)
        runner = Hy4SpecRunner(dec, mtp, K, gr, PAGES)
        ctx = dec.ctx
        mesh_map = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
        mp = mesh_map

        def seed_state(first, hns, cap):
            """MTP prefill rows (pair (h_p, token p+1) at position p) + seed round (m = 0: pair (h_{L-1}, x_L) at L-1 gives the first drafts); then cap."""
            maxlen = max(lens)
            for p0 in range(0, maxlen - 1, S):
                tk = ids[:, p0 + 1 : p0 + S + 1]
                if tk.shape[1] < S:
                    tk = torch.cat([tk, torch.full((R, S - tk.shape[1]), PAD, dtype=torch.long)], 1)
                tt_tok = ttnn.from_torch(
                    tk.to(torch.int32),
                    device=md,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_map,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                c, o_ = p0 // PF_ROWS, p0 % PF_ROWS
                hh = ttnn.slice(hns[c], [0, 0, o_, 0], [1, 1, o_ + S, HID])
                ctx.mode = MODE
                ctx.set_step([p0] * R)
                tabs = rows_tabs(ly.attn, [p0] * R, S)
                o = mtp.forward(tt_tok, hh, tabs, ctx.dense_ids8() if MODE == "dense" else None)
                for t in (tt_tok, hh, o):
                    ttnn.deallocate(t)
            ttnn.synchronize_device(md)
            runner.load(first, [l - 1 for l in lens], [[-1] * K] * R, cap=[0] * R)
            a_seed = ttnn.from_torch(
                torch.tensor([[first[r]] * S for r in range(R)], dtype=torch.int32).reshape(R * S, 1),
                device=md,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_map,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            hl = []
            for r in range(R):
                c, o_ = (lens[r] - 1) // PF_ROWS, (lens[r] - 1) % PF_ROWS
                hl.append(
                    ttnn.to_torch(
                        ttnn.get_device_tensors(ttnn.slice(hns[c], [0, 0, o_, 0], [1, 1, o_ + 1, HID]))[
                            r * int(md.shape[1])
                        ]
                    ).reshape(1, HID)
                )
            h_seed = upload(
                md,
                torch.stack(hl).reshape(R, 1, 1, HID).repeat(1, 1, S, 1).bfloat16(),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                row_dim=0,
            )
            runner.sd.derive(runner.tail.pos)
            ctx.mode = MODE
            runner.tail.run(a_seed, h_seed, derived=True)
            ttnn.synchronize_device(md)
            ttnn.deallocate(a_seed)
            ttnn.deallocate(h_seed)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.full((R, 1, 1, 1), float(cap)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mp
                ),
                runner.tail.cap,
            )

        caps = [int(x) for x in os.environ.get("CAPS", str(K)).split(",")]
        for ci, cap in enumerate(caps):
            if ci > 0:  # fresh state: prefill again (the trace and persistent tensors are reused)
                first, hns = prefill_all()
            seed_state(first, hns, cap)
            toks = [[t] for t in first]
            hist = [0] * (K + 1)
            rounds = 0
            if runner.trace is None:
                t0 = time.time()
                warm = runner.capture()  # the eager warm round is a real round: its tokens count
                res(event="trace_captured", sec=round(time.time() - t0, 1), dram=dram_view(md))
                rounds = 1
                for r in range(R):
                    toks[r] += warm[r][0]
                    hist[warm[r][1]] += 1
            t0 = time.perf_counter()
            while min(len(t) for t in toks) < NSTEPS + 1:
                runner.submit()
                out = runner.read()
                rounds += 1
                for r in range(R):
                    em, m, pnew = out[r]
                    toks[r] += em
                    hist[m] += 1
            wall = time.perf_counter() - t0
            t1 = time.perf_counter()
            for _ in range(10):
                runner.submit()
            ttnn.synchronize_device(md)
            tr = (time.perf_counter() - t1) / 10
            res(
                event="spec_done",
                K=K,
                cap=cap,
                rounds=rounds,
                m_hist=hist,
                tokens_per_round=round(sum(len(t) - 1 for t in toks) / (R * rounds), 3),
                host_loop_ms_per_round=round(wall / rounds * 1e3, 2),
                trace_only_ms_per_round=round(tr * 1e3, 2),
            )
            tag = TAG if len(caps) == 1 else f"{TAG}_cap{cap}"
            json.dump(
                {"prompts": prompts, "generated": toks, "lens": lens, "m_hist": hist},
                open(f"{OUT}/spec_tokens_{tag}.json", "w"),
            )
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()

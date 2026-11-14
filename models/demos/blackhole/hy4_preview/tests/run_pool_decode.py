# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""integ step gates on the full 78-layer chain, 4x8 mesh, pool decode (HY4_POOL=1 HY4_KV_FMT=scaled_fp8 set by the runner):
 stage A: 4 short prompts -> chain prefill -> hand-off to the pool -> traced greedy decode NSTEPS (dense ids, <= 2048 keys) -> tokens, ms/step, DRAM leak, decode-vs-prefill logits PCC (teacher forced),
          repeated-call gate (the decode steps themselves vary tokens / positions >= 5 calls), comparison with the recorded bf16-chain tokens (decode_tokens_d6.json).
 stage B (CTXS): synthetic fill of the pool to C tokens, traced decode at position C-1 (dense mode for C <= 2048, else topk mode), ms/step per bucket.
Env: NL (78), MTP (1), NSTEPS (40), PAGES, CTXS ('' = skip stage B), NSTEPS_B (20), TAGNAME, HY4_CKPT, OUT, PROMPTS, TRACE_BYTES, NOSTAGEA."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_chain import CHUNK, Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CKPT = os.environ.get("HY4_CKPT", "/ckpt")
OUT = os.environ.get("OUT", "/out")
NL = int(os.environ.get("NL", "78"))
NSTEPS = int(os.environ.get("NSTEPS", "40"))
NSTEPS_B = int(os.environ.get("NSTEPS_B", "20"))
CTXS = [int(x) for x in os.environ.get("CTXS", "").split(",") if x]
PAGES = int(os.environ.get("PAGES", str(max([c // 512 for c in CTXS] + [8]))))
TAG = os.environ.get("TAGNAME", "p")
PAD = 120002


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    prompts = torch.load(os.environ.get("PROMPTS", f"{OUT}/decode_prompts.pt"))
    lens = [len(p) for p in prompts]
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "700e6"))))
    try:
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=CKPT)
        t0 = time.time()
        ch.add_layers(0, NL, log=lambda s: None)
        if os.environ.get("MTP", "1") == "1":
            ch.add_layers(78, 79, log=lambda s: None)
        ch.build_head()
        res(event="built", sec=round(time.time() - t0, 1), **{"dram": dram_view(md)}, kv_fmt=ch.kv_fmt)
        dec = Hy4PoolDecoder(ch, PAGES, mode="dense")
        res(event="pool_built", pages=PAGES, **{"dram": dram_view(md)})
        ids = torch.full((4, CHUNK), PAD, dtype=torch.long)
        for r in range(4):
            ids[r, : lens[r]] = torch.tensor(prompts[r])
        toks = pos = None
        if os.environ.get("NOSTAGEA", "0") != "1":
            t0 = time.time()
            x4, _ = ch.prefill(ids)
            hn, logits = ch.head(x4)
            lg = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape))).reshape(4, CHUNK, -1).float()
            first = [int(lg[r, lens[r] - 1].argmax()) for r in range(4)]
            ttnn.synchronize_device(md)
            res(event="prefill", sec=round(time.time() - t0, 2), first_tokens=first, lens=lens)
            ttnn.deallocate(x4)
            dec.handoff(CHUNK)
            ch.reset()
            res(event="dram_before_trace", **dram_view(md))
            toks = [[t] for t in first]
            pos = list(lens)
            dec.set_inputs([t[-1] for t in toks], pos)
            t0 = time.time()
            dec.capture()
            res(event="trace_captured", sec=round(time.time() - t0, 1), **dram_view(md))
            dram0 = dram_view(md)
            times, keep = [], []
            for s in range(NSTEPS):
                dec.set_inputs([t[-1] for t in toks], pos)
                t1 = time.perf_counter()
                out = dec.run()
                nxt, lgs = dec.read_argmax(out)
                if s < 8:
                    keep.append(lgs.clone())
                times.append(time.perf_counter() - t1)
                for r in range(4):
                    toks[r].append(nxt[r])
                    pos[r] += 1
                if s in (0, 1, 9, 39):
                    res(event="step", s=s, ms=round(times[-1] * 1e3, 2), toks=[t[-1] for t in toks])
            dram1 = dram_view(md)
            ts = sorted(times[3:])
            res(event="decode_done", steps=NSTEPS, median_ms=round(ts[len(ts) // 2] * 1e3, 2), tok_s_per_user=round(1.0 / (sum(ts) / len(ts)), 2), dram_alloc_delta_per_bank=dram1["alloc_per_bank"] - dram0["alloc_per_bank"])
            json.dump({"prompts": prompts, "generated": toks, "lens": lens}, open(f"{OUT}/decode_tokens_{TAG}.json", "w"))
            try:
                ref = json.load(open(f"{OUT}/decode_tokens_d6.json"))["generated"]
                agree = []
                for r in range(4):
                    a, b = toks[r], ref[r]
                    n = 0
                    while n < min(len(a), len(b)) and a[n] == b[n]:
                        n += 1
                    agree.append(dict(row=r, common_prefix=n, of=min(len(a), len(b))))
                res(event="vs_bf16_chain_tokens", agree=agree)
            except Exception as e:  # noqa: BLE001
                res(event="vs_bf16_chain_tokens", err=repr(e)[:200])
            K = 8
            ids2 = torch.full((4, CHUNK), PAD, dtype=torch.long)
            for r in range(4):
                seq = list(prompts[r]) + toks[r][:K]
                ids2[r, : len(seq)] = torch.tensor(seq)
            ch.reset()
            x4b, _ = ch.prefill(ids2)
            _, lgb = ch.head(x4b)
            lgb = ttnn.to_torch(lgb, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape))).reshape(4, CHUNK, -1).float()
            for r in range(4):
                pcs, agree = [], []
                for j in range(K):
                    refl = lgb[r, lens[r] + j]
                    pcs.append(round(pcc(keep[j][r], refl), 4))
                    agree.append(int(keep[j][r].argmax()) == int(refl.argmax()))
                res(event="decode_vs_prefill", row=r, pcc=pcs, argmax_agree=agree)
            ch.reset()
            ttnn.synchronize_device(md)
            t1 = time.perf_counter()
            for _ in range(10):
                ttnn.execute_trace(md, dec.trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            res(event="trace_only_ms", ctx="short", ms=round((time.perf_counter() - t1) / 10 * 1e3, 2))
        # ---------------- stage B: synthetic long-context
        if CTXS:
            gen = torch.Generator().manual_seed(11)
            dec.release()
            t0 = time.time()
            dec.fill_synthetic(max(CTXS) // 512 * 512, gen)
            res(event="synthetic_fill", tokens=max(CTXS), sec=round(time.time() - t0, 1), **dram_view(md))
            for C in CTXS:
                dec.release()
                mode = "dense" if C <= 2048 else "topk"
                dec.mode = mode
                dec.ctx.set_bucket(C, mode)
                p = [C - 1] * 4
                dec.set_inputs([1000, 2000, 3000, 4000], p)
                t0 = time.time()
                dec.capture()
                tc = time.time() - t0
                dram_a = dram_view(md)
                ts, tl = [], []
                for s in range(NSTEPS_B):
                    dec.set_inputs([1000 + s, 2000 + s, 3000 + s, 4000 + s], [x + s for x in p] if C > 2048 and s < 0 else p)
                    t1 = time.perf_counter()
                    out = dec.run()
                    nxt, lgs = dec.read_argmax(out)
                    tl.append(time.perf_counter() - t1)
                t1 = time.perf_counter()
                for _ in range(10):
                    ttnn.execute_trace(md, dec.trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                tr = (time.perf_counter() - t1) / 10
                res(event="ctx_decode", ctx=C, mode=mode, capture_s=round(tc, 1), host_loop_median_ms=round(statistics.median(tl[2:]) * 1e3, 2), trace_only_ms=round(tr * 1e3, 2), finite=bool(torch.isfinite(lgs).all()), next=nxt, dram_after=dram_view(md)["alloc_per_bank"] - dram_a["alloc_per_bank"], **{"free_per_bank": dram_a["free_per_bank"]})
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()

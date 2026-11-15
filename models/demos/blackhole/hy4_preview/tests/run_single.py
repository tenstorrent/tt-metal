# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tp32-single driver: NL layers, NO prefill program (a TP32-only model has no ring experts for the prefill path).
Prompts are fed through the traced decode step one token per step (decode-as-prefill, teacher forced); then NGEN greedy steps (or the reference tokens when REF is set).
Logits of every step are saved (SAVE tag) and, when REF=<tag of an EP run>, compared in-process with the reference run (same teacher-forced tokens): PCC / argmax agreement per step.
Then trace-only ms (short context) and, for CTXS, synthetic-fill long-context trace-only ms. DRAM accounting events after every stage.
Env: NL, NGEN, TAGNAME, REF, CTXS, PAGES, NSTEPS_B, TRACE_BYTES, OUT, PROMPTS."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CKPT = os.environ.get("HY4_CKPT", "/ckpt")
OUT = os.environ.get("OUT", "/out")
NL = int(os.environ.get("NL", "78"))
NGEN = int(os.environ.get("NGEN", "12"))
NSTEPS_B = int(os.environ.get("NSTEPS_B", "10"))
CTXS = [int(x) for x in os.environ.get("CTXS", "").split(",") if x]
PAGES = int(os.environ.get("PAGES", str(max([c // 512 for c in CTXS] + [8]))))
TAG = os.environ.get("TAGNAME", "s")
REF = os.environ.get("REF", "")


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


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
        t0 = time.time()
        ch.add_layers(0, NL, log=lambda s: None)
        ch.build_head()
        res(event="built", sec=round(time.time() - t0, 1), dram=dram_view(md), kv_fmt=ch.kv_fmt)
        dec = Hy4PoolDecoder(ch, PAGES, mode="dense")
        res(event="pool_built", pages=PAGES, dram=dram_view(md))
        res(
            event="moe_classes",
            classes=sorted(
                {type(getattr(ly, "blk_tp", None) or ly.blk).__name__ for ly in ch.layers if hasattr(ly, "blk")}
            ),
            tp32_layers=sum(1 for ly in ch.layers if getattr(ly, "blk_tp", None) is not None),
            ring_layers=sum(1 for ly in ch.layers if getattr(ly, "um", None) is not None),
        )
        ref = torch.load(f"{OUT}/single_{REF}.pt") if REF else None
        nst = max(lens) + NGEN
        toks = [[p[0]] for p in prompts]  # token fed at step s (row r)
        gen = [[] for _ in range(4)]
        dec.set_inputs([t[0] for t in toks], [0] * 4)
        t0 = time.time()
        dec.capture()
        res(event="trace_captured", sec=round(time.time() - t0, 1), dram=dram_view(md))
        dram0 = dram_view(md)
        logits, times, cmp = [], [], []
        for s in range(nst):
            fed = []
            for r in range(4):
                if ref is not None:
                    fed.append(int(ref["fed"][s][r]))
                elif s < lens[r]:
                    fed.append(int(prompts[r][s]))
                else:
                    fed.append(gen[r][-1])
            dec.set_inputs(fed, [s] * 4)
            t1 = time.perf_counter()
            out = dec.run()
            nxt, lgs = dec.read_argmax(out)
            times.append(time.perf_counter() - t1)
            for r in range(4):
                gen[r].append(int(nxt[r]))
            logits.append((fed, lgs.clone().float()))
            if ref is not None:
                rl = ref["logits"][s]
                pc = [round(float(pcc(lgs[r].float(), rl[r])), 5) for r in range(4)]
                ag = [int(lgs[r].argmax()) == int(rl[r].argmax()) for r in range(4)]
                cmp.append((pc, ag))
                res(event="vs_ref", s=s, pcc=pc, argmax_agree=ag, in_prompt=[s < lens[r] for r in range(4)])
            if s in (0, 1, nst - 1):
                res(event="step", s=s, ms=round(times[-1] * 1e3, 2), next=[int(x) for x in nxt])
        torch.save(
            {
                "fed": [f for f, _ in logits],
                "logits": torch.stack([l for _, l in logits]).to(torch.bfloat16),
                "gen": gen,
                "lens": lens,
            },
            f"{OUT}/single_{TAG}.pt",
        )
        dram1 = dram_view(md)
        ts = sorted(times[3:])
        res(
            event="decode_done",
            steps=nst,
            median_ms=round(ts[len(ts) // 2] * 1e3, 2),
            dram_alloc_delta_per_bank=dram1["alloc_per_bank"] - dram0["alloc_per_bank"],
            gen_tokens=gen,
        )
        if cmp:
            gen_part = [(pc, ag) for s, (pc, ag) in enumerate(cmp)]
            res(
                event="vs_ref_summary",
                min_pcc=min(min(p) for p, _ in gen_part),
                argmax_agree_frac=sum(sum(a) for _, a in gen_part) / (4 * len(gen_part)),
                steps=len(gen_part),
            )
        ttnn.synchronize_device(md)
        t1 = time.perf_counter()
        for _ in range(10):
            ttnn.execute_trace(md, dec.trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        res(event="trace_only_ms", ctx="short", ms=round((time.perf_counter() - t1) / 10 * 1e3, 2))
        if CTXS:
            g = torch.Generator().manual_seed(11)
            dec.release()
            t0 = time.time()
            dec.fill_synthetic(max(CTXS) // 512 * 512, g)
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
                tl = []
                for s in range(NSTEPS_B):
                    dec.set_inputs([1000 + s, 2000 + s, 3000 + s, 4000 + s], p)
                    t1 = time.perf_counter()
                    out = dec.run()
                    nxt, lgs = dec.read_argmax(out)
                    tl.append(time.perf_counter() - t1)
                t1 = time.perf_counter()
                for _ in range(10):
                    ttnn.execute_trace(md, dec.trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                tr = (time.perf_counter() - t1) / 10
                try:  # attn-fusion: save the last-step logits of this ctx and compare with the REF run's (same synthetic fill, same inputs)
                    torch.save(lgs.to(torch.bfloat16), f"{OUT}/ctxlg_{TAG}_{C}.pt")
                    rp = f"{OUT}/ctxlg_{REF}_{C}.pt"
                    if REF and os.path.exists(rp):
                        rl = torch.load(rp).float()
                        res(
                            event="ctx_vs_ref",
                            ctx=C,
                            pcc=[
                                round(float(torch.corrcoef(torch.stack([lgs[r].float(), rl[r]]))[0, 1]), 6)
                                for r in range(lgs.shape[0])
                            ],
                            argmax_agree=[bool(lgs[r].argmax() == rl[r].argmax()) for r in range(lgs.shape[0])],
                        )
                except Exception as e:  # noqa
                    res(event="ctx_vs_ref_err", err=str(e)[:200])
                res(
                    event="ctx_decode",
                    ctx=C,
                    mode=mode,
                    capture_s=round(tc, 1),
                    host_loop_median_ms=round(statistics.median(tl[2:]) * 1e3, 2),
                    trace_only_ms=round(tr * 1e3, 2),
                    finite=bool(torch.isfinite(lgs).all()),
                    next=nxt,
                    dram_after=dram_view(md)["alloc_per_bank"] - dram_a["alloc_per_bank"],
                    free_per_bank=dram_a["free_per_bank"],
                )
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()

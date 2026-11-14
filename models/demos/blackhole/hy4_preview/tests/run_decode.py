# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chain stage B: NL layers (+MTP weights) -> 4-prompt prefill (the chain prefill path) -> traced greedy decode, NSTEPS steps; checks first generated tokens, DRAM/L1 leak, tok/s.
Env: NL (78), MTP (1), NSTEPS (100), TCAP (1024), TAGNAME, HY4_CKPT, OUT, PROMPTS (decode_prompts.pt)."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.hy4_chain import CHUNK, Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_decode import Hy4Decoder
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CKPT = os.environ.get("HY4_CKPT", "/ckpt")
OUT = os.environ.get("OUT", "/out")
NL = int(os.environ.get("NL", "78"))
NSTEPS = int(os.environ.get("NSTEPS", "100"))
TCAP = int(os.environ.get("TCAP", "1024"))
TAG = os.environ.get("TAGNAME", "d")
PAD = 120002


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def l1_view(md):
    out = {}
    for n in ("L1", "L1_SMALL"):
        try:
            v = ttnn.get_memory_view(md, getattr(ttnn.BufferType, n))
            out[n] = dict(alloc_per_bank=v.total_bytes_allocated_per_bank, free_per_bank=v.total_bytes_free_per_bank)
        except Exception as e:  # noqa: BLE001
            out[n] = repr(e)
    return out


def main():
    prompts = torch.load(os.environ.get("PROMPTS", f"{OUT}/decode_prompts.pt"))
    lens = [len(p) for p in prompts]
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "600e6"))))
    try:
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=CKPT)
        t0 = time.time()
        ch.add_layers(0, NL, log=lambda s: print(s, flush=True))
        if os.environ.get("MTP", "0") == "1":
            ch.add_layers(78, 79, log=lambda s: print(s, flush=True))
        ch.build_head()
        res(event="built", sec=round(time.time() - t0, 1), **{"dram": dram_view(md)})
        ids = torch.full((4, CHUNK), PAD, dtype=torch.long)
        for r in range(4):
            ids[r, : lens[r]] = torch.tensor(prompts[r])
        # tolerate NL < 78 (debug): the layers present are used
        t0 = time.time()
        x4, _ = ch.prefill(ids)
        hn, logits = ch.head(x4)
        lg = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape)))
        lg = lg.reshape(4, CHUNK, -1).float()
        first = [int(lg[r, lens[r] - 1].argmax()) for r in range(4)]
        ttnn.synchronize_device(md)
        res(event="prefill", sec=round(time.time() - t0, 2), first_tokens=first, lens=lens)
        ttnn.deallocate(x4)
        dec = Hy4Decoder(ch, TCAP)
        dec.load_prefill_cache()
        ch.reset()  # free the concat-grown prefill caches and key caches
        res(event="dram_before_trace", **dram_view(md), l1=l1_view(md))
        toks = [[t] for t in first]
        pos = list(lens)
        dec.set_inputs([t[-1] for t in toks], pos)
        t0 = time.time()
        dec.capture()
        res(event="trace_captured", sec=round(time.time() - t0, 1), **dram_view(md))
        dram0 = dram_view(md)
        times = []
        keep = []
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
            if s in (0, 1, 9, 49):
                res(event="step", s=s, ms=round(times[-1] * 1e3, 2), toks=[t[-1] for t in toks], **dram_view(md))
        dram1 = dram_view(md)
        ts = sorted(times[3:])
        res(
            event="decode_done",
            steps=NSTEPS,
            median_ms=round(ts[len(ts) // 2] * 1e3, 2),
            mean_ms=round(sum(ts) / len(ts) * 1e3, 2),
            tok_s_per_user=round(1.0 / (sum(ts) / len(ts)), 2),
            tok_s_total=round(4.0 / (sum(ts) / len(ts)), 2),
            dram_alloc_delta_per_bank=dram1["alloc_per_bank"] - dram0["alloc_per_bank"],
            l1=l1_view(md),
        )
        json.dump({"prompts": prompts, "generated": toks, "lens": lens}, open(f"{OUT}/decode_tokens_{TAG}.json", "w"))
        # consistency: decode logits of the first 8 steps vs the chain PREFILL path on prompt + generated tokens (same device, teacher forced)
        K = 8
        ids2 = torch.full((4, CHUNK), PAD, dtype=torch.long)
        for r in range(4):
            seq = list(prompts[r]) + toks[r][:K]
            ids2[r, : len(seq)] = torch.tensor(seq)
        ch.reset()
        x4b, _ = ch.prefill(ids2)
        _, lgb = ch.head(x4b)
        lgb = (
            ttnn.to_torch(lgb, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape)))
            .reshape(4, CHUNK, -1)
            .float()
        )
        for r in range(4):
            pcs, agree = [], []
            for j in range(K):
                ref = lgb[r, lens[r] + j]
                pcs.append(round(pcc(keep[j][r], ref), 4))
                agree.append(int(keep[j][r].argmax()) == int(ref.argmax()))
            res(event="decode_vs_prefill", row=r, pcc=pcs, argmax_agree=agree)
        # device-time of the traced step only (no host argmax / input copies)
        ttnn.synchronize_device(md)
        t1 = time.perf_counter()
        for _ in range(10):
            ttnn.execute_trace(md, dec.trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        res(event="trace_only_ms", ms=round((time.perf_counter() - t1) / 10 * 1e3, 2))
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()

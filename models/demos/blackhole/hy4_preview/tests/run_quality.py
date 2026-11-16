# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tp32-quality driver (task tp32-quality): one boot of an NL-layer model (EP ring or TP32-only, per env flags), decode-as-prefill (one token per step, B=4 rows).
Pass TF  : teacher-forced real text (TFP=<torch file: list of 4 token lists>), logits of every step saved (QTAG) -> offline PCC / argmax / top-5 vs other variants. TFREP=2 repeats the pass (determinism + pos-reset check).
Pass GSM : greedy generation of GSM8K prompts (GSMP=<torch file: list of {ids,target,q}>, GSMN problems, batches of 4, positions restart at 0 per batch), stops on EOS / GMAX tokens; json with generated ids.
Env: NL, PAGES, QTAG, OUT, TFP, TFREP, GSMP, GSMN, GMAX, EOS, TRACE_BYTES."""
import json
import os
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
PAGES = int(os.environ.get("PAGES", "8"))
TAG = os.environ.get("QTAG", "q")
TFP = os.environ.get("TFP", "")
TFREP = int(os.environ.get("TFREP", "1"))
GSMP = os.environ.get("GSMP", "")
GSMN = int(os.environ.get("GSMN", "50"))
GMAX = int(os.environ.get("GMAX", "384"))
EOS = int(os.environ.get("EOS", "120025"))
TFMAX = int(os.environ.get("TFMAX", "100000"))


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
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
        res(
            event="moe_classes",
            classes=sorted(
                {type(getattr(ly, "blk_tp", None) or ly.blk).__name__ for ly in ch.layers if hasattr(ly, "blk")}
            ),
            tp32_layers=sum(1 for ly in ch.layers if getattr(ly, "blk_tp", None) is not None),
            ring_layers=sum(1 for ly in ch.layers if getattr(ly, "um", None) is not None),
        )
        dec.set_inputs([1, 2, 3, 4], [0] * 4)
        t0 = time.time()
        dec.capture()
        res(event="trace_captured", sec=round(time.time() - t0, 1))

        def step(fed, s):
            dec.set_inputs(fed, [s] * 4)
            out = dec.run()
            return dec.read_argmax(out)

        if TFP:
            tf = torch.load(TFP)
            n = min(min(len(p) for p in tf), TFMAX)
            first = None
            for rep in range(TFREP):
                t0 = time.time()
                lg_all, nx_all = [], []
                for s in range(n):
                    nxt, lgs = step([int(p[s]) for p in tf], s)
                    lg_all.append(lgs.to(torch.bfloat16))
                    nx_all.append(nxt)
                L = torch.stack(lg_all)
                if rep == 0:
                    first = L
                    torch.save(
                        {"tokens": [p[:n] for p in tf], "logits": L, "argmax": nx_all}, f"{OUT}/quality_{TAG}_tf.pt"
                    )
                else:
                    pc = [float(pcc(L[s, r].float(), first[s, r].float())) for s in range(n) for r in range(4)]
                    same = sum(int(L[s, r].argmax()) == int(first[s, r].argmax()) for s in range(n) for r in range(4))
                    res(
                        event="tf_repeat",
                        rep=rep,
                        min_pcc=min(pc),
                        mean_pcc=sum(pc) / len(pc),
                        argmax_same=same / (4 * n),
                        bit_equal=bool(torch.equal(L, first)),
                    )
                res(event="tf_pass", rep=rep, steps=n, sec=round(time.time() - t0, 1))
        if GSMP:
            probs = torch.load(GSMP)[:GSMN]
            gens = {}
            t0 = time.time()
            for b0 in range(0, len(probs), 4):
                idxs = [min(b0 + r, len(probs) - 1) for r in range(4)]
                real = [b0 + r < len(probs) for r in range(4)]
                ids = [probs[i]["ids"] for i in idxs]
                gen = [[] for _ in range(4)]
                done = [not real[r] for r in range(4)]
                s = 0
                while not all(done):
                    fed = []
                    for r in range(4):
                        if s < len(ids[r]):
                            fed.append(int(ids[r][s]))
                        else:
                            fed.append(gen[r][-1] if gen[r] else EOS)
                    nxt, _ = step(fed, s)
                    for r in range(4):
                        if s >= len(ids[r]) - 1 and not done[r]:
                            gen[r].append(int(nxt[r]))
                            if int(nxt[r]) == EOS or len(gen[r]) >= GMAX:
                                done[r] = True
                    s += 1
                    if s >= PAGES * 512 - 8:
                        break
                for r in range(4):
                    if real[r]:
                        gens[idxs[r]] = gen[r]
                res(event="gsm_batch", b0=b0, steps=s, sec=round(time.time() - t0, 1), lens=[len(g) for g in gen])
                json.dump(gens, open(f"{OUT}/quality_{TAG}_gsm.json", "w"))
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()

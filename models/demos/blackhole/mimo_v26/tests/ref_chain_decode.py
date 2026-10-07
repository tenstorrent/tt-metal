# SPDX-License-Identifier: Apache-2.0
"""CPU oracle for a short chain (layers 0..L-1 + final norm + lm_head, bf16 weights/matmuls like the track-0 dumps): greedy generation for N GSM8K prompts
(non-thinking chat template). Saves prompts, generated tokens and the logits of every step (first token from the prefill + each decode step) for the device test.
    python ref_chain_decode.py --layers 4 --n 16 --steps 12 --out /mnt/tt-data/ssinghal/mimo-dumps/decode_chain4.pt"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402

from models.demos.blackhole.mimo_v26.reference import loader as Ld  # noqa: E402
from models.demos.blackhole.mimo_v26.reference import ref_layer as R  # noqa: E402
from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "16")))
    tok = tokenizer()
    items = gsm_prompts(a.n, tok, a.start)
    ck = Ld.Ckpt()
    E = ck.get("model.embed_tokens.weight")
    sds = []
    for L in range(a.layers):
        t0 = time.time()
        sds.append(Ld.load_layer(L, torch.bfloat16, ck))
        print(f"loaded layer {L} in {time.time()-t0:.0f}s", flush=True)
    wn, lm = ck.get("model.norm.weight"), ck.get("lm_head.weight").float()

    def logits_of(h_last):
        hn = R.rms_norm(h_last, wn).float()
        return hn @ lm.T

    res = []
    for it in items:
        ids = torch.tensor(it["prompt_ids"])[None]
        h = E[ids]
        kv = []
        for L in range(a.layers):
            r = R.layer_forward(h, sds[L], L)
            kv.append((r["k"], r["v"]))
            h = r["h_out"]
        lg = [logits_of(h[:, -1])[0]]
        toks = [int(lg[0].argmax())]
        pos = ids.shape[1]
        for s in range(a.steps - 1):
            h = E[torch.tensor([[toks[-1]]])]
            for L in range(a.layers):
                r = R.layer_forward(h, sds[L], L, kv_past=kv[L], q0=pos)
                kv[L] = (torch.cat([kv[L][0], r["k"]], 2), torch.cat([kv[L][1], r["v"]], 2))
                h = r["h_out"]
            pos += 1
            lg.append(logits_of(h[:, -1])[0])
            toks.append(int(lg[-1].argmax()))
        res.append(dict(idx=it["idx"], prompt_ids=it["prompt_ids"], tokens=toks, logits=torch.stack(lg).half()))
        print("done", it["idx"], len(it["prompt_ids"]), toks[:6], flush=True)
    torch.save(dict(layers=a.layers, steps=a.steps, items=res), a.out)


if __name__ == "__main__":
    main()

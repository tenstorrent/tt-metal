# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced MTP acceptance (alpha_1..k) from saved post-norm main hidden states, CPU oracle MTP.
Usage: python -m ...spec.alpha_cpu OUT.json [--dump DIR] [--k 3] [--zero0 1]
Per prompt ids [N] (BOS first) and post-norm hidden H [N,D]:
 row t = (H[t], token t+1) at position t -> d1[t] (predicts t+2); step j feeds (d_j, own out) at position t+j.
 alpha_j = P(d_j[t] == main_argmax[t+j] | d_1..d_{j-1} matched). Chain sequences are rebuilt per t (rows 0..t are the
 step-0 rows, rows t+1.. the recursive rows) = the MTP KV a real decode round would see. zero0: position-0 embedding zeroed (vLLM rule)."""
import json
import os
import sys

import torch

from models.demos.blackhole.hy4_preview.reference.dump_ref import SHORT, long_prompt  # noqa: F401
from models.demos.blackhole.hy4_preview.reference.oracle import Conv, Hy4Oracle, mtp_forward

CHUNK = int(os.environ.get("ALPHA_CHUNK", "24"))
STRIDE = int(os.environ.get("ALPHA_STRIDE", "1"))  # recursion (alpha_2, alpha_3) on every STRIDE-th row t only


def chain_alpha(o, ids, H, k=3, zero0=False, conv=None):
    conv = conv or Conv()
    N = ids.shape[0]
    embed = o.embed
    if zero0:
        embed = embed.clone()  # only used through tok_next; zeroing is applied via a sentinel row below
    lg_main = o.logits(H)  # [N,V]
    main = lg_main.argmax(-1)  # main[r] predicts token r+1
    D = H.shape[-1]
    res = {"n": int(N), "k": k}
    # step 0 over all rows (teacher forced)
    o0 = mtp_forward(o.ck, H[None, :-1], ids[None, 1:], conv, embed=embed, zero_pos0=zero0)[0]  # [N-1,D]
    d = [o.logits(o0).argmax(-1)]  # d[0][t] predicts t+2
    outs = [o0]
    hit = [(d[0][: N - 1] == main[1:N])]  # vs main[t+1]
    res["alpha_1"] = hit[0].float().mean().item()
    res["n1"] = int(hit[0].numel())
    ok = {t: bool(hit[0][t]) for t in range(N - 1)}  # chain alive after step 1 (d_1 accepted)
    dmap = [{t: d[0][t] for t in range(N - 1)}]  # dmap[i][t] = draft token of step i+1 for row t
    omap = [{t: o0[t] for t in range(N - 1)}]  # omap[i][t] = MTP output hidden of step i+1 for row t
    for j in range(1, k):
        ts = [
            t for t in range(0, N - 1 - j) if t % STRIDE == 0 and t in dmap[j - 1]
        ]  # rows with main[t+j+1] defined (t+j+1 <= N-1)
        if not ts:
            break
        L = N  # padded length
        newd, newo = {}, {}
        for c0 in range(
            0, len(ts), CHUNK
        ):  # sequences of different t share only the prefix: chunked (attention scores are [chunk, 64, L, L] fp32)
            tt_ = ts[c0 : c0 + CHUNK]
            toks = torch.zeros(len(tt_), L, dtype=torch.long)
            hid = torch.zeros(len(tt_), L, D)
            for b, t in enumerate(tt_):
                toks[b, : t + 1] = ids[1 : t + 2]
                hid[b, : t + 1] = H[: t + 1]
                for i in range(1, j + 1):  # recursive rows t+i
                    toks[b, t + i] = dmap[i - 1][t]
                    hid[b, t + i] = omap[i - 1][t]
            oj = mtp_forward(o.ck, hid, toks, conv, embed=embed, zero_pos0=zero0)  # [B,L,D]
            last = torch.stack([oj[b, t + j] for b, t in enumerate(tt_)])
            dj = o.logits(last).argmax(-1)
            for b, t in enumerate(tt_):
                newd[t], newo[t] = dj[b], last[b]
        dmap.append(newd)
        omap.append(newo)
        hj = {t: bool(newd[t] == main[t + j + 1]) for t in ts}
        alive = [t for t in ts if ok[t]]
        res[f"alpha_{j+1}_cond"] = sum(hj[t] for t in alive) / max(1, len(alive))
        res[f"alpha_{j+1}_uncond"] = sum(hj.values()) / len(ts)
        res[f"n{j+1}"] = len(alive)
        ok = {t: ok[t] and hj[t] for t in ts}
    return res


def main():
    out = sys.argv[1]
    dump = (
        sys.argv[sys.argv.index("--dump") + 1]
        if "--dump" in sys.argv
        else "/mnt/tt-data/ssinghal/hy4-logs/ref_dumps/dump_v1"
    )
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 3
    torch.set_num_threads(int(os.environ.get("HY4_THREADS", "24")))
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("/mnt/tt-data/Hy4-Preview/tokenizer.json")
    texts = SHORT  # the 7 short dump prompts (the 2.6k one is a synthetic repeat: step-0 only, see --long)
    if "--long" in sys.argv:
        texts = texts + [long_prompt()]
    zero0 = "--zero0" in sys.argv
    o = Hy4Oracle()
    allr = []
    meta = os.path.join(dump, "meta.json")
    if os.path.exists(meta) and "ids" in json.load(open(meta)):  # text_dump.py dumps carry their token ids and kinds
        mj = json.load(open(meta))
        items = [(torch.tensor(i), kd) for i, kd in zip(mj["ids"], mj["kinds"])]
    else:
        items = [(torch.tensor([120000] + tok.encode(t).ids), "dump") for t in texts]
    for p, (ids, kind) in enumerate(items):
        H = torch.load(os.path.join(dump, f"final_hidden_p{p}.pt"))["final_normed"][0]
        assert H.shape[0] == ids.shape[0], (H.shape, ids.shape)
        r = chain_alpha(o, ids, H, k=k, zero0=zero0)
        r["prompt"], r["kind"] = p, kind
        print(r, flush=True)
        allr.append(r)
        json.dump(allr, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 speculative greedy decode on the resident engine (qwen36_resident.dflash.ResidentDFlash), one user.

Per prompt (the serving chat prompts with thinking on and off, and coding prompts): the spec decode's tokens
against the verify model's own one-token greedy (lossless: must be identical), its tok/s, accepted drafts per
iteration and, for the first --timing prompts, the median draft / verify / extend time of an iteration.

usage: python -m qwen36_resident.tests.dfl_loop --out loop.json [--new 128] [--timing]
"""
import argparse
import faulthandler
import json
import sys
import time

import torch
from transformers import AutoTokenizer

from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlash
from qwen36_resident.serve.prompts import PROMPTS, chat_ids
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

CODE = [
    "Write a Python function that merges two sorted lists into one sorted list, with type hints and a docstring.",
    "Implement a binary search tree in C++ with insert, find and in-order traversal.",
    "Write a Python class for an LRU cache with get and put in O(1), using a dict and a doubly linked list.",
    "Write a bash script that finds the ten largest files under a directory and prints their sizes.",
    "Write a SQL query that returns, for each customer, their total order amount in 2025, highest first.",
    "Write a JavaScript function that debounces another function, and show how to use it.",
]


def prompts():
    out = [
        dict(name=f"{p['name']}/think={t}", content=p["content"], thinking=t, kind="chat")
        for p in PROMPTS
        for t in (True, False)
    ]
    out += [
        dict(name=f"code{i}/think={t}", content=c, thinking=t, kind="code")
        for i, c in enumerate(CODE)
        for t in (False, True)
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--timing", type=int, default=3, help="prompts that also get a timed pass (parts synchronized)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--long",
        default="",
        help="corpus prompt lengths: the prefill + handoff path (all but the drafter's window prefilled) against the "
        "prompt through verify steps only (builds the pipeline-parallel prefill)",
    )
    ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen36_27b/profiles/resident/256k/corpus_ids.pt")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(600, repeat=True, file=sys.stderr)
    path = Q.checkpoint_dir()
    ck = Q.Checkpoint(path)
    tok = AutoTokenizer.from_pretrained(path)
    mesh = open_mesh()
    rows = []
    try:
        t0 = time.time()
        longs = [int(x) for x in a.long.split(",") if x]
        spec = ResidentDFlash(mesh, ck, max_pos=max(longs + [2048]) + a.new + 256, prefill=bool(longs))
        log(f"built in {time.time() - t0:.0f} s")
        corpus = torch.load(a.corpus)["ids"] if longs else None
        for T in longs:
            ids = corpus[:T].tolist()
            time.sleep(2)
            gen_pp, st_pp = spec.generate(ids, a.new)
            window = spec.window
            spec.window = 1 << 30  # every prompt token through verify steps
            gen_v, st_v = spec.generate(ids, a.new)
            spec.window = window
            first = next((k for k in range(a.new) if gen_pp[k] != gen_v[k]), None)
            rows.append(
                dict(
                    name=f"long{T}",
                    kind="long",
                    prompt_len=T,
                    spec=gen_pp,
                    verify_prompt=gen_v,
                    identical=gen_pp == gen_v,
                    first_diff=first,
                    stats=st_pp,
                    stats_verify_prompt=st_v,
                )
            )
            log(
                f"long T={T}: prefill+handoff vs verify-step prompt: identical {gen_pp == gen_v} (first diff {first}); "
                f"first token {st_pp['first_token_s']:.2f} s vs {st_v['first_token_s']:.2f} s, tok/s {st_pp['tok_s']:.1f} "
                f"vs {st_v['tok_s']:.1f}, accepted/iter {st_pp['mean_accept']:.2f} vs {st_v['mean_accept']:.2f}"
            )
        ps = prompts()[: a.limit or None] if not longs else []
        for p in ps:
            ids = chat_ids(tok, dict(content=p["content"], thinking=p["thinking"]))
            time.sleep(2)  # chip 1 throttles under sustained load: short bursts
            gen, stats = spec.generate(ids, a.new)
            if len(rows) < a.timing:
                _, timed = spec.generate(ids, a.new, timing=True)
                stats.update({k: v for k, v in timed.items() if k.endswith("_ms") and k != "ms_per_iter"})
            ref, gaps = spec.greedy(ids, a.new, other=gen)
            first = next((k for k in range(a.new) if gen[k] != ref[k]), None)
            # where the spec tokens first differ: the reference's margin of its token over the spec token (a near-tie
            # when within a bf16 step of the logits, about 0.06-0.25 at their scale)
            first_gap = gaps[first] if first is not None else None
            rows.append(
                dict(
                    p,
                    prompt_len=len(ids),
                    spec=gen,
                    greedy=ref,
                    identical=gen == ref,
                    first_diff=first,
                    first_diff_margin=first_gap,
                    stats=stats,
                    text=tok.decode(gen, skip_special_tokens=True),
                )
            )
            s = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items() if k != "accepts"}
            log(
                f"{p['name']}: P={len(ids)} lossless {gen == ref} (first diff {first}, reference margin there "
                f"{first_gap}) {json.dumps(s)}"
            )
        for kind in ("chat", "code"):
            rs = [r for r in rows if r["kind"] == kind]
            if rs:
                log(
                    f"{kind}: {len(rs)} prompts, lossless {sum(r['identical'] for r in rs)}/{len(rs)}, tok/s "
                    f"{min(r['stats']['tok_s'] for r in rs):.1f}-{max(r['stats']['tok_s'] for r in rs):.1f} "
                    f"(mean {sum(r['stats']['tok_s'] for r in rs) / len(rs):.1f}), accepted drafts per iteration "
                    f"{sum(r['stats']['mean_accept'] for r in rs) / len(rs):.2f}"
                )
    finally:
        json.dump(rows, open(a.out, "w"), indent=1)
        close_mesh(mesh)


if __name__ == "__main__":
    main()

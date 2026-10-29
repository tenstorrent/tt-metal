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
    a = ap.parse_args()
    faulthandler.dump_traceback_later(600, repeat=True, file=sys.stderr)
    path = Q.checkpoint_dir()
    ck = Q.Checkpoint(path)
    tok = AutoTokenizer.from_pretrained(path)
    mesh = open_mesh()
    rows = []
    try:
        t0 = time.time()
        spec = ResidentDFlash(mesh, ck)
        log(f"built in {time.time() - t0:.0f} s")
        ps = prompts()[: a.limit or None]
        for p in ps:
            ids = chat_ids(tok, dict(content=p["content"], thinking=p["thinking"]))
            time.sleep(2)  # chip 1 throttles under sustained load: short bursts
            gen, stats = spec.generate(ids, a.new)
            if len(rows) < a.timing:
                _, timed = spec.generate(ids, a.new, timing=True)
                stats.update({k: v for k, v in timed.items() if k.endswith("_ms") and k != "ms_per_iter"})
            ref = spec.greedy(ids, a.new)
            first = next((k for k in range(a.new) if gen[k] != ref[k]), None)
            rows.append(
                dict(
                    p,
                    prompt_len=len(ids),
                    spec=gen,
                    greedy=ref,
                    identical=gen == ref,
                    first_diff=first,
                    stats=stats,
                    text=tok.decode(gen, skip_special_tokens=True),
                )
            )
            s = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items() if k != "accepts"}
            log(f"{p['name']}: P={len(ids)} lossless {gen == ref} (first diff {first}) {json.dumps(s)}")
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

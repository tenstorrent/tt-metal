# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user serving checks against a running serve_local.sh multi server (greedy, token ids compared exactly).

identity: the prompts of prompts.py decoded (a) one at a time, (b) all at once beside fillers (C=8), (c) after
    8 long fillers took launch 0, so the probes run in another launch / other rows; (b) and (c) must equal (a) token
    for token (the rows-mode build is compiled at width 8 for every launch: a user's arithmetic does not depend on
    its neighbours, its row or its launch). The fillers of (c) are aborted mid-stream.
churn: N requests of mixed prompt lengths (an instruction plus the start of a text file) arriving staggered, some
    aborted mid-stream by closing the connection; then every completed one again alone: its tokens must equal the
    concurrent run's. A request that does not finish within the timeout fails the check (hang). The mix always holds
    LONG prompts over the PP chunk unit: arriving while others decode they are split (the server logs each "split
    prefill"), alone they run whole, so equal tokens are the end-to-end check that a split prompt computes what it
    would whole (one PP chunk per sequence); the run script counts the split prompts.

usage: python -m qwen36_resident.serve.mu_checks identity --port 8000 --out identity.json
       python -m qwen36_resident.serve.mu_checks churn --port 8000 --out churn.json --doc FILE [--n 16] [--aborts 2]
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.request

from qwen36_resident.serve.chat_client import model_name
from qwen36_resident.serve.prompts import PROMPTS, messages

# churn: prompt lengths over the PP chunk unit (1024) in every mix, none a multiple of it
LONG = [1100, 2100, 3333, 5000, 6000, 8000]


def stream(port, model, msgs, max_tokens, thinking=True, abort_after=None, timeout=1800):
    """one greedy streamed request: generated token ids, finish reason, seconds; abort_after: close the connection
    after that many tokens (the server aborts the request)"""
    body = dict(
        model=model,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        return_token_ids=True,
        chat_template_kwargs=dict(enable_thinking=thinking),
    )
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    ids, finish, t0 = [], None, time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            for c in json.loads(line[5:]).get("choices", []):
                ids += c.get("token_ids") or []
                finish = c.get("finish_reason") or finish
            if abort_after is not None and len(ids) >= abort_after:
                finish = "aborted"
                break
    return dict(token_ids=ids, finish=finish, seconds=time.time() - t0)


def concurrent(jobs):
    """run [(key, fn)] in threads, start order as given (each after the previous one started); {key: result}"""
    out, threads = {}, []

    def run(key, fn):
        try:
            out[key] = fn()
        except Exception as e:  # recorded, the check fails on it
            out[key] = dict(error=repr(e))

    for key, fn in jobs:
        t = threading.Thread(target=run, args=(key, fn))
        t.start()
        threads.append(t)
        time.sleep(0.05)
    for t in threads:
        t.join()
    return out


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((k for k in range(n) if a[k] != b[k]), None if len(a) == len(b) else n)


def run_identity(a):
    model = model_name(a.port)
    probes = {p["name"]: p for p in PROMPTS}
    filler = dict(content="Write a long, detailed essay about the history of lighthouses.", thinking=False)
    solo = {n: stream(a.port, model, messages(p), a.max_tokens, p["thinking"]) for n, p in probes.items()}
    print("solo done", {n: len(r["token_ids"]) for n, r in solo.items()}, flush=True)
    # (b) all probes at once beside fillers: 8 users in one launch
    jobs = [(n, lambda p=p: stream(a.port, model, messages(p), a.max_tokens, p["thinking"])) for n, p in probes.items()]
    jobs += [(f"fill{i}", lambda: stream(a.port, model, messages(filler), a.max_tokens, False)) for i in range(3)]
    together = concurrent(jobs)
    # (c) 8 fillers take launch 0 first, the probes then run in launch 1; the fillers abort after filler_tokens
    # (they must outlast the probes)
    fill_jobs = []
    for i in range(8):

        def fill(i=i):
            return stream(a.port, model, messages(filler), 4096, False, abort_after=a.filler_tokens)

        fill_jobs.append((f"long{i}", fill))
    holder = threading.Thread(target=lambda: together.setdefault("_fillers", concurrent(fill_jobs)))
    holder.start()
    time.sleep(a.filler_head_start)
    other = concurrent(jobs[: len(probes)])
    holder.join()
    rows, ok = [], True
    for n in probes:
        s = solo[n]["token_ids"]
        db, dc = first_diff(s, together[n].get("token_ids", [])), first_diff(s, other[n].get("token_ids", []))
        ok &= db is None and dc is None and "error" not in together[n] and "error" not in other[n]
        rows.append(dict(name=n, solo=len(s), with_neighbours_first_diff=db, other_launch_first_diff=dc))
        print(f"{n}: {len(s)} tokens; beside 7 others first diff {db}; in another launch first diff {dc}", flush=True)
    json.dump(dict(ok=ok, rows=rows, solo=solo, together=together, other_launch=other), open(a.out, "w"), indent=1)
    print("IDENTITY " + ("PASS" if ok else "FAIL"), flush=True)


def run_churn(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])
    model = model_name(a.port)
    rng = random.Random(a.seed)
    text = tok(open(a.doc).read(), add_special_tokens=False)["input_ids"]
    lengths = LONG[: a.n] + [rng.choice([16, 100, 500, 1000, 2000, 3000, 6000, 8000]) for _ in range(a.n - len(LONG))]
    rng.shuffle(lengths)
    reqs = []
    for i, n in enumerate(lengths):
        start = rng.randrange(0, max(1, len(text) - n))
        body = "Summarize the following text in a few sentences.\n\n" + tok.decode(text[start : start + n])
        reqs.append(dict(key=f"r{i}", msgs=[{"role": "user", "content": body}], max_tokens=rng.randint(32, 160), isl=n))
    aborted = set(rng.sample([r["key"] for r in reqs], a.aborts))
    jobs = []
    for r in reqs:
        abort_after = 8 if r["key"] in aborted else None

        def job(r=r, abort_after=abort_after):
            time.sleep(rng.uniform(0, a.stagger))
            return stream(a.port, model, r["msgs"], r["max_tokens"], False, abort_after, timeout=a.timeout)

        jobs.append((r["key"], job))
    t0 = time.time()
    together = concurrent(jobs)
    print(f"concurrent phase: {time.time() - t0:.1f} s", flush=True)
    rows, ok = [], True
    for r in reqs:
        got = together.get(r["key"], {})
        if "error" in got:
            ok = False
            rows.append(dict(key=r["key"], isl=r["isl"], error=got["error"]))
            print(f"{r['key']} (isl {r['isl']}): ERROR {got['error']}", flush=True)
            continue
        if r["key"] in aborted:
            rows.append(dict(key=r["key"], isl=r["isl"], aborted=True))
            continue
        solo = stream(a.port, model, r["msgs"], r["max_tokens"], False, timeout=a.timeout)
        d = first_diff(solo["token_ids"], got["token_ids"])
        ok &= d is None
        rows.append(dict(key=r["key"], isl=r["isl"], tokens=len(got["token_ids"]), first_diff=d))
        print(f"{r['key']} (isl {r['isl']}): {len(got['token_ids'])} tokens, vs solo first diff {d}", flush=True)
    json.dump(dict(ok=ok, rows=rows, aborted=sorted(aborted)), open(a.out, "w"), indent=1)
    print("CHURN " + ("PASS" if ok else "FAIL"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["identity", "churn"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--filler-tokens", type=int, default=2500, help="identity (c): tokens before a filler aborts")
    ap.add_argument("--filler-head-start", type=float, default=10.0, help="identity (c): seconds before the probes")
    ap.add_argument("--doc")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--aborts", type=int, default=2)
    ap.add_argument("--stagger", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    run_identity(a) if a.mode == "identity" else run_churn(a)


if __name__ == "__main__":
    main()

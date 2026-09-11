"""Does client abort + slot reuse corrupt the survivors' output?

The eval dropped 6 of 10 connections simultaneously at t=1800 s and retried
them. This reproduces that shape cheaply: start N streaming requests, abort a
subset mid-flight, let the survivors run on, then check whether the survivors'
token stream corrupts and whether corruption starts near the abort.
"""
import argparse
import asyncio
import json
import os
import re
import time

import aiohttp
from transformers import AutoTokenizer

PROMPT = (
    "What is the correct answer to this question: Two quantum states with energies "
    "E1 and E2 have lifetimes of 1e-9 s and 1e-8 s respectively. We want to clearly "
    "distinguish these two energy levels. Which one of the following options could "
    "be their energy difference so that they can be clearly resolved?\n"
    "Choices:\n(A) 10^-4 eV\n(B) 10^-11 eV\n(C) 10^-8 eV\n(D) 10^-9 eV\n"
    "Please reason step by step, and your final answer must be only (A,B,C or D) "
    "within \\boxed\nAnswer:"
)
DUP = re.compile(r"\b(\w{3,})\1", re.I)
REP = re.compile(r"\b(\w+)( \1\b){2,}", re.I)


async def one(session, i, max_tokens, out, t0, abort_at=None):
    body = {
        "model": "Qwen/Qwen3.8-27B",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 2000 + i,
        "stream": True,
        "logprobs": True,
        "top_logprobs": 0,
    }
    toks = []  # (t_rel, token_id)
    aborted = False
    try:
        async with session.post("http://127.0.0.1:8000/v1/chat/completions", json=body) as r:
            async for raw in r.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "):
                    continue
                p = line[6:]
                if p == "[DONE]":
                    break
                try:
                    c = json.loads(p)
                except json.JSONDecodeError:
                    continue
                for ch in c.get("choices", []):
                    lp = (ch.get("logprobs") or {}).get("content") or []
                    for e in lp:
                        m = re.fullmatch(r"token_id:(\d+)", e["token"])
                        if m:
                            toks.append((time.perf_counter() - t0, int(m.group(1))))
                if abort_at is not None and (time.perf_counter() - t0) >= abort_at:
                    aborted = True
                    break
    except Exception as e:
        out[i] = {"toks": toks, "aborted": aborted, "error": repr(e)}
        return
    out[i] = {"toks": toks, "aborted": aborted, "error": None}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--abort", type=int, default=6, help="how many to drop mid-flight")
    ap.add_argument("--abort-at", type=float, default=120.0, help="seconds")
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--weights", default=os.environ.get("QWEN38_WEIGHTS", "/proj_sw/user_dev/weights/Qwen3.8-27B"))
    a = ap.parse_args()

    out = {}
    t0 = time.perf_counter()
    conn = aiohttp.TCPConnector(limit=a.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200), connector=conn) as s:
        tasks = [
            one(s, i, a.max_tokens, out, t0, abort_at=(a.abort_at if i < a.abort else None))
            for i in range(a.concurrency)
        ]
        await asyncio.gather(*tasks)

    tok = AutoTokenizer.from_pretrained(a.weights, trust_remote_code=True)
    print(f"aborted {a.abort} of {a.concurrency} at t={a.abort_at}s\n")
    print(f"{'req':>3} {'role':>9} {'toks':>5} {'dup':>4} {'first-dup-at':>13}")
    for i in sorted(out):
        v = out[i]
        ids = [t[1] for t in v["toks"]]
        role = "ABORTED" if i < a.abort else "survivor"
        text = tok.decode(ids, skip_special_tokens=True) if ids else ""
        hits = list(DUP.finditer(text)) + list(REP.finditer(text))
        first = ""
        if hits:
            pos = min(h.start() for h in hits)
            frac = pos / max(len(text), 1)
            # map char position back to a wall-clock time via token fraction
            ti = int(frac * len(v["toks"]))
            first = f"{v['toks'][min(ti, len(v['toks'])-1)][0]:.0f}s"
        print(f"{i:>3} {role:>9} {len(ids):>5} {len(hits):>4} {first:>13}")
        v["text"] = text

    surv = [out[i] for i in out if i >= a.abort]
    bad = [v for v in surv if DUP.findall(v["text"]) or REP.findall(v["text"])]
    print(f"\nSURVIVORS with duplication: {len(bad)}/{len(surv)}")
    if bad:
        w = max(bad, key=lambda v: len(DUP.findall(v["text"])))
        print("\n===== a corrupted survivor, tail:")
        print(repr(w["text"][-700:]))
    json.dump(
        {str(k): {"aborted": v["aborted"], "n": len(v["toks"]), "text": v["text"]} for k, v in out.items()},
        open("./abort_out.json", "w"),
    )


asyncio.run(main())

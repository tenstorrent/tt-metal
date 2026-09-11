"""Same discriminating test, but at eval concurrency.

For each of N concurrent requests: capture the sampled token ids, detokenize
them offline in one shot, and compare against the string the server assembled.
Reports corruption markers so a batched-decode defect separates from an
assembly defect.
"""
import argparse
import asyncio
import json
import os
import re

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

# fragment-duplication signature from doc/SAMPLING_TEXT_QUALITY.md
DUP = re.compile(r"\b(\w{3,})\1", re.I)  # boxedboxed
REP = re.compile(r"\b(\w+)( \1\b){2,}", re.I)  # state state state


async def one(session, i, max_tokens, out):
    body = {
        "model": "Qwen/Qwen3.8-27B",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 1000 + i,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 0,
    }
    async with session.post("http://127.0.0.1:8000/v1/chat/completions", json=body) as r:
        d = await r.json()
    ch = d["choices"][0]
    msg = ch["message"]
    lp = (ch.get("logprobs") or {}).get("content") or []
    ids = []
    for e in lp:
        m = re.fullmatch(r"token_id:(\d+)", e["token"])
        if m:
            ids.append(int(m.group(1)))
    out[i] = {
        "ids": ids,
        "server": (msg.get("reasoning_content") or "") + (msg.get("content") or ""),
        "content": msg.get("content") or "",
        "finish": ch.get("finish_reason"),
        "usage": d.get("usage", {}).get("completion_tokens"),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--weights", default=os.environ.get("QWEN38_WEIGHTS", "/proj_sw/user_dev/weights/Qwen3.8-27B"))
    ap.add_argument("--out", default="./conc_out.json")
    a = ap.parse_args()
    out = {}
    conn = aiohttp.TCPConnector(limit=a.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200), connector=conn) as s:
        await asyncio.gather(*[one(s, i, a.max_tokens, out) for i in range(a.concurrency)])

    tok = AutoTokenizer.from_pretrained(a.weights, trust_remote_code=True)
    print(f"{'req':>3} {'toks':>5} {'finish':>7} {'dup(ids)':>9} {'dup(srv)':>9} {'match':>6}")
    bad_ids = bad_srv = 0
    for i in sorted(out):
        v = out[i]
        one_shot = tok.decode(v["ids"], skip_special_tokens=True) if v["ids"] else ""
        d_ids = len(DUP.findall(one_shot)) + len(REP.findall(one_shot))
        d_srv = len(DUP.findall(v["server"])) + len(REP.findall(v["server"]))
        # does the server string appear inside the offline decode?
        tail = v["server"][-60:]
        match = bool(tail) and tail in one_shot
        bad_ids += d_ids > 0
        bad_srv += d_srv > 0
        print(f"{i:>3} {v['usage']:>5} {str(v['finish']):>7} {d_ids:>9} {d_srv:>9} {str(match):>6}")
        v["one_shot"] = one_shot
    print(f"\nrequests whose OFFLINE decode shows duplication: {bad_ids}/{len(out)}")
    print(f"requests whose SERVER string shows duplication:  {bad_srv}/{len(out)}")
    worst = max(out.values(), key=lambda v: len(DUP.findall(v["server"])))
    print("\n===== worst server string, tail:")
    print(repr(worst["server"][-500:]))
    print("\n===== its OWN ids, one-shot offline decode, tail:")
    print(repr(worst["one_shot"][-500:]))
    json.dump({str(k): v for k, v in out.items()}, open(a.out, "w"))
    print(f"\nsaved -> {a.out}")


asyncio.run(main())

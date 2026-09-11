"""Replay the eval's ACTUAL 10 GPQA prompts at eval concurrency.

Same prompts, same sampling, same seed=42 the harness sends, but a token cap
small enough to finish in minutes and no client aborts. Isolates "real eval
prompts" from "aborts" as the corruption trigger.
"""
import argparse
import asyncio
import json
import os
import re

import aiohttp
from transformers import AutoTokenizer

DUP = re.compile(r"\b(\w{3,})\1", re.I)
REP = re.compile(r"\b(\w+)( \1\b){2,}", re.I)


def load_prompts(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        raw = r["arguments"]["gen_args_0"]["arg_0"][0]
        msgs = json.loads(raw)
        content = (
            "".join(p["content"] for p in msgs if p.get("type") == "text")
            if isinstance(msgs, list) and msgs and "type" in msgs[0]
            else msgs
        )
        out.append((content, r.get("target")))
    return out


async def one(session, i, prompt, max_tokens, out):
    body = {
        "model": "Qwen/Qwen3.8-27B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 42,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 0,
    }
    async with session.post("http://127.0.0.1:8000/v1/chat/completions", json=body) as r:
        d = await r.json()
    ch = d["choices"][0]
    m = ch["message"]
    lp = (ch.get("logprobs") or {}).get("content") or []
    ids = [int(x.group(1)) for x in (re.fullmatch(r"token_id:(\d+)", e["token"]) for e in lp) if x]
    out[i] = {
        "ids": ids,
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning_content") or "",
        "finish": ch.get("finish_reason"),
        "n": d.get("usage", {}).get("completion_tokens"),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--weights", default=os.environ.get("QWEN38_WEIGHTS", "/proj_sw/user_dev/weights/Qwen3.8-27B"))
    a = ap.parse_args()

    prompts = load_prompts(a.samples)
    print(f"replaying {len(prompts)} real eval prompts at concurrency {len(prompts)}")
    out = {}
    conn = aiohttp.TCPConnector(limit=len(prompts))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200), connector=conn) as s:
        await asyncio.gather(*[one(s, i, p, a.max_tokens, out) for i, (p, _t) in enumerate(prompts)])

    tok = AutoTokenizer.from_pretrained(a.weights, trust_remote_code=True)
    print(f"\n{'doc':>3} {'toks':>5} {'finish':>7} {'dup(offline ids)':>17} {'dup(server)':>12} {'target':>7}")
    bad_off = bad_srv = 0
    for i in sorted(out):
        v = out[i]
        offline = tok.decode(v["ids"], skip_special_tokens=True) if v["ids"] else ""
        server = v["reasoning"] + v["content"]
        d_off = len(DUP.findall(offline)) + len(REP.findall(offline))
        d_srv = len(DUP.findall(server)) + len(REP.findall(server))
        bad_off += d_off > 0
        bad_srv += d_srv > 0
        print(f"{i:>3} {v['n']:>5} {str(v['finish']):>7} {d_off:>17} {d_srv:>12} {str(prompts[i][1]):>7}")
        v["offline"] = offline
    print(f"\ndocs whose OFFLINE id-decode shows duplication: {bad_off}/{len(out)}")
    print(f"docs whose SERVER string shows duplication:     {bad_srv}/{len(out)}")
    worst = max(out.values(), key=lambda v: len(DUP.findall(v["offline"])))
    if DUP.findall(worst["offline"]):
        print("\n===== worst OFFLINE decode, sample:")
        print(repr(worst["offline"][:400]))
        print(" ... tail ...")
        print(repr(worst["offline"][-400:]))
    json.dump(
        {
            str(k): {"n": v["n"], "finish": v["finish"], "offline": v["offline"], "content": v["content"]}
            for k, v in out.items()
        },
        open("./replay_out.json", "w"),
    )


asyncio.run(main())

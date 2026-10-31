#!/usr/bin/env python3
"""GSM8K spot check against the server (no structured requests): first N test problems, greedy chat completions, C concurrent.
usage: gsm_check.py N C out.json   (4-shot-free: 'think step by step, finish with: The answer is <number>')"""
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request

N, C, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
G = [json.loads(l) for l in open("/mnt/tt-data/ssinghal/datasets/gsm8k_test.jsonl")][:N]


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def one(g):
    gold = num(g["answer"].split("####")[-1])
    body = dict(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        temperature=0.0,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": g["question"] + "\n\nThink step by step and finish with the line: The answer is <number>.",
            }
        ],
    )
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=900).read())
        txt = d["choices"][0]["message"].get("content") or ""
        m = re.search(r"answer is[:\s]*\**\s*\$?(-?[\d,]*\.?\d+)", txt, re.I)
        pred = num(m.group(1)) if m else num(txt[-200:])
        return pred is not None and gold is not None and abs(pred - gold) < 1e-4, d["usage"]["completion_tokens"]
    except Exception as e:
        return False, -1


t0 = time.time()
with cf.ThreadPoolExecutor(C) as ex:
    res = list(ex.map(one, G))
ok = sum(1 for r, _ in res if r)
errs = sum(1 for _, n in res if n < 0)
out = dict(
    n=len(res),
    correct=ok,
    acc=100.0 * ok / len(res),
    errors=errs,
    secs=time.time() - t0,
    mean_tokens=sum(n for _, n in res if n > 0) / max(1, len(res) - errs),
)
print(json.dumps(out))
json.dump(out, open(OUT, "w"))

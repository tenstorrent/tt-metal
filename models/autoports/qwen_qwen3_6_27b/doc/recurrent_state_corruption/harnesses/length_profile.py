"""Is the corruption a function of generation LENGTH alone?

One request, no concurrency, no aborts. Decode the sampled ids in fixed
token-position windows and score each window, so onset is located in token
space rather than character space.
"""
import argparse
import json
import os
import re

import requests
from transformers import AutoTokenizer

MERGE = re.compile(r"[a-z]{2}[A-Z]")
DUPFRAG = re.compile(r"\b(\w{3,})\1", re.I)
REPW = re.compile(r"\b(\w+)( \1\b){1,}", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--max-tokens", type=int, default=6000)
ap.add_argument("--window", type=int, default=500)
ap.add_argument("--weights", default=os.environ.get("QWEN38_WEIGHTS", "/proj_sw/user_dev/weights/Qwen3.8-27B"))
ap.add_argument("--out", default="./length_profile.json")
a = ap.parse_args()

# a prompt that reliably produces a long chain
PROMPT = (
    "What is the correct answer to this question: In a genetics experiment on white "
    "lupine, three candidate resistance genes G1, G2 and G3 were knocked out singly "
    "and in all pairwise combinations, and anthracnose resistance was scored. At "
    "least one gene is a transcription factor acting upstream of the others. Work "
    "through every possible epistatic arrangement in full detail, considering each "
    "single and double mutant phenotype in turn, before concluding.\n"
    "Choices:\n(A) G1 is upstream\n(B) G2 is upstream\n(C) G3 is upstream\n"
    "(D) cannot be determined\n"
    "Please reason step by step at length, and your final answer must be only "
    "(A,B,C or D) within \\boxed\nAnswer:"
)

r = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3.8-27B",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": a.max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 42,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 0,
    },
    timeout=7200,
)
r.raise_for_status()
d = r.json()
ch = d["choices"][0]
lp = (ch.get("logprobs") or {}).get("content") or []
ids = [int(m.group(1)) for m in (re.fullmatch(r"token_id:(\d+)", e["token"]) for e in lp) if m]
print(f"finish={ch.get('finish_reason')} completion_tokens={d['usage']['completion_tokens']} ids={len(ids)}")

tok = AutoTokenizer.from_pretrained(a.weights, trust_remote_code=True)
print(f"\n{'tokens':>12} {'merge':>6} {'dupfrag':>8} {'repword':>8} {'total':>6}")
prof = []
for s in range(0, len(ids), a.window):
    w = ids[s : s + a.window]
    t = tok.decode(w, skip_special_tokens=True)
    m, dfr, rp = len(MERGE.findall(t)), len(DUPFRAG.findall(t)), len(REPW.findall(t))
    prof.append({"start": s, "merge": m, "dupfrag": dfr, "repword": rp})
    print(f"{s:>6}-{s+len(w):<5} {m:>6} {dfr:>8} {rp:>8} {m+dfr+rp:>6}")

first_bad = next((p["start"] for p in prof if p["merge"] + p["dupfrag"] + p["repword"] >= 3), None)
print(f"\nonset (first window with >=3 hits): {first_bad}")
full = tok.decode(ids, skip_special_tokens=True)
print("\n===== head (300 chars)")
print(repr(full[:300]))
print("\n===== tail (500 chars)")
print(repr(full[-500:]))
json.dump({"ids": ids, "profile": prof, "text": full}, open(a.out, "w"))

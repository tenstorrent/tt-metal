# SPDX-License-Identifier: Apache-2.0
"""Accuracy + speed benchmark against the running Qwen3.6-27B vLLM server (localhost:8000).
Config under test: TP=8, enforce-eager, max_num_seqs=1, max_model_len=1024, bf8 dense weights.

Speed: streaming /v1/completions -> TTFT (prefill) + decode tok/s, for a few prompt/out sizes.
Accuracy: cloze/short-answer completions (the base model answers BEFORE its <think> block,
so small max_tokens captures the direct answer), temp=0, case-insensitive answer-contains.
This probes the served weights+pipeline on facts/arithmetic; it is NOT a chain-of-thought
eval (eager B=1 + 1024 ctx make long-reasoning evals impractical here)."""
import json
import re
import time

import requests

BASE = "http://localhost:8000/v1"
MODEL = "qwen3.6-27b"


def stream_complete(prompt, max_tokens, temperature=0.0):
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/completions", stream=True, timeout=900, json={
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": temperature, "stream": True})
    ttft = None
    toks = 0
    text = ""
    for line in r.iter_lines():
        if not line:
            continue
        s = line.decode()
        if s.startswith("data: "):
            s = s[6:]
            if s.strip() == "[DONE]":
                break
            piece = json.loads(s)["choices"][0].get("text", "")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                toks += 1
                text += piece
    return ttft, time.perf_counter() - t0, toks, text


def complete(prompt, max_tokens, temperature=0.0):
    r = requests.post(f"{BASE}/completions", timeout=900, json={
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature})
    return r.json()["choices"][0]["text"]


def speed():
    print("=== SPEED (TP=8, eager, B=1) ===", flush=True)
    stream_complete("Hello,", 8)  # warmup (first request also JITs/traces)
    cases = [
        ("short prompt, 64 out", "The capital of France is", 64),
        ("short prompt, 128 out", "The capital of France is", 128),
        ("long prompt (~220 tok), 64 out", ("Summarize the following. " + ("The quick brown fox jumps over the lazy dog. " * 40)), 64),
    ]
    for name, prompt, ol in cases:
        ttft, total, toks, _ = stream_complete(prompt, ol)
        dec = (toks - 1) / (total - ttft) if toks > 1 and total > ttft else 0.0
        print(f"  [{name}] TTFT(prefill)={ttft*1000:.0f} ms | out_toks={toks} | "
              f"decode={dec:.2f} tok/s ({1000.0/dec:.0f} ms/tok) | total={total:.1f}s", flush=True)


# (prompt, list of acceptable answer substrings, lowercased)
ACC = [
    ("The capital of Japan is", ["tokyo"]),
    ("The capital of Italy is", ["rome"]),
    ("The capital of Canada is", ["ottawa"]),
    ("The chemical symbol for gold is", ["au"]),
    ("The largest planet in our solar system is", ["jupiter"]),
    ("The author of the play Romeo and Juliet is", ["shakespeare"]),
    ("The speed of light is approximately 300,000 kilometers per", ["second"]),
    ("Water is made of hydrogen and", ["oxygen"]),
    ("The first President of the United States was", ["washington"]),
    ("The opposite of 'hot' is", ["cold"]),
    ("17 + 28 =", ["45"]),
    ("9 * 7 =", ["63"]),
    ("100 - 37 =", ["63"]),
    ("144 / 12 =", ["12"]),
    ("The number of days in a leap year is", ["366"]),
    ("Two plus two equals", ["4", "four"]),
    ("The square root of 81 is", ["9", "nine"]),
    ("The freezing point of water in Celsius is", ["0", "zero"]),
    ("The largest ocean on Earth is the", ["pacific"]),
    ("DNA stands for deoxyribonucleic", ["acid"]),
]


def accuracy():
    print("\n=== ACCURACY (cloze, temp=0, answer-contains) ===", flush=True)
    ok = 0
    fails = []
    for prompt, answers in ACC:
        out = complete(prompt, 16, 0.0)
        # the direct answer precedes any <think>; check the pre-think span
        head = out.split("<think>")[0].lower()
        hit = any(re.search(r"\b" + re.escape(a) + r"\b", head) for a in answers)
        if hit:
            ok += 1
        else:
            fails.append((prompt, answers, out[:60].replace("\n", " ")))
    print(f"  ACCURACY: {ok}/{len(ACC)} = {100.0*ok/len(ACC):.1f}%", flush=True)
    for p, a, o in fails:
        print(f"   MISS: {p!r} (want {a}) got {o!r}", flush=True)


if __name__ == "__main__":
    speed()
    accuracy()
    print("\nBENCH_DONE", flush=True)

#!/usr/bin/env python3
"""Known-answer check of the server on :8000 before benchmarking. Exits 1 if any answer is wrong.

Chips left in a bad state by a hung, killed or crashed process can make the next server return fluent-looking garbage
while every latency number still looks normal. This catches that in ~30 s: short questions sent one at a time and all
at once (batched decode), and a long number sequence the model must continue (long-context prefill + decode).
usage: sanity_check.py [base_url]
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
QA = [
    ("What is 17 + 25? Answer with just the number.", "42"),
    ("What is the capital of Japan? One word.", "tokyo"),
    ("Name the largest planet in the solar system. One word.", "jupiter"),
    ("How many legs does a spider have? Just the number.", "8"),
    ("What is the chemical symbol for gold? Just the symbol.", "au"),
    ("What is 9 times 8? Just the number.", "72"),
    ("What colour do you get mixing blue and yellow? One word.", "green"),
    ("What is the opposite of 'cold'? One word.", "hot"),
]


def chat(content, max_tokens=16):
    body = {
        "model": "llama70b",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    bad = []
    for label, answers in [
        ("sequential", [chat(q) for q, _ in QA]),
        ("concurrent", list(ThreadPoolExecutor(len(QA)).map(lambda qa: chat(qa[0]), QA))),
    ]:
        for (q, want), got in zip(QA, answers):
            if want not in got.lower():
                bad.append(f"{label}: {q!r} -> {got!r}")
    k = 2500  # ~9K-token prompt: crosses several prefill chunks
    seq = ", ".join(str(i) for i in range(1, k + 1))
    got = chat(f"Continue this sequence exactly, with no other text:\n{seq},", max_tokens=12)
    if f"{k + 1}, {k + 2}" not in got:
        bad.append(f"long ({k} numbers): -> {got!r}")
    if bad:
        print("SANITY FAILED (reset the Galaxy with tt-smi -glx_reset_auto, then restart the server):")
        print("\n".join(bad))
        sys.exit(1)
    print(f"SANITY OK ({2 * len(QA)} short answers, {k}-number continuation)")


if __name__ == "__main__":
    main()

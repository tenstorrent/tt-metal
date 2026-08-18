# SPDX-License-Identifier: Apache-2.0
"""Isolate which release setting degrades generation: batch size or device sampling mode.

Baselines, measured on this port at max_num_seqs=1 + sample_on_device_mode=all +
FABRIC_1D_RING (tests/thinking_mode_probe.py, all with enable_thinking=false):

    trivial  "What is 2 + 2?"        ->     2 words, stop, correct
    easy     harmonic-oscillator MCQ ->   472 words, stop, \\boxed{A} correct
    diamond0 real GPQA Diamond row 0 -> 1,849 words, stop, \\boxed{A} correct

The CI-faithful run (max_num_seqs=32 + decode_only + FABRIC_1D) produced completed
documents of 1-196 words, corrupted from the first token ("LetLet's", "of of",
"larger larger", answer options losing their exponents). Two release-only variables
changed at once, so this probe re-runs the same three prompts under one variable at a
time.

Thinking is disabled for every prompt so responses are short and bounded: corruption
shows up as word-count collapse and duplicated fragments rather than as a runaway.
"""

import collections
import csv
import json
import os
import re
import sys
import urllib.request

BASE = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.6-27B"
GPQA_CSV = ("/huggingface/hub/datasets--Idavidrein--gpqa/snapshots/"
            "633f5ee89ab8ad4522a9f850766b73f62147ffdd/gpqa_diamond.csv")

TEMPLATE = (
    "What is the correct answer to this question:{question}\n"
    "Choices:\n(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n"
    "Please reason step by step, and put your final answer (only the letter A, B, C, "
    "or D) within \\boxed{{}}.\nAnswer:"
)


def diamond(idx):
    with open(GPQA_CSV) as f:
        rows = list(csv.DictReader(f))
    r = rows[idx]
    return TEMPLATE.format(
        question=" " + (r["Question"] or "").strip(),
        a=(r["Correct Answer"] or "").strip(),
        b=(r["Incorrect Answer 1"] or "").strip(),
        c=(r["Incorrect Answer 2"] or "").strip(),
        d=(r["Incorrect Answer 3"] or "").strip())


EASY = TEMPLATE.format(
    question=(" A particle of mass m moves in a one-dimensional potential "
              "V(x) = kx^2/2. What is the ground state energy?"),
    a="hbar*omega/2", b="hbar*omega", c="2*hbar*omega", d="0")

# (name, prompt, max_tokens, gold, baseline_words)
CASES = [
    ("trivial", "What is 2 + 2? Answer with just the number.", 128, None, 2),
    ("easy", EASY, 1024, "A", 472),
    ("diamond0", diamond(0), 4096, "A", 1849),
]


def ask(prompt, max_tokens):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def adjacent_dupes(text):
    """Count 'of of' / 'larger larger' style adjacent word repeats."""
    w = re.findall(r"[A-Za-z']+", text)
    return sum(1 for a, b in zip(w, w[1:]) if a.lower() == b.lower() and len(a) > 1)


def intraword_dupes(text):
    """Count 'LetLet' / 'CalculateCalcul' style repeated-stem words."""
    n = 0
    for word in re.findall(r"[A-Za-z]{6,}", text):
        half = len(word) // 2
        for k in range(3, half + 1):
            if word[:k].lower() == word[k:2 * k].lower():
                n += 1
                break
    return n


def repetition(text, n=12):
    w = text.split()
    if len(w) < n * 2:
        return 0.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = collections.Counter(grams)
    return sum(v - 1 for v in c.values() if v > 1) / len(grams)


def boxed_letter(text):
    i = text.rfind("boxed")
    if i < 0:
        return None
    for ch in text[i:i + 24]:
        if ch in "ABCD":
            return ch
    return None


label = os.environ.get("PROBE_LABEL", "unlabelled")
print(f"=== isolation probe: {label}")
out = {"label": label, "cases": {}}
for name, prompt, budget, gold, baseline in CASES:
    print(f"\n--- {name} (max_tokens={budget}, baseline {baseline} words)", flush=True)
    try:
        d = ask(prompt, budget)
    except Exception as e:
        print(f"    REQUEST FAILED: {type(e).__name__}: {e}", flush=True)
        out["cases"][name] = {"error": f"{type(e).__name__}: {e}"}
        continue
    ch = d["choices"][0]
    text = ch["message"].get("content") or ""
    ct = d.get("usage", {}).get("completion_tokens")
    words = len(text.split())
    rec = {
        "finish_reason": ch.get("finish_reason"),
        "completion_tokens": ct,
        "words": words,
        "baseline_words": baseline,
        "words_vs_baseline": round(words / baseline, 3) if baseline else None,
        "adjacent_dupes": adjacent_dupes(text),
        "intraword_dupes": intraword_dupes(text),
        "repetition_12gram": round(repetition(text), 4),
        "boxed": boxed_letter(text),
        "gold": gold,
        "correct": (boxed_letter(text) == gold) if gold else None,
        "head": text[:150],
        "tail": text[-120:],
    }
    out["cases"][name] = rec
    print(f"    finish_reason   : {rec['finish_reason']}")
    print(f"    tokens / words  : {ct} / {words}   (baseline {baseline}, ratio {rec['words_vs_baseline']})")
    print(f"    adjacent dupes  : {rec['adjacent_dupes']}   intraword dupes: {rec['intraword_dupes']}")
    print(f"    12-gram rep     : {rec['repetition_12gram']}")
    print(f"    boxed / correct : {rec['boxed']!r} / {rec['correct']}")
    print(f"    head            : {rec['head']!r}")
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=2)

print("\n=== SUMMARY")
bad = 0
for name, r in out["cases"].items():
    if "error" in r:
        print(f"  {name}: ERROR")
        bad += 1
        continue
    flags = []
    if r["words_vs_baseline"] is not None and r["words_vs_baseline"] < 0.5:
        flags.append("WORD-COUNT COLLAPSE")
    if r["adjacent_dupes"] + r["intraword_dupes"] > 2:
        flags.append("DUPLICATION")
    if r["gold"] and not r["correct"]:
        flags.append("WRONG/NO ANSWER")
    if flags:
        bad += 1
    print(f"  {name}: {r['words']}w (x{r['words_vs_baseline']}) dupes={r['adjacent_dupes']}+{r['intraword_dupes']} "
          f"correct={r['correct']}  {' '.join(flags) if flags else 'OK'}")
print(f"  -> {bad}/{len(out['cases'])} cases degraded for {label}")

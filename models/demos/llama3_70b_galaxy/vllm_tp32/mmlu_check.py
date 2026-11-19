#!/usr/bin/env python3
"""Accuracy + garbage count from an lm-eval output dir (run_mmlu.sh writes runs/mmlu_<tag>/).

usage: mmlu_check.py <mmlu_output_dir>
Garbage = a response containing a run of backslashes (the TP32 async-scheduling failure mode)
or an empty response. Exits 0 only if garbage == 0 and accuracy >= 0.80.
"""
import glob, json, os, re, sys

d = sys.argv[1]
res = sorted(glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True))
samples = glob.glob(os.path.join(d, "**", "samples_*.jsonl"), recursive=True)
if not res or not samples:
    print(f"missing results or samples under {d}")
    sys.exit(2)
r = json.load(open(res[-1]))["results"]
key = "mmlu_llama" if "mmlu_llama" in r else sorted(r)[0]
acc = next(v for k, v in r[key].items() if k.startswith("exact_match") and "stderr" not in k)
bad = re.compile(r"\\\\{2,}")
n = garbage = empty = 0
for f in samples:
    for line in open(f):
        s = json.loads(line)
        text = s["resps"][0][0] if s.get("resps") else ""
        n += 1
        if not text.strip():
            empty += 1
        elif bad.search(text):
            garbage += 1
print(json.dumps({"accuracy": round(acc, 4), "answered": n, "garbage": garbage, "empty": empty}))
sys.exit(0 if garbage == 0 and empty == 0 and acc >= 0.80 else 1)

#!/usr/bin/env python3
"""python score_segments.py <session log> -- GSM8K accuracy (vs datasets/gsm8k_test.jsonl, first N questions) of every gsm8k_* scenario segment of a demo session log."""
import json
import re
import sys

sys.path.insert(0, "/mnt/tt-data/ssinghal/tests/tt-metal")
from models.demos.blackhole.deepseek_v41_flash.reference.e2e_analyze import extract, norm

txt = open(sys.argv[1]).read()
gold = [json.loads(l) for l in open("/mnt/tt-data/ssinghal/datasets/gsm8k_test.jsonl")]
for s in re.split(r"(?=\S* ?- === session scenario )", txt):
    m = re.search(r"=== session scenario (gsm8k\S*) \((.*?)\) ===(.*?MODE (\S+))?", s[:400])
    if not m:
        continue
    out = {
        int(u): o
        for u, o in re.findall(r"==USER (\d+) - OUTPUT\n(.*?)(?=\n==USER|\n==REPEAT|\n\d{4}-\d\d-\d\d |\Z)", s, re.S)
    }
    ok = [u for u in sorted(out) if extract(out[u]) == norm(gold[u]["answer"].split("####")[-1].strip())]
    print(f"{m.group(1)} {m.group(2)} {m.group(4) or ''}: {len(ok)}/{len(out)} correct")

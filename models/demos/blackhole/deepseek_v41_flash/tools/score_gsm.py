"""usage: score_gsm.py <log> <scenario id> : GSM8K accuracy of the demo's ==USER i - OUTPUT blocks vs gold (first N test questions)"""
import json
import re
import sys

sys.path.insert(0, "/mnt/tt-data/ssinghal/tests/tt-metal")
from models.demos.blackhole.deepseek_v41_flash.reference.e2e_analyze import extract, norm

log, scen = sys.argv[1], sys.argv[2]
txt = open(log).read()
txt = txt[txt.index(f"=== session scenario {scen} ===") :]
nxt = txt.find("=== session scenario ", 10)
txt = txt if nxt < 0 else txt[:nxt]
out = {}
for m in re.finditer(r"==USER (\d+) - OUTPUT\n(.*?)(?=\n==USER|\n==REPEAT|\n\d{4}-\d\d-\d\d |\Z)", txt, re.S):
    out[int(m.group(1))] = m.group(2)
gold = [json.loads(l) for l in open("/mnt/tt-data/ssinghal/datasets/gsm8k_test.jsonl")]
ok = [u for u in sorted(out) if extract(out[u]) == norm(gold[u]["answer"].split("####")[-1].strip())]
print(f"{scen}: {len(ok)}/{len(out)} correct; correct users {ok}")

"""Paired GSM8K scoring of the demo's plain vs spec outputs: python score_pair.py <log> <scenario id>  (session logs)."""
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


def grab(tag):
    out = {}
    for m in re.finditer(rf"==USER (\d+) - {tag}\n(.*?)(?=\n==USER|\n==REPEAT|\n\d{{4}}-\d\d-\d\d |\Z)", txt, re.S):
        out[int(m.group(1))] = m.group(2)
    return out


plain, spec = grab("OUTPUT"), grab("SPEC OUTPUT")
gold = [json.loads(l) for l in open("/mnt/tt-data/ssinghal/datasets/gsm8k_test.jsonl")]
users = sorted(set(plain) & set(spec))
both = pc = sc = ps = sp = 0
for u in users:
    g = norm(gold[u]["answer"].split("####")[-1].strip())
    a, b = extract(plain[u]) == g, extract(spec[u]) == g
    both += a and b
    ps += a and not b
    sp += b and not a
    pc += a
    sc += b
print(
    f"{scen}: {len(users)} questions; plain correct {pc}, spec correct {sc}; both correct {both}, plain-only {ps}, spec-only {sp}, both wrong {len(users) - both - ps - sp}; "
    f"identical outputs {sum(plain[u].strip() == spec[u].strip() for u in users)}"
)

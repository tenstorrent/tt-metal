import json
import os
import sys
from collections import Counter

from transformers import AutoTokenizer

W = os.environ.get("QWEN38_WEIGHTS", "/proj_sw/user_dev/weights/Qwen3.8-27B")
tok = AutoTokenizer.from_pretrained(W, trust_remote_code=True)
d = json.load(open(sys.argv[1]))
ids = d["ids"]
print(f"{sys.argv[2]}: {len(ids)} tokens, </think> x{d['text'].count('</think>')}")
print(f"{'tokens':>12} {'round-trip':>11}")
for s in range(0, len(ids), 250):
    w = ids[s : s + 250]
    if len(w) < 20:
        break
    back = tok.encode(tok.decode(w, skip_special_tokens=True), add_special_tokens=False)
    ca, cb = Counter(w), Counter(back)
    r = (sum((ca - cb).values()) + sum((cb - ca).values())) / len(w) * 100
    print(f"{s:>6}-{s+250:<5} {r:>10.1f}%  {'#'*int(r*10)}")
print("\ntail:", repr(d["text"][-400:]))

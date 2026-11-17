#!/usr/bin/env python3
"""mix.py <host> <n_requests> <threads>: random prefix-sharing mix (6 documents, random lengths 600-9000 tokens, T 0 / 0.7, 24 new tokens); counts errors and CJK strays."""
import glob
import json
import random
import re
import sys
import threading
import time
import urllib.request

from transformers import AutoTokenizer

H, N, TH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tok = AutoTokenizer.from_pretrained("/mnt/tt-data/Hy4-Preview")
t = ""
for f in sorted(glob.glob("/mnt/tt-data/ssinghal/tests/tt-metal/tech_reports/**/*.md", recursive=True)):
    t += open(f, errors="ignore").read() + "\n\n"
    if len(t) > 500000:
        break
A = tok(t, add_special_tokens=False)["input_ids"]
DOCS = [A[i * 9500 : (i + 1) * 9500] for i in range(6)]
CJK = re.compile(r"[　-鿿]")
lock = threading.Lock()
st = dict(ok=0, err=0, cjk=0)
k = [0]


def work(w):
    r = random.Random(100 + w)
    while True:
        with lock:
            if k[0] >= N:
                return
            k[0] += 1
            i = k[0]
        d = r.randrange(6)
        L = r.randrange(600, 9000)
        T = r.choice([0.0, 0.7])
        b = dict(model="tencent/Hy4-preview", prompt=DOCS[d][:L], max_tokens=24, temperature=T, stream=False)
        q = urllib.request.Request(
            f"http://{H}:8000/v1/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}
        )
        t0 = time.time()
        try:
            o = json.loads(urllib.request.urlopen(q, timeout=1800).read())
            txt = o["choices"][0]["text"]
            with lock:
                st["ok"] += 1
                st["cjk"] += len(CJK.findall(txt))
            print(f"mix {i}/{N} doc={d} L={L} T={T} wall={time.time()-t0:.1f}s ok", flush=True)
        except Exception as e:
            with lock:
                st["err"] += 1
            print(f"mix {i}/{N} doc={d} L={L} ERR {e}", flush=True)


th = [threading.Thread(target=work, args=(w,)) for w in range(TH)]
[x.start() for x in th]
[x.join() for x in th]
print("MIX", st, "PASS" if st["err"] == 0 and st["cjk"] == 0 else "FAIL")

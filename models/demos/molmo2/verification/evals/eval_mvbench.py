"""MVBench over the OpenAI-compatible server (greedy, MCQ letter). Usage: run_mvbench.py [--limit N] [--conc 4]"""
import argparse
import collections
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

R = "/home/ubuntu/ssinghal/molmo2/evals/mvbench"
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--out", default=f"{R}/results.jsonl")
ap.add_argument("--tasks", default="")
ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
a = ap.parse_args()
# index of every extracted file by relative-path suffix and basename
idx = collections.defaultdict(list)
for p in glob.glob(f"{R}/vid/**/*", recursive=True):
    idx[os.path.basename(p)].append(p)


def resolve(v):
    cands = idx.get(os.path.basename(v), [])
    tail = v.replace("\\", "/")
    for c in cands:
        if c.endswith(tail):
            return c
    return cands[0] if cands and os.path.isfile(cands[0]) else None


items = []
for f in sorted(glob.glob(f"{R}/json/*.json")):
    t = os.path.basename(f)[:-5]
    if a.tasks and t not in a.tasks.split(","):
        continue
    d = json.load(open(f))
    if a.limit:
        d = d[: a.limit]
    for i, q in enumerate(d):
        items.append((t, i, q))
L = "ABCDEFGH"


def run(it):
    t, i, q = it
    p = resolve(q["video"])
    if p is None or os.path.isdir(p):
        return dict(task=t, i=i, skipped=True)
    opts = "\n".join(f"{L[k]}. {c}" for k, c in enumerate(q["candidates"]))
    gt = L[q["candidates"].index(q["answer"])]
    txt = f"{q['question']}\n{opts}\nPlease respond with only the letter of the correct answer."
    body = {
        "model": "allenai/Molmo2-8B",
        "temperature": 0,
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": txt}, {"type": "video_url", "video_url": {"url": "file://" + p}}],
            }
        ],
    }
    t0 = time.time()
    try:
        r = requests.post(a.url, json=body, timeout=900)
        out = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else f"HTTP{r.status_code}"
        err = r.status_code != 200
    except Exception as e:
        out, err = repr(e), True
    m = re.match(r"\s*\(?([A-H])\b", out or "")
    return dict(task=t, i=i, gt=gt, out=out, pred=m.group(1) if m else None, err=err, sec=time.time() - t0)


res = []
with ThreadPoolExecutor(a.conc) as ex, open(a.out, "w") as fo:
    for n, r in enumerate(ex.map(run, items)):
        res.append(r)
        fo.write(json.dumps(r) + "\n")
        fo.flush()
        if n % 100 == 0:
            print(n, len(items), flush=True)
by = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in res:
    s = by[r["task"]]
    if r.get("skipped"):
        s[2] += 1
        continue
    s[1] += 1
    s[0] += r["pred"] == r["gt"]
    s[3] += r["err"]
tot = [0, 0]
for t, (c, n, sk, e) in sorted(by.items()):
    print(f"{t:26s} {c}/{n} = {100*c/max(n,1):5.1f}%  skipped={sk} errors={e}")
    tot[0] += c
    tot[1] += n
print(
    f"OVERALL {tot[0]}/{tot[1]} = {100*tot[0]/max(tot[1],1):.2f}%  (task-macro: {sum(100*c/max(n,1) for c,n,_,_ in by.values() if n)/max(sum(1 for v in by.values() if v[1]),1):.2f}%)"
)

import collections
import concurrent.futures as cf
import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "meta-models/Muse-Glimmer-30B"


def ask(idx_seed):
    idx, seed = idx_seed
    t0 = time.time()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Generate a list of 10 random colors."}],
        "max_tokens": 50,
        "temperature": 0.9,
        "seed": seed,
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return idx, seed, (r["choices"][0]["message"]["content"] or ""), r["id"], round(time.time() - t0, 2)


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


runs = []
for trial in range(3):
    seeds = []
    for i in range(1, 17):
        seeds += [i, 0]
    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        out = list(ex.map(ask, list(enumerate(seeds))))
    zero = [(i, t, rid, sec) for i, s, t, rid, sec in out if s == 0]
    counts = collections.Counter(t for _, t, _, _ in zero)
    majority = counts.most_common(1)[0][0]
    odd = [(i, t, rid, sec) for i, t, rid, sec in zero if t != majority]
    print(f"trial {trial}: distinct={len(counts)} outliers={[(i, sec) for i,_,_,sec in odd]}")
    for i, t, rid, sec in odd:
        d = first_div(majority, t)
        print(f"   pos={i} id={rid} secs={sec} first_char_divergence={d}")
        print(f"     majority[{max(0,d-40)}:{d+60}] = {majority[max(0,d-40):d+60]!r}")
        print(f"     outlier [{max(0,d-40)}:{d+60}] = {t[max(0,d-40):d+60]!r}")
    runs.append(len(counts))
print("distinct-per-trial:", runs)

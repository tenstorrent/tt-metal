"""Discriminate: is the seed=0 divergence in the device sampler, or in the logits?

Arm A: 32-way, device sampling (baseline repro).
Arm B: 32-way, same requests + logprobs -> the TT plugin routes to HOST sampling
       on a 4-die mesh, so vLLM's own per-request seeded RNG picks the token from
       the device logits.

If B is uniform and A is not, the logits agree across rows and the defect is in the
device seeding / device RNG.  If B also splits, the per-row logits differ.
"""
import collections
import concurrent.futures as cf
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "meta-models/Muse-Glimmer-30B"


def ask(arg):
    idx, seed, logprobs = arg
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Generate a list of 10 random colors."}],
        "max_tokens": 50,
        "temperature": 0.9,
        "seed": seed,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 1
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return idx, seed, (r["choices"][0]["message"]["content"] or "")


def run(label, logprobs, total=32):
    seeds = []
    for i in range(1, total // 2 + 1):
        seeds += [i, 0]
    args = [(i, s, logprobs) for i, s in enumerate(seeds)]
    with cf.ThreadPoolExecutor(max_workers=total) as ex:
        out = list(ex.map(ask, args))
    zero = [(i, t) for i, s, t in out if s == 0]
    nonzero = [t for i, s, t in out if s != 0]
    cz = collections.Counter(t for _, t in zero)
    print(f"--- {label} (logprobs={logprobs})")
    print(f"    seed=0: n={len(zero)} distinct={len(cz)}")
    for text, n in cz.most_common():
        idxs = [i for i, t in zero if t == text]
        print(f"      n={n} positions={idxs} :: {text[:80]!r}")
    print(f"    seed!=0: n={len(nonzero)} distinct={len(set(nonzero))}")
    sys.stdout.flush()
    return len(cz), len(set(nonzero))


if __name__ == "__main__":
    res = {}
    res["A_device"] = run("A device sampling", False)
    res["B_host"] = run("B host sampling (logprobs)", True)
    res["B_host_2"] = run("B host sampling (logprobs) trial2", True)
    print("SUMMARY (distinct_seed0, distinct_nonzero):", res)

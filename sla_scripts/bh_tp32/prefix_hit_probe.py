#!/usr/bin/env python3
# Measures the real prefix-cache hit rate from usage.prompt_tokens_details.cached_tokens.
# Server must run with enable_prompt_tokens_details. Usage: prefix_hit_probe.py <tag> [n] [rps] [prefix] [unique]
import json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

tag = sys.argv[1]
n, rps = int(sys.argv[2]) if len(sys.argv) > 2 else 120, float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
prefix_len, unique_len = (int(sys.argv[4]) if len(sys.argv) > 4 else 2048), (
    int(sys.argv[5]) if len(sys.argv) > 5 else 3072
)
url, out_tokens = "http://localhost:8000/v1/completions", 64

tok = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])
model = json.load(urllib.request.urlopen("http://localhost:8000/v1/models"))["data"][0]["id"]
rng = random.Random(4242)
vocab = [i for i in range(1000, 120000)]
prefix = [rng.choice(vocab) for _ in range(prefix_len)]


def send(i):
    ids = prefix + [rng.choice(vocab) for _ in range(unique_len)]
    body = {
        "model": model,
        "prompt": ids,
        "max_tokens": out_tokens,
        "ignore_eos": True,
        "top_k": 32,
        "top_p": 0.9,
        "temperature": 0.6,
    }
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    usage = json.load(urllib.request.urlopen(req, timeout=900))["usage"]
    details = usage.get("prompt_tokens_details") or {}
    return usage["prompt_tokens"], details.get("cached_tokens") or 0, time.time() - t0


with ThreadPoolExecutor(64) as pool:
    futs = []
    for i in range(n):
        futs.append(pool.submit(send, i))
        time.sleep(rng.expovariate(rps))
    rows = [f.result() for f in futs]

prompt = sum(r[0] for r in rows)
cached = sum(r[1] for r in rows)
res = {
    "tag": tag,
    "n": n,
    "rps": rps,
    "prefix_len": prefix_len,
    "unique_len": unique_len,
    "prompt_tokens": prompt,
    "cached_tokens": cached,
    "hit_rate": cached / prompt,
    "designed_hit_rate": prefix_len / (prefix_len + unique_len),
    "requests_with_hit": sum(1 for r in rows if r[1] > 0),
    "first_10_cached": [r[1] for r in rows[:10]],
}
print(json.dumps(res, indent=1))
with open(os.path.join(os.environ.get("OUT", "."), f"hit_{tag}.json"), "w") as f:
    json.dump(res, f, indent=1)

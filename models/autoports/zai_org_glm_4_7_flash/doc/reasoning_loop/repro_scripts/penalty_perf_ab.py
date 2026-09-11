"""Decode-rate A/B: same prompt (doc1, terminates ~1.5k tokens greedy), penalties off vs on."""
import json, sys, time, urllib.request
URL = "http://127.0.0.1:8000/v1/chat/completions"
rows = [json.loads(l) for l in open(sys.argv[1])]
prompt = rows[1]["doc"]["prompt"]
for tag, extra in [("no_penalty", {}), ("rep1.05", {"repetition_penalty": 1.05}), ("no_penalty_again", {})]:
    payload = {"model": "zai-org/GLM-4.7-Flash", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 4096, "temperature": 0.0, "seed": 42, **extra}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=3000)); dt = time.time() - t0
    n = d["usage"]["completion_tokens"]
    print(f"[{tag:16s}] tokens={n:5d} wall={dt:6.1f}s  {n/dt:5.1f} tok/s  finish={d['choices'][0]['finish_reason']}", flush=True)

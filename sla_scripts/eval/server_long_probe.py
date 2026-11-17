#!/usr/bin/env python3
"""Long counting prompts (many prefill chunks): the continuation must be the next number."""
import json
import random
import urllib.request

for k in [2500, 4000, 6000, 9500]:
    h = f"Session {random.randint(10**8, 10**9)}. Continue the list.\n"
    text = h + ", ".join(str(i) for i in range(1, k + 1)) + ","
    body = {"model": "llama70b", "prompt": text, "max_tokens": 8, "temperature": 0, "top_k": 32}
    req = urllib.request.Request(
        "http://localhost:8000/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    r = json.load(urllib.request.urlopen(req, timeout=900))
    out = r["choices"][0]["text"].strip()
    print(f"k={k} prompt_tokens={r['usage']['prompt_tokens']} {'OK' if out.startswith(str(k + 1)) else 'WRONG'} {out[:20]!r}", flush=True)

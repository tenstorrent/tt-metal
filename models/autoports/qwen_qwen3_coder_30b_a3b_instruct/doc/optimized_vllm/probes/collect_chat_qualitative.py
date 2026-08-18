"""Chat-templated qualitative pass against a live server, matching stage 08's by-hand collection.

This checkpoint declares a chat template and the shared runner sends raw
/v1/completions, so the canonical form for an instruct model is only exercised
here. Same six prompts, same greedy + sampled pair, as stage 08.
"""
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100/v1/chat/completions"
OUT = sys.argv[2]
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
PROMPTS = [
    "Write a haiku about machine learning.",
    "Explain the difference between supervised and unsupervised learning in simple terms.",
    "Complete this story: Once upon a time, in a faraway kingdom, there lived a curious young inventor who discovered",
    "What are the three laws of thermodynamics?",
    'Translate the following to French: "Hello, how are you today?"',
    "Write a Python function to calculate the Fibonacci sequence.",
]


def ask(prompt, temperature):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": temperature,
    }
    if temperature == 0.0:
        body["top_p"] = 1.0
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


out = []
for p in PROMPTS:
    g = ask(p, 0.0)
    s = ask(p, 0.7)
    out.append({"prompt": p, "greedy_completion": g, "sampled_completion": s})
    print(f"[ok] {p[:60]!r} greedy={len(g.split())}w sampled={len(s.split())}w", flush=True)

with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {OUT}")

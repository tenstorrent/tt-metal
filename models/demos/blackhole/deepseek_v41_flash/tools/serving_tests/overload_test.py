#!/usr/bin/env python3
"""More concurrent requests than slots (the vLLM 0.31 TTScheduler bug: the engine died on the (slots+1)-th concurrent request): 4 x slots requests at once, two waves, with a long-ish prompt so that prefills queue; every request must return 200 with the right answer and the server must stay healthy and serve a normal request afterwards. usage: overload_test.py <slots>"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

slots = int(sys.argv[1])
N = 4 * slots
U = "http://localhost:8000"
M = "deepseek-ai/DeepSeek-V4.1-Flash"
filler = " ".join(["alpha beta gamma delta river stone cloud forest engine window"] * 60)


def ask(i):
    a, b = 11 + i % 40, 7 + i % 13
    body = dict(
        model=M,
        messages=[
            {
                "role": "user",
                "content": f"{filler}\n\nIgnore the text above. What is {a}*{b}? Answer with just the number.",
            }
        ],
        max_tokens=16,
        temperature=0,
    )
    t = time.time()
    try:
        d = json.load(
            urllib.request.urlopen(
                urllib.request.Request(
                    U + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
                ),
                timeout=1800,
            )
        )
        out = d["choices"][0]["message"]["content"]
        return i, 200, str(a * b) in out, time.time() - t
    except Exception as e:
        return i, repr(e)[:80], False, time.time() - t


ok = True
for wave in (1, 2):
    t0 = time.time()
    with cf.ThreadPoolExecutor(N) as ex:
        res = list(ex.map(ask, range(N)))
    good = sum(1 for r in res if r[1] == 200 and r[2])
    bad = [r for r in res if not (r[1] == 200 and r[2])]
    print(
        f"wave {wave}: {N} concurrent requests on {slots} slots: {good}/{N} correct, wall {time.time()-t0:.1f}s, max latency {max(r[3] for r in res):.1f}s"
        + (f", failures {bad[:3]}" if bad else "")
    )
    ok &= good == N
try:
    h = urllib.request.urlopen(U + "/health", timeout=20).status
except Exception as e:
    h = repr(e)[:60]
print("health after the waves:", h)
ok &= h == 200
i, st, good, dt = ask(0)
print("normal request afterwards:", st, good)
ok &= st == 200 and good
print("OVERLOAD_OK" if ok else "OVERLOAD_FAIL")
sys.exit(0 if ok else 1)

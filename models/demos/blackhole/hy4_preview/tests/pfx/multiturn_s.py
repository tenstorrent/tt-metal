#!/usr/bin/env python3
"""multiturn.py <host> [turns=12] [block_sentences=190] : multi-turn chat with growing context (~30k at 12 turns). Each turn the user adds a filler block with a planted fact and asks a recall question on an earlier fact; the assistant reply is appended to history. Reports per-turn prompt tokens, TTFT/wall, recall correctness, CJK strays."""
import json
import os
import random
import re
import sys
import time
import urllib.request

H = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SENT = int(sys.argv[3]) if len(sys.argv) > 3 else 190
U = f"http://{H}:8000/v1/chat/completions"
CJK = re.compile(r"[　-鿿]")
W = "river stone cloud lamp garden window engine paper silver forest harbor bridge candle market winter violin copper meadow ladder thunder".split()


def block(t):
    rnd = random.Random(500 + t + 1000 * int(os.environ.get("SEED", "0")))
    return "".join(
        " ".join(rnd.choice(W) for _ in range(9)).capitalize() + f" number {rnd.randrange(10**6)}. "
        for _ in range(SENT)
    )


rnd = random.Random(77 + int(os.environ.get("SEED", "0")))
codes = [str(rnd.randrange(10000, 99999)) for _ in range(T)]
msgs = []
bad = 0
allok = True
for t in range(T):
    q = f"Note for the record: the secret code of vault {t+1} is {codes[t]}.\n{block(t)}\n"
    if t == 0:
        q += "Reply with just OK."
        exp = None
    else:
        k = rnd.randrange(0, t)
        q += f"Question: what is the secret code of vault {k+1}? Answer with the number only."
        exp = codes[k]
    msgs.append({"role": "user", "content": q})
    b = dict(
        model="tencent/Hy4-preview",
        messages=msgs,
        max_tokens=32,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"reasoning_effort": "no_think"},
    )
    r = urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    txt = ""
    pt = ct = 0
    for line in urllib.request.urlopen(r, timeout=1200):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        d = json.loads(line[5:])
        if d.get("usage"):
            pt, ct = d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]
        if d.get("choices") and d["choices"][0]["delta"].get("content"):
            if ttft is None:
                ttft = time.time() - t0
            txt += d["choices"][0]["delta"]["content"]
    ok = True if exp is None else (exp in txt)
    allok &= ok
    bad += len(CJK.findall(txt))
    print(
        f"turn {t+1}/{T} prompt_tokens={pt} ttft={ttft and round(ttft,2)}s wall={time.time()-t0:.1f}s ct={ct} recall={'n/a' if exp is None else ok} reply={txt.strip()[:60]!r}",
        flush=True,
    )
    msgs.append({"role": "assistant", "content": txt})
print(f"MULTITURN {'PASS' if allok and not bad else 'FAIL'} turns={T} cjk={bad} final_prompt_tokens={pt}")
sys.exit(0 if allok and not bad else 1)

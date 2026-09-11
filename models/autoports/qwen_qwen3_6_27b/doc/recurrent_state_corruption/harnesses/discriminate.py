"""The discriminating test from doc/SAMPLING_TEXT_QUALITY.md.

Mechanism A (wrong logits/sampling) vs B (detokenization / text assembly):
take the token ids the server actually sampled, detokenize them offline in ONE
shot with the HF tokenizer, and compare against the server-assembled string.

  one-shot decode clean, server string garbled  -> B (assembly)
  one-shot decode garbled                       -> A (the ids themselves)
"""
import argparse
import json
import os
import re
import sys

import requests
from transformers import AutoTokenizer

PROMPT = (
    "What is the correct answer to this question: Two quantum states with energies "
    "E1 and E2 have lifetimes of 1e-9 s and 1e-8 s respectively. We want to clearly "
    "distinguish these two energy levels. Which one of the following options could "
    "be their energy difference so that they can be clearly resolved?\n"
    "Choices:\n(A) 10^-4 eV\n(B) 10^-11 eV\n(C) 10^-8 eV\n(D) 10^-9 eV\n"
    "Please reason step by step, and your final answer must be only (A,B,C or D) "
    "within \\boxed\nAnswer:"
)

ap = argparse.ArgumentParser()
ap.add_argument("--max-tokens", type=int, default=1200)
ap.add_argument("--weights", default=os.environ.get("QWEN38_WEIGHTS", "/proj_sw/user_dev/weights/Qwen3.8-27B"))
ap.add_argument("--out", default="./discriminate_out.json")
a = ap.parse_args()

r = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3.8-27B",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": a.max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 42,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 0,
    },
    timeout=3600,
)
r.raise_for_status()
d = r.json()
ch = d["choices"][0]
msg = ch["message"]
server_text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")

lp = (ch.get("logprobs") or {}).get("content") or []
raw_tokens = [e["token"] for e in lp]
ids = []
for t in raw_tokens:
    m = re.fullmatch(r"token_id:(\d+)", t)
    ids.append(int(m.group(1)) if m else None)

print(f"finish_reason={ch.get('finish_reason')}  usage={d.get('usage')}")
print(f"logprob entries={len(lp)}  ids parsed={sum(i is not None for i in ids)}")
if not lp or any(i is None for i in ids):
    print("!! token ids unavailable; sample of raw tokens:", raw_tokens[:8])
    sys.exit(2)

tok = AutoTokenizer.from_pretrained(a.weights, trust_remote_code=True)
one_shot = tok.decode(ids, skip_special_tokens=False)
piecewise = "".join(tok.convert_ids_to_tokens(ids))

print("\n===== server-assembled TAIL (last 600 chars)")
print(repr(server_text[-600:]))
print("\n===== ONE-SHOT offline detokenization of SAME ids, TAIL (last 600 chars)")
print(repr(one_shot[-600:]))
# the answer region: everything the model emitted after </think>
if "</think>" in one_shot:
    ans = one_shot.split("</think>", 1)[1]
    print("\n===== ONE-SHOT post-</think> answer region")
    print(repr(ans[:600]))
    print("\n===== SERVER content field (what lm-eval scores)")
    print(repr((msg.get("content") or "")[:600]))

# adjacent duplicate ids: a token literally sampled twice in a row
dups = [(i, ids[i]) for i in range(1, len(ids)) if ids[i] == ids[i - 1]]
print(f"\nadjacent duplicate ids: {len(dups)} / {len(ids)}")
if dups[:10]:
    print("  e.g.", [(i, tok.decode([t])) for i, t in dups[:10]])

stripped = re.sub(r"<think>|</think>", "", one_shot)
print(f"\nlen(server_text)={len(server_text)}  len(one_shot)={len(one_shot)}")
print(f"one_shot contains server_text tail? {server_text[-40:] in one_shot if server_text else 'n/a'}")

json.dump(
    {
        "ids": ids,
        "server_text": server_text,
        "one_shot": one_shot,
        "piecewise": piecewise,
        "usage": d.get("usage"),
        "finish_reason": ch.get("finish_reason"),
    },
    open(a.out, "w"),
)
print(f"\nsaved -> {a.out}")

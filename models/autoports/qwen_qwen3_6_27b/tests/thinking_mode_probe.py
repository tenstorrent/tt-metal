# SPDX-License-Identifier: Apache-2.0
"""Does thinking mode explain the non-termination, and does disabling it fix it?

Qwen3.6-27B is a thinking model. Its chat template ends the generation prompt with
an OPEN <think> tag unless enable_thinking=false is passed:

    {%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\\n\\n</think>\\n\\n' }}      # closed empty block -> answer directly
    {%- else %}
    {{- '<think>\\n' }}                       # default -> open block

lm-eval's --apply_chat_template passes no enable_thinking, so every graded response
was generated in thinking mode and had to emit </think> before any \\boxed{} answer.

Two arms per prompt, everything else identical:
  A. default (thinking ON)  -- does </think> ever arrive, and at what token?
  B. enable_thinking=false  -- does generation terminate promptly with an answer?
"""

import json
import sys
import urllib.request
import collections

BASE = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.6-27B"

GPQA_STYLE = (
    "What is the correct answer to this question: A particle of mass m moves in a "
    "one-dimensional potential V(x) = kx^2/2. What is the ground state energy?\n"
    "Choices:\n(A) hbar*omega/2\n(B) hbar*omega\n(C) 2*hbar*omega\n(D) 0\n"
    "Please reason step by step, and put your final answer (only the letter A, B, C, "
    "or D) within \\boxed{}.\nAnswer:"
)

CASES = [
    ("trivial_thinkOFF", "What is 2 + 2? Answer with just the number.", 128, False),
    ("trivial_thinkON", "What is 2 + 2? Answer with just the number.", 4096, None),
    ("gpqa_thinkOFF", GPQA_STYLE, 1024, False),
    ("gpqa_thinkON", GPQA_STYLE, 8192, None),
]


def ask(prompt, max_tokens, enable_thinking):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def repetition(text, n=12):
    w = text.split()
    if len(w) < n * 2:
        return 0.0, None
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = collections.Counter(grams)
    dup = sum(v - 1 for v in c.values() if v > 1)
    worst, cnt = max(c.items(), key=lambda kv: kv[1])
    return dup / len(grams), (cnt, worst[:90])


out = {}
for name, prompt, budget, think in CASES:
    label = "thinking OFF" if think is False else "thinking ON (default)"
    print(f"\n=== {name}: {label}, max_tokens={budget}", flush=True)
    try:
        d = ask(prompt, budget, think)
    except Exception as e:
        print(f"  REQUEST FAILED: {type(e).__name__}: {e}", flush=True)
        out[name] = {"error": f"{type(e).__name__}: {e}"}
        continue
    ch = d["choices"][0]
    msg = ch["message"]
    text = msg.get("content") or ""
    # a configured reasoning parser would populate this; server showed reasoning_parser=''
    reasoning = msg.get("reasoning_content") or ""
    usage = d.get("usage", {})
    ct = usage.get("completion_tokens")
    rate, worst = repetition(text)
    close_idx = text.find("</think>")
    rec = {
        "arm": label,
        "finish_reason": ch.get("finish_reason"),
        "completion_tokens": ct,
        "max_tokens": budget,
        "hit_cap": ct == budget,
        "words": len(text.split()),
        "has_reasoning_content_field": bool(reasoning),
        "close_think_char_index": close_idx,
        "contains_boxed": "boxed" in text.lower(),
        "repetition_rate_12gram": round(rate, 4),
        "most_repeated": worst,
        "head": text[:180],
        "tail": text[-180:],
    }
    out[name] = rec
    print(f"  finish_reason     : {rec['finish_reason']}")
    print(f"  completion_tokens : {ct} / {budget}{'   <-- HIT CAP' if rec['hit_cap'] else ''}")
    print(f"  </think> found at : {close_idx}{'   (never closed)' if close_idx < 0 else ''}")
    print(f"  contains 'boxed'  : {rec['contains_boxed']}")
    print(f"  12-gram repetition: {rate:.1%}{'   <-- DEGENERATE LOOP' if rate > 0.30 else ''}")
    if worst:
        print(f"  most repeated x{worst[0]}: {worst[1]!r}")
    print(f"  head: {rec['head']!r}")
    print(f"  tail: {rec['tail']!r}")

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)

print("\n=== VERDICT")
off = [out.get(k, {}) for k in ("trivial_thinkOFF", "gpqa_thinkOFF")]
on = [out.get(k, {}) for k in ("trivial_thinkON", "gpqa_thinkON")]
if all(o.get("finish_reason") == "stop" and not o.get("hit_cap") for o in off if o):
    print("  thinking OFF terminates normally -> stop tokens ARE honoured, and the")
    print("  non-termination is a property of thinking mode, not of the port's sampler.")
    g = out.get("gpqa_thinkOFF", {})
    print(f"  gpqa thinkOFF: {g.get('completion_tokens')} tokens, boxed={g.get('contains_boxed')}")
else:
    print("  thinking OFF also fails to terminate -> the defect is NOT thinking mode.")
for k, o in (("trivial_thinkON", out.get("trivial_thinkON", {})), ("gpqa_thinkON", out.get("gpqa_thinkON", {}))):
    if not o:
        continue
    print(f"  {k}: {o.get('completion_tokens')} tokens, </think> at {o.get('close_think_char_index')}, "
          f"repetition {o.get('repetition_rate_12gram')}")

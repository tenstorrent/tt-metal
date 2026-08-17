# SPDX-License-Identifier: Apache-2.0
"""Does thinking mode explain the runaway generation, and on which questions?

Context. Qwen3.6-27B is a thinking model. Its chat template ends the generation
prompt with an OPEN <think> tag unless enable_thinking=false is passed:

    {%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\\n\\n</think>\\n\\n' }}      # closed empty block -> answer directly
    {%- else %}
    {{- '<think>\\n' }}                       # default -> open block

lm-eval's --apply_chat_template passes no enable_thinking, so every graded response
was produced in thinking mode.

What the first probe established (2026-08-17):
  * trivial prompts ("what is 2+2") hit a 64-token cap emitting coherent
    "Thinking Process: 1. **Identify the user's core question:** ..." at 0%
    repetition -- thinking mode consuming the budget, not a broken sampler;
  * an EASY physics multiple-choice question STOPPED on its own at 1362/4096 tokens,
    finish_reason=stop, ending "...matches option (A).\\n\\n\\boxed{A}" -- correct
    answer, 0% repetition.

So stop tokens ARE honoured and the model DOES terminate on easy items. Yet the real
GPQA Diamond document 1 consumed all 32,768 tokens. The open question is therefore
not "does it stop" but "what about the real questions makes it not stop".

This probe uses the ACTUAL GPQA Diamond rows, formatted exactly as the task's
doc_to_text formats them. Diamond row 0 is very likely the graded document 1: it is
the two-lifetimes energy-resolution physics item, and the retained 256-token sample
was severed mid-expression at "$\\Gamma_2 \\approx \\frac{\\hbar}{", the linewidth
calculation for precisely that question.
"""

import collections
import csv
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.6-27B"
GPQA_CSV = ("/huggingface/hub/datasets--Idavidrein--gpqa/snapshots/"
            "633f5ee89ab8ad4522a9f850766b73f62147ffdd/gpqa_diamond.csv")

# The task's doc_to_text, verbatim from lm_eval/tasks/gpqa/cot_zeroshot/
TEMPLATE = (
    "What is the correct answer to this question:{question}\n"
    "Choices:\n(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n"
    "Please reason step by step, and put your final answer (only the letter A, B, C, "
    "or D) within \\boxed{{}}.\nAnswer:"
)

EASY = TEMPLATE.format(
    question=(" A particle of mass m moves in a one-dimensional potential "
              "V(x) = kx^2/2. What is the ground state energy?"),
    a="hbar*omega/2", b="hbar*omega", c="2*hbar*omega", d="0")


def diamond(idx):
    """Real Diamond row, correct answer placed at (A). Returns (prompt, 'A')."""
    with open(GPQA_CSV) as f:
        rows = list(csv.DictReader(f))
    r = rows[idx]
    return TEMPLATE.format(
        question=" " + (r["Question"] or "").strip(),
        a=(r["Correct Answer"] or "").strip(),
        b=(r["Incorrect Answer 1"] or "").strip(),
        c=(r["Incorrect Answer 2"] or "").strip(),
        d=(r["Incorrect Answer 3"] or "").strip()), "A", (r.get("Subdomain") or "?")


d0, gold0, sub0 = diamond(0)
d1, gold1, sub1 = diamond(1)

# (name, prompt, max_tokens, enable_thinking, gold_letter)
CASES = [
    ("trivial_thinkOFF", "What is 2 + 2? Answer with just the number.", 128, False, None),
    ("easy_thinkOFF", EASY, 1024, False, "A"),
    ("diamond0_thinkOFF", d0, 4096, False, gold0),
    ("diamond0_thinkON", d0, 16384, None, gold0),
    ("diamond1_thinkON", d1, 16384, None, gold1),
]

print(f"diamond row 0 subdomain: {sub0}")
print(f"diamond row 1 subdomain: {sub1}")


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


def boxed_letter(text):
    i = text.rfind("boxed")
    if i < 0:
        return None
    frag = text[i:i + 24]
    for ch in frag:
        if ch in "ABCD":
            return ch
    return None


out = {}
for name, prompt, budget, think, gold in CASES:
    arm = "thinking OFF" if think is False else "thinking ON (default)"
    print(f"\n=== {name}: {arm}, max_tokens={budget}", flush=True)
    try:
        d = ask(prompt, budget, think)
    except Exception as e:
        print(f"  REQUEST FAILED: {type(e).__name__}: {e}", flush=True)
        out[name] = {"error": f"{type(e).__name__}: {e}"}
        continue
    ch = d["choices"][0]
    text = ch["message"].get("content") or ""
    reasoning = ch["message"].get("reasoning_content") or ""
    ct = d.get("usage", {}).get("completion_tokens")
    rate, worst = repetition(text)
    got = boxed_letter(text)
    rec = {
        "arm": arm,
        "finish_reason": ch.get("finish_reason"),
        "completion_tokens": ct,
        "max_tokens": budget,
        "hit_cap": ct == budget,
        "words": len(text.split()),
        "has_reasoning_content_field": bool(reasoning),
        "close_think_char_index": text.find("</think>"),
        "contains_boxed": "boxed" in text.lower(),
        "boxed_letter": got,
        "gold_letter": gold,
        "correct": (got == gold) if (got and gold) else None,
        "repetition_rate_12gram": round(rate, 4),
        "most_repeated": worst,
        "head": text[:180],
        "tail": text[-220:],
    }
    out[name] = rec
    print(f"  finish_reason     : {rec['finish_reason']}")
    print(f"  completion_tokens : {ct} / {budget}{'   <-- HIT CAP' if rec['hit_cap'] else ''}")
    print(f"  </think> at index : {rec['close_think_char_index']}"
          f"{'   (never closed)' if rec['close_think_char_index'] < 0 else ''}")
    print(f"  boxed answer      : {got!r}  gold={gold!r}  correct={rec['correct']}")
    print(f"  12-gram repetition: {rate:.1%}"
          f"{'   <-- DEGENERATE LOOP' if rate > 0.30 else ''}")
    if worst:
        print(f"  most repeated x{worst[0]}: {worst[1]!r}")
    print(f"  tail: {rec['tail']!r}")
    with open(sys.argv[1], "w") as f:      # checkpoint after every case
        json.dump(out, f, indent=2)

print("\n=== VERDICT")
off = {k: v for k, v in out.items() if v.get("arm") == "thinking OFF"}
if off and all(v.get("finish_reason") == "stop" for v in off.values()):
    print("  thinking OFF terminates on every prompt, including real Diamond items.")
    print("  -> non-termination is a property of thinking mode plus budget, and the")
    print("     eval invocation can fix it without touching the port.")
else:
    for k, v in off.items():
        if v.get("finish_reason") != "stop":
            print(f"  thinking OFF did NOT terminate on {k} -> port-side defect suspected.")

d0on = out.get("diamond0_thinkON", {})
if d0on:
    if d0on.get("hit_cap"):
        if (d0on.get("repetition_rate_12gram") or 0) > 0.30:
            print("  Diamond row 0 thinking ON: hit cap WITH high repetition -> degenerate")
            print("  greedy loop on hard items. This is a real quality defect.")
        else:
            print("  Diamond row 0 thinking ON: hit cap WITHOUT repetition -> genuinely")
            print("  long reasoning that does not converge within the budget.")
    else:
        print(f"  Diamond row 0 thinking ON stopped at {d0on.get('completion_tokens')} tokens,"
              f" correct={d0on.get('correct')} -- so the 32,768-token eval run needs another")
        print("  explanation (harness-side: double template application, or a different item).")
for k in ("diamond0_thinkOFF", "diamond0_thinkON", "diamond1_thinkON"):
    v = out.get(k)
    if v:
        print(f"  {k}: {v.get('completion_tokens')} tok, correct={v.get('correct')}, "
              f"rep={v.get('repetition_rate_12gram')}")

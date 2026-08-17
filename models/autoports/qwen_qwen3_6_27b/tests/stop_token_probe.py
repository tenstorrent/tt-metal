# SPDX-License-Identifier: Apache-2.0
"""Why does generation not stop?

GPQA at max_gen_toks=32768 consumed the whole budget on document 1 (1,887.95 s at
17.8 tok/s ~= 33.6k tokens), so removing the 256-token cap did not fix the score --
the model simply does not terminate. Three candidate causes:

  A. stop-token handling is broken, so EOS is generated but never honoured;
  B. degenerate repetition under greedy decoding (temperature 0);
  C. genuinely very long reasoning that 32768 does not fit.

These are distinguishable. A trivial prompt that cannot need many tokens
separates A from B/C: if "what is 2+2" also runs to the cap, EOS is not being
honoured. Repetition analysis on a long generation separates B from C.
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

PROBES = [
    ("trivial_math", "What is 2 + 2? Answer with just the number.", 64),
    ("trivial_fact", "Name the capital of France in one word.", 64),
    ("gpqa_style", GPQA_STYLE, 4096),
]


def ask(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def repetition(text, n=12):
    """Fraction of n-word windows that are duplicates. High => degenerate loop."""
    w = text.split()
    if len(w) < n * 2:
        return 0.0, None
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = collections.Counter(grams)
    dup = sum(v - 1 for v in c.values() if v > 1)
    worst, worst_n = max(c.items(), key=lambda kv: kv[1])
    return dup / len(grams), (worst_n, worst[:90])


out = {}
for name, prompt, budget in PROBES:
    print(f"\n=== {name}  (max_tokens={budget})", flush=True)
    try:
        d = ask(prompt, budget)
    except Exception as e:
        print(f"  REQUEST FAILED: {type(e).__name__}: {e}", flush=True)
        out[name] = {"error": f"{type(e).__name__}: {e}"}
        continue
    ch = d["choices"][0]
    text = ch["message"].get("content") or ""
    usage = d.get("usage", {})
    fr = ch.get("finish_reason")
    rate, worst = repetition(text)
    rec = {
        "finish_reason": fr,
        "completion_tokens": usage.get("completion_tokens"),
        "max_tokens": budget,
        "hit_cap": usage.get("completion_tokens") == budget,
        "words": len(text.split()),
        "ends_terminal": text.strip()[-1:] in ".!?\"')]}" if text.strip() else False,
        "contains_boxed": "boxed" in text.lower(),
        "repetition_rate_12gram": round(rate, 4),
        "most_repeated": worst,
        "head": text[:200],
        "tail": text[-200:],
    }
    out[name] = rec
    print(f"  finish_reason      : {fr}")
    print(f"  completion_tokens  : {usage.get('completion_tokens')} / {budget}"
          f"{'   <-- HIT CAP' if rec['hit_cap'] else ''}")
    print(f"  words              : {rec['words']}")
    print(f"  ends terminal punct: {rec['ends_terminal']}")
    print(f"  contains 'boxed'   : {rec['contains_boxed']}")
    print(f"  12-gram repetition : {rate:.1%}"
          f"{'   <-- DEGENERATE LOOP' if rate > 0.30 else ''}")
    if worst:
        print(f"  most repeated x{worst[0]}: {worst[1]!r}")
    print(f"  head: {rec['head']!r}")
    print(f"  tail: {rec['tail']!r}")

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)

print("\n=== VERDICT")
# CONFOUND, found when this probe was first run: it sends no chat_template_kwargs,
# so the chat template's default branch applies and the prompt ends with an OPEN
# <think>. A trivial prompt then spends its whole budget on reasoning -- observed
# 2026-08-17, both trivial prompts hit a 64-token cap emitting coherent
# "Thinking Process: 1. **Identify the user's core question:** ..." text at 0%
# repetition. That is thinking mode working, NOT stop tokens being ignored.
#
# So "trivial prompt hit the cap" does not license conclusion A. Use
# thinking_mode_probe.py, which runs the same prompts with
# chat_template_kwargs={"enable_thinking": false} as a control, to separate the two.
triv = [out.get(n, {}) for n in ("trivial_math", "trivial_fact")]
if any(t.get("hit_cap") for t in triv):
    print("  INCONCLUSIVE: a trivial prompt ran to the cap, but this probe leaves")
    print("  thinking mode ON, so the cap may simply be smaller than the reasoning")
    print("  preamble. Run thinking_mode_probe.py for the enable_thinking=false arm.")
    print("  Check the head/tail above: reasoning prose at low repetition means")
    print("  thinking mode; repeated n-grams would mean a degenerate loop.")
elif all(t.get("finish_reason") == "stop" for t in triv if t):
    print("  stop tokens ARE honoured on short answers.")
    g = out.get("gpqa_style", {})
    if g.get("hit_cap"):
        if (g.get("repetition_rate_12gram") or 0) > 0.30:
            print("  B: degenerate repetition on the long prompt -- greedy loop.")
        else:
            print("  C: long reasoning without repetition -- genuinely does not fit 4096.")
    else:
        print(f"  gpqa-style stopped on its own at {g.get('completion_tokens')} tokens"
              f" (finish_reason={g.get('finish_reason')}).")
else:
    print("  inconclusive: see per-probe finish_reason above.")

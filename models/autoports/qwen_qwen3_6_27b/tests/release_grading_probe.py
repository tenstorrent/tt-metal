# SPDX-License-Identifier: Apache-2.0
"""Reproduce the RELEASE grading condition: thinking ON plus reasoning_parser=qwen3.

Why. The release spec for this model (tt-inference-server
workflows/model_specs/dev/llm.yaml, Qwen/Qwen3.6-27B, P300X2) sets

    vllm_args:
      reasoning_parser: qwen3

which the stage-11 run did NOT (it carried reasoning_parser_name only under
metadata, which never becomes a CLI flag; the server dump showed
reasoning_parser=''). With a reasoning parser, vLLM splits the response: everything
up to </think> goes to message.reasoning_content, and message.content holds only
what follows. lm-eval grades message.content.

The same spec sets no gen_kwargs, so EvalTask's default {"stream": "False"} applies
(eval_config.py:209) and lm-eval falls back to its API default of 256 tokens for
gpqa_diamond_cot_zeroshot, whose YAML sets no max_gen_toks. Other reasoning models in
the same file DO set a budget for the same task -- zai-org/GLM-5.2 uses
max_gen_toks 200*1024, and the Kimi entries use 256*1024 for r1_gpqa_diamond.

Prediction under test: at 256 tokens the model is still inside <think>, so the parser
sends everything to reasoning_content and leaves content EMPTY, and the release GPQA
score is 0.00 rather than the 0.30 the autoport recorded without a parser.

Each case reports content and reasoning_content separately, which is the whole point.
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

TEMPLATE = (
    "What is the correct answer to this question:{question}\n"
    "Choices:\n(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n"
    "Please reason step by step, and put your final answer (only the letter A, B, C, "
    "or D) within \\boxed{{}}.\nAnswer:"
)


def diamond(idx):
    with open(GPQA_CSV) as f:
        rows = list(csv.DictReader(f))
    r = rows[idx]
    return TEMPLATE.format(
        question=" " + (r["Question"] or "").strip(),
        a=(r["Correct Answer"] or "").strip(),
        b=(r["Incorrect Answer 1"] or "").strip(),
        c=(r["Incorrect Answer 2"] or "").strip(),
        d=(r["Incorrect Answer 3"] or "").strip())


D0 = diamond(0)
EASY = TEMPLATE.format(
    question=(" A particle of mass m moves in a one-dimensional potential "
              "V(x) = kx^2/2. What is the ground state energy?"),
    a="hbar*omega/2", b="hbar*omega", c="2*hbar*omega", d="0")

# Sampling profiles that the release actually uses. Every graded run on this
# branch, and every probe until now, used greedy temperature 0 -- which is NOT
# what either the task YAML or the release spec asks for, and greedy decoding is a
# known non-convergence mode for thinking models.
GREEDY = None                                                   # what was measured
R1_TASK = {"temperature": 0.6, "top_k": 40, "top_p": 0.95}      # r1_gpqa_diamond.yaml
RELEASE = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}      # model card / spec gen_kwargs

# (name, prompt, max_tokens, enable_thinking, gold, sampling)
CASES = [
    # the exact release condition: thinking default ON, lm-eval's 256 fallback
    ("greedy_diamond0_256", D0, 256, None, "A", GREEDY),
    ("greedy_easy_256", EASY, 256, None, "A", GREEDY),
    # same, with a budget like the one GLM-5.2 gets for this very task
    ("r1sampling_easy_4096", EASY, 4096, None, "A", R1_TASK),
    ("r1sampling_diamond0_32768", D0, 32768, None, "A", R1_TASK),
    ("release_sampling_diamond0_32768", D0, 32768, None, "A", RELEASE),
    # thinking off, for contrast: does a small budget suffice then?
    ("thinkoff_diamond0_4096", D0, 4096, False, "A", GREEDY),
]


def ask(prompt, max_tokens, enable_thinking, sampling=None):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if sampling:
        payload.update(sampling)
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def boxed_letter(text):
    i = text.rfind("boxed")
    if i < 0:
        return None
    for ch in text[i:i + 24]:
        if ch in "ABCD":
            return ch
    return None


def repetition(text, n=12):
    w = text.split()
    if len(w) < n * 2:
        return 0.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = collections.Counter(grams)
    return sum(v - 1 for v in c.values() if v > 1) / len(grams)


out = {}
for name, prompt, budget, think, gold, sampling in CASES:
    arm = "thinking OFF" if think is False else "thinking ON (release default)"
    samp = "greedy t=0" if not sampling else ", ".join(f"{k}={v}" for k, v in sampling.items())
    print(f"\n=== {name}: {arm}, max_tokens={budget}, sampling: {samp}", flush=True)
    try:
        d = ask(prompt, budget, think, sampling)
    except Exception as e:
        print(f"  REQUEST FAILED: {type(e).__name__}: {e}", flush=True)
        out[name] = {"error": f"{type(e).__name__}: {e}"}
        continue
    ch = d["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    ct = d.get("usage", {}).get("completion_tokens")
    got = boxed_letter(content)
    rec = {
        "arm": arm,
        "sampling": samp,
        "finish_reason": ch.get("finish_reason"),
        "completion_tokens": ct,
        "max_tokens": budget,
        "hit_cap": ct == budget,
        # the decisive split
        "content_chars": len(content),
        "content_is_empty": len(content.strip()) == 0,
        "reasoning_content_chars": len(reasoning),
        "parser_active": bool(reasoning),
        "boxed_letter_from_content": got,
        "gold_letter": gold,
        "graded_correct": (got == gold) if (got and gold) else False,
        "repetition_rate_12gram": round(repetition(content or reasoning), 4),
        "content_tail": content[-160:],
        "reasoning_tail": reasoning[-160:],
    }
    out[name] = rec
    print(f"  finish_reason        : {rec['finish_reason']}")
    print(f"  completion_tokens    : {ct} / {budget}"
          f"{'   <-- HIT CAP' if rec['hit_cap'] else ''}")
    print(f"  reasoning_content    : {rec['reasoning_content_chars']} chars"
          f"{'   (parser ACTIVE)' if rec['parser_active'] else '   (parser INACTIVE)'}")
    print(f"  content              : {rec['content_chars']} chars"
          f"{'   <-- EMPTY, grades as [invalid]' if rec['content_is_empty'] else ''}")
    print(f"  boxed from content   : {got!r}  gold={gold!r}  graded_correct={rec['graded_correct']}")
    print(f"  content tail         : {rec['content_tail']!r}")
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=2)

print("\n=== VERDICT")
p = [v for v in out.values() if isinstance(v, dict) and v.get("parser_active")]
if not p:
    print("  reasoning_content was empty in every case -- the parser did NOT engage.")
    print("  Check that the server was started with --reasoning_parser qwen3.")
else:
    print(f"  parser engaged in {len(p)}/{len([v for v in out.values() if 'error' not in v])} cases.")
for k in ("greedy_diamond0_256", "greedy_easy_256"):
    v = out.get(k)
    if isinstance(v, dict) and "error" not in v:
        verdict = ("CONFIRMED: content empty -> release GPQA scores 0.00 on this item"
                   if v.get("content_is_empty") else
                   f"content NOT empty ({v['content_chars']} chars), boxed="
                   f"{v.get('boxed_letter_from_content')!r}")
        print(f"  {k}: {verdict}")
for k in ("r1sampling_easy_4096", "r1sampling_diamond0_32768",
          "release_sampling_diamond0_32768", "thinkoff_diamond0_4096"):
    v = out.get(k)
    if isinstance(v, dict) and "error" not in v:
        print(f"  {k}: {v.get('completion_tokens')} tok, content={v.get('content_chars')} chars, "
              f"correct={v.get('graded_correct')}")

print("\n=== SAMPLING COMPARISON (the untested variable)")
for k in ("r1sampling_diamond0_32768", "release_sampling_diamond0_32768"):
    v = out.get(k)
    if isinstance(v, dict) and "error" not in v:
        conv = "did NOT converge (hit cap)" if v.get("hit_cap") else \
               f"CONVERGED at {v.get('completion_tokens')} tokens"
        print(f"  {k}: {conv}, content={v.get('content_chars')} chars, "
              f"correct={v.get('graded_correct')}, rep={v.get('repetition_rate_12gram')}")
print("  Compare against the greedy thinking-ON run, which consumed all 32,768 tokens.")
print("  If non-greedy converges, the runaway was a sampling-configuration artifact")
print("  that the release gen_kwargs already avoids.")

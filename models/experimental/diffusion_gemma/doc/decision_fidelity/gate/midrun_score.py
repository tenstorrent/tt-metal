#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Score a GPQA run against the CUDA reference WHILE IT RUNS, from the server log.

``lm_eval`` writes its samples only after the last question, so a 198-question run offers no score
for ~3 h. ``compare_same_questions.py`` is the authoritative tool once that file exists; this is the
same comparison computed from ``DG_VLLM_METRIC {"event": "block_ids"}`` lines instead, so a bad
configuration is visible at question twenty rather than question 198.

Getting a mid-run number right needs three things that are easy to get silently wrong. Each is
CHECKED here, and the script refuses to print a score rather than print a plausible wrong one:

1. **The choice shuffle.** lm_eval reshuffles the multiple-choice options per run, so gold letters
   are per-run: the same question is "(A)" in one run and "(B)" in another. Re-rendering the task at
   the run's seed reproduces the shuffle it actually served -- verified at seed 42 against the
   finished 07-28 run, matching target 198/198, choices 198/198 and doc_id 198/198.

2. **The answer extraction.** lm_eval's own ``BoxedChoiceFilter`` is IMPORTED, not reimplemented.
   A hand-written mirror of it agreed with lm_eval on only 54 of 61 non-empty responses: it had the
   ``\\boxed`` and answer-marker stages but not the third-stage delegation to an ``[A-D]``-constrained
   ``multi_choice_regex``, nor the ``<think>`` stripping, so it silently under-counted by ~11 pp.
   The import agrees 61/61 and cannot drift when the filter changes.

3. **The request -> question mapping.** ``block_ids`` carries ``row`` and ``prompt_len``, not a
   question index. One ``server.log`` also carries EVERY stage: the smoke stage's requests come
   first, then the full stage restarts the doc sequence from zero, so "request i is doc i" mixes the
   two. The start index is found by requiring the ``prompt_len`` offset to be constant across every
   remaining request, rather than assumed or hardcoded.

Requires the venv that has lm_eval (the one the eval itself runs from), because of (1) and (2)::

    .venv_evals_common/bin/python midrun_score.py /home/zni/dg_runs/cot_rerun
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

METRIC = re.compile(r"DG_VLLM_METRIC (\{.*\})\s*$")
CANVAS = 256
DEFAULT_REFERENCE = Path(__file__).resolve().parent / "gpu_reference_by_record.json"


def rate(flags):
    n = len(flags)
    if not n:
        return 0, 0, 0.0, 0.0
    c = sum(1 for f in flags if f)
    p = c / n
    return c, n, 100.0 * p, 100.0 * math.sqrt(p * (1 - p) / n)


def answer_stage(choice_filter, text, doc):
    """WHICH filter stage produced the letter: 1 and 2 are answers, 3 is a guess, None is nothing.

    This distinction is the difference between a score and a fiction. BoxedChoiceFilter's stage 3
    delegates to an ``[A-D]``-constrained ``multi_choice_regex``: the last "(A-D)" paren ANYWHERE in
    the response, or a mention of a choice's TEXT. On a response that never stated an answer -- one
    truncated at the generation cap mid-reasoning, or a degenerate canvas -- stage 3 still returns a
    letter, and it is right about a quarter of the time.

    Measured on this model at 129 responses: stage 1 74%, stage 2 2%, **stage 3 19.5%**, nothing 4%.
    The A100 reference: stage 1 91-93%, stage 2 6-8%, stage 3 1.5%, nothing 0%. So counting stage-3
    letters as answers credited ~7 pp of pure chance to the TT side and none to the reference, and
    made a 76% real-answer rate print as "extractable 100%".
    """
    if not text or not text.strip():
        return None
    stripped = choice_filter._strip_think(text)
    if choice_filter._clean_boxed_letter(stripped):
        return 1
    if choice_filter._marker_letter(stripped):
        return 2
    got = str(choice_filter.apply([[text]], [doc])[0][0]).strip().upper().strip("()")
    return 3 if got in ("A", "B", "C", "D") else None


def render_docs(task_name: str, checkpoint: str, thinking: bool, seed: int):
    """Every question in served order, with the gold this run's own shuffle produced."""
    random.seed(seed)  # before the task is built: process_docs shuffles the choices from `random`
    from lm_eval.tasks import TaskManager, get_task_dict
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    task = get_task_dict([task_name], TaskManager())[task_name]
    while isinstance(task, dict):
        task = next(iter(task.values()))

    docs = []
    for doc in task.eval_docs:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": task.doc_to_text(doc)}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        ids = enc["input_ids"] if hasattr(enc, "keys") or isinstance(enc, dict) else enc
        while isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        docs.append(
            {
                "record": doc.get("Record ID"),
                "target": str(task.doc_to_target(doc)).strip(),
                "raw_len": len(ids),
                "doc": doc,
            }
        )
    return docs, tok


def read_requests(server_log: Path):
    """Per-request committed ids, in arrival order, across every stage in the log."""
    requests, current = [], None
    for line in server_log.open(errors="replace"):
        if "ending request at block" in line and current is not None:
            current["guard_ended"] = True
        m = METRIC.search(line)
        if not m:
            continue
        try:
            ev = json.loads(m.group(1))
        except ValueError:
            continue
        if ev.get("event") != "block_ids":
            continue
        if ev.get("block_idx") == 0:
            current = {"prompt_len": ev.get("prompt_len"), "ids": [], "guard_ended": False}
            requests.append(current)
        if current is not None:
            current["ids"].extend(ev.get("ids") or [])
    return requests


def gold_is_trustworthy(rows, ref, reps):
    """Sanity-check the re-rendered gold before any score built on it is printed.

    The gold letters here come from re-rendering the task at ``--seed``. If the run under analysis
    used a different lm_eval seed, the choice shuffle differs and every gold letter is wrong -- and
    the prompt-length alignment check CANNOT catch it, because reordering four choices does not change
    the prompt's length. That combination produced a score of 24.75% for a run lm_eval itself scored
    59.6%: a 35-point error with no warning.

    Two shuffle-independent tells, both from the 07-28 16384 CI run:

    * accuracy on responses that DID state an answer collapsing to ~the 25% random baseline while the
      stated-answer rate is healthy is not a model failure, it is a key mismatch;
    * per-question agreement with the reference far below what the two scores imply. For scores p (TT)
      and q (ref), agreement should sit near p*q + (1-p)*(1-q); the run showed 38-40% against an
      implied ~54%, while the reference's own two reps agree 80%.

    Returns a list of warning strings; empty means nothing looked wrong.
    """
    warn = []
    real = [r for r in rows if r["stage"] in (1, 2)]
    if len(real) >= 30:
        acc_real = sum(1 for r in real if r["correct"]) / len(real)
        stated = len(real) / max(1, len(rows))
        if acc_real < 0.35 and stated > 0.6:
            warn.append(
                f"accuracy on stated answers is {100*acc_real:.1f}% (near the 25% random floor) while "
                f"{100*stated:.0f}% of responses stated an answer -- that combination usually means the "
                f"gold letters do not match this run's choice shuffle, not that the model failed"
            )
    if reps and len(rows) >= 30:
        p = sum(1 for r in rows if r["correct"]) / len(rows)
        for rep in reps:
            pairs = [(r, ref[r["record"]][rep]) for r in rows if r["record"] in ref]
            if not pairs:
                continue
            q = sum(1 for _, g in pairs if g) / len(pairs)
            agree = sum(1 for r, g in pairs if r["correct"] == g) / len(pairs)
            implied = p * q + (1 - p) * (1 - q)
            if agree < implied - 0.12:
                warn.append(
                    f"per-question agreement with {rep} is {100*agree:.0f}% but the two scores imply "
                    f"~{100*implied:.0f}% -- suspect a gold/shuffle mismatch"
                )
    return warn


def align(requests, docs):
    """(start, offset) such that requests[start + i] is docs[i], or None.

    Found rather than assumed. A single server.log holds every stage, and the full stage restarts
    the doc sequence, so the served list is not one run of doc order.
    """
    for start in range(0, max(1, len(requests) - 4)):
        tail = requests[start:]
        if len(tail) < 5:
            break
        offsets = {int(r["prompt_len"]) - docs[i]["raw_len"] for i, r in enumerate(tail[: len(docs)])}
        if len(offsets) == 1:
            return start, offsets.pop()
    return None


def _finished_count(run_dir):
    """Requests lm_eval has FINISHED, from its progress bar; None when it cannot be read."""
    for name in ("full.log", "eval.log"):
        log = run_dir / name
        if log.exists():
            hits = re.findall(r"(\d+)/\d+ \[", log.read_text(errors="replace")[-6000:])
            if hits:
                return int(hits[-1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--task", default="gpqa_diamond_cot_zeroshot")
    ap.add_argument("--checkpoint", default="/home/zni/dg_models/diffusiongemma-26B-A4B-it")
    ap.add_argument("--thinking", type=int, choices=(0, 1), default=1)
    ap.add_argument("--seed", type=int, default=42, help="the eval's --seed; it fixes the choice shuffle")
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument(
        "--max-gen-toks",
        type=int,
        default=None,
        help="the eval's max_gen_toks, used to derive the truncation threshold (max_gen_toks - 256). "
        "Omit to infer it from the run's own longest response. NEVER hardcode it: it is a function of "
        "max_model_len, so a literal from one context silently reports zero truncation at another.",
    )
    args = ap.parse_args()

    server_log = args.run_dir / "server.log"
    if not server_log.exists():
        sys.exit(f"no server.log under {args.run_dir}")
    if not args.reference.exists():
        sys.exit(f"no reference at {args.reference}")

    requests = read_requests(server_log)
    print(f"run: {args.run_dir}")
    print(f"requests with committed blocks: {len(requests)}")
    if not requests:
        sys.exit("no block_ids lines -- this run predates the per-block ids audit line")

    from lm_eval.filters.extraction import BoxedChoiceFilter

    choice_filter = BoxedChoiceFilter()
    docs, tok = render_docs(args.task, args.checkpoint, bool(args.thinking), args.seed)

    aligned = align(requests, docs)
    if aligned is None:
        sys.exit(
            "no start index gives a constant prompt_len offset, so the served order is not doc order "
            "and cannot be recovered from lengths alone. Refusing to report a score -- wait for the "
            "samples file and use compare_same_questions.py."
        )
    start, offset = aligned
    if start:
        print(f"  -> doc order holds from request {start} on (constant offset {offset}); the {start}")
        print("     earlier request(s) belong to an earlier stage in the same log and are dropped")
    requests = requests[start:]

    # Drop the request still being generated: a partial completion's extracted letter is not final,
    # and including it also makes this denominator disagree with any subset derived from the same log.
    finished = _finished_count(args.run_dir)
    if finished is not None and len(requests) > finished:
        live = len(requests) - finished
        requests = requests[:finished]
        print(
            f"  -> {live} request(s) still generating dropped (lm_eval reports {finished} finished); "
            "a partial completion has no final answer"
        )

    ref = json.loads(args.reference.read_text())["per_record"]
    rows = []
    for i, req in enumerate(requests[: len(docs)]):
        d = docs[i]
        text = tok.decode(req["ids"], skip_special_tokens=True)
        got = str(choice_filter.apply([[text]], [d["doc"]])[0][0]).strip().upper()
        gold = d["target"].strip().upper()
        rows.append(
            {
                "record": d["record"],
                "correct": got.strip("()") == gold.strip("()"),
                "pick": got,
                "stage": answer_stage(choice_filter, text, d["doc"]),
                "tokens": len(req["ids"]),
                "empty": not text.strip(),
                "guard": req["guard_ended"],
            }
        )

    # EVERY served question is in the denominator. An empty or answer-free response is WRONG, which
    # is what lm_eval scores it -- dropping them is a one-sided correction, since the reference has
    # no empty responses at all.
    scored = [r for r in rows if r["record"] in ref]
    if not scored:
        sys.exit("no served question overlaps the reference yet")
    real = [r for r in scored if r["stage"] in (1, 2)]
    guessed = [r for r in scored if r["stage"] == 3]
    nothing = [r for r in scored if r["stage"] is None]
    print()
    print(
        f"served questions:  {len(scored)} of {len(docs)}   (guard-truncated " f"{sum(1 for r in rows if r['guard'])})"
    )
    print(
        f"  stated an answer (\\boxed or marker): {len(real)}/{len(scored)} = "
        f"{100.0*len(real)/len(scored):.1f}%   <- the real answer rate"
    )
    print(
        f"  letter only from the filter's GUESS stage: {len(guessed)}   "
        f"(right by chance in {sum(1 for r in guessed if r['correct'])} of them)"
    )
    print(f"  no letter at all: {len(nothing)}  (empty: {sum(1 for r in nothing if r['empty'])})")
    if len(real) < 0.95 * len(scored):
        print(f"  !! the A100 reference states an answer on 98.5% of questions. A gap here is a")
        print(f"     TERMINATION problem, not a reasoning one -- check how many hit the generation cap")
        # The cap is DERIVED, never hardcoded. It is one canvas below the eval's max_gen_toks, which
        # is itself (max_model_len - longest_prompt) floored to a canvas -- so it moves with the
        # served context. A literal 5376 (correct only for max_model_len 8192) silently marks nothing
        # as truncated at 16384, which would turn "the budget was the problem" into "the budget was
        # not the problem". With no --max-gen-toks given, fall back to the run's own observed maximum:
        # a request that stopped at the cap is by definition the longest one there can be.
        cap = (args.max_gen_toks - CANVAS) if args.max_gen_toks else max(r["tokens"] for r in scored)
        capped = [r for r in scored if r["tokens"] >= cap]
        print(
            f"     generation cap taken as {cap} tokens "
            f"({'from --max-gen-toks' if args.max_gen_toks else 'observed maximum'})"
        )
        print(
            f"     responses at/near the generation cap: {len(capped)}, of which answer-free "
            f"{sum(1 for r in capped if r['stage'] in (3, None))}"
        )

    reps_for_check = sorted(k for k in next(iter(ref.values())) if k.startswith("rep") and not k.endswith("_stage"))
    warnings = gold_is_trustworthy(scored, ref, reps_for_check)
    if warnings:
        print()
        print("  *** THE GOLD LETTERS LOOK WRONG FOR THIS RUN -- the scores below are NOT usable:")
        for w in warnings:
            print(f"  ***   {w}")
        print("  *** Gold is re-rendered at --seed, and lm_eval reshuffles the choices per run, so a")
        print("  *** different eval seed silently invalidates every letter. The prompt-length alignment")
        print("  *** check cannot catch it (reordering choices does not change prompt length).")
        print("  *** Use the run's OWN samples file via compare_same_questions.py, or pass the eval's")
        print("  *** actual --seed. The stated-answer rate and stage split above ARE still valid:")
        print("  *** they do not depend on which letter is correct.")

    print()
    print("  THREE denominators. The first is the one comparable to the reference.")
    ct, nt, pt, et = rate([r["correct"] and r["stage"] in (1, 2) for r in scored])
    print(f"  {'TT (answer-free = wrong)':<30} {ct:3d}/{nt:<3d} = {pt:6.2f}%  +/- {et:.2f} pp")
    c2, n2, p2, _ = rate([r["correct"] for r in scored])
    print(f"  {'TT (guessed letters counted)':<30} {c2:3d}/{n2:<3d} = {p2:6.2f}%   <- inflated by chance")
    if real:
        c3, n3, p3, _ = rate([r["correct"] for r in real])
        print(f"  {'TT (only where it answered)':<30} {c3:3d}/{n3:<3d} = {p3:6.2f}%   <- reasoning quality")
    print()
    answered = scored
    reps = sorted(k for k in next(iter(ref.values())) if k.startswith("rep") and not k.endswith("_stage"))
    scores = []
    for rep in reps:
        c, n, p, _ = rate([ref[r["record"]][rep] for r in answered])
        scores.append(p)
        print(f"  {'reference ' + rep:<22} {c:3d}/{n:<3d} = {p:6.2f}%")
    bar = sum(scores) / len(scores)
    gap = pt - bar
    sigma = gap / et if et else 0.0
    print(f"  {'reference mean':<22} {'':8} {bar:6.2f}%   <- the bar, on THESE questions")
    print()
    verdict = (
        "indistinguishable from the reference"
        if abs(sigma) < 2
        else ("ABOVE the reference" if gap > 0 else "BELOW the reference")
    )
    print(f"  gap: {gap:+.2f} pp = {sigma:+.2f} sigma  -> {verdict}")
    print(f"  (n={nt}: one question is worth {100.0/nt:.1f} pp)")
    print()
    for rep in reps:
        a, n, p, _ = rate([r["correct"] == ref[r["record"]][rep] for r in answered])
        print(f"  per-question agreement, TT vs {rep}: {a}/{n} = {p:.0f}%")
    if len(reps) >= 2:
        a, n, p, _ = rate([ref[r["record"]][reps[0]] == ref[r["record"]][reps[1]] for r in answered])
        print(f"  {reps[0]} vs {reps[1]}: {a}/{n} = {p:.0f}%   <- the reference's own noise floor")

    print()
    print(f"  answer distribution: {dict(sorted(Counter(r['pick'] for r in answered).items()))}")
    print("    (a collapse onto one letter is a tell, not a score)")
    # The scored set is a PREFIX of the task, so its subject mix can be skewed. Worth seeing.
    mix = Counter(ref[r["record"]].get("domain") for r in answered)
    full = Counter(v.get("domain") for v in ref.values())
    print("  subject mix (scored set is a prefix, not a sample):")
    for k in sorted(full, key=str):
        print(f"    {str(k):<12} {mix.get(k,0):3d}/{full[k]:<3d}  {100.0*mix.get(k,0)/len(answered):4.0f}% of scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

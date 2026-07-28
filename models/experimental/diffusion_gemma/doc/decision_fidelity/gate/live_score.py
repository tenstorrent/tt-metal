#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Score a GPQA run WHILE it runs, from the server log.

``lm_eval`` writes its samples and results only after the last question, so for the three-plus hours
a 198-question run takes there is no way to see whether the model is answering correctly — and
nothing else in the serving path carries the generated text. That made a long eval unobservable
until it finished, which is how a whole run gets spent on a configuration that was wrong in the
first ten questions.

``generator_vllm`` now emits a ``DG_VLLM_METRIC {"event": "block_ids", ...}`` line per committed
block carrying the committed token ids (ids, not text: the generator owns no detokenizer, and a text
field would couple the log to whichever tokenizer the server holds). This reassembles those into
per-request completions, detokenizes with the checkpoint tokenizer, extracts the answer the same way
lm_eval's ``flexible-extract`` filter does, and compares against the gold letters in
``gpu_reference.jsonl``.

Matching a served request to a question. The log carries ``row`` and ``prompt_len``, not a question
index — ``row`` is a batch slot and is reused. So the prompts are re-rendered through lm_eval's own
task object (the same rendering ``compute_prefill_whitelist.py`` uses), tokenized, and matched on
**padded prompt length plus arrival order**. That is unambiguous while requests are served one at a
time (``--max-num-seqs 1``, ``num_concurrent=1``), which is how these runs are configured; the script
CHECKS that assumption and refuses to guess when a length is ambiguous rather than reporting a
plausible wrong number.

Usage::

    live_score.py /home/zni/dg_runs/flip_8192/both
    live_score.py /home/zni/dg_runs/flip_8192/both --follow

Reports the extractable-answer rate, the answer distribution, and how many completions the guard
truncated — which is what distinguishes "reasoning wrong" from "never got to an answer". It does
NOT report accuracy: gold answers are LETTERS, and lm_eval reshuffles the choices per run, so this
run's "(B)" and the reference run's "(B)" are different answers. ``compare_same_questions.py`` is
the tool for a real score; it joins on the stable ``doc["Record ID"]`` and compares each run's own
``exact_match`` rather than letters.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

METRIC = re.compile(r"DG_VLLM_METRIC (\{.*\})\s*$")
TILE = 32

# lm_eval's flexible-extract for this task is a boxed_choice filter: the \boxed{} value first, then
# explicit answer markers, and only A-D is accepted. Mirrored here so a mid-run number and the final
# lm_eval number mean the same thing.
BOXED = re.compile(r"\\boxed\s*\{\s*\(?([A-D])\)?\s*\}")
MARKERS = [
    re.compile(r"[Tt]he answer is[^A-D]{0,12}\(?([A-D])\)?"),
    re.compile(r"[Aa]nswer\s*[::]\s*\(?([A-D])\)?"),
    re.compile(r"\*\*\s*\(?([A-D])\)?\s*\*\*\s*$"),
]


def extract_choice(text: str):
    m = BOXED.search(text)
    if m:
        return m.group(1)
    for pat in MARKERS:
        found = pat.findall(text)
        if found:
            return found[-1]
    return None


def load_gold(gate_dir: Path):
    """Gold letters per question index, from the CUDA reference arm.

    OFF BY DEFAULT, and that is not caution -- these letters are only valid for the choice shuffle
    of the run that produced them. lm_eval reshuffles the multiple-choice options per run, so the
    same question is "(A)" in one run and "(B)" in another; scoring this run's extracted letters
    against another run's gold silently produces a number that means nothing. (The same reshuffling
    is why doc_hash matches only 18 of 198 across two runs.)

    For a real accuracy number use ``compare_same_questions.py``, which joins on the stable
    ``doc["Record ID"]``, compares each run's own ``exact_match`` rather than letters, and verifies
    the join against the shuffle-independent correct-answer TEXT.

    What this script reports without gold -- extractable-answer rate, answer distribution, guard
    truncations -- is shuffle-independent and is the part that is actually useful mid-run: it
    separates "reasoning is wrong" from "never reached an answer".
    """
    path = gate_dir / "gpu_reference.jsonl"
    if not path.exists():
        return {}
    gold = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        idx = row.get("index", row.get("idx"))
        letter = row.get("gold", row.get("gold_letter", row.get("answer")))
        if idx is not None and letter:
            gold[int(idx)] = str(letter).strip().upper()[:1]
    return gold


def load_tokenizer(checkpoint: str):
    """Just the tokenizer -- all the default path needs, and it does not require lm_eval."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)


def render_prompts(task_name: str, tok, thinking: bool):
    """(padded_len, index) for every question, rendered through lm_eval's task object.

    Only needed to map a served request back to a question index, which in turn is only needed to
    score against gold letters -- so it is called only under --unsafe-gold. Keeping it out of the
    default path also keeps lm_eval out of it, which is why this runs from any venv mid-run.
    """
    from lm_eval.tasks import TaskManager, get_task_dict

    task = get_task_dict([task_name], TaskManager())[task_name]
    while isinstance(task, dict):
        task = next(iter(task.values()))
    out = []
    for idx, doc in enumerate(task.eval_docs):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": task.doc_to_text(doc)}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        ids = enc["input_ids"] if hasattr(enc, "keys") or isinstance(enc, dict) else enc
        while isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        out.append((max(TILE, -(-len(ids) // TILE) * TILE), idx))
    return out


def read_completions(server_log: Path):
    """Group block_ids lines into per-request completions, in arrival order."""
    requests, current, guard_trips = [], None, 0
    for line in server_log.open(errors="replace"):
        if "ending request at block" in line:
            guard_trips += 1
            if current is not None:
                current["guard_ended"] = True
            continue
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
            current = {"prompt_len": ev.get("prompt_len"), "ids": [], "blocks": 0, "guard_ended": False}
            requests.append(current)
        if current is None:  # log started mid-request
            continue
        current["ids"].extend(ev.get("ids") or [])
        current["blocks"] += 1
    return requests, guard_trips


def score(run_dir: Path, task: str, checkpoint: str, thinking: bool, gate_dir: Path, unsafe_gold: bool = False):
    server_log = run_dir / "server.log"
    if not server_log.exists():
        return f"no server.log under {run_dir}"

    requests, guard_trips = read_completions(server_log)
    if not requests:
        return (
            "no block_ids lines yet — this run predates the per-block ids audit line, or has not\n"
            "emitted a block. A run started before that landed can only be scored at the end."
        )

    tok = load_tokenizer(checkpoint)
    gold = load_gold(gate_dir) if unsafe_gold else {}

    served_order, unmatched, ambiguous = [None] * len(requests), 0, 0
    if gold:
        # Match on padded prompt length + arrival order. Ambiguity is reported, never guessed.
        # Only needed to reach gold; skipped entirely otherwise.
        by_len = {}
        for plen, idx in render_prompts(task, tok, thinking):
            by_len.setdefault(plen, []).append(idx)
        served_order = []
        cursor = {k: 0 for k in by_len}
        for req in requests:
            plen = max(TILE, -(-int(req["prompt_len"] or 0) // TILE) * TILE)
            pool = by_len.get(plen)
            if not pool:
                unmatched += 1
                served_order.append(None)
                continue
            i = cursor[plen]
            if i >= len(pool):
                ambiguous += 1
                served_order.append(None)
                continue
            cursor[plen] = i + 1
            served_order.append(pool[i])

    n = correct = extractable = truncated = 0
    letters = Counter()
    for req, idx in zip(requests, served_order):
        text = tok.decode(req["ids"], skip_special_tokens=True)
        pick = extract_choice(text)
        n += 1
        if req["guard_ended"]:
            truncated += 1
        if pick:
            extractable += 1
            letters[pick] += 1
            if idx is not None and gold.get(idx) and pick == gold[idx]:
                correct += 1

    L = [f"run: {run_dir}", f"requests seen: {n}   (guard-truncated: {truncated})"]
    if unmatched or ambiguous:
        L.append(f"  !! {unmatched} unmatched prompt length(s), {ambiguous} ambiguous — those are")
        L.append("     excluded from the accuracy numerator, so the score below is a LOWER bound.")
    L.append(f"extractable answers: {extractable}/{n}" + (f"  ({100.0*extractable/n:.0f}%)" if n else ""))
    if gold:
        L.append(
            f"correct: {correct}/{n}  = {100.0*correct/n:.1f}%   (of extractable: " f"{100.0*correct/extractable:.1f}%)"
            if extractable
            else f"correct: {correct}/{n}"
        )
        L.append("  !! --unsafe-gold: these letters come from ANOTHER run's choice shuffle, so this")
        L.append("     number is only valid if both runs shuffled identically. It is not a score.")
    else:
        L.append("correct: not reported here on purpose — gold LETTERS are per-run (lm_eval reshuffles")
        L.append("  the choices), so scoring this run against another run's letters is meaningless.")
        L.append("  For the real number: compare_same_questions.py <run_dir>  (joins on Record ID).")
    if letters:
        L.append(
            f"answer distribution: {dict(sorted(letters.items()))}"
            "   (a collapse onto one letter is a tell, not a score)"
        )
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--task", default="gpqa_diamond_cot_zeroshot")
    ap.add_argument("--checkpoint", default="/home/zni/dg_models/diffusiongemma-26B-A4B-it")
    ap.add_argument("--thinking", type=int, choices=(0, 1), default=1)
    ap.add_argument("--gate-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument(
        "--unsafe-gold",
        action="store_true",
        help="also score against gpu_reference.jsonl's gold LETTERS. Only meaningful if that file "
        "came from a run with the same choice shuffle as this one -- lm_eval reshuffles per run. "
        "Use compare_same_questions.py for a real score.",
    )
    ap.add_argument("-f", "--follow", action="store_true")
    ap.add_argument("-i", "--interval", type=float, default=60.0)
    args = ap.parse_args()

    run_dir = args.run_dir
    if not (run_dir / "server.log").exists():
        cands = [d for d in run_dir.iterdir() if d.is_dir() and (d / "server.log").exists()] if run_dir.is_dir() else []
        if cands:
            run_dir = max(cands, key=lambda d: (d / "server.log").stat().st_mtime)

    while True:
        print(time.strftime("%H:%M:%S"))
        print(
            score(
                run_dir,
                args.task,
                args.checkpoint,
                bool(args.thinking),
                args.gate_dir,
                unsafe_gold=args.unsafe_gold,
            ),
            flush=True,
        )
        if not args.follow:
            return 0
        time.sleep(args.interval)
        print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())

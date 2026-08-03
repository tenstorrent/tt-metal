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


def _real_filter():
    """lm_eval's own boxed_choice filter, when it is importable.

    The mirror below is NOT equivalent, and the difference is measurable: on 61 non-empty responses
    it agreed with lm_eval on 54 (88.5%), because it lacks the filter's third stage (delegation to an
    ``[A-D]``-constrained ``multi_choice_regex``, which catches a trailing "(A)" in prose) and its
    ``<think>`` stripping. Under-counting extractable answers by ~11 pp would make a healthy run look
    like a degenerate one. So prefer the real thing and keep the mirror only for the venvs that do
    not have lm_eval.
    """
    try:
        from lm_eval.filters.extraction import BoxedChoiceFilter
    except Exception:  # noqa: BLE001 - any import failure means "fall back"
        return None
    return BoxedChoiceFilter()


FILTER = _real_filter()


def extract_choice(text: str):
    if FILTER is not None:
        got = str(FILTER.apply([[text]], [{}])[0][0]).strip().upper().strip("()")
        return got if len(got) == 1 and got in "ABCD" else None
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


def render_prompts(task_name: str, tok, thinking: bool, seed: int = 42, limit: int | None = None):
    """(padded_len, index) per question, plus this run's OWN gold letters, keyed by index.

    Returns ``(pairs, gold_by_index, records_by_index)``.

    Seeding first is what makes the gold trustworthy. lm_eval seeds the global ``random`` (and numpy /
    torch) from its ``--seed`` before building the task dict, and ``process_docs`` shuffles each
    question's four choices out of that RNG -- so seeding identically reproduces the exact same
    shuffle, and therefore the same gold LETTER, as the run being watched. Verified against a finished
    run: 198/198 letters and 198/198 choice orders identical at seed 42. ``--validate-against`` re-runs
    that check rather than trusting this comment.

    ``limit`` mirrors lm_eval's ``--limit``, which takes the FIRST n docs -- without it a limited run's
    requests get matched against questions it never served.
    """
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass

    from lm_eval.tasks import TaskManager, get_task_dict

    task = get_task_dict([task_name], TaskManager())[task_name]
    while isinstance(task, dict):
        task = next(iter(task.values()))
    out, gold_by_idx, rec_by_idx = [], {}, {}
    docs = list(task.eval_docs)
    if limit is not None:
        docs = docs[:limit]
    for idx, doc in enumerate(docs):
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
        letter = str(doc.get("answer", "")).strip().upper().strip("()")
        if letter[:1] in ("A", "B", "C", "D"):
            gold_by_idx[idx] = letter[:1]
        rec_by_idx[idx] = doc.get("Record ID")
    return out, gold_by_idx, rec_by_idx


def validate_shuffle(task_name: str, tok, thinking: bool, seed: int, samples: Path) -> str:
    """Prove the reproduced shuffle equals a finished run's, joined on the stable Record ID."""
    import json as _json

    _, gold_by_idx, rec_by_idx = render_prompts(task_name, tok, thinking, seed=seed)
    repro = {rec_by_idx[i]: g for i, g in gold_by_idx.items() if rec_by_idx.get(i)}
    actual = {}
    for line in samples.open(errors="replace"):
        row = _json.loads(line)
        if row.get("filter") not in (None, "flexible-extract"):
            continue
        d = row.get("doc") or {}
        rec = d.get("Record ID")
        if rec:
            actual[rec] = str(d.get("answer", "")).strip().upper().strip("()")[:1]
    common = [k for k in actual if k in repro]
    if not common:
        return f"validate: no Record ID overlap with {samples.name} -- cannot prove the shuffle"
    same = sum(1 for k in common if repro[k] == actual[k])
    verdict = "EXACT" if same == len(common) else "MISMATCH"
    return f"validate: reproduced gold matches {samples.name} on {same}/{len(common)} " f"Record IDs -> {verdict}"


def read_completions(server_log: Path):
    """Group block_ids lines into per-request completions, in arrival order.

    Also counts ``prefill_rejected``. An unwarmed aligned prefill length ends ONE request with an
    empty answer while the server stays up, and it emits NO block_ids at all -- so a rejected request
    is invisible in this arrival-order walk and every question index after it pairs with another
    question's gold letter, exactly like the smoke-stage offset that produced 60.0% instead of 72.9%.
    Since ``DG_UPFRONT_STRICT_PREFILL_LENS`` was deleted 2026-08-03 there is no engine-fatal arm to
    stop the run, so this counter is the only thing standing between a rejection and a published
    score. It is a hard refusal, not a warning.
    """
    requests, current, guard_trips, rejected = [], None, 0, []
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
        if ev.get("event") == "prefill_rejected":
            rejected.append(ev)
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
    return requests, guard_trips, rejected


def smoke_stage_requests(run_dir: Path) -> int:
    """How many leading requests belong to the runner's SMOKE stage, per that stage's own samples.

    The runner smoke-tests a couple of questions and then runs the full eval against the SAME server,
    and both stages start at doc 0. Walking requests in arrival order without dropping the smoke ones
    shifts every question index after them, which silently pairs each answer with another question's
    gold letter -- 60.0% instead of 72.9% on the run this was found on. The count is read off the smoke
    stage's finished samples file rather than assumed to be 2.
    """
    files = sorted((run_dir / "smoke").rglob("samples_*.jsonl"))
    if not files:
        return 0
    docs = set()
    for line in files[-1].open(errors="replace"):
        try:
            docs.add(json.loads(line).get("doc_id"))
        except ValueError:
            continue
    return len(docs)


def completed_count(run_dir: Path):
    """How many requests lm_eval has FINISHED, from its progress bar; None if it cannot be read.

    Anything past that count is still generating, and a partial completion's extracted letter changes
    as blocks land -- so scoring it makes the reported rate drift with no new question answered.
    """
    log = run_dir / "full.log"
    if not log.exists():
        return None
    hits = re.findall(r"(\d+)/\d+ \[", log.read_text(errors="replace")[-6000:])
    return int(hits[-1]) if hits else None


def score(
    run_dir: Path,
    task: str,
    checkpoint: str,
    thinking: bool,
    gate_dir: Path,
    unsafe_gold: bool = False,
    seed: int | None = 42,
    limit: int | None = None,
):
    server_log = run_dir / "server.log"
    if not server_log.exists():
        return f"no server.log under {run_dir}"

    requests, guard_trips, rejected = read_completions(server_log)
    if rejected:
        lens = sorted({r.get("cache_len") for r in rejected})
        return (
            f"REFUSING TO SCORE: {len(rejected)} request(s) were REJECTED at unwarmed aligned prefill\n"
            f"length(s) {lens}. A rejected request emits no block_ids, so every question index after it\n"
            f"pairs with another question's gold letter and the score below would be meaningless.\n"
            f"Add {lens} to DG_UPFRONT_PREFILL_WARMUP_LENS and re-run."
        )
    if not requests:
        return (
            "no block_ids lines yet — this run predates the per-block ids audit line, or has not\n"
            "emitted a block. A run started before that landed can only be scored at the end."
        )

    # Score the full stage's SETTLED requests only: the smoke stage restarts at doc 0 (so keeping it
    # misaligns every later question index), and a request still generating has a partial completion
    # whose extracted letter is not final yet.
    n_smoke = smoke_stage_requests(run_dir)
    requests = requests[n_smoke:]
    done = completed_count(run_dir)
    n_live = 0 if done is None else max(0, len(requests) - done)
    if n_live:
        requests = requests[: len(requests) - n_live]
    if not requests:
        return (
            f"nothing settled yet: {n_smoke} smoke-stage request(s) dropped and "
            f"{n_live} still generating, leaving no finished request from the full stage."
        )

    tok = load_tokenizer(checkpoint)
    # Gold from THIS run's own shuffle, reproduced from the seed (see render_prompts). --unsafe-gold
    # remains the old cross-run behaviour and is strictly worse; it is kept only so existing callers
    # do not break.
    gold, gold_src = {}, ""
    if unsafe_gold:
        gold, gold_src = load_gold(gate_dir), "ANOTHER run's letters (--unsafe-gold)"
    elif seed is not None:
        gold_src = f"reproduced from --seed {seed}"

    served_order, unmatched, ambiguous = [None] * len(requests), 0, 0
    if gold or (seed is not None and not unsafe_gold):
        # Match on padded prompt length + arrival order. Ambiguity is reported, never guessed.
        pairs, gold_by_idx, _rec = render_prompts(task, tok, thinking, seed=seed or 42, limit=limit)
        if not unsafe_gold:
            gold = gold_by_idx
        by_len = {}
        for plen, idx in pairs:
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

    L = [
        f"run: {run_dir}",
        f"requests scored: {n} settled   (guard-truncated: {truncated})",
        f"  excluded: {n_smoke} smoke-stage request(s)"
        + (f", {n_live} still generating" if n_live else "")
        + (f"; lm_eval reports {done} finished" if done is not None else ""),
    ]
    if FILTER is None:
        L.append("  (approximate extractor: lm_eval is not importable here, so the built-in mirror is")
        L.append("   in use -- it agreed with lm_eval on 54/61 non-empty responses, undercounting.")
        L.append("   Run from the evals venv, or use midrun_score.py, for the exact filter.)")
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
        if unsafe_gold:
            L.append("  !! --unsafe-gold: these letters come from ANOTHER run's choice shuffle, so this")
            L.append("     number is only valid if both runs shuffled identically. It is not a score.")
        else:
            L.append(f"  gold: {gold_src}. lm_eval seeds the global RNG from --seed before building the")
            L.append("     task, and process_docs shuffles the choices out of it, so the same seed gives")
            L.append("     the same letters. Verified 198/198 against a finished run; re-check any time")
            L.append("     with --validate-against <finished samples_*.jsonl>.")
        if unmatched or ambiguous:
            L.append(f"  !! {unmatched} request(s) matched no question and {ambiguous} were ambiguous on")
            L.append("     padded prompt length -- those are EXCLUDED, so the denominator is short. If a")
            L.append("     limited run, pass --limit so the candidate set matches what was served.")
        L.append("  final number still comes from the samples file: compare_same_questions.py <run_dir>")
    else:
        L.append("correct: not reported (--seed -1). Pass the run's --seed to score, or use")
        L.append("  compare_same_questions.py <run_dir> once samples are written (joins on Record ID).")
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
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the run's lm_eval --seed. Gold letters are reproduced from it, which is what makes a "
        "mid-run accuracy trustworthy (verified 198/198 against a finished run). Pass -1 to skip "
        "scoring and report only extractability / distribution / guard trips.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="the run's lm_eval --limit, if it used one. --limit takes the FIRST n docs, so without "
        "this a limited run's requests are matched against questions it never served.",
    )
    ap.add_argument(
        "--validate-against",
        type=Path,
        default=None,
        help="a FINISHED samples_*.jsonl. Proves the reproduced shuffle equals that run's, joined on "
        "doc['Record ID'], and prints the verdict instead of scoring.",
    )
    ap.add_argument("-f", "--follow", action="store_true")
    ap.add_argument("-i", "--interval", type=float, default=60.0)
    args = ap.parse_args()

    if args.validate_against is not None:
        print(
            validate_shuffle(
                args.task, load_tokenizer(args.checkpoint), bool(args.thinking), args.seed, args.validate_against
            )
        )
        return 0

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
                seed=(None if args.seed is not None and args.seed < 0 else args.seed),
                limit=args.limit,
            ),
            flush=True,
        )
        if not args.follow:
            return 0
        time.sleep(args.interval)
        print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())

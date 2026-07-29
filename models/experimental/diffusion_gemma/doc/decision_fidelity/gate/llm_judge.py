#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ask Claude whether our model's output is a MEANINGFUL response -- instead of regex-extracting it.

The regex extractors in this directory decide "did the model answer?" and get it wrong both ways:
``boxed_choice`` stage 3 hands a letter to responses that never answered (19.5% of the TT side against
the reference's 1.5% on 2026-07-28), and nothing regex-shaped sees language drift, noise bursts, or a
block that stopped mid-thought -- those all produce text a regex is happy with.

So this sends the raw completion to Claude Opus 5 and gets back a structured verdict: is it coherent
on-task text, which language, did it commit to an answer, which choice. **Meaningful is not correct**
-- a well-argued wrong answer is meaningful, and the judge is told so.

The judge never sees the answer key: it reports which choice the response selected, and correctness is
computed locally against gold. Otherwise it could launder a non-answer into the right letter.

Usage::

    export ANTHROPIC_API_KEY=...
    pip install anthropic

    llm_judge.py /home/zni/dg_runs/cot_rerun --out /tmp/verdicts.jsonl
    llm_judge.py responses.jsonl --votes 3
    cat one_response.txt | llm_judge.py -

Three Opus 5 properties shape the request: structured outputs constrain the reply to the verdict
schema; thinking is on by default and ``max_tokens`` covers thinking plus text, so the budget is sized
for both (``--effort low`` is the cost lever); and temperature does not exist -- Opus 5 rejects it --
so ``--votes N`` gets its spread from ordinary sampling non-determinism, not a knob.

Concurrency is per ITEM (128 by default, and the httpx pool is sized to match, since httpx's own
default of 100 would otherwise cap it silently). Votes within one item run in sequence, so the calls in
flight top out at the item count.

Measured against the real API at ``--effort low``: 60 questions in **7.0 s (8.6 call/s)**, where one at
a time would be about five minutes. The vote caveat is visible in the same numbers -- 5 items at
``--votes 3`` is 12 calls but only 4 in flight, so it takes 20 s. That is fine at 198 questions and
worth knowing if you ever run many votes over a handful of responses.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = os.environ.get("LLM_JUDGE_MODEL", "claude-opus-5")
CONCURRENCY = int(os.environ.get("LLM_JUDGE_CONCURRENCY", "128"))
MAX_TOKENS = 2048  # covers thinking + the verdict; thinking is on by default on Opus 5
LETTERS = "ABCD"
MODES = ("none", "empty", "repetition", "incoherent_noise", "wrong_language", "truncated_midthought", "off_task")

SYSTEM = """You classify RAW language-model output. You are not answering the question.

MEANINGFUL is not CORRECT. A response that argues carefully and reaches the wrong conclusion is
meaningful. A response is NOT meaningful only when it is broken as text: empty, degenerate repetition,
random-token noise, in a language the prompt did not ask for, cut off mid-thought before any
conclusion, or off-task.

Judge only what is present. Never infer an answer the response does not state: if it never commits to
a choice, say so, even if one is clearly implied. A parenthesised letter somewhere in prose is NOT a
selection."""

SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "meaningful": {"type": "boolean"},
            "failure_mode": {"type": "string", "enum": list(MODES)},
            "language": {"type": "string", "description": "ISO 639-1 code of the bulk of it, or 'mixed'"},
            "answered": {"type": "boolean", "description": "did it explicitly state a final answer?"},
            # A nullable enum has to be an anyOf: `{"type": ["string","null"], "enum": [...]}` is
            # rejected with "Enum value 'A' does not match declared type" -- the enum values are
            # checked against the declared type union rather than either arm of it.
            "selected_letter": {"anyOf": [{"type": "string", "enum": list(LETTERS)}, {"type": "null"}]},
            "selected_answer_text": {"type": ["string", "null"], "description": "what it said the answer is"},
            "notes": {"type": "string", "description": "one short sentence naming the evidence"},
        },
        "required": [
            "meaningful",
            "failure_mode",
            "language",
            "answered",
            "selected_letter",
            "selected_answer_text",
            "notes",
        ],
        "additionalProperties": False,
    },
}


def truncate(text: str, budget: int) -> str:
    """Keep the head and the TAIL -- the final answer lives at the end, so a head-only cut loses it."""
    if len(text) <= budget:
        return text
    head = budget // 4
    return f"{text[:head]}\n[... {len(text) - budget} chars elided ...]\n{text[head - budget:]}"


def prompt_for(item: dict, budget: int) -> str:
    parts = []
    if item.get("question"):
        parts.append("QUESTION:\n" + truncate(item["question"], 4000))
    if item.get("choices"):
        parts.append(
            "CHOICES (as the model saw them):\n"
            + "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(item["choices"]) if i < len(LETTERS))
        )
    parts.append("RESPONSE to classify:\n<<<\n" + truncate(item.get("text") or "", budget) + "\n>>>")
    return "\n\n".join(parts)


class Tally:
    """Token counter. Locked because the calls run from many threads at once."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.input = self.output = self.cache_read = 0

    def add(self, usage) -> None:
        with self.lock:
            self.calls += 1
            self.input += int(getattr(usage, "input_tokens", 0) or 0)
            self.output += int(getattr(usage, "output_tokens", 0) or 0)
            # Every call repeats the same system prompt, so a lot of the input is served from cache and
            # billed at ~0.1x. Counting it separately keeps `input` from reading as the whole bill.
            self.cache_read += int(getattr(usage, "cache_read_input_tokens", 0) or 0)

    def line(self) -> str:
        return (
            f"tokens: {self.input:,} input + {self.output:,} output"
            + (f"  ({self.cache_read:,} of the input served from cache)" if self.cache_read else "")
            + f"   over {self.calls} call(s)"
        )


def judge_one(client, item: dict, cfg: dict, tally: Tally | None = None) -> dict:
    """One call. A refusal is HTTP 200 with stop_reason 'refusal', and with thinking on the verdict is
    not content[0] -- so neither is assumed here."""
    resp = client.beta.messages.create(
        model=cfg["model"],
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt_for(item, cfg["max_chars"])}],
        output_config={"effort": cfg["effort"], "format": SCHEMA},
        # Safety classifiers can decline; "default" retries on another model inside the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    # Count before the refusal check: a mid-stream decline still bills what it produced.
    if tally is not None:
        tally.add(getattr(resp, "usage", None))
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"judge declined ({getattr(getattr(resp, 'stop_details', None), 'category', None)})")
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
    got = json.loads(text)
    got["selected_letter"] = got.get("selected_letter") if got.get("selected_letter") in tuple(LETTERS) else None
    return got


def judge_item(client, item: dict, cfg: dict, tally: Tally | None = None) -> dict:
    """An item's verdict: empty text is settled locally, otherwise N votes majority-voted."""
    if not (item.get("text") or "").strip():
        return {
            "meaningful": False,
            "failure_mode": "empty",
            "language": None,
            "answered": False,
            "selected_letter": None,
            "selected_answer_text": None,
            "notes": "empty -- no call made",
        }
    votes = []
    for _ in range(cfg["votes"]):
        try:
            votes.append(judge_one(client, item, cfg, tally))
        except Exception as exc:  # noqa: BLE001 - one bad item must not lose the run
            return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return majority(votes)


def majority(votes: list[dict]) -> dict:
    """Field-wise majority, plus whether the votes disagreed at all."""
    out = dict(votes[0])
    for field in ("meaningful", "answered"):
        out[field] = sum(bool(v.get(field)) for v in votes) > len(votes) / 2
    for field in ("failure_mode", "language", "selected_letter"):
        out[field] = Counter(v.get(field) for v in votes).most_common(1)[0][0]
    out["votes"] = len(votes)
    out["split"] = len({(bool(v["meaningful"]), v["selected_letter"]) for v in votes}) > 1
    return out


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def judge_letter(verdict: dict, item: dict) -> str | None:
    """The letter the response picked: the judge's letter, else its verbatim answer text matched back
    against the choices. The text path is what makes this shuffle-independent.

    A response the judge marked ``answered: false`` gets no letter even if it mentioned a choice --
    crediting that is the same false credit as the regex extractor's stage 3, which is the whole
    reason for this tool.
    """
    if verdict.get("answered") is False:
        return None
    if verdict.get("selected_letter"):
        return verdict["selected_letter"]
    said, choices = norm(verdict.get("selected_answer_text") or ""), item.get("choices") or []
    if not said:
        return None
    hits = [i for i, c in enumerate(choices) if norm(c) and (norm(c) in said or said in norm(c))]
    return LETTERS[hits[0]] if len(hits) == 1 and hits[0] < len(LETTERS) else None


def load(target: Path, stage: str) -> tuple[str, list[dict]]:
    """lm_eval samples (a file or the newest under a run dir), a jsonl of responses, or one text blob."""
    if target.is_dir():
        found = sorted(target.rglob(f"{stage}/**/samples_*.jsonl")) or sorted(target.rglob("samples_*.jsonl"))
        if not found:
            sys.exit(f"no samples_*.jsonl under {target}")
        target = found[-1]

    if str(target) != "-" and "samples_" in target.name:
        items = []
        for line in target.open(errors="replace"):
            row = json.loads(line)
            if row.get("filter") not in (None, "flexible-extract"):
                continue  # one row per filter; flexible-extract is the scored one
            doc = row.get("doc") or {}
            gold = str(doc.get("answer", "")).strip().upper().strip("()")[:1]
            items.append(
                {
                    "id": doc.get("Record ID") or row.get("doc_id"),
                    "question": doc.get("Question"),
                    "choices": list(doc.get("choices") or []),
                    "gold_letter": gold if gold in LETTERS else None,
                    "text": (row.get("resps") or [[""]])[0][0],
                    "regex_correct": bool(row["exact_match"]) if "exact_match" in row else None,
                }
            )
        return f"{target} ({len(items)} questions)", items

    raw = sys.stdin.read() if str(target) == "-" else target.read_text(errors="replace")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if lines and all(ln.lstrip().startswith("{") for ln in lines):
        items = []
        for i, ln in enumerate(lines):
            row = json.loads(ln)
            row.setdefault("id", i)
            row["text"] = row.get("text") or row.get("response") or row.get("output") or ""
            items.append(row)
        return f"{target} (jsonl, {len(items)} rows)", items
    return f"{target} (single response)", [{"id": 0, "text": raw}]


def report(
    source: str,
    items: list[dict],
    verdicts: list[dict],
    cfg: dict,
    tally: Tally | None = None,
    elapsed: float | None = None,
) -> None:
    ok = [(i, v) for i, v in zip(items, verdicts) if not v.get("error")]
    errors = [v for v in verdicts if v.get("error")]
    n = len(ok)
    pct = (lambda c: f"{100.0 * c / n:.1f}%") if n else (lambda c: "n/a")

    print(f"source: {source}")
    print(f"judge: {cfg['model']}  effort={cfg['effort']}  votes={cfg['votes']}")
    print(f"items: {len(items)}   judged: {n}" + (f"   ERRORED: {len(errors)}" if errors else ""))
    print()
    meaningful = sum(1 for _i, v in ok if v["meaningful"])
    answered = sum(1 for _i, v in ok if v["answered"])
    print(f"meaningful: {meaningful}/{n}  ({pct(meaningful)})   <- coherent on-task text, correct or not")
    print(f"answered:   {answered}/{n}  ({pct(answered)})   <- explicitly stated a final answer")
    split = sum(1 for _i, v in ok if v.get("split"))
    if split:
        print(f"  ({split} item(s) split across votes -- those are the majority, not unanimous)")
    print()
    print(
        "failure modes: "
        + (", ".join(f"{k}={v}" for k, v in Counter(v["failure_mode"] for _i, v in ok).most_common()) or "none")
    )
    print(
        "languages:     "
        + ", ".join(f"{k}={v}" for k, v in Counter(v.get("language") or "?" for _i, v in ok).most_common())
    )

    graded = [(i, v) for i, v in ok if i.get("gold_letter")]
    if graded:
        correct = sum(1 for i, v in graded if judge_letter(v, i) == i["gold_letter"])
        print(
            f"\ncorrect: {correct}/{len(graded)} = {100.0 * correct / len(graded):.1f}%   "
            "(gold was never shown to the judge)"
        )
    both = [(i, v) for i, v in graded if i.get("regex_correct") is not None]
    if both:
        laundered = sum(1 for i, v in both if i["regex_correct"] and not v["answered"])
        disagree = sum(1 for i, v in both if i["regex_correct"] != (judge_letter(v, i) == i["gold_letter"]))
        print(
            f"vs the regex extractor on {len(both)} item(s): {disagree} disagreement(s); "
            f"{laundered} response(s) it scored correct that never answered"
        )
    if errors:
        print(f"\n!! {len(errors)} item(s) failed; first: {errors[0]['error']}")
        print("   Excluded from every rate above, so the denominators are short.")
    if tally is not None and tally.calls:
        print()
        print(tally.line())
        if elapsed:
            print(f"wall clock: {elapsed:.1f}s   ({tally.calls / elapsed:.1f} call/s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="a run dir, samples_*.jsonl, a jsonl of responses, or - for stdin")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="low", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--votes", type=int, default=1, help="independent calls per item, majority-voted")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY, help=f"calls in flight (default {CONCURRENCY})")
    ap.add_argument("--stage", default="full", help="which lm_eval stage's samples to read")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-chars", type=int, default=24000, help="response truncation budget (head+TAIL kept)")
    ap.add_argument("--out", type=Path, default=None, help="write one verdict per line as jsonl")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit, spending nothing")
    args = ap.parse_args()

    source, items = load(args.target, args.stage)
    if args.limit:
        items = items[: args.limit]
    if not items:
        sys.exit(f"no items in {source}")

    cfg = {"model": args.model, "effort": args.effort, "votes": max(1, args.votes), "max_chars": args.max_chars}
    if args.dry_run:
        print(f"{source}\ncalls: {sum(1 for i in items if (i.get('text') or '').strip()) * cfg['votes']}")
        print(f"\n--- system ---\n{SYSTEM}\n\n--- first prompt ---\n{prompt_for(items[0], args.max_chars)}")
        return 0

    try:
        import anthropic
        import httpx
    except ImportError as exc:
        sys.exit(f"needs the Anthropic SDK: pip install anthropic   ({exc})")

    workers = max(1, min(args.concurrency, len(items) * cfg["votes"]))
    # httpx defaults to max_connections=100, which would silently cap concurrency above that.
    client = anthropic.Anthropic(
        http_client=anthropic.DefaultHttpxClient(
            limits=httpx.Limits(max_connections=workers + 8, max_keepalive_connections=workers + 8)
        )
    )
    tally, started = Tally(), time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(lambda it: judge_item(client, it, cfg, tally), items))
    elapsed = time.monotonic() - started

    if args.out:
        with args.out.open("w") as fh:
            for item, verdict in zip(items, verdicts):
                fh.write(
                    json.dumps(
                        {
                            "id": item.get("id"),
                            "gold_letter": item.get("gold_letter"),
                            "judge_letter": None if verdict.get("error") else judge_letter(verdict, item),
                            "regex_correct": item.get("regex_correct"),
                            **verdict,
                        }
                    )
                    + "\n"
                )

    report(source, items, verdicts, cfg, tally, elapsed)
    if args.out:
        print(f"\nper-item verdicts: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

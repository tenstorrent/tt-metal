#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ask an LLM whether our model's output is a MEANINGFUL response -- instead of regex-extracting it.

Every existing scorer in this directory decides "did the model answer?" with a pile of regexes, and
that decision is measurably wrong in both directions:

* ``boxed_choice`` stage 3 hands a letter to responses that never answered -- 19.5% of the TT side
  against the reference's 1.5% on 2026-07-28 -- so the official number credits TT for prose that
  merely happens to contain a parenthesised letter.
* the mirror in ``live_score.py`` agreed with the real filter on only 54/61 non-empty responses.
* nothing regex-shaped detects the failures we actually chase: language drift, unrevealed-canvas
  noise bursts, a block that stopped mid-thought. Those all produce text a regex is happy with.

So this tool sends the raw completion to a judge model behind the Tenstorrent LiteLLM proxy and gets
back a structured verdict: is it coherent on-task text, which language is it in, did it commit to a
final answer, and which choice did it pick. **Meaningful is not the same as correct** -- a
well-argued wrong answer is meaningful, and the judge is told so explicitly. That separation is the
whole point: it splits "the model is reasoning badly" from "the model is emitting garbage", which is
the distinction every DG regression so far has turned on.

THE JUDGE NEVER SEES THE ANSWER KEY. It is asked which choice the response selected; correctness is
computed locally by comparing that against the gold letter. Putting gold in the prompt would let the
judge launder a non-answer into the right letter -- exactly the failure mode we are replacing.

Usage::

    export API_KEY=...          # or LITELLM_API_KEY, or --api-key-file
    llm_judge.py --list-models

    # a finished lm_eval run (reads doc + resps + its own exact_match, so it can also report
    # where the regex extractor and the judge disagree)
    llm_judge.py /home/zni/dg_runs/cot_rerun --out /tmp/verdicts.jsonl

    # mid-run, before lm_eval has written any samples: reassemble from the server log
    llm_judge.py /home/zni/dg_runs/flip_8192/both --from-server-log

    # anything else: one response per line of jsonl, or a whole file as one response
    llm_judge.py responses.jsonl
    cat one_response.txt | llm_judge.py -

Costs money per call, so verdicts are cached on disk keyed by (model, prompt version, text): re-runs
of the same run are free, and ``--votes N`` majority-votes N independent calls when a single
judgement is not enough.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_BASE = os.environ.get("LITELLM_BASE_URL", "https://litellm-proxy--tenstorrent.workload.tenstorrent.com")
DEFAULT_MODEL = os.environ.get("LITELLM_JUDGE_MODEL", "azure/gpt-4o")
KEY_ENV = ("LITELLM_API_KEY", "TT_LITELLM_API_KEY", "API_KEY", "OPENAI_API_KEY")

# Bumped whenever the rubric changes: it is part of the cache key, so old verdicts do not survive a
# prompt edit and silently mix two rubrics into one number.
PROMPT_VERSION = "v1"

LETTERS = "ABCDEFGH"

FAILURE_MODES = (
    "none",
    "empty",
    "repetition",
    "incoherent_noise",
    "wrong_language",
    "truncated_midthought",
    "off_task",
    "refusal",
)

JUDGE_SYSTEM = """You are a strict evaluator of RAW language-model output during hardware bring-up.

You are NOT answering the question. You are classifying the RESPONSE text you are given.

Critical distinction: MEANINGFUL is not the same as CORRECT. A response that argues carefully and
reaches the wrong conclusion is meaningful. A response is NOT meaningful only when it is broken as
text: empty, degenerate repetition, random-token noise, in a language the prompt did not ask for,
cut off mid-thought before any conclusion, off-task, or a refusal.

Judge only what is present. Never infer an answer the response does not state: if it never commits
to a choice, say so, even if one choice is clearly implied or the reasoning is heading there. A
parenthesised letter appearing somewhere in prose is NOT a selection.

Reply with a single JSON object, no prose and no code fence:
{
  "meaningful": bool,
  "failure_mode": one of ["none","empty","repetition","incoherent_noise","wrong_language",
                          "truncated_midthought","off_task","refusal"],
  "language": ISO 639-1 code of the bulk of the response, e.g. "en", "zh", "ko", or "mixed",
  "answered": bool,            // did it explicitly state a final answer?
  "selected_letter": "A".."D" or null,   // only if choices were given AND it selected one
  "selected_answer_text": string or null, // what it said the answer is, verbatim, <=120 chars
  "reasoning_present": bool,   // is there substantive reasoning, not just an answer?
  "notes": string              // <=200 chars, why. Name the evidence.
}"""


# ---------------------------------------------------------------------------- transport


def resolve_key(explicit: str | None, key_file: Path | None) -> str:
    if explicit:
        return explicit.strip()
    if key_file:
        return key_file.read_text().strip()
    for name in KEY_ENV:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    sys.exit(
        "no API key. Set one of " + ", ".join(KEY_ENV) + ", or pass --api-key-file.\n"
        "The proxy is the Tenstorrent LiteLLM gateway; the key is the same one the curl examples use."
    )


def http_json(url: str, key: str, payload: dict | None = None, timeout: float = 180.0, retries: int = 4):
    """POST (or GET, when payload is None) JSON, retrying the failures that are worth retrying.

    429 and 5xx are transient on a shared proxy; 4xx otherwise is a bad request and retrying it just
    burns wall clock. The error body is surfaced -- LiteLLM puts the real reason (unknown model, key
    out of budget) in there, and swallowing it turns a one-line fix into a debugging session.
    """
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    last = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="GET" if data is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:400]
            last = f"HTTP {exc.code}: {body}"
            if exc.code not in (408, 429) and exc.code < 500:
                raise RuntimeError(last) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(30.0, 2.0**attempt) * (0.7 + 0.6 * random.random()))
    raise RuntimeError(f"{url} failed after {retries + 1} attempts -- {last}")


def list_models(base: str, key: str) -> int:
    got = http_json(f"{base.rstrip('/')}/models", key)
    ids = sorted(str(m.get("id")) for m in (got.get("data") or []) if m.get("id"))
    print(f"{len(ids)} model(s) on {base}:")
    for mid in ids:
        print(f"  {mid}")
    if DEFAULT_MODEL not in ids:
        print(f"\n!! the default --model {DEFAULT_MODEL!r} is not in that list; pass --model explicitly")
    return 0


# ---------------------------------------------------------------------------- prompt


def truncate(text: str, max_chars: int) -> str:
    """Keep the head and the TAIL. The final answer lives at the end; a head-only cut loses it."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 4
    tail = max_chars - head
    return f"{text[:head]}\n\n[... {len(text) - max_chars} chars elided ...]\n\n{text[-tail:]}"


def build_user_prompt(item: dict, max_chars: int) -> str:
    parts = []
    if item.get("question"):
        parts.append("QUESTION the model was given:\n" + truncate(str(item["question"]), 4000))
    choices = item.get("choices") or []
    if choices:
        parts.append(
            "CHOICES (letters as the model saw them):\n"
            + "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices) if i < len(LETTERS))
        )
    parts.append("RESPONSE to classify:\n<<<RESPONSE\n" + truncate(item.get("text") or "", max_chars) + "\nRESPONSE>>>")
    if not choices:
        parts.append('There were no multiple-choice options: leave "selected_letter" null.')
    return "\n\n".join(parts)


def parse_verdict(content: str) -> dict:
    """Parse the judge's JSON, tolerating a code fence or a stray sentence around it."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        got = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"judge did not return JSON: {content[:200]!r}") from None
        got = json.loads(m.group(0))
    if not isinstance(got, dict):
        raise ValueError(f"judge returned {type(got).__name__}, not an object")
    mode = str(got.get("failure_mode") or "none")
    # A null letter is the COMMON case -- "it never selected one" is the finding, not an error path --
    # so this has to survive None without reaching `None in LETTERS`.
    raw = got.get("selected_letter")
    letter = str(raw).strip().upper().strip("()")[:1] if raw else ""
    return {
        "meaningful": bool(got.get("meaningful")),
        "failure_mode": mode if mode in FAILURE_MODES else "other",
        "language": (str(got.get("language") or "").strip().lower() or None),
        "answered": bool(got.get("answered")),
        # `letter and` is load-bearing: "" is a substring of every string, so an empty letter would
        # otherwise pass the membership test and be reported as a selection.
        "selected_letter": letter if letter and letter in LETTERS else None,
        "selected_answer_text": (
            str(got.get("selected_answer_text"))[:200] if got.get("selected_answer_text") else None
        ),
        "reasoning_present": bool(got.get("reasoning_present")),
        "notes": str(got.get("notes") or "")[:300],
    }


def judge_call(item: dict, cfg: dict, vote: int) -> dict:
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_user_prompt(item, cfg["max_chars"])},
        ],
        # Deterministic for vote 0; later votes need spread or majority voting is theatre.
        "temperature": 0.0 if vote == 0 else 0.7,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }
    got = http_json(f"{cfg['base'].rstrip('/')}/chat/completions", cfg["key"], payload, timeout=cfg["timeout"])
    choice = (got.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    verdict = parse_verdict(content)
    usage = got.get("usage") or {}
    verdict["_usage"] = {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }
    return verdict


# ---------------------------------------------------------------------------- cache


class Cache:
    """Disk cache of verdicts. Keyed on the response text, so it survives re-ordered inputs."""

    def __init__(self, path: Path | None):
        self.path = path
        self.data = {}
        self.hits = 0
        self.writes = 0  # calls actually paid for this run, as opposed to replayed
        self.lock = threading.Lock()
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text())
            except ValueError:
                self.data = {}  # a truncated cache is not worth a crash; rebuild it

    @staticmethod
    def key(item: dict, cfg: dict, vote: int) -> str:
        blob = json.dumps(
            [
                cfg["model"],
                PROMPT_VERSION,
                cfg["max_chars"],
                vote,
                item.get("text") or "",
                item.get("question") or "",
                item.get("choices") or [],
            ],
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def get(self, key: str):
        with self.lock:  # --concurrency means these run from several threads
            got = self.data.get(key)
            if got is not None:
                self.hits += 1
            return got

    def put(self, key: str, verdict: dict) -> None:
        with self.lock:
            self.data[key] = verdict
            self.writes += 1

    def flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data))


# ---------------------------------------------------------------------------- inputs


def _gold_letter(doc: dict) -> str | None:
    letter = str(doc.get("answer", "")).strip().upper().strip("()")[:1]
    return letter if letter in LETTERS else None


def load_samples(target: Path, stage: str) -> tuple[str, list[dict]]:
    """Items from an lm_eval samples_*.jsonl (or the newest under a run dir).

    Carries the run's own ``exact_match`` so the report can show where the regex extractor and the
    judge disagree -- which is the number that says whether the regexes were lying.
    """
    if target.is_dir():
        cands = sorted(target.rglob(f"{stage}/**/samples_*.jsonl")) or sorted(target.rglob("samples_*.jsonl"))
        if not cands:
            sys.exit(f"no samples_*.jsonl under {target} (mid-run? try --from-server-log)")
        path = cands[-1]
    else:
        path = target

    items = []
    for line in path.open(errors="replace"):
        if not line.strip():
            continue
        row = json.loads(line)
        # One record per filter, so every raw count is doubled. flexible-extract is the scored one.
        if row.get("filter") not in (None, "flexible-extract"):
            continue
        doc = row.get("doc") or {}
        items.append(
            {
                "id": doc.get("Record ID") or row.get("doc_id"),
                "question": doc.get("Question") or doc.get("question"),
                "choices": list(doc.get("choices") or []),
                "gold_letter": _gold_letter(doc),
                "gold_text": str(doc.get("Correct Answer", "")).strip() or None,
                "text": (row.get("resps") or [[""]])[0][0],
                "regex_correct": bool(row.get("exact_match")) if "exact_match" in row else None,
            }
        )
    return str(path), items


def load_server_log(run_dir: Path, checkpoint: str, stage_skip: bool = True) -> tuple[str, list[dict]]:
    """Items reassembled from the server log's per-block token ids, for a run still in flight.

    Reuses ``live_score.py`` rather than re-deriving the log format: the smoke-stage drop and the
    block grouping are subtle enough that a second copy would drift out of agreement with the mid-run
    scorer, and two disagreeing mid-run numbers is worse than one.
    """
    import importlib.util

    src = Path(__file__).resolve().parent / "live_score.py"
    spec = importlib.util.spec_from_file_location("dg_live_score", src)
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)

    log = run_dir / "server.log"
    if not log.exists():
        cands = [d for d in run_dir.iterdir() if d.is_dir() and (d / "server.log").exists()] if run_dir.is_dir() else []
        if not cands:
            sys.exit(f"no server.log under {run_dir}")
        run_dir = max(cands, key=lambda d: (d / "server.log").stat().st_mtime)
        log = run_dir / "server.log"

    requests, _trips = live.read_completions(log)
    if stage_skip:
        requests = requests[live.smoke_stage_requests(run_dir) :]
    tok = live.load_tokenizer(checkpoint)
    items = [
        {
            "id": f"req{i}",
            "text": tok.decode(req["ids"], skip_special_tokens=True),
            "guard_ended": bool(req.get("guard_ended")),
            "blocks": req.get("blocks"),
        }
        for i, req in enumerate(requests)
    ]
    return str(log), items


def load_plain(target: Path) -> tuple[str, list[dict]]:
    """jsonl with a text-ish field per line, or a whole file (or stdin) as one response."""
    raw = sys.stdin.read() if str(target) == "-" else target.read_text(errors="replace")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if lines and all(ln.lstrip().startswith("{") for ln in lines):
        items = []
        for i, ln in enumerate(lines):
            row = json.loads(ln)
            text = row.get("text") or row.get("response") or row.get("output") or row.get("completion")
            if text is None and isinstance(row.get("resps"), list):
                text = (row["resps"] or [[""]])[0][0]
            items.append(
                {
                    "id": row.get("id", i),
                    "question": row.get("question"),
                    "choices": list(row.get("choices") or []),
                    "gold_letter": row.get("gold_letter"),
                    "gold_text": row.get("gold_text"),
                    "text": text or "",
                }
            )
        return f"{target} (jsonl, {len(items)} rows)", items
    return f"{target} (single response)", [{"id": 0, "text": raw}]


# ---------------------------------------------------------------------------- grading


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def derive_letter(verdict: dict, item: dict) -> str | None:
    """The letter the response selected: the judge's letter, else its verbatim answer text matched
    back against the choices. The text path is what makes this shuffle-independent -- and it is a
    LOCAL match against choices the judge already saw, never against gold."""
    if verdict.get("selected_letter"):
        return verdict["selected_letter"]
    said = verdict.get("selected_answer_text")
    choices = item.get("choices") or []
    if not said or not choices:
        return None
    said_n = norm(said)
    if not said_n:
        return None
    hits = [i for i, c in enumerate(choices) if norm(c) and (norm(c) in said_n or said_n in norm(c))]
    return LETTERS[hits[0]] if len(hits) == 1 and hits[0] < len(LETTERS) else None


def majority(verdicts: list[dict]) -> dict:
    """Field-wise majority over N votes, plus how split they were."""
    out = dict(verdicts[0])
    for field in ("meaningful", "answered", "reasoning_present"):
        votes = [bool(v.get(field)) for v in verdicts]
        out[field] = sum(votes) > len(votes) / 2
    for field in ("failure_mode", "language", "selected_letter"):
        counted = Counter(v.get(field) for v in verdicts)
        out[field] = counted.most_common(1)[0][0]
    out["_votes"] = len(verdicts)
    out["_split"] = len({(bool(v.get("meaningful")), v.get("selected_letter")) for v in verdicts}) > 1
    out["_usage"] = {
        "prompt_tokens": sum(v.get("_usage", {}).get("prompt_tokens", 0) for v in verdicts),
        "completion_tokens": sum(v.get("_usage", {}).get("completion_tokens", 0) for v in verdicts),
    }
    return out


EMPTY_VERDICT = {
    "meaningful": False,
    "failure_mode": "empty",
    "language": None,
    "answered": False,
    "selected_letter": None,
    "selected_answer_text": None,
    "reasoning_present": False,
    "notes": "empty response -- classified locally, no API call",
    "_usage": {"prompt_tokens": 0, "completion_tokens": 0},
}


def judge_item(item: dict, cfg: dict, cache: Cache) -> dict:
    """One item's final verdict. Empty text is settled locally -- paying for that call is waste."""
    if not (item.get("text") or "").strip():
        return dict(EMPTY_VERDICT)
    votes = []
    for vote in range(cfg["votes"]):
        key = Cache.key(item, cfg, vote)
        got = cache.get(key)
        if got is None:
            try:
                got = judge_call(item, cfg, vote)
            except Exception as exc:  # noqa: BLE001 - one bad item must not lose the whole run
                return {"error": f"{type(exc).__name__}: {exc}"[:300]}
            cache.put(key, got)
        votes.append(got)
    return majority(votes) if len(votes) > 1 else votes[0]


# ---------------------------------------------------------------------------- report


def report(source: str, items: list[dict], verdicts: list[dict], cfg: dict, cache: Cache) -> tuple[str, float]:
    n = len(items)
    errors = [v for v in verdicts if v.get("error")]
    ok = [(i, v) for i, v in zip(items, verdicts) if not v.get("error")]
    modes = Counter(v["failure_mode"] for _i, v in ok)
    langs = Counter(v.get("language") or "?" for _i, v in ok)
    meaningful = sum(1 for _i, v in ok if v["meaningful"])
    answered = sum(1 for _i, v in ok if v["answered"])
    split = sum(1 for _i, v in ok if v.get("_split"))

    graded = [(i, v) for i, v in ok if i.get("gold_letter")]
    correct = sum(1 for i, v in graded if derive_letter(v, i) == i["gold_letter"])

    # Where the regexes lied. Both directions matter: a false credit inflates the official score, a
    # missed answer deflates it, and the two do not cancel.
    comparable = [(i, v) for i, v in graded if i.get("regex_correct") is not None]
    judge_correct = {id(i): derive_letter(v, i) == i["gold_letter"] for i, v in comparable}
    regex_only = sum(1 for i, _v in comparable if i["regex_correct"] and not judge_correct[id(i)])
    judge_only = sum(1 for i, _v in comparable if not i["regex_correct"] and judge_correct[id(i)])
    laundered = sum(1 for i, v in comparable if i["regex_correct"] and not v["answered"])

    prompt_tok = sum(v.get("_usage", {}).get("prompt_tokens", 0) for _i, v in ok)
    comp_tok = sum(v.get("_usage", {}).get("completion_tokens", 0) for _i, v in ok)

    pct = (lambda c: f"{100.0 * c / len(ok):.1f}%") if ok else (lambda c: "n/a")
    L = [
        f"source: {source}",
        f"judge: {cfg['model']}  votes={cfg['votes']}  prompt={PROMPT_VERSION}",
        f"items: {n}   judged: {len(ok)}" + (f"   ERRORED: {len(errors)}" if errors else ""),
        "",
        f"meaningful:  {meaningful}/{len(ok)}  ({pct(meaningful)})   <- coherent on-task text, correct or not",
        f"answered:    {answered}/{len(ok)}  ({pct(answered)})   <- explicitly stated a final answer",
    ]
    if split:
        L.append(f"  ({split} item(s) split across votes -- those verdicts are the majority, not unanimous)")
    L.append("")
    L.append("failure modes: " + (", ".join(f"{k}={v}" for k, v in modes.most_common()) or "none"))
    L.append("languages:     " + ", ".join(f"{k}={v}" for k, v in langs.most_common()))
    if graded:
        L.append("")
        L.append(
            f"correct (judge-selected letter vs gold): {correct}/{len(graded)}  "
            f"= {100.0 * correct / len(graded):.1f}%"
        )
        L.append("  gold was NEVER shown to the judge; the letter is matched locally.")
    if comparable:
        L.append("")
        L.append(f"vs lm_eval's regex extractor, on {len(comparable)} item(s) with both:")
        L.append(f"  regex correct but judge says wrong/no-answer: {regex_only}")
        L.append(f"  judge correct but regex missed it:            {judge_only}")
        L.append(
            f"  regex credited a response that never answered:  {laundered}"
            "   <- the laundering this tool exists to find"
        )
    if errors:
        L.append("")
        L.append(f"!! {len(errors)} item(s) failed to judge; first: {errors[0]['error']}")
        L.append("   They are excluded from every rate above, so the denominators are short.")
    L.append("")
    L.append(
        f"calls: {cache.writes} made this run, {cache.hits} replayed from cache"
        + (f" -> {cache.path}" if cache.path else "  (cache disabled)")
    )
    # Cached verdicts carry the usage of the call that produced them, so this is the cost of judging
    # this input ONCE -- not what was spent this run. Said plainly rather than quietly conflated.
    L.append(f"tokens across all verdicts (cached included): {prompt_tok} prompt + {comp_tok} completion")
    return "\n".join(L), (meaningful / len(ok) if ok else 0.0)


# ---------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "target",
        nargs="?",
        type=Path,
        help="a run dir, an lm_eval samples_*.jsonl, a jsonl of responses, a text file, or - for stdin",
    )
    ap.add_argument("--list-models", action="store_true", help="print the models the proxy exposes and exit")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"judge model (default {DEFAULT_MODEL})")
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--api-key", default=None, help="prefer the env vars: " + ", ".join(KEY_ENV))
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument(
        "--from-server-log", action="store_true", help="reassemble completions from a live run's server.log"
    )
    ap.add_argument(
        "--checkpoint", default="/home/zni/dg_models/diffusiongemma-26B-A4B-it", help="--from-server-log tokenizer"
    )
    ap.add_argument("--stage", default="full", help="which lm_eval stage's samples to read (full|smoke)")
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N items")
    ap.add_argument("--votes", type=int, default=1, help="independent judge calls per item, majority-voted")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=24000, help="response truncation budget (head+TAIL kept)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / ".llm_judge_cache.json")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write one verdict per line as jsonl")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and the item count, spend nothing")
    ap.add_argument(
        "--min-meaningful",
        type=float,
        default=None,
        help="exit 1 if the meaningful rate is below this fraction (for gating a run)",
    )
    args = ap.parse_args()

    if args.list_models:
        return list_models(args.base_url, resolve_key(args.api_key, args.api_key_file))
    if args.target is None:
        ap.error("a target is required (or --list-models)")

    if args.from_server_log:
        source, items = load_server_log(args.target, args.checkpoint)
    elif str(args.target) == "-" or (args.target.is_file() and "samples_" not in args.target.name):
        source, items = load_plain(args.target)
    else:
        source, items = load_samples(args.target, args.stage)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        print(f"no items in {source}")
        return 0

    cfg = {
        "model": args.model,
        "base": args.base_url,
        "key": "" if args.dry_run else resolve_key(args.api_key, args.api_key_file),
        "votes": max(1, args.votes),
        "max_chars": args.max_chars,
        "timeout": args.timeout,
    }

    if args.dry_run:
        print(
            f"source: {source}\nitems: {len(items)}   calls that would be made: "
            f"{sum(1 for i in items if (i.get('text') or '').strip()) * cfg['votes']}"
        )
        print(f"\n--- system ---\n{JUDGE_SYSTEM}\n\n--- first user prompt ---")
        print(build_user_prompt(items[0], args.max_chars))
        return 0

    cache = Cache(None if args.no_cache else args.cache)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        verdicts = list(pool.map(lambda it: judge_item(it, cfg, cache), items))
    cache.flush()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            for item, verdict in zip(items, verdicts):
                row = {
                    "id": item.get("id"),
                    "gold_letter": item.get("gold_letter"),
                    "judge_letter": None if verdict.get("error") else derive_letter(verdict, item),
                    "regex_correct": item.get("regex_correct"),
                    "chars": len(item.get("text") or ""),
                    **{k: v for k, v in verdict.items() if not k.startswith("_")},
                }
                fh.write(json.dumps(row) + "\n")

    text, rate = report(source, items, verdicts, cfg, cache)
    print(text)
    if args.out:
        print(f"per-item verdicts: {args.out}")
    if args.min_meaningful is not None and rate < args.min_meaningful:
        print(f"\nFAIL: meaningful rate {rate:.3f} < --min-meaningful {args.min_meaningful}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

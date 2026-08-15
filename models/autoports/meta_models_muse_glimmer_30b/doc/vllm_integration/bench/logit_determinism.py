# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Logit-level determinism through the live vLLM server.

`$vllm-integration`: "When determinism tests fail in vLLM, validate that logits
output by the model for a given prompt are reproducible across runs and batch
positions. Check both standalone model and running through vllm."

Four of this stage's sampling failures are seeded-reproducibility tests, so that
clause is triggered. `determinism_vllm.json` already pins *token* determinism —
run-to-run and across 8 concurrent batch positions — but tokens are an argmax, and
an argmax can hide a logit that wobbles below the winning margin. This measures
the logits themselves.

The logits are read through `logprobs`, which on a 4-device mesh routes the
request to vLLM's host sampler and returns `raw_logprobs` — the model's own
pre-penalty, pre-sampling output. That is exactly what should be compared:

  A. run-to-run     — same prompt twice, sequentially.
  B. cross-position — the same prompt sent N times concurrently, so the requests
                      occupy different rows of the 32-row decode batch. Each row
                      indexes its own cache slot and page-table row, so a
                      row-indexing bug shows up as one row's logits differing.
  C. vs standalone  — a sanity cross-check only. The standalone arms record token
                      *ids* and this API returns token *text*, so the two are not
                      directly comparable at logit level; the byte-level equality
                      of the two arms' full completions lives in
                      determinism_baseline_recheck.json, and that is the real
                      standalone comparison. This section records the pair so the
                      mismatch is visible rather than implied.

Usage::

    python doc/vllm_integration/bench/logit_determinism.py --server-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

FULL_MODEL = ROOT / "doc/full_model/qualitative"
OUT_DEFAULT = ROOT / "doc/vllm_integration/logit_determinism.json"


def say(*a) -> None:
    print(*a, flush=True)


def _top_logprobs(choice) -> list[dict]:
    """Per-position ``{token: logprob}`` maps, oldest first."""
    lp = choice.logprobs
    if lp is None or not lp.top_logprobs:
        return []
    return [dict(d) for d in lp.top_logprobs]


def _canonical(maps: list[dict]) -> list[list[tuple[str, float]]]:
    """The FULL per-position distribution, key-sorted so comparison is order-free.

    Emphatically not a top-1 reduction. The whole point of measuring logits rather
    than tokens is that an argmax hides a logit wobbling below the winning margin --
    and on a confident greedy step the top-1 logprob saturates at 0.0, so a top-1
    comparison is nearly information-free exactly where it needs to discriminate. A
    seeded sampler draws from the whole distribution, so the whole distribution is
    what has to be reproducible.
    """
    return [sorted(((str(k), float(v)) for k, v in m.items()), key=lambda kv: kv[0]) for m in maps]


def _max_delta(a: list[list[tuple[str, float]]], b: list[list[tuple[str, float]]]) -> float:
    """Largest absolute logprob difference over every position and every candidate."""
    worst = 0.0
    for pa, pb in zip(a, b):
        da, db = dict(pa), dict(pb)
        for tok in set(da) | set(db):
            if tok not in da or tok not in db:
                return float("inf")  # candidate sets differ: not comparable, so not identical
            worst = max(worst, abs(da[tok] - db[tok]))
    return worst


def _candidates(a: list[list[tuple[str, float]]]) -> int:
    return sum(len(p) for p in a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://localhost:8000")
    ap.add_argument("--hf-model", default="meta-models/Muse-Glimmer-30B")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--concurrent", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    import openai

    client = openai.OpenAI(base_url=f"{args.server_url.rstrip('/')}/v1", api_key="dummy", timeout=1800)
    pinned = json.loads((FULL_MODEL / "qualitative_prompts.json").read_text())
    probe = pinned[0]

    def ask():
        r = client.completions.create(
            model=args.hf_model,
            prompt=probe["token_ids"],
            max_tokens=args.max_tokens,
            temperature=0.0,
            logprobs=args.top_logprobs,
        )
        return _canonical(_top_logprobs(r.choices[0]))

    report: dict = {
        "prompt_id": probe["id"],
        "prompt_tokens": len(probe["token_ids"]),
        "max_tokens": args.max_tokens,
        "top_logprobs": args.top_logprobs,
        "note": (
            "logprobs route to vLLM's host sampler on this 4-device mesh and return raw "
            "(pre-penalty) logprobs, i.e. the model's own output; that is what makes them a "
            "valid probe of logit determinism"
        ),
    }

    # --- A. run to run -------------------------------------------------------
    first, second = ask(), ask()
    report["run_to_run"] = {
        "positions": len(first),
        "candidates_compared": _candidates(first),
        "candidate_sets_match": [[t for t, _ in p] for p in first] == [[t for t, _ in p] for p in second],
        "max_abs_logprob_delta": _max_delta(first, second),
        "bitwise_identical": first == second,
        "top1_per_position": [p and max(p, key=lambda kv: kv[1])[0] for p in first],
        "full_distribution": first,
    }
    say(
        f"RUN_TO_RUN bitwise_identical={report['run_to_run']['bitwise_identical']} "
        f"candidates={report['run_to_run']['candidates_compared']} "
        f"max_delta={report['run_to_run']['max_abs_logprob_delta']}"
    )

    # --- B. cross batch position --------------------------------------------
    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        rows = [f.result() for f in [pool.submit(ask) for _ in range(args.concurrent)]]
    distinct = {json.dumps(r, sort_keys=True) for r in rows}
    deltas = [_max_delta(r, first) for r in rows]
    report["cross_batch_position"] = {
        "concurrent_requests": args.concurrent,
        "candidates_compared_per_request": _candidates(first),
        "distinct_logit_distributions": len(distinct),
        "all_identical": len(distinct) == 1,
        "matches_sequential_run": all(r == first for r in rows),
        "max_abs_logprob_delta_vs_sequential": max(deltas, default=0.0),
    }
    say(
        f"CROSS_BATCH distinct={len(distinct)} candidates={_candidates(first)} all_identical="
        f"{report['cross_batch_position']['all_identical']} "
        f"max_delta={report['cross_batch_position']['max_abs_logprob_delta_vs_sequential']}"
    )

    # --- C. against the standalone model ------------------------------------
    sweep = ROOT / "doc/datatype_sweep/qualitative/qualitative_tt_chat.json"
    if sweep.is_file():
        base = {i["id"]: i for i in json.loads(sweep.read_text())}.get(probe["id"])
        if base:
            served_first_tok = max(first[0], key=lambda kv: kv[1])[0] if first and first[0] else None
            report["vs_standalone"] = {
                "source": str(sweep.relative_to(REPO)),
                "standalone_first_generated_token_id": base["token_ids"][0],
                "served_top1_token_repr": served_first_tok,
                "note": (
                    "the standalone arm records token ids and the API returns token text, so this "
                    "row is a sanity cross-check; the byte-level equality of the two arms' full "
                    "completions is in determinism_baseline_recheck.json"
                ),
            }

    report["verdict"] = {
        "logits_reproducible_run_to_run": report["run_to_run"]["bitwise_identical"],
        "logits_reproducible_across_batch_positions": report["cross_batch_position"]["all_identical"],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    say(f"LOGIT_DETERMINISM_OK -> {args.out}")
    ok = all(report["verdict"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

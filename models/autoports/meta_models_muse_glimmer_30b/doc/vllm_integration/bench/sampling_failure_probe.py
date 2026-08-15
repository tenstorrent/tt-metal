# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Measure -- not assert -- the two non-reproducibility sampling failures.

Both failures are of the form "the test asserts an *observable* consequence and
the observable did not happen".  Neither test can distinguish "the feature does
nothing" from "the feature does exactly what it should and the consequence the
test looks for does not follow on this checkpoint".  This probe measures the
underlying quantity through the live server so the distinction is decidable.

ITEM 1  test_tt_penalties.py::TestPresencePenalty (both)
    The tests sweep presence_penalty and require the *greedy text* to differ.
    Presence penalty subtracts `p` (|p| <= 2 by API) from the logit of every
    token that has already been emitted, once, regardless of count.  So it can
    only change a greedy argmax when the winning token has already appeared and
    leads the best not-yet-appeared candidate by less than `p`.
    We request logprobs and measure that margin at every decode step, plus the
    exact per-token logit shift the penalty produced, via the identity

        [lp_i(p) - lp_i(0)] - [lp_j(p) - lp_j(0)] = -(p) for i appeared, j not

    (the log-partition term cancels), which is a *direct* measurement that the
    penalty reached the logits through vLLM.

ITEM 2  test_host_only_params.py::TestHostOnlyParameters::test_allowed_token_ids
    Asserts non-empty text.  Request 0 (allowed ids [1,2,3]) returned ''.  We
    show whether the request actually generated its 10 tokens (usage +
    token ids) -- i.e. whether the empty string is a detokenizer property of
    byte-fallback ids or an actual serving failure.

Usage (against a live server, see bench/serve.sh hold):

    python models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/bench/sampling_failure_probe.py \
        --server-url http://localhost:8000 \
        --out models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/sampling_failure_probe.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

PROMPT = "a b c a b c a b c"
SWEEP = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
N_LOGPROBS = 20  # legacy-completions cap is 20 in most builds


async def _completion(client, model, **kw):
    return await client.completions.create(model=model, prompt=kw.pop("prompt", PROMPT), **kw)


def _steps(resp):
    """-> list of {token, logprob, top: {token_id_str: logprob}} in emission order."""
    lp = resp.choices[0].logprobs
    out = []
    for i, tok in enumerate(lp.tokens):
        out.append(
            {
                "token": tok,
                "logprob": lp.token_logprobs[i],
                "top": dict(lp.top_logprobs[i]) if lp.top_logprobs else {},
            }
        )
    return out


# --------------------------------------------------------------------------
# ITEM 1
# --------------------------------------------------------------------------
async def item1(client, model, max_tokens: int) -> dict:
    print("\n=== ITEM 1: presence penalty ===", flush=True)

    # (a) reproduce the test's own observable: 8 concurrent, greedy, text only.
    reqs = [
        _completion(
            client,
            model,
            max_tokens=max_tokens,
            temperature=0,
            presence_penalty=p,
            extra_body={"add_special_tokens": True},
        )
        for p in SWEEP
    ]
    texts = [r.choices[0].text for r in await asyncio.gather(*reqs)]
    uniq = len(set(texts))
    print(f"  concurrent greedy sweep {SWEEP}: {uniq} unique of {len(texts)}", flush=True)

    # (b) the measurement: same sweep, but with logprobs, so the logit shift is visible.
    #     Run one at a time so batch composition cannot be blamed for the numbers.
    per_penalty = {}
    for p in (0.0, 2.0, -1.5):
        r = await _completion(
            client,
            model,
            max_tokens=max_tokens,
            temperature=0,
            presence_penalty=p,
            logprobs=N_LOGPROBS,
            extra_body={"return_tokens_as_token_ids": True},
        )
        per_penalty[p] = _steps(r)
        print(f"  penalty={p:>5}: emitted {[s['token'] for s in per_penalty[p]][:12]} ...", flush=True)

    base = per_penalty[0.0]

    # measured shift: for each step, tokens in both top-k sets, split by
    # "already emitted before this step" vs not.
    shifts = []
    for p in (2.0, -1.5):
        cur = per_penalty[p]
        n = min(len(base), len(cur))
        rows = []
        for t in range(n):
            emitted = {s["token"] for s in base[:t]}
            common = set(base[t]["top"]) & set(cur[t]["top"])
            appeared = sorted(common & emitted)
            fresh = sorted(common - emitted)
            if not appeared or not fresh:
                continue
            d_app = [cur[t]["top"][k] - base[t]["top"][k] for k in appeared]
            d_new = [cur[t]["top"][k] - base[t]["top"][k] for k in fresh]
            # the log-partition term cancels in the difference of differences
            rel = sum(d_app) / len(d_app) - sum(d_new) / len(d_new)
            rows.append(
                {
                    "step": t,
                    "n_appeared_in_topk": len(appeared),
                    "n_fresh_in_topk": len(fresh),
                    "measured_relative_shift": round(rel, 4),
                }
            )
        meas = [r["measured_relative_shift"] for r in rows]
        summary = {
            "presence_penalty": p,
            "expected_relative_shift": -p,
            "steps_measured": len(rows),
            "measured_min": round(min(meas), 4) if meas else None,
            "measured_max": round(max(meas), 4) if meas else None,
            "measured_mean": round(sum(meas) / len(meas), 4) if meas else None,
            "rows": rows,
        }
        print(
            f"  penalty={p:>5}: expected relative logit shift {-p:+.2f}, "
            f"measured mean {summary['measured_mean']} over {len(rows)} steps "
            f"(min {summary['measured_min']}, max {summary['measured_max']})",
            flush=True,
        )
        shifts.append(summary)

    # (c) the margin: can any |p| <= 2 flip greedy at all?
    margins = []
    for t, s in enumerate(base):
        emitted = {x["token"] for x in base[:t]}
        top = s["top"]
        if not top:
            continue
        ranked = sorted(top.items(), key=lambda kv: -kv[1])
        win, win_lp = ranked[0]
        fresh = [(k, v) for k, v in ranked if k not in emitted]
        old = [(k, v) for k, v in ranked if k in emitted]
        row = {"step": t, "winner": win, "winner_appeared_before": win in emitted}
        if win in emitted and fresh:
            # positive penalty demotes the winner; needs p > gap to lose
            row["gap_to_best_fresh"] = round(win_lp - fresh[0][1], 4)
        if old and win not in emitted:
            # negative penalty promotes already-seen tokens; needs |p| > gap
            row["gap_to_best_already_seen"] = round(win_lp - old[0][1], 4)
        margins.append(row)

    pos_gaps = [r["gap_to_best_fresh"] for r in margins if "gap_to_best_fresh" in r]
    neg_gaps = [r["gap_to_best_already_seen"] for r in margins if "gap_to_best_already_seen" in r]
    print(
        f"  greedy flip margins over {len(margins)} steps: "
        f"positive-penalty min gap = {min(pos_gaps) if pos_gaps else 'n/a'} "
        f"({len(pos_gaps)} steps where the winner had already appeared); "
        f"negative-penalty min gap = {min(neg_gaps) if neg_gaps else 'n/a'} "
        f"({len(neg_gaps)} steps where it had not)",
        flush=True,
    )

    return {
        "concurrent_greedy_sweep": {"penalties": SWEEP, "unique_outputs": uniq, "texts": texts},
        "measured_logit_shift": shifts,
        "greedy_flip_margins": {
            "steps": len(margins),
            "min_gap_winner_appeared_to_best_fresh": min(pos_gaps) if pos_gaps else None,
            "min_gap_winner_fresh_to_best_already_seen": min(neg_gaps) if neg_gaps else None,
            "rows": margins,
        },
    }


# --------------------------------------------------------------------------
# ITEM 2
# --------------------------------------------------------------------------
ALLOWED = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]


async def item2(client, model) -> dict:
    print("\n=== ITEM 2: allowed_token_ids ===", flush=True)
    reqs = [
        _completion(
            client,
            model,
            prompt="Allowed: ",
            max_tokens=10,
            temperature=1.0,
            logprobs=5,
            extra_body={"allowed_token_ids": ids, "return_tokens_as_token_ids": True},
        )
        for ids in ALLOWED
    ]
    resps = await asyncio.gather(*reqs, return_exceptions=True)
    rows = []
    for i, (ids, r) in enumerate(zip(ALLOWED, resps)):
        if isinstance(r, BaseException):
            rows.append({"request": i, "allowed_token_ids": ids, "error": repr(r)})
            print(f"  req {i} allowed={ids}: ERROR {r!r}", flush=True)
            continue
        ch = r.choices[0]
        toks = ch.logprobs.tokens if ch.logprobs else []
        ids_emitted = [int(t.split(":")[-1]) for t in toks] if toks else []
        row = {
            "request": i,
            "allowed_token_ids": ids,
            "text": ch.text,
            "text_len": len(ch.text),
            "finish_reason": ch.finish_reason,
            "completion_tokens": r.usage.completion_tokens if r.usage else None,
            "emitted_token_ids": ids_emitted,
            "all_emitted_ids_allowed": bool(ids_emitted) and set(ids_emitted) <= set(ids),
        }
        rows.append(row)
        print(
            f"  req {i} allowed={ids}: completion_tokens={row['completion_tokens']} "
            f"finish={row['finish_reason']} emitted_ids={ids_emitted} "
            f"all_allowed={row['all_emitted_ids_allowed']} text={ch.text!r}",
            flush=True,
        )

    # Control: identical host-sampling path, but with ids that decode to real text.
    ctrl = await _completion(
        client,
        model,
        prompt="Allowed: ",
        max_tokens=10,
        temperature=1.0,
        logprobs=5,
        extra_body={"allowed_token_ids": [13, 14, 15], "return_tokens_as_token_ids": True},
    )
    print(f"  control allowed=[13,14,15]: text={ctrl.choices[0].text!r}", flush=True)
    return {
        "requests": rows,
        "control_13_14_15_text": ctrl.choices[0].text,
        "control_completion_tokens": ctrl.usage.completion_tokens if ctrl.usage else None,
    }


async def main_async(args) -> dict:
    client = AsyncOpenAI(base_url=f"{args.server_url}/v1", api_key="EMPTY", timeout=600.0)
    models = await client.models.list()
    model = models.data[0].id
    print(f"server model: {model}", flush=True)
    report = {"server_url": args.server_url, "model": model}
    if args.item in ("1", "all"):
        report["item1_presence_penalty"] = await item1(client, model, args.max_tokens)
    if args.item in ("2", "all"):
        report["item2_allowed_token_ids"] = await item2(client, model)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://localhost:8000")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--item", choices=["1", "2", "all"], default="all")
    args = ap.parse_args()
    report = asyncio.run(main_async(args))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

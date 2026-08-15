# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Does the presence penalty reach the *device-sampled* logits, end to end?

``TestPresencePenalty`` cannot answer this.  It sweeps presence_penalty over
greedy requests and asserts the text changes; on this checkpoint it does not,
and "the penalty is ignored" and "the penalty is applied and is too small to
move a peaked argmax" look identical from the outside.

Direct measurement is unavailable on this mesh: the plugin routes *any* logprobs
request to host sampling (``model_runner.check_perform_device_sampling`` needs
``num_devices in (8, 32)`` for device logprobs; this mesh has 4), and vLLM's host
sampler returns ``raw_logprobs`` -- computed *before* penalties by design -- so
the returned logprobs can never show a device penalty.

So instead: make a falsifiable prediction and test it.

  PHASE A (host path, logprobs=20, presence=0): read the true pre-penalty logit
    margins.  For every greedy step whose winning token has already been emitted,
    the margin to the best not-yet-emitted candidate is exactly how much presence
    penalty it takes to unseat it.  Search prompts for a step with margin < 2.0
    (2.0 is the OpenAI/vLLM cap, so a larger margin is *unflippable by any legal
    presence penalty* and the test's observable is unreachable there).

  PHASE B (device path, no logprobs, greedy): run that prompt with presence
    penalties straddling the measured margin.  The prediction is exact:
    outputs identical to the baseline for penalty < margin, and first divergence
    at exactly the measured step for penalty > margin.

Passing PHASE B is a positive, quantitative demonstration that the on-device
penalty path receives the per-request presence value and subtracts it from the
right logits at the right step.

    python .../bench/presence_flip_probe.py --server-url http://localhost:8000 \
        --out .../doc/vllm_integration/presence_flip_probe.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
from pathlib import Path

from openai import AsyncOpenAI

PROMPTS = [
    "a b c a b c a b c",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    "Once upon a time, there was a small village by the sea. The village",
    "In 2020 the world changed. In 2021 the world",
    "banana apple banana apple banana",
    "The capital of France is Paris. The capital of Germany is",
    "Write a short poem about the ocean:\n",
    "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return",
    "Reasons to learn a musical instrument:\n1.",
    "She opened the door and saw",
    "Item, quantity, price\napple, 3, 1.20\nbanana, 5, 0.60\ncherry,",
    "Q: What is machine learning?\nA: Machine learning is",
    "one two three one two three one",
    "The weather today is quite nice, and tomorrow it will be",
]
MAX_TOKENS = 48
CAP = 2.0  # vLLM/OpenAI clamp on presence_penalty


async def _phase_a(client, model, prompt: str) -> dict:
    """Host path: raw (pre-penalty) logprobs -> true logit margins."""
    r = await client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=0,
        presence_penalty=0.0,
        logprobs=20,
        extra_body={"return_tokens_as_token_ids": True},
    )
    lp = r.choices[0].logprobs
    toks = list(lp.tokens)
    tops = [dict(d) for d in (lp.top_logprobs or [])]
    cands = []
    for t, top in enumerate(tops):
        emitted = set(toks[:t])
        if not top:
            continue
        ranked = sorted(top.items(), key=lambda kv: -kv[1])
        win, win_lp = ranked[0]
        if win not in emitted:
            continue  # presence penalty does not touch the winner here
        fresh = [(k, v) for k, v in ranked if k not in emitted]
        if not fresh:
            continue
        cands.append(
            {
                "step": t,
                "winner": win,
                "challenger": fresh[0][0],
                "margin": round(win_lp - fresh[0][1], 4),
            }
        )
    flippable = [c for c in cands if c["margin"] < CAP]
    best = min(cands, key=lambda c: c["margin"]) if cands else None
    return {
        "prompt": prompt,
        "text": r.choices[0].text,
        "steps_with_repeated_winner": len(cands),
        "min_margin": best["margin"] if best else None,
        "first_flippable": min(flippable, key=lambda c: c["step"]) if flippable else None,
        "smallest_margin_step": best,
    }


async def _phase_b(client, model, prompt: str, penalties: list[float], max_tokens: int) -> list[dict]:
    """Device path: greedy, no logprobs -> real on-device split sampling."""
    # No logprobs anywhere here: asking for them (even logprobs=0) routes the
    # request to host sampling on a 4-device mesh, which is the path this phase
    # exists to avoid.  Sequential, so batch composition cannot explain a diff.
    out = []
    for p in penalties:
        r = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            presence_penalty=p,
        )
        out.append({"presence_penalty": p, "text": r.choices[0].text})
    return out


def _first_diff(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if a == b else min(len(a), len(b))


async def main_async(args) -> dict:
    client = AsyncOpenAI(base_url=f"{args.server_url}/v1", api_key="EMPTY", timeout=900.0)
    model = (await client.models.list()).data[0].id
    print(f"server model: {model}\n", flush=True)

    print("=== PHASE A: measure the true pre-penalty greedy margins (host path) ===", flush=True)
    surveys = []
    for prompt in PROMPTS:
        s = await _phase_a(client, model, prompt)
        surveys.append(s)
        ff = s["first_flippable"]
        print(
            f"  {prompt[:44]!r:48} repeated-winner steps={s['steps_with_repeated_winner']:>3} "
            f"min_margin={s['min_margin']} "
            f"first_flippable={'step %d margin %s' % (ff['step'], ff['margin']) if ff else 'none'}",
            flush=True,
        )

    with_flip = [s for s in surveys if s["first_flippable"]]
    report = {"phase_a": surveys}
    if not with_flip:
        print(
            "\n  no prompt has a greedy step flippable by any legal presence penalty "
            f"(cap {CAP}); PHASE B cannot be run",
            flush=True,
        )
        report["phase_b"] = None
        return report

    chosen = min(with_flip, key=lambda s: s["first_flippable"]["margin"])
    ff = chosen["first_flippable"]
    margin = ff["margin"]
    # Generate exactly up to and including the predicted step.  `first_flippable`
    # is the *earliest* step with margin < 2.0, so every earlier step is unflippable
    # by any legal penalty and the predicted step is the only one that can move --
    # which makes the observed threshold unambiguous.  (Over a longer generation,
    # later steps with even smaller margins flip first and mask it.)
    b_max_tokens = ff["step"] + 1
    print(
        f"\n=== PHASE B: device path on {chosen['prompt']!r} ===\n"
        f"  max_tokens={b_max_tokens} (steps 0..{ff['step']-1} all have margin >= {CAP}, "
        f"so step {ff['step']} is the only step any legal penalty can move)\n"
        f"  prediction: presence penalty > {margin} unseats {ff['winner']} at step {ff['step']} "
        f"in favour of {ff['challenger']}; penalty < {margin} changes nothing",
        flush=True,
    )
    # A ladder straddling the measured margin, so the *threshold* is measured and
    # not just the presence of an effect: the first penalty that changes the output
    # should be the first one above `margin`.
    ladder = [margin * f for f in (0.0, 0.25, 0.5, 0.75, 0.95, 1.05, 1.25, 1.5, 2.0)]
    penalties = sorted({round(min(max(p, 0.0), CAP), 4) for p in ladder} | {CAP})
    runs = await _phase_b(client, model, chosen["prompt"], penalties, b_max_tokens)
    base = runs[[r["presence_penalty"] for r in runs].index(0.0)]["text"]
    for r in runs:
        r["identical_to_zero_penalty"] = r["text"] == base
        r["first_char_divergence"] = _first_diff(base, r["text"])
        print(
            f"  presence={r['presence_penalty']:>5}: identical_to_baseline={r['identical_to_zero_penalty']} "
            f"first_char_div={r['first_char_divergence']} text={r['text'][:70]!r}",
            flush=True,
        )

    derived = phase_b_verdict(runs, margin)
    print(
        f"\n  predicted flip threshold (measured logit margin) = {margin}\n"
        f"  observed  flip threshold (smallest penalty that changed the output) = "
        f"{derived['observed_flip_threshold']}"
        f"  (largest penalty that did not: {derived['largest_penalty_with_no_change']})\n"
        f"  monotone={derived['monotone_step']} within_tolerance={derived['within_tolerance']} "
        f"(tol {derived['tolerance']})",
        flush=True,
    )
    print(
        f"  PRESENCE_PENALTY_REACHES_DEVICE_LOGITS = " f"{derived['presence_penalty_reaches_device_logits']}",
        flush=True,
    )
    report["phase_b"] = {
        "prompt": chosen["prompt"],
        "max_tokens": b_max_tokens,
        "predicted_step": ff["step"],
        "measured_margin": margin,
        "winner": ff["winner"],
        "challenger": ff["challenger"],
        "runs": runs,
        **derived,
    }
    return report


def phase_b_verdict(runs, margin):
    """Derive phase B's verdict from its ladder.  One implementation, two callers.

    Kept as a pure function of ``(runs, margin)`` so the same rule can be applied to
    a *committed* ladder offline.  That matters: an earlier revision of this file
    wrote a stricter verdict than the one it printed, so the artifact on disk
    disagreed with its own data.  ``--recompute-verdict`` re-derives the fields from
    the committed ``runs`` with this function and rewrites only the derived fields,
    never the measurements.
    """
    changed = [r["presence_penalty"] for r in runs if not r["identical_to_zero_penalty"]]
    unchanged = [r["presence_penalty"] for r in runs if r["identical_to_zero_penalty"]]
    observed_threshold = min(changed) if changed else None
    # Monotone step: everything at or above the threshold flips, everything below
    # does not.  A penalty that never reaches the logits has no threshold at all.
    monotone = bool(changed) and (not unchanged or max(unchanged) < observed_threshold)
    # Tolerance is one ladder rung.  Exact agreement is not expected: the penalty
    # buffer, the logits and the subtraction are all bfloat16, so the effective
    # threshold sits within one bf16 ulp of the logit magnitude involved (near the
    # +/-20 softcap that ulp is 0.0625-0.125), and an exact post-subtraction tie
    # resolves to the lower token id.
    tol = max(0.15, 0.25 * margin)
    close = observed_threshold is not None and abs(observed_threshold - margin) <= tol
    return {
        "observed_flip_threshold": observed_threshold,
        "largest_penalty_with_no_change": max(unchanged) if unchanged else None,
        "monotone_step": monotone,
        "tolerance": tol,
        "within_tolerance": close,
        "presence_penalty_reaches_device_logits": bool(monotone and close),
    }


def recompute_verdict(path) -> int:
    """Re-derive phase B's verdict fields from a committed ladder, offline.

    Rewrites only the derived fields; ``runs``, ``measured_margin`` and every other
    measurement are left byte-for-byte alone.  Needs no server and no device.
    """
    import json as _json

    path = pathlib.Path(path)
    report = _json.loads(path.read_text())
    pb = report.get("phase_b")
    if not pb or not pb.get("runs"):
        print(f"no phase_b ladder in {path}", flush=True)
        return 2
    before = pb.get("presence_penalty_reaches_device_logits")
    derived = phase_b_verdict(pb["runs"], pb["measured_margin"])
    pb.update(derived)
    pb["verdict_provenance"] = (
        "derived fields recomputed offline by presence_flip_probe.phase_b_verdict() from the "
        "committed runs; measurements untouched. The original file was written by an earlier "
        f"revision whose rule disagreed with the printed verdict (recorded {before})."
    )
    path.write_text(_json.dumps(report, indent=2) + "\n")
    print(f"recomputed {path}: {before} -> {derived['presence_penalty_reaches_device_logits']}", flush=True)
    for k, v in derived.items():
        print(f"  {k} = {v}", flush=True)
    return 0 if derived["presence_penalty_reaches_device_logits"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recompute-verdict",
        metavar="JSON",
        default=None,
        help="Re-derive phase B's verdict fields from a committed ladder and exit. No device.",
    )
    ap.add_argument("--server-url", default="http://localhost:8000")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.recompute_verdict:
        return recompute_verdict(args.recompute_verdict)
    report = asyncio.run(main_async(args))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}", flush=True)
    pb = report.get("phase_b")
    return 0 if (pb and pb["presence_penalty_reaches_device_logits"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

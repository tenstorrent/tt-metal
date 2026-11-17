# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One table for an eval_runs/<ts> directory: every endpoint (tt/, openrouter_<provider>/) x every bench, with the
harness score (failed requests count as wrong, as lmms-eval scores them), the score over answered items only, the
failure count, tokens, an OpenRouter cost estimate and the wall clock.

Layouts understood: run_suite.sh (<endpoint>/<bench>/full/) and a flat <endpoint>/<bench>/. A request counts as
answered when its doc is in the step's journal (only successful responses are journaled), as failed otherwise. Only
the journal records of the step's last run count (the request keys in journal.keys.json, which the client writes):
a journal also keeps the answers of earlier runs into the same directory with other settings (temperature, thinking,
token limit), and those must not make a failed request look answered or add their tokens and cost. A journal written
before journal.keys.json existed is taken whole, as before (the row then says "journal": "all records").

"Answered only" is computed for the per-item-mean metrics (OCRBench, MMAU, Video-MME, MMMU-Pro AA, VGI-Bench), and
for those the "failed = wrong" score is recomputed here from the samples too: a failed request's text is an error
message, and a task's answer extraction can find a letter in it (the plugin's own tasks score it wrong themselves,
lmms-eval's do not), so the harness's own number ("score_harness" in --json) can count failures as correct.
LibriSpeech's WER is corpus-level and shown as the harness computed it.

  compare_runs.py <run dir> [--prices '{"gmicloud": [0.112, 0.224], "xiaomi": [0.14, 0.28]}'] [--json out.json]
"""

import argparse
import glob
import json
import os

BENCHES = {  # bench -> (lmms task, harness metric key, per-sample correctness, scale)
    "ocrbench": ("ocrbench", "ocrbench_accuracy,none", lambda r: r["ocrbench_accuracy"]["score"], 100.0),
    "mmau_mini": ("mmau_test_mini", "accuracy,none", lambda r: r["accuracy"]["overall"], 100.0),
    "librispeech_clean": ("librispeech_test_clean", "wer,none", None, 1.0),
    "videomme_short": (
        "videomme_short",
        "videomme_perception_score,none",
        lambda r: float(r["videomme_perception_score"]["pred_answer"] == r["videomme_perception_score"]["answer"]),
        100.0,
    ),
    "mmmu_pro_aa": ("mmmu_pro_aa", "aa_accuracy,none", lambda r: r["aa_accuracy"], 100.0),
    "vgi_bench": ("vgi_bench", "vgi_accuracy,none", lambda r: r["vgi_accuracy"], 100.0),
}
HARNESS_SCALE = {
    "ocrbench": 100.0,
    "mmmu_pro_aa": 100.0,
    "vgi_bench": 100.0,
}  # these report a fraction; the rest are already percent


def step_dir(endpoint_dir, bench):
    for d in (os.path.join(endpoint_dir, bench, "full"), os.path.join(endpoint_dir, bench)):
        if os.path.exists(os.path.join(d, "summary.json")):
            return d
    return None


def last_run_keys(d):
    """The request keys of the step's last run (journal.keys.json), or None for a journal from before that file."""
    try:
        with open(os.path.join(d, "journal.keys.json")) as f:
            return set(json.load(f)["keys"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def journal_docs(d):
    """(docs answered in the last run, [input tokens, output tokens, cost] of those answers, whether the journal was
    restricted to the last run's keys)."""
    docs, usage = set(), [0, 0, 0.0]
    keys = last_run_keys(d)
    path = os.path.join(d, "journal.jsonl")
    if os.path.exists(path):
        latest = {}
        for line in open(path):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if keys is not None and rec.get("key") not in keys:
                continue  # an earlier run's answer to another request (other settings) for this doc
            latest[rec.get("key", len(latest))] = rec  # one answer per request
        for rec in latest.values():
            docs.add(rec["doc_id"])
            u = rec.get("usage") or {}
            usage[0] += u.get("prompt_tokens") or 0
            usage[1] += u.get("completion_tokens") or 0
            usage[2] += float(u.get("cost") or 0)
    return docs, usage, keys is not None


def row(endpoint, bench, d, prices):
    task, key, correct, scale = BENCHES[bench]
    summary = json.load(open(os.path.join(d, "summary.json")))
    metric = (summary.get("metrics") or {}).get(key)
    n = (summary.get("n_samples") or {}).get("effective")
    docs, (tin, tout, cost), last_run_only = journal_docs(d)
    out = {"endpoint": endpoint, "bench": bench, "dir": d, "n": n, "answered": len(docs)}
    out["journal"] = "last run" if last_run_only else "all records"
    out["failed"] = (n - len(docs)) if n is not None else None
    out["score"] = out["score_harness"] = None if metric is None else metric * HARNESS_SCALE.get(bench, 1.0)
    samples = sorted(glob.glob(os.path.join(d, "lmms", "**", f"*samples_{task}.jsonl"), recursive=True))
    if correct and samples:
        vals, total = [], 0
        for line in open(samples[-1]):
            r = json.loads(line)
            total += 1
            if r["doc_id"] in docs:
                vals.append(float(correct(r)))
        out["score_answered"] = 100.0 * sum(vals) / len(vals) if vals else None
        if total:  # a failed request is wrong whatever the task's extraction read in its error text
            out["score"] = 100.0 * sum(vals) / total
    out["input_tokens"], out["output_tokens"] = tin, tout
    provider = endpoint.split("openrouter_", 1)[1] if endpoint.startswith("openrouter_") else None
    if cost:
        out["cost_usd"] = round(cost, 3)
    elif provider and provider in prices:
        p_in, p_out = prices[provider]
        out["cost_usd_est"] = round((tin * p_in + tout * p_out) / 1e6, 3)
    out["wall_s"] = summary.get("wall_s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--prices", default='{"gmicloud": [0.112, 0.224], "xiaomi": [0.14, 0.28]}')
    ap.add_argument("--json")
    args = ap.parse_args()
    prices = json.loads(args.prices)
    rows = []
    for endpoint_dir in sorted(glob.glob(os.path.join(args.run_dir, "*/"))):
        endpoint = os.path.basename(os.path.normpath(endpoint_dir))
        for bench in BENCHES:
            d = step_dir(endpoint_dir, bench)
            if d:
                rows.append(row(endpoint, bench, d, prices))
    print("| bench | endpoint | score (failed = wrong) | answered-only | n | failed | out tokens | cost $ | wall s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (list(BENCHES).index(r["bench"]), r["endpoint"])):
        fmt = lambda v: "" if v is None else f"{v:.2f}"  # noqa: E731
        cost = r.get("cost_usd", r.get("cost_usd_est"))
        print(
            f"| {r['bench']} | {r['endpoint']} | {fmt(r['score'])} | {fmt(r.get('score_answered'))} | {r['n']} | "
            f"{r['failed']} | {r['output_tokens']} | {'' if cost is None else cost} | {r['wall_s']} |"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

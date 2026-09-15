# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render completed, adjudicated qualification results as Markdown on stdout."""

import collections
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
rows = list(map(json.loads, (p / "accepted-results.jsonl").read_text().splitlines()))
oracle = {(r["id"], r["mode"]): r for r in map(json.loads, (p / "accepted-oracle.jsonl").read_text().splitlines())}
assert len(rows) == 1320 and all(r["status"] != "ERROR" for r in rows)


def subset(mode, group=None, dist=None, n=None):
    return [
        r
        for r in rows
        if r["mode"] == mode
        and (group is None or r["group"] == group)
        and (dist is None or r["distribution"] == dist)
        and (n is None or r["kv_len"] == n)
    ]


def counts(rr):
    c = collections.Counter(r["status"] for r in rr)
    return f"{c['PASS']} / {c['FAIL']} / {c['UNSUPPORTED']}"


def metric(rr, field, minimum=False):
    xs = [h[field] for r in rr for h in r.get("per_head", []) if h[field] is not None]
    if not xs:
        return "—"
    value = min(xs) if minimum else max(xs)
    return f"{value:.8f}" if field == "pcc" else f"{value:.6f}"


print("# SDPA numerical qualification v1\n")
print(
    "**Neither candidate qualifies against the agreed numerical/structural contract.** "
    "The complete 660-input synthetic matrix was evaluated on one Blackhole P100A "
    "(yyzo-bh-26, reservation 215262): 1320 requested mode cases, with unsupported "
    "FP32 requests explicitly excluded rather than run on fallback. Performance "
    "and end-to-end model evaluation were not tested.\n"
)
print("| Mode | Pass | Fail | Unsupported |\n|---|---:|---:|---:|")
for mode in ("fast", "accurate"):
    c = collections.Counter(r["status"] for r in subset(mode))
    print(f"| {mode} | {c['PASS']} | {c['FAIL']} | {c['UNSUPPORTED']} |")
print(
    "\nCounts are per input case/mode; every tested head must pass. Unsupported "
    "does not count as a pass. Long-context references cover 512 stratified query "
    "rows/head; Q<=2048 cases check every query row. Full device operations, not "
    "sampled-device operations, were executed. These are absolute acceptance "
    "failures, not a main-versus-candidate regression comparison.\n"
)
print("## Results by test group\n")
print("Cells are pass / fail / unsupported.\n")
print("| Group | Fast | Accurate streaming |\n|---|---:|---:|")
for group in ("normal", "stress", "structural", "boundary"):
    print(f"| {group} | {counts(subset('fast', group))} | {counts(subset('accurate', group))} |")
print("\n## Standard normal inputs\n")
print(
    "Maximum per-head L2 percent and minimum defined per-head PCC, across both "
    "head counts and all five seeds. These are worst cases, not means.\n"
)
print("| Length | Fast L2 % / PCC | Accurate L2 % / PCC | Accurate pass / fail / unsupported |\n|---|---:|---:|---:|")
for n in (2048, 8192, 25920, 32768, 65536, 75600, 131072, 262144):
    f, a = subset("fast", "normal", n=n), subset("accurate", "normal", n=n)
    print(
        f"| {n:,} | {metric(f, 'l2_pct')} / {metric(f, 'pcc', True)} | "
        f"{metric(a, 'l2_pct')} / {metric(a, 'pcc', True)} | {counts(a)} |"
    )
print(
    "\nThe previous approximately 0.49% aggregate FP32 result does not imply every "
    "head is below 0.5%. This qualification exposes that margin issue directly. "
    "The 25,920 and 75,600 FP32 results from earlier experiments were fallback "
    "measurements and are deliberately NOT reused here.\n"
)
print("## Stress inputs\n")
print(
    "Each distribution is tested at 32K and 256K, H=5 and H=10, five seeds. "
    "Common-mode rows combine offsets -32,-8,+8,+32 applied to only that tensor.\n"
)
print(
    "| Distribution | Fast max head L2 % | Accurate max head L2 % | Fast pass / fail | Accurate pass / fail |\n|---|---:|---:|---:|---:|"
)
for dist in ("scaled_low", "scaled_qk", "outliers", "common_q", "common_k", "common_v"):
    f, a = subset("fast", "stress", dist), subset("accurate", "stress", dist)
    print(
        f"| {dist} | {metric(f, 'l2_pct')} | {metric(a, 'l2_pct')} | "
        f"{counts(f).rsplit(' / ', 1)[0]} | {counts(a).rsplit(' / ', 1)[0]} |"
    )
print("\n| Distribution | Fast min PCC / worst row L2 % | Accurate min PCC / worst row L2 % |\n|---|---:|---:|")
for dist in ("scaled_low", "scaled_qk", "outliers", "common_q", "common_k", "common_v"):
    print(
        f"| {dist} | {metric(subset('fast', 'stress', dist), 'pcc', True)} / {metric(subset('fast', 'stress', dist), 'row_max_pct')} | "
        f"{metric(subset('accurate', 'stress', dist), 'pcc', True)} / {metric(subset('accurate', 'stress', dist), 'row_max_pct')} |"
    )
print(
    "\nRow denominators use the agreed 1%-of-head-RMS floor. Raw unfloored row "
    "summaries, maximum absolute error, and raw maximum elementwise relative "
    "error are also in the JSONL. Maxima apply to tested rows, not every long-context output.\n"
)
print("## Structural checks\n")
print("| Input | Fast pass / fail / unsupported | Accurate pass / fail / unsupported |\n|---|---:|---:|")
for dist in ("zero_v", "constant_v", "uniform", "cancellation", "single_key"):
    print(
        f"| {dist} | {counts(subset('fast', 'structural', dist))} | {counts(subset('accurate', 'structural', dist))} |"
    )
for mode in ("fast", "accurate"):
    rr = subset(mode, "structural")
    ulps = [h["structural_max_ulp"] for r in rr for h in r.get("per_head", []) if "structural_max_ulp" in h]
    print(f"\n{mode}: maximum structural error where measured = {max(ulps, default=0):g} BF16 ULP.")
print("\n## Common-V rounding-floor audit\n")
print(
    "The original proposal exempts undefined PCC for constant outputs. The online "
    "scorer initially missed the constant-actual exception; adjudicate.py applies "
    "it to the preserved raw metrics. It does not change thresholds, waive defined "
    "PCC failures, or alter any other gate.\n"
)
for mode in ("fast", "accurate"):
    oo = [r for r in oracle.values() if r["mode"] == mode]
    rr = subset(mode, "stress", "common_v")
    same = sum(r["sampled_output_sha256"] == oracle[r["id"], mode]["ideal_output_sha256"] for r in rr)
    residual_failures = sum(any(f.endswith("common_v_residual") for f in r["failed_gates"]) for r in rr)
    print(
        f"- {mode}: ideal FP64-reference-rounded-to-BF16 oracle: {counts(oo)}; "
        f"device output exactly matches that oracle in {same}/{len(rr)} cases; "
        f"{residual_failures}/{len(rr)} cases fail the common-V residual budget."
    )
print(
    "\nDefined PCC still rejects some ideally rounded results: the residual signal "
    "can be poorly resolved in BF16 even when ordinary L2 is at its rounding "
    "floor. These remain formal failures under the current contract, flagged "
    "separately from additional operator error. A floor-aware PCC rule needs "
    "agreement before treating this as a release contract.\n"
)
print("## Coverage and implementation provenance\n")
print(
    "- Fast uses compensated BF16 streaming, untouched Q, Q/K chunks 128/512, "
    "and unchanged input buffering. A qualification-only host patch enables "
    "compensation below the default 64-K-chunk cutoff.\n"
    "- Accurate uses the retained improved FP32 streaming algorithm and its "
    "six-bit Q preprocessing/1.0027 scale compensation. K chunks are 1024, or "
    "512 where that permits streaming (33,280). No FP32 fallback was executed.\n"
    "- The three negative FP32 probes verify host rejection for short/padded "
    "inputs. Explicit-mask structural requests are unsupported by both "
    "specializations; their guard probes are recorded separately.\n"
    "- Numerical kernels were not tuned or changed. The qualification-only "
    "host patch, source hashes, build logs, input hashes, and output hashes "
    "are retained.\n"
    "- One harness-only interruption came from assuming centered K was always "
    "exactly BF16-representable. The runner was corrected to account for "
    "requantization, and the affected case was retried. Historical records "
    "remain in raw results; this was not a hardware/kernel failure.\n"
    "- **Not qualified:** Galaxy hardware, two model-family Q/K/V capture "
    "sets, and an additional release holdout suite. Captures were requested "
    "but not available; no Blackhole Galaxy cards were available at discovery. "
    "The available Wormhole Galaxies cannot run these Blackhole-only "
    "specializations unchanged.\n"
    "- Causal/GQA/D!=128/multibatch behavior is outside the proposed v1 scope. "
    "Performance and model-score evaluation were explicitly excluded.\n"
)
print("## Artifacts\n")
print(
    "- [Frozen contract and matrix](SPEC.md), [reproduction procedure](REPRODUCE.md).\n"
    "- [Accepted scores](accepted-results.jsonl), [raw measurements and initial scores](results.jsonl).\n"
    "- [Accepted rounding oracle](accepted-oracle.jsonl), [raw oracle](rounding-oracle.jsonl).\n"
    "- [Qualification host patch](qualification-host.patch), [source hashes](SOURCE-SHA256.txt).\n"
    "- [Summary](summary.json), [completeness/provenance audit](audit.json)."
)

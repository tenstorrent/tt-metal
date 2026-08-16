#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-sample health of an lm-eval run, without copying the samples.

The `samples_*.jsonl` an eval writes are tens of MB of model text and are
deliberately left in the TTI cache. But the two measurement defects this stage
found -- `aime25` responses truncated to nothing at 32768, and `ifeval` turns
graded on their analysis channel at 8192 -- were both visible *only* in those
files, and neither the release report nor any copied artifact would have shown
them. This writes the few KB that would: per document, the response length, the
score, and whether the turn reached the model's visible channel. No model text.

"Reached the visible channel" is the property `tt/reasoning_parser.py` keys on:
a turn that opened `to=self` and ran out of budget before writing
`assistant to=user` has no reply in it, and whatever the harness scored was
reasoning.

Usage::

    python eval_sample_health.py --samples-dir <lm-eval output dir> \\
        --out doc/tti_release/evals/ifeval_sample_health.json --task ifeval
"""

from __future__ import annotations

import argparse
import glob
import json
import os

ANALYSIS_HEADER = " to=self"
VISIBLE_HEADER = "assistant to=user"


def response_of(record: dict) -> str:
    resp = record.get("resps") or record.get("filtered_resps") or [""]
    first = resp[0]
    return first[0] if isinstance(first, list) else first


def score_of(record: dict):
    for key in ("prompt_level_strict_acc", "exact_match", "acc"):
        if key in record:
            return key, record[key]
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    matches = sorted(glob.glob(f"{args.samples_dir}/samples_{args.task}_*.jsonl"))
    if not matches:
        print(f"no samples_{args.task}_*.jsonl under {args.samples_dir}")
        return 1
    path = matches[-1]

    rows, truncated, empty = [], [], []
    score_key = None
    for line in open(path):
        record = json.loads(line)
        text = response_of(record)
        key, value = score_of(record)
        score_key = score_key or key
        reached = VISIBLE_HEADER in text or not text.startswith(ANALYSIS_HEADER)
        row = {
            "doc_id": record.get("doc_id"),
            "key": record.get("doc", {}).get("key"),
            "chars": len(text),
            "score": value,
            "reached_visible_channel": reached,
        }
        rows.append(row)
        if not reached:
            truncated.append(row)
        if not text:
            empty.append(row)

    doc = {
        "_what": (
            "per-document health of one lm-eval run: response length, score, and whether the "
            "turn reached the model's visible channel. No model text. This is the artifact that "
            "makes an under-budgeted generation limit visible without copying samples_*.jsonl."
        ),
        "task": args.task,
        "source": os.path.basename(path),
        "n": len(rows),
        "empty_responses": len(empty),
        "truncated_inside_analysis_channel": len(truncated),
        "truncated_docs": truncated,
        "max_chars": max((r["chars"] for r in rows), default=0),
        "score_key": score_key,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(
        f"{args.task}: n={doc['n']} empty={doc['empty_responses']} "
        f"truncated_in_analysis={doc['truncated_inside_analysis_channel']} "
        f"max_chars={doc['max_chars']} -> {args.out}"
    )
    for row in truncated:
        print(f"   truncated doc_id={row['doc_id']} key={row['key']} chars={row['chars']} score={row['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline audit of final source provenance, equality, and reverse-order checks."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = HERE.parent / "fp32-util-v1"


def read(path):
    return json.loads(path.read_text())


def same(a, b):
    for key in ("full_output_sha256", "l2_pct", "pcc"):
        assert a[key] == b[key], (a["label"], b["label"], key)
    assert a["full_trace_equality"] and b["full_trace_equality"]
    assert a["full_output_finite_checked"] and b["full_output_finite_checked"]


count = 0
for path in sorted(HERE.glob("final-v3-*.jsonl")):
    record = read(path)
    provenance = read(path.with_suffix(".provenance.json"))
    for source, expected in provenance["source_sha256"].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected, source
    label = path.stem
    if "-perf-l1macro-" in label:
        continue
    if "-perf-refine-" in label:
        same(record, read(HERE / path.name.replace("-refine-", "-l1macro-")))
        frozen = path.name.replace("final-v3-perf-refine-", "final-v1-perf-baseline-")
    else:
        section = "stress" if "-stress-" in label else "holdout"
        frozen = path.name.replace(f"final-v3-{section}-", f"final-v1-{section}-baseline-")
    same(record, read(FROZEN / frozen.replace(".jsonl", "-b1.jsonl")))
    count += 1
assert count == 16, count
for n in (32768, 262144):
    same(read(HERE / f"reverse-v3-n{n}-normal.jsonl"), read(HERE / f"reverse-control-v3-n{n}-normal.jsonl"))
    same(read(HERE / f"reverse-v3-n{n}-normal.jsonl"), read(HERE / f"final-v3-perf-refine-n{n}-normal.jsonl"))
print("PASS: 16 final qualification comparisons; current source hashes; 2 reverse-order pairs")

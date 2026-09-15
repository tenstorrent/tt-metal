# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate qualification provenance, completeness, and status consistency."""

import argparse
import collections
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("--require-complete", action="store_true")
parser.add_argument("--results", default="results.jsonl")
args = parser.parse_args()
manifest = json.loads((args.directory / "manifest-final.json").read_text())
expected = {(c["id"], mode) for c in manifest for mode in ("fast", "accurate")}
history = list(map(json.loads, (args.directory / args.results).read_text().splitlines()))
rows = {(r["id"], r["mode"]): r for r in history}
assert rows.keys() <= expected
assert len({c["id"] for c in manifest}) == len(manifest)
trace_count = padding_count = pair_count = 0
for key, r in rows.items():
    if r["status"] == "UNSUPPORTED":
        assert r["mode"] == "accurate" and r["fallback_executed"] is False
        assert r["kv_len"] < 32768 or r["kv_len"] % 512 or r["q_len"] % 128
        continue
    if r["status"] == "ERROR":
        continue
    assert r["status"] in ("PASS", "FAIL")
    assert (r["status"] == "FAIL") == bool(r["failed_gates"])
    assert r["full_device_operation"] is True
    assert len(r["per_head"]) == r["heads"]
    assert r["reference_rows"] == (r["q_len"] if r["q_len"] <= 2048 else 512)
    assert len(set(r["positions"])) == r["reference_rows"]
    assert min(r["positions"]) >= 0 and max(r["positions"]) < r["q_len"]
    assert r["fp32_streaming"] == (r["mode"] == "accurate")
    assert r["bf16_compensated"] == (r["mode"] == "fast")
    assert r["q_preprocessing"] == ("none" if r["mode"] == "fast" else "bitceil6_scale1.0027")
    if r["mode"] == "accurate":
        assert r["kv_len"] >= 32768 and r["kv_len"] % r["k_chunk"] == 0 and r["q_len"] % 128 == 0
    else:
        assert r["k_chunk"] == 512
    trace_count += bool(r["trace_full_equality"])
    if "padding_full_output_equal" in r:
        padding_count += 1
        assert r["padding_full_output_equal"] or "padding_invariance" in r["failed_gates"]
    partner = rows.get((r["id"], "fast" if r["mode"] == "accurate" else "accurate"))
    if partner and "input_sha256" in partner:
        assert r["input_sha256"] == partner["input_sha256"]
        assert r["positions"] == partner["positions"]
        pair_count += 1
missing = expected - rows.keys()
errors = [r["id"] for r in rows.values() if r["status"] == "ERROR"]
if args.require_complete:
    assert not missing and not errors, (len(missing), errors)
print(
    json.dumps(
        dict(
            planned=len(expected),
            completed=len(rows),
            missing=len(missing),
            errors=errors,
            historical_records=len(history),
            historical_errors=sum(r["status"] == "ERROR" for r in history),
            full_trace_equality_checks=trace_count,
            full_padding_equality_checks=padding_count,
            matching_mode_input_pairs=pair_count // 2,
            status_counts={
                mode: dict(collections.Counter(r["status"] for r in rows.values() if r["mode"] == mode))
                for mode in ("fast", "accurate")
            },
        ),
        indent=2,
    )
)

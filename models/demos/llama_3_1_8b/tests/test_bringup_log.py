# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Recipe §7 lint for ``bringup_log.jsonl``. Host only.

The recipe points at ``models/demos/common/prefill/tools/bringup_digest.py --lint``. That tool does
not exist in this workspace — ``models/demos/common/prefill/tools/`` is absent on this branch — so
the lint it describes is implemented here instead and runs as part of the package suite, which is
strictly more useful than a command nobody can run: a malformed record fails on the first test run
rather than whenever someone remembers to check.

Checks, straight from §7.2-7.4: the first line is a ``start`` record, timestamps are ISO-8601 UTC to
the minute, stages are from the closed set, ``judgment.kind`` is from the closed vocabulary, a
failing ``verify`` carries a non-empty ``failed`` list, and ``issue`` / ``fix`` / ``why`` are one
line of at most 160 characters.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

LOG = Path(__file__).resolve().parents[1] / "bringup_log.jsonl"

STAGES = {"E", "D1", "D2", "D3", "M1", "M2", "M3", "P1", "P2"}
KINDS = {"spec_gap", "recipe_gap", "reference_gap", "ttnn_gap", "model_quirk", "env"}
EVENTS = {"start", "enter", "verify", "judgment", "fallback", "skip", "source"}
TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z$")
CAPPED = ("issue", "fix", "why")

REQUIRED = {
    "start": {"model", "mesh", "recipe_sha"},
    "enter": {"t", "stage"},
    "verify": {"t", "stage", "result", "failed"},
    "judgment": {"t", "stage", "kind", "issue", "fix", "failed"},
    "fallback": {"t", "stage", "block", "why"},
    "skip": {"t", "stage", "row", "why"},
    "source": {"t", "stage", "part", "chosen", "envelope", "rejected"},
}


def _records():
    lines = [ln for ln in LOG.read_text().splitlines() if ln.strip()]
    return [(i + 1, json.loads(ln)) for i, ln in enumerate(lines)]


def test_log_exists_and_is_one_object_per_line():
    assert LOG.exists(), f"{LOG} is missing; the process log is committed with the model"
    for lineno, rec in _records():
        assert isinstance(rec, dict), f"line {lineno} is not a JSON object"


def test_first_line_is_start():
    lineno, rec = _records()[0]
    assert rec["ev"] == "start", "the first line must be the `start` record (§7.2)"
    assert rec["model"] == "llama_3_1_8b"
    assert rec["mesh"] == "8x4"
    assert rec["recipe_sha"], "recipe_sha attributes the run to a recipe version"
    assert sum(1 for _, r in _records() if r.get("ev") == "start") == 1, "`start` is written once"


def test_records_have_their_required_fields():
    for lineno, rec in _records():
        ev = rec.get("ev")
        assert ev in EVENTS, f"line {lineno}: unknown event {ev!r}"
        missing = REQUIRED[ev] - set(rec)
        assert not missing, f"line {lineno} ({ev}): missing {sorted(missing)}"
        if ev != "start":
            assert TS.match(rec["t"]), f"line {lineno}: `t` must be ISO-8601 UTC to the minute, got {rec['t']!r}"
            assert rec["stage"] in STAGES, f"line {lineno}: unknown stage {rec['stage']!r}"


def test_verify_and_judgment_semantics():
    for lineno, rec in _records():
        if rec.get("ev") == "verify":
            assert rec["result"] in ("pass", "fail"), f"line {lineno}: bad result {rec['result']!r}"
            if rec["result"] == "fail":
                assert rec["failed"], f"line {lineno}: a failing verify must name the failing node ids"
            else:
                assert rec["failed"] == [], f"line {lineno}: a passing verify carries no failures"
        if rec.get("ev") == "judgment":
            assert rec["kind"] in KINDS, f"line {lineno}: `kind` {rec['kind']!r} is outside the closed vocabulary"


def test_capped_fields_are_one_short_line():
    for lineno, rec in _records():
        for field in CAPPED:
            if field in rec:
                value = rec[field]
                assert "\n" not in value, f"line {lineno}: `{field}` must be one line"
                assert len(value) <= 160, f"line {lineno}: `{field}` is {len(value)} chars, cap is 160"


def test_every_entered_stage_ends_green():
    """A stage that was entered must have a passing `verify`, and none may be entered twice."""
    entered = [r["stage"] for _, r in _records() if r.get("ev") == "enter"]
    assert len(entered) == len(set(entered)), f"a stage was entered twice: {entered}"
    passed = {r["stage"] for _, r in _records() if r.get("ev") == "verify" and r["result"] == "pass"}
    assert set(entered) <= passed | {"E"}, f"entered but never green: {sorted(set(entered) - passed - {'E'})}"


def test_exploration_recorded_a_source_per_part():
    """Stage E's deliverable: one `source` (or an explicit `skip`) for every part in recipe §2.2."""
    parts = {r["part"] for _, r in _records() if r.get("ev") == "source"}
    skipped = {r["row"] for _, r in _records() if r.get("ev") == "skip" and r["stage"] == "E"}
    expected = {"weight_loading", "dequant", "norm_embedding", "mlp", "attention", "rope", "kv_cache", "runtime"}
    assert expected <= (parts | skipped), f"no source or skip for: {sorted(expected - parts - skipped)}"

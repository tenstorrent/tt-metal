# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: bringup_log.jsonl follows the recipe's record schema (section 7).

Stands in for ``bringup_digest.py --lint``, which is not present in this checkout: one JSON object
per line, a single leading ``start`` record, the closed ``ev`` / ``stage`` / ``kind`` vocabularies,
per-event required fields, non-empty ``failed`` on a failing verify, and the 160-character one-line
cap on ``issue`` / ``fix`` / ``why``.
"""

import json
import re

from models.demos.mistral_medium_3_5_128b.config import PACKAGE_DIR

STAGES = {"E", "D1", "D2", "D3", "M1", "M2", "M3", "P1", "P2"}
KINDS = {"spec_gap", "recipe_gap", "reference_gap", "ttnn_gap", "model_quirk", "env"}
REQUIRED = {
    "enter": {"t", "stage"},
    "verify": {"t", "stage", "result", "failed"},
    "judgment": {"t", "stage", "kind", "issue", "fix", "failed"},
    "fallback": {"t", "stage", "block", "why"},
    "skip": {"t", "stage", "row", "why"},
    "source": {"t", "stage", "part", "chosen", "envelope", "rejected"},
}
TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z$")


def test_bringup_log_is_well_formed():
    lines = (PACKAGE_DIR / "bringup_log.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    start = records[0]
    assert start["ev"] == "start" and {"model", "mesh", "recipe_sha"} <= set(start), start
    assert start["model"] == "mistral_medium_3_5_128b"
    assert sum(r["ev"] == "start" for r in records) == 1
    for n, rec in enumerate(records[1:], start=2):
        where = f"line {n}: {rec}"
        assert rec["ev"] in REQUIRED, where
        assert REQUIRED[rec["ev"]] <= set(rec), where
        assert TIME.match(rec["t"]), where
        assert rec["stage"] in STAGES, where
        for field in ("issue", "fix", "why"):
            if field in rec:
                assert isinstance(rec[field], str) and 0 < len(rec[field]) <= 160 and "\n" not in rec[field], where
        if rec["ev"] == "verify":
            assert rec["result"] in ("pass", "fail"), where
            assert isinstance(rec["failed"], list) and (rec["result"] == "pass") == (not rec["failed"]), where
        if rec["ev"] == "judgment":
            assert rec["kind"] in KINDS and isinstance(rec["failed"], list), where
        if rec["ev"] == "source":
            assert isinstance(rec["rejected"], list) and isinstance(rec["chosen"], str), where

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Floors, golden paths and the record stamp shared by the tests."""
import os
from pathlib import Path

# CPU fp32 goldens from tests/golden/make_goldens.py (test/) and make_bench_goldens.py (bench/), kept out of git
# (README.md). KAIROS_GOLDEN_DIR overrides the default.
GOLDENS = Path(os.environ.get("KAIROS_GOLDEN_DIR", Path(__file__).resolve().parents[1] / "goldens"))
TEST_GOLDENS = GOLDENS / "test"
BENCH_GOLDENS = GOLDENS / "bench"

# Floors on every suite and bench case (None = reported, not gated).
FLOORS = {
    "pcc_normalized": 0.999,  # minimum over the case
    "pcc_normalized_full": 0.999,  # both decoder tokens, 128 steps
    "pcc_final": 0.99,
    "route_mismatch_rows": 0,  # exact routing
    "bias_max_abs_diff": 1e-6,
    "module_pcc": 0.999,
    "pcc_spread": None,
    "pcc_row_p01": None,
    "pcc_row_min": None,
    "spread_ratio_p50": None,
    "abs_err_max": None,
}
FLOORS_PERFORMANCE = {**FLOORS, "pcc_normalized": 0.995, "pcc_normalized_full": 0.995, "pcc_final": 0.98}

# Device module tests (tests/pcc/test_modules.py), asserted in both precisions. Set by the rule m - 2 (1 - m), rounded
# down to 4 digits, from the minimum m over the test goldens (pcc job 1047).
MODULE_FLOORS = {
    "cross_attn": {"accuracy": 0.9999, "performance": 0.9992},
    "dec_self": {"accuracy": 0.9999, "performance": 0.9995},
    "embed": {"accuracy": 0.9999, "performance": 0.9998},
    "enc_block": {"accuracy": 0.9999, "performance": 0.9967},
    "enc_chain": {"accuracy": 0.999, "performance": 0.9655},
    "enc_out": {"accuracy": 0.9994, "performance": 0.9848},
    "rope_qk": {"accuracy": 0.9999, "performance": 0.9998},
}

_SESSION = {}


def git_stamp() -> dict:
    """HEAD and the porcelain status of the model's tt/ and tests/ (untracked files included), for the records."""
    import subprocess

    model = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.run(["git", *a], cwd=model, capture_output=True, text=True, check=True).stdout.strip()
    return {"git_head": git("rev-parse", "HEAD"), "git_dirty": git("status", "--porcelain", "--", "tt", "tests")}


def session_stamp() -> dict:
    """The stamp taken once, at the first call in this process (tests call it at import, i.e. at session start),
    with ``started_at`` (unix time); every record of the session carries this same stamp."""
    import time

    if not _SESSION:
        _SESSION.update(started_at=time.time(), **git_stamp())
    return dict(_SESSION)

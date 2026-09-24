# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Say loudly when the suite has nothing to test against.

The device tests skip when `tests/golden` lacks the goldens (`scripts/gen_golden.py`) or the
weight exports (`scripts/export_weights.py`), and pytest reports a run in which every one of
them skipped as green. The header names what is absent before the run; the summary counts the
tests it cost after it, so such a run does not read as a pass.
"""
from __future__ import annotations

import glob
import os

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
WEIGHT_EXPORTS = ("hift_weights.npz", "flow_weights.npz", "llm_weights.npz")
# Words the absent-asset skip reasons across the suite share.
ASSET_SKIP_WORDS = ("gen_golden", "export_weights", "goldens first", "weights first", "prepare_inputs")


def _missing_assets() -> list[str]:
    missing = [w for w in WEIGHT_EXPORTS if not os.path.exists(os.path.join(GOLDEN_DIR, w))]
    goldens = [p for p in glob.glob(os.path.join(GOLDEN_DIR, "*.npz")) if os.path.basename(p) not in WEIGHT_EXPORTS]
    if not goldens:
        missing.insert(0, "the goldens")
    return missing


def pytest_report_header(config):
    missing = _missing_assets()
    if missing:
        return (
            f"CosyVoice: tests/golden lacks {', '.join(missing)}; the tests that need them will skip. "
            "Run scripts/gen_golden.py and scripts/export_weights.py first (README.md, Quick start, step 3)."
        )
    return None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    skipped = [
        r
        for r in terminalreporter.stats.get("skipped", [])
        if isinstance(r.longrepr, tuple) and any(w in str(r.longrepr[-1]) for w in ASSET_SKIP_WORDS)
    ]
    if not skipped:
        return
    terminalreporter.section("CosyVoice: tests skipped for absent goldens, weights or inputs", red=True, bold=True)
    terminalreporter.line(
        f"{len(skipped)} tests did not run because tests/golden or the prompt inputs are absent "
        f"({', '.join(_missing_assets()) or 'the prompt inputs'}). A green result here is not a pass for them: "
        "generate the goldens with scripts/gen_golden.py and export the weights with scripts/export_weights.py, "
        "then run again.",
        red=True,
        bold=True,
    )

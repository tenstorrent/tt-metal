# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the daily PR gate summary (lives in tt_metal/tt-llk/perf)."""

import pathlib
import sys

_PERF = pathlib.Path(__file__).parents[2] / "perf"
sys.path.insert(0, str(_PERF))
from pr_gate_digest import build_text


def test_a_day_without_runs_still_posts_a_line():
    text = build_text([], 24)
    assert "last 24 h" in text and "no PR gate run" in text


def test_the_summary_counts_each_arch_and_names_the_quiet_passes():
    entries = [
        {"arch": "blackhole", "status": "clean", "pr": 58689},
        {"arch": "wormhole", "status": "regressed", "pr": 58689},
        {"arch": "blackhole", "status": "skipped", "pr": 58668},
        {"arch": "wormhole", "status": "clean", "pr": 57460},
    ]
    text = build_text(entries, 24)
    assert "4 verdict(s) on 3 PR(s)" in text
    assert "• blackhole: 1 passed, 0 regressed, 1 skipped" in text
    assert "• wormhole: 1 passed, 1 regressed, 0 skipped" in text
    assert "Passed, posted only here: #57460, #58689" in text

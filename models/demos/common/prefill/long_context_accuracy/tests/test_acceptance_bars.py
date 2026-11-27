# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance bars and the trend they check growth with, on small hand-computed values. CPU only, no device."""

import math

import pytest

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar, GrowthBar, check
from models.demos.common.prefill.long_context_accuracy.windowed_errors import trend

WINDOWS = [
    {"start": 0, "end": 10, "e": 0.010, "p": 0.999},
    {"start": 10, "end": 20, "e": 0.012, "p": 0.998},
    {"start": 20, "end": 30, "e": 0.011},  # this window does not measure p
]


def test_trend_reports_first_last_and_worst() -> None:
    assert trend(WINDOWS, "e", "max") == {
        "first": 0.010,
        "last": 0.011,
        "worst": 0.012,
        "worst_start": 10,
        "worst_over_first": pytest.approx(1.2),
    }
    assert trend(WINDOWS, "p", "min")["worst"] == 0.998 and trend(WINDOWS, "p", "min")["last"] == 0.998


def test_bars_pass_at_the_value_and_name_each_breaking_window() -> None:
    assert (
        check(WINDOWS, [AcceptanceBar("e", "max", 0.012), AcceptanceBar("p", "min", 0.998), GrowthBar("e", 1.2)]) == []
    )
    failures = check(
        WINDOWS, [AcceptanceBar("e", "max", 0.0115), AcceptanceBar("p", "min", 0.9985), GrowthBar("e", 1.1)]
    )
    assert failures == [
        "e 0.012 > 0.0115 in [10, 20)",
        "p 0.998 < 0.9985 in [10, 20)",
        "e grows 1.2x (> 1.1) from 0.01 to 0.012 at row 10",
    ]


def test_bars_fail_closed() -> None:
    nan = [{"start": 0, "end": 1, "e": math.nan}]
    assert check(nan, [AcceptanceBar("e", "max", 1.0)]) == ["e nan is not finite in [0, 1)"]
    assert check(nan, [GrowthBar("e", 2.0)]) == ["e nan is not finite in [0, 1)"]
    assert check(WINDOWS, [AcceptanceBar("missing", "max", 1.0)]) == ["missing: not measured by any window"]
    exact_first = [{"start": 0, "end": 1, "e": 0.0}, {"start": 1, "end": 2, "e": 1e-9}]
    assert check(exact_first, [GrowthBar("e", 1000.0)]) == ["e grows infx (> 1000.0) from 0 to 1e-09 at row 1"]

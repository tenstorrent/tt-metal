# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance bars on windowed errors: a limit per window (``AcceptanceBar``) or on growth over context
(``GrowthBar``), and ``check``, which returns every violation.

Bars are checked fail closed: a NaN or infinite metric fails, and a bar on a metric no window measured fails.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from models.demos.common.prefill.long_context_accuracy.windowed_errors import Kind, trend


@dataclass(frozen=True)
class AcceptanceBar:
    """``metric`` must stay at most (``max``) or at least (``min``) ``value`` in every window that measures it."""

    metric: str
    kind: Kind
    value: float


@dataclass(frozen=True)
class GrowthBar:
    """A larger-is-worse ``metric``'s worst window must stay within ``factor`` times its first window."""

    metric: str
    factor: float


def check(windows: Sequence[dict], bars: Sequence[AcceptanceBar | GrowthBar]) -> list[str]:
    """Every bar failure, as text naming the metric, value and window. Fail closed: a non-finite value fails, and so
    does a bar whose metric no window measures."""
    failures = []
    for bar in bars:
        measured = [w for w in windows if bar.metric in w]
        if not measured:
            failures.append(f"{bar.metric}: not measured by any window")
            continue
        bad = [w for w in measured if not math.isfinite(w[bar.metric])]
        failures += [f"{bar.metric} {w[bar.metric]} is not finite in [{w['start']}, {w['end']})" for w in bad]
        if bad:
            continue
        if isinstance(bar, AcceptanceBar):
            for w in measured:
                value = w[bar.metric]
                if value > bar.value if bar.kind == "max" else value < bar.value:
                    sign = ">" if bar.kind == "max" else "<"
                    failures.append(f"{bar.metric} {value:.6g} {sign} {bar.value} in [{w['start']}, {w['end']})")
        else:
            t = trend(measured, bar.metric, "max")
            if t["worst_over_first"] > bar.factor:
                failures.append(
                    f"{bar.metric} grows {t['worst_over_first']:.4g}x (> {bar.factor}) from {t['first']:.6g} "
                    f"to {t['worst']:.6g} at row {t['worst_start']}"
                )
    return failures

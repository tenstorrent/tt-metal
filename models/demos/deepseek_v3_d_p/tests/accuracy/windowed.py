# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared windowed scoring core for long-context accuracy: per-row sums, fixed row windows, trends and bars.

A device result over a long prompt is scored chunk by chunk without holding its rows: each chunk becomes per-row
FP64 sufficient statistics (``row_sums``), from which the NRMSE and PCC of any set of rows follow exactly. Windows
are aligned to absolute row 0, so two runs with different chunk sizes or layouts score identical row windows.

Bars are checked fail closed: a NaN or infinite metric fails, and a bar on a metric no window measured fails.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

# Columns of ``row_sums``: with x the device (actual) and y the golden (expected) values of one row.
SUM_Y, SUM_X, SUM_YY, SUM_XX, SUM_XY, SUM_EE, COUNT = range(7)


def row_sums(expected: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Per row ``[rows, 7]`` FP64: Σy, Σx, Σy², Σx², Σxy, Σ(x − y)², element count, for ``[rows, ...]`` tensors."""
    if expected.shape != actual.shape or expected.ndim < 1:
        raise ValueError(
            f"expected matching [rows, ...] tensors, got {tuple(expected.shape)} and {tuple(actual.shape)}"
        )
    y = expected.reshape(expected.shape[0], -1).double()
    x = actual.reshape(actual.shape[0], -1).double()
    count = torch.full((y.shape[0],), float(y.shape[1]), dtype=torch.float64)
    return torch.stack(
        [y.sum(1), x.sum(1), (y * y).sum(1), (x * x).sum(1), (x * y).sum(1), (x - y).square().sum(1), count], 1
    )


def nrmse(sums: torch.Tensor) -> float:
    """``‖x − y‖ / ‖y‖`` over the rows of ``sums``: 0 for an exact match (also on an all-zero golden), inf for any
    error on an all-zero golden."""
    error, norm = float(sums[:, SUM_EE].sum()), float(sums[:, SUM_YY].sum())
    if error == 0:
        return 0.0
    return math.sqrt(error / norm) if norm > 0 else math.inf


def pcc(sums: torch.Tensor) -> float:
    """Pearson correlation of all elements of the rows of ``sums``; NaN when either side is constant."""
    sy, sx, syy, sxx, sxy, _, n = sums.sum(0).tolist()
    variance = (sxx - sx * sx / n) * (syy - sy * sy / n)
    return (sxy - sx * sy / n) / math.sqrt(variance) if variance > 0 else math.nan


def row_nrmse(sums: torch.Tensor, floor: float) -> torch.Tensor:
    """Per-row NRMSE with each row's golden norm floored at ``floor`` (the golden's typical scale, so a near-zero row
    does not report its own magnitude); an exact row is 0."""
    error = sums[:, SUM_EE].sqrt()
    return torch.where(error == 0, torch.zeros_like(error), error / sums[:, SUM_YY].sqrt().clamp_min(floor))


Metric = Callable[[torch.Tensor], float]


class RowSeries:
    """Per-row sums of a result streamed in contiguous chunks from absolute row ``start``."""

    def __init__(self, start: int = 0) -> None:
        self.start = start
        self._chunks: list[torch.Tensor] = []
        self._end = start

    @property
    def end(self) -> int:
        return self._end

    def append(self, start: int, sums: torch.Tensor) -> None:
        if start != self._end:
            raise ValueError(f"chunk at row {start} does not continue the series, which ends at row {self._end}")
        if sums.ndim != 2 or sums.shape[1] != COUNT + 1 or sums.dtype != torch.float64:
            raise ValueError(f"expected [rows, {COUNT + 1}] float64 row sums, got {tuple(sums.shape)} {sums.dtype}")
        self._chunks.append(sums)
        self._end += sums.shape[0]

    def sums(self) -> torch.Tensor:
        if not self._chunks:
            raise ValueError("no rows accumulated")
        return torch.cat(self._chunks)

    def windows(self, size: int, metrics: dict[str, Metric]) -> list[dict]:
        """``metrics`` of each ``size``-row window aligned to absolute row 0 and clipped to the series; each window is
        ``{"start", "end", "rows", <metric>: value}`` (``rows`` < ``size`` marks a partial window)."""
        if size <= 0:
            raise ValueError(f"window size must be positive, got {size}")
        sums = self.sums()
        windows = []
        for lo in range((self.start // size) * size, self._end, size):
            start, end = max(lo, self.start), min(lo + size, self._end)
            part = sums[start - self.start : end - self.start]
            windows.append(
                {"start": start, "end": end, "rows": end - start, **{n: f(part) for n, f in metrics.items()}}
            )
        return windows


Kind = Literal["max", "min"]


@dataclass(frozen=True)
class Bar:
    """``metric`` must stay at most (``max``) or at least (``min``) ``value`` in every window that measures it."""

    metric: str
    kind: Kind
    value: float


@dataclass(frozen=True)
class GrowthBar:
    """A larger-is-worse ``metric``'s worst window must stay within ``factor`` times its first window."""

    metric: str
    factor: float


def trend(windows: Sequence[dict], metric: str, worse: Kind) -> dict:
    """First, last and worst (``worse``: which way is worse) value of ``metric`` over the windows that measure it,
    where the worst one starts, and worst over first."""
    measured = [w for w in windows if metric in w]
    if not measured:
        raise ValueError(f"no window measures {metric!r}")
    first, last = measured[0][metric], measured[-1][metric]
    worst = (max if worse == "max" else min)(measured, key=lambda w: w[metric])
    ratio = worst[metric] / first if first else (0.0 if worst[metric] == 0 else math.inf)
    return {
        "first": first,
        "last": last,
        "worst": worst[metric],
        "worst_start": worst["start"],
        "worst_over_first": ratio,
    }


def check(windows: Sequence[dict], bars: Sequence[Bar | GrowthBar]) -> list[str]:
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
        if isinstance(bar, Bar):
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

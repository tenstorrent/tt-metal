# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared windowed scoring core for long-context accuracy: per-row sums, fixed row windows, trends and bars.

A device result over a long prompt is scored chunk by chunk without holding its rows: each chunk becomes per-row
FP64 sufficient statistics (``row_sums``), from which the NRMSE, PCC and absolute RMS error of any set of rows follow
exactly. Windows are aligned to absolute row 0, so two runs with different chunk sizes or layouts score identical row
windows.

Every continuous product is reported in absolute and relative terms together (``error_metrics``): a relative error
that stays flat while the golden grows is a growing absolute error downstream, so NRMSE alone misleads. Absolute
error is the RMS over all elements of the window (``rms_err``), next to the golden's own RMS (``rms_ref``);
``nrmse = rms_err / rms_ref``.

Bars are checked fail closed: a NaN or infinite metric fails, and a bar on a metric no window measured fails. A
metric a window does not define is left out of that window (its function returns ``None``), never reported as NaN.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

# Columns of ``row_sums``, with x the device (actual) and y the golden (expected) values of one row: the row's means,
# its centred moments (PCC from these stays exact on data with a large mean, where raw Σx², Σxy cancel), and the raw
# sums NRMSE needs.
MEAN_Y, MEAN_X, M2_Y, M2_X, C_XY, SUM_YY, SUM_EE, COUNT = range(8)


def row_sums(expected: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Per row ``[rows, 8]`` FP64: ȳ, x̄, Σ(y − ȳ)², Σ(x − x̄)², Σ(x − x̄)(y − ȳ), Σy², Σ(x − y)², element count, for
    ``[rows, ...]`` tensors."""
    if expected.shape != actual.shape or expected.ndim < 1:
        raise ValueError(
            f"expected matching [rows, ...] tensors, got {tuple(expected.shape)} and {tuple(actual.shape)}"
        )
    y = expected.reshape(expected.shape[0], -1).double()
    x = actual.reshape(actual.shape[0], -1).double()
    mean_y, mean_x = y.mean(1, keepdim=True), x.mean(1, keepdim=True)
    dy, dx = y - mean_y, x - mean_x
    count = torch.full((y.shape[0],), float(y.shape[1]), dtype=torch.float64)
    return torch.stack(
        [
            mean_y[:, 0],
            mean_x[:, 0],
            dy.square().sum(1),
            dx.square().sum(1),
            (dx * dy).sum(1),
            y.square().sum(1),
            (x - y).square().sum(1),
            count,
        ],
        1,
    )


def nrmse(sums: torch.Tensor) -> float:
    """``‖x − y‖ / ‖y‖`` over the rows of ``sums``: 0 for an exact match (also on an all-zero golden), inf for any
    error on an all-zero golden."""
    error, norm = float(sums[:, SUM_EE].sum()), float(sums[:, SUM_YY].sum())
    if error == 0:
        return 0.0
    return math.sqrt(error / norm) if norm > 0 else math.inf


def pcc(sums: torch.Tensor) -> float:
    """Pearson correlation of all elements of the rows of ``sums``; NaN when either side is constant.

    Rows are merged by their centred moments (Chan et al.): each row's moments about its own mean plus the spread of
    the row means about the overall mean, so no large raw sums cancel."""
    n = sums[:, COUNT]
    total = n.sum()
    mean_y, mean_x = (n * sums[:, MEAN_Y]).sum() / total, (n * sums[:, MEAN_X]).sum() / total
    shift_y, shift_x = sums[:, MEAN_Y] - mean_y, sums[:, MEAN_X] - mean_x
    m2_y = float(sums[:, M2_Y].sum() + (n * shift_y * shift_y).sum())
    m2_x = float(sums[:, M2_X].sum() + (n * shift_x * shift_x).sum())
    c_xy = float(sums[:, C_XY].sum() + (n * shift_x * shift_y).sum())
    return c_xy / math.sqrt(m2_x * m2_y) if m2_x > 0 and m2_y > 0 else math.nan


def row_nrmse(sums: torch.Tensor, floor: float) -> torch.Tensor:
    """Per-row NRMSE with each row's golden norm floored at ``floor`` (the golden's typical scale, so a near-zero row
    does not report its own magnitude); an exact row is 0."""
    error = sums[:, SUM_EE].sqrt()
    return torch.where(error == 0, torch.zeros_like(error), error / sums[:, SUM_YY].sqrt().clamp_min(floor))


def rms_err(sums: torch.Tensor) -> float:
    """Absolute error: the RMS of ``x − y`` over all elements of the rows of ``sums``."""
    return math.sqrt(float(sums[:, SUM_EE].sum()) / float(sums[:, COUNT].sum()))


def rms_ref(sums: torch.Tensor) -> float:
    """The golden's own scale: the RMS of ``y`` over all elements of the rows of ``sums``."""
    return math.sqrt(float(sums[:, SUM_YY].sum()) / float(sums[:, COUNT].sum()))


def quantile(t: torch.Tensor, q: float) -> float:
    """The ``q`` quantile of all of ``t``'s values (linear interpolation, as ``torch.quantile``), NaN if empty; for any
    size (``torch.quantile`` refuses more than 2^24 values)."""
    v = t.flatten().float().sort().values
    if not v.numel():
        return math.nan
    pos = q * (v.numel() - 1)
    lo = int(pos)
    hi = min(lo + 1, v.numel() - 1)
    return float(v[lo] + (v[hi] - v[lo]) * (pos - lo))


Window = Mapping[str, object]
"""One window's slice of a ``RowSeries``: each row column as a ``[rows, ...]`` tensor, each event column as a dict of
1-D tensors (its ``row`` field absolute)."""

Metric = Callable[[Window], "float | None"]


def of(column: str, f: Callable[[torch.Tensor], float]) -> Metric:
    """A window metric computing ``f`` on one row column."""
    return lambda window: f(window[column])


ERRORS: dict[str, Callable[[torch.Tensor], float]] = {
    "nrmse": nrmse,
    "rms_err": rms_err,
    "rms_ref": rms_ref,
    "pcc": pcc,
}


def errors(sums: torch.Tensor, prefix: str = "") -> dict[str, float]:
    """The absolute and relative error of the rows of ``sums`` together: ``<prefix>nrmse``, ``<prefix>rms_err``,
    ``<prefix>rms_ref`` and ``<prefix>pcc``."""
    return {f"{prefix}{name}": f(sums) for name, f in ERRORS.items()}


def error_metrics(prefix: str, column: str = "sums") -> dict[str, Metric]:
    """``errors`` as window metrics of a continuous product whose ``row_sums`` are in ``column``:
    ``<prefix>_nrmse``, ``<prefix>_rms_err``, ``<prefix>_rms_ref`` and ``<prefix>_pcc``."""
    return {f"{prefix}_{name}": of(column, f) for name, f in ERRORS.items()}


class RowSeries:
    """Named per-row columns of a result streamed in contiguous chunks from row 0, plus row-tagged event columns.

    ``append(start, **columns)`` adds the rows ``[start, start + n)`` of every row column (each ``[n, ...]``; the same
    names on every call). ``add_events(name, row=..., **fields)`` adds events (for example top-k misses) at absolute
    rows, in any order. ``windows`` slices both by row."""

    def __init__(self) -> None:
        self._chunks: dict[str, list[torch.Tensor]] = {}
        self._events: dict[str, dict[str, list[torch.Tensor]]] = {}
        self._end = 0

    @property
    def end(self) -> int:
        return self._end

    def append(self, start: int, **columns: torch.Tensor) -> None:
        if start != self._end:
            raise ValueError(f"chunk at row {start} does not continue the series, which ends at row {self._end}")
        if not columns:
            raise ValueError("a chunk needs at least one column")
        if self._chunks and set(columns) != set(self._chunks):
            raise ValueError(f"chunk columns {sorted(columns)} differ from the series' {sorted(self._chunks)}")
        rows = {name: t.shape[0] if t.ndim else -1 for name, t in columns.items()}
        if len(set(rows.values())) != 1 or -1 in rows.values():
            raise ValueError(f"every column needs the same leading row count, got {rows}")
        for name, t in columns.items():
            self._chunks.setdefault(name, []).append(t)
        self._end += next(iter(rows.values()))

    def add_events(self, name: str, *, row: torch.Tensor, **fields: torch.Tensor) -> None:
        if name in self._chunks:
            raise ValueError(f"{name!r} is a row column")
        fields = {"row": row, **fields}
        if any(t.ndim != 1 or t.shape[0] != row.shape[0] for t in fields.values()):
            raise ValueError(f"event fields must be 1-D of one length, got {[tuple(t.shape) for t in fields.values()]}")
        events = self._events.setdefault(name, {k: [] for k in fields})
        if set(events) != set(fields):
            raise ValueError(f"event fields {sorted(fields)} differ from {name!r}'s {sorted(events)}")
        for k, t in fields.items():
            events[k].append(t)

    def column(self, name: str) -> torch.Tensor:
        """All rows of row column ``name``."""
        if name not in self._chunks:
            raise ValueError(f"no row column {name!r}" if self._chunks else "no rows accumulated")
        chunks = self._chunks[name]
        if len(chunks) > 1:
            chunks[:] = [torch.cat(chunks)]
        return chunks[0]

    def events(self, name: str) -> dict[str, torch.Tensor]:
        """All events of event column ``name`` (``{}`` fields empty when none were added)."""
        fields = self._events[name]
        for chunks in fields.values():
            if len(chunks) > 1:
                chunks[:] = [torch.cat(chunks)]
        return {k: v[0] for k, v in fields.items()}

    def windows(self, size: int, metrics: dict[str, Metric]) -> list[dict]:
        """``metrics`` of each ``size``-row window aligned to row 0 and clipped to the series; each window is
        ``{"start", "end", "rows", <metric>: value}`` (``rows`` < ``size`` marks a partial window). A metric that
        returns ``None`` is left out of that window."""
        if size <= 0:
            raise ValueError(f"window size must be positive, got {size}")
        if not self._chunks:
            raise ValueError("no rows accumulated")
        columns = {name: self.column(name) for name in self._chunks}
        events = {name: self.events(name) for name in self._events}
        windows = []
        for start in range(0, self._end, size):
            end = min(start + size, self._end)
            view: dict[str, object] = {name: t[start:end] for name, t in columns.items()}
            for name, fields in events.items():
                inside = (fields["row"] >= start) & (fields["row"] < end)
                view[name] = {k: v[inside] for k, v in fields.items()}
            window = {"start": start, "end": end, "rows": end - start}
            for metric, f in metrics.items():
                value = f(view)
                if value is not None:
                    window[metric] = value
            windows.append(window)
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

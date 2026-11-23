# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared windowed scoring core for long-context accuracy: per-row sums, fixed row windows, trends and bars.

A device result over a long prompt is scored chunk by chunk without holding its rows: each chunk becomes per-row
FP64 sufficient statistics (``row_sums``), from which the NRMSE, PCC and absolute RMS error of any set of rows follow
exactly. Windows are aligned to absolute row 0, so two runs with different chunk sizes or layouts score identical row
windows.

Every continuous product is reported in absolute and relative terms together (``errors``): a relative error
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


def _centred(sums: torch.Tensor) -> tuple[float, float, float]:
    """(Σ(y − ȳ)², Σ(x − x̄)², Σ(x − x̄)(y − ȳ)) over all elements of the rows of ``sums``. Rows are merged by their
    centred moments (Chan et al.): each row's moments about its own mean plus the spread of the row means about the
    overall mean, so no large raw sums cancel."""
    n = sums[:, COUNT]
    total = n.sum()
    mean_y, mean_x = (n * sums[:, MEAN_Y]).sum() / total, (n * sums[:, MEAN_X]).sum() / total
    shift_y, shift_x = sums[:, MEAN_Y] - mean_y, sums[:, MEAN_X] - mean_x
    m2_y = float(sums[:, M2_Y].sum() + (n * shift_y * shift_y).sum())
    m2_x = float(sums[:, M2_X].sum() + (n * shift_x * shift_x).sum())
    c_xy = float(sums[:, C_XY].sum() + (n * shift_x * shift_y).sum())
    return m2_y, m2_x, c_xy


def pcc(sums: torch.Tensor) -> float:
    """Pearson correlation of all elements of the rows of ``sums``. Fails closed: a varying golden against a constant
    device side (an all-zero cache never written, say) is 0 (no correlation), and non-finite data give NaN. Only a
    constant golden leaves it undefined (NaN; ``errors`` leaves it out)."""
    m2_y, m2_x, c_xy = _centred(sums)
    if m2_y == 0:
        return math.nan
    if m2_x == 0:
        return 0.0
    return c_xy / math.sqrt(m2_x * m2_y)


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


def errors(sums: torch.Tensor, prefix: str = "") -> dict[str, float]:
    """The absolute and relative error of the rows of ``sums`` together: ``<prefix>nrmse``, ``<prefix>rms_err``,
    ``<prefix>rms_ref``, and ``<prefix>pcc`` unless the golden is constant (PCC undefined: left out, not NaN). A
    constant device side gives PCC 0 and non-finite data NaN, so a minimum-PCC bar fails them (fail closed)."""
    out = {f"{prefix}nrmse": nrmse(sums), f"{prefix}rms_err": rms_err(sums), f"{prefix}rms_ref": rms_ref(sums)}
    if _centred(sums)[0] != 0:
        out[f"{prefix}pcc"] = pcc(sums)
    return out


@dataclass(frozen=True)
class Window:
    """Rows ``[start, end)`` of a ``RowSeries``: ``window[name]`` is a row column's ``[rows, ...]`` slice, or an
    event column's events inside the window (a dict of tensors, ``row`` absolute)."""

    start: int
    end: int
    data: Mapping[str, object]

    def __getitem__(self, name: str):
        return self.data[name]

    def __contains__(self, name: str) -> bool:
        return name in self.data


Measure = Callable[[Window], dict]


class RowSeries:
    """Named per-row columns of a result from row 0, plus row-tagged event columns, measured per window.

    ``append(start, **columns)`` adds the rows ``[start, start + n)`` of every row column (each ``[n, ...]``; the same
    names on every call, contiguous from row 0). ``add_events(name, row=..., **fields)`` adds events (top-k misses,
    or quantities known on sampled rows only) at absolute rows, in any order; every field has one entry per event.
    ``windows`` and ``whole`` apply one ``measure`` to each window's slice of both. A measure returns ``None`` for a
    quantity its window does not define; that key is left out of the window, never reported as NaN."""

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
        if row.ndim != 1 or any(t.ndim < 1 or t.shape[0] != row.shape[0] for t in fields.values()):
            raise ValueError(
                f"every event field needs one entry per event, got {[tuple(t.shape) for t in fields.values()]}"
            )
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

    def _events_of(self, name: str) -> dict[str, torch.Tensor]:
        fields = self._events[name]
        for chunks in fields.values():
            if len(chunks) > 1:
                chunks[:] = [torch.cat(chunks)]
        return {k: v[0] for k, v in fields.items()}

    def _measure(self, start: int, end: int, measure: Measure) -> dict:
        data: dict[str, object] = {name: self.column(name)[start:end] for name in self._chunks}
        for name in self._events:
            fields = self._events_of(name)
            inside = (fields["row"] >= start) & (fields["row"] < end)
            data[name] = {k: v[inside] for k, v in fields.items()}
        values = measure(Window(start, end, data))
        return {"start": start, "end": end, "rows": end - start, **{k: v for k, v in values.items() if v is not None}}

    def windows(self, size: int, measure: Measure) -> list[dict]:
        """``measure`` of each ``size``-row window aligned to row 0 and clipped to the series: ``{"start", "end",
        "rows", **measure(window)}`` (``rows`` < ``size`` marks a partial window)."""
        if size <= 0:
            raise ValueError(f"window size must be positive, got {size}")
        if not self._chunks:
            raise ValueError("no rows accumulated")
        return [self._measure(start, min(start + size, self._end), measure) for start in range(0, self._end, size)]

    def whole(self, measure: Measure) -> dict:
        """``measure`` of every row of the series, as one window."""
        if not self._chunks:
            raise ValueError("no rows accumulated")
        return self._measure(0, self._end, measure)


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

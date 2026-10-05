# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Terminal and Markdown summaries of a comparison. Both come from the same rows."""

from __future__ import annotations

from collections import Counter

from tests.perf.compare import FAILING, CaseResult, Comparison

STATUS_ORDER = ("REGRESSION", "STALE", "ERROR", "MISSING", "NEW", "PASS")


def format_value(value: float | None, unit: str) -> str:
    if value is None:
        return "-"
    if unit == "s":
        for scale, suffix in ((1e-9, "ns"), (1e-6, "us"), (1e-3, "ms")):
            if value < scale * 1000:
                return f"{value / scale:.2f} {suffix}"
        return f"{value:.3f} s"
    if unit == "B/s":
        return f"{value / 1e9:.3f} GB/s"
    return f"{value:.4g} {unit}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1f}%"


def _context_line(context: dict[str, str], golden_context: dict[str, str]) -> str:
    parts = []
    for key in sorted(set(context) | set(golden_context)):
        now, then = context.get(key, "?"), golden_context.get(key)
        if then is None or then == now:
            parts.append(f"{key}={now}")
        else:
            parts.append(f"{key}={now} (golden {then})")
    return ", ".join(parts)


def _detail(result: CaseResult, unit: str, show_metric: bool) -> list[str]:
    status = result.status if result.first_status is None else f"PASS (first run {result.first_status})"
    retry = f"retry {_pct(result.retry_change_pct)}" if result.retry_actual is not None else ""
    note = result.error or retry
    return [
        status,
        f"{result.case} [{result.metric}]" if show_metric and result.metric != "-" else result.case,
        format_value(result.actual, unit),
        format_value(result.golden, unit),
        _pct(result.change_pct),
        note,
    ]


def render(
    title: str,
    comparison: Comparison,
    units: dict[str, str],
    *,
    context: dict[str, str] | None = None,
    golden_context: dict[str, str] | None = None,
    enforce: bool = True,
    show_all: bool = False,
    markdown: bool = False,
    baseline_label: str = "golden",
    notes: list[str] | tuple = (),
) -> str:
    counts = Counter(r.status for r in comparison.results)
    totals = "  ".join(f"{counts[s]} {s}" for s in STATUS_ORDER if counts[s])
    out = []
    heading = f"{title}   {totals}" + ("" if enforce else "   (report only, not enforced)")
    out.append(f"### {heading}" if markdown else heading)
    if context or golden_context:
        line = _context_line(context or {}, golden_context or {})
        out.append(f"Context: {line}" if not markdown else f"Context: `{line}`")
    for error in comparison.config_errors:
        out.append(f"CONFIG MISMATCH: {error}")
    out += list(notes)
    out.append("")

    statuses = [s for s in STATUS_ORDER if counts[s]]
    groups: dict[str, Counter] = {}
    for result in comparison.results:
        groups.setdefault(result.group, Counter())[result.status] += 1
    out += _table(
        ["group", *[s.lower() for s in statuses]],
        [[g, *[str(c[s]) for s in statuses]] for g, c in sorted(groups.items())],
        markdown,
    )
    out.append("")

    rows = [
        _detail(r, units.get(r.metric, ""), show_metric=len(units) > 1)
        for r in sorted(comparison.results, key=lambda r: (STATUS_ORDER.index(r.status), r.case))
        if show_all or r.status in FAILING or r.first_status is not None
    ]
    if rows:
        out += _table(["status", "case", "actual", baseline_label, "change", "note"], rows, markdown)
    return "\n".join(out) + "\n"


def _table(header: list[str], rows: list[list[str]], markdown: bool) -> list[str]:
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        return lines + ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = lambda row: "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip()
    return [fmt(header)] + [fmt(row) for row in rows]

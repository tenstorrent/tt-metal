#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Render a polished single-page HTML report from a JUnit XML produced by the
ttsim regression harness.

Adds value over `junit2html`:
  * Big-number summary cards with pass/fail/skip percentages.
  * A donut chart (pure CSS).
  * Top-N ttsim error categories (extracted from <system-out>).
  * Per-file summary table.
  * Per-test detail with the captured ttsim stdout inlined.

Self-contained: zero JS framework, no external assets at runtime.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from junitparser import JUnitXml
except ImportError:
    sys.stderr.write("ERROR: junitparser not installed. Run: pip install junitparser\n")
    sys.exit(2)


_TTSIM_ERR_RE = re.compile(r"^\[\d+\] ERROR: (\w+): (\w+): (.*)$", re.MULTILINE)


@dataclass
class Case:
    classname: str
    name: str
    duration: float
    outcome: str
    short: str
    stdout: str
    ttsim_category: Optional[str] = None
    ttsim_func: Optional[str] = None
    ttsim_detail: Optional[str] = None


@dataclass
class FileStats:
    file: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errored: int = 0
    cases: list[Case] = field(default_factory=list)


@dataclass
class RunMeta:
    """Run-level metadata derived from the XML and the file on disk.

    `cases_time` is the SUM of per-test durations (this can exceed wall-clock
    when tests run in parallel under pytest-xdist). `wall_time` is an
    approximation of the actual end-to-end runtime computed from the
    suite's start `timestamp` attribute and the XML file's mtime; it is
    `None` when either piece is unavailable.
    """

    cases_time: float = 0.0
    wall_time: Optional[float] = None
    started_at: Optional[str] = None


def _parse_iso(ts: str) -> Optional[float]:
    """Parse an ISO-8601 timestamp into a POSIX seconds value, or None."""
    if not ts:
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def parse(xml_path: Path) -> tuple[list[FileStats], list[Case], RunMeta]:
    xml = JUnitXml.fromfile(str(xml_path))
    by_file: dict[str, FileStats] = {}
    all_cases: list[Case] = []

    earliest_start: Optional[float] = None
    started_at_iso: Optional[str] = None

    for suite in xml:
        ts = getattr(suite, "timestamp", None)
        if ts:
            started_at_iso = started_at_iso or ts
            t = _parse_iso(ts)
            if t is not None and (earliest_start is None or t < earliest_start):
                earliest_start = t
        for case in suite:
            classname = case.classname or "<no-class>"
            stats = by_file.setdefault(classname, FileStats(classname))

            if case.is_passed:
                outcome = "passed"
                short = "PASSED"
                stats.passed += 1
            elif any(getattr(r, "_tag", "") == "skipped" for r in (case.result or [])):
                outcome = "skipped"
                short = (case.result[0].message or "skipped")[:120]
                stats.skipped += 1
            elif any(getattr(r, "_tag", "") == "failure" for r in (case.result or [])):
                outcome = "failed"
                fr = next(r for r in case.result if getattr(r, "_tag", "") == "failure")
                short = (fr.message or "failed")[:160]
                stats.failed += 1
            elif any(getattr(r, "_tag", "") == "error" for r in (case.result or [])):
                outcome = "errored"
                er = next(r for r in case.result if getattr(r, "_tag", "") == "error")
                short = (er.message or "errored")[:160]
                stats.errored += 1
            else:
                outcome = "unknown"
                short = "unknown"

            stdout_parts = []
            so = getattr(case, "system_out", None) or ""
            if so:
                stdout_parts.append(so)
            se = getattr(case, "system_err", None) or ""
            if se:
                stdout_parts.append(se)
            stdout = "\n".join(stdout_parts)

            ttsim_cat = ttsim_func = ttsim_detail = None
            m = _TTSIM_ERR_RE.search(stdout)
            if m:
                ttsim_cat, ttsim_func, ttsim_detail = (
                    m.group(1),
                    m.group(2),
                    m.group(3).strip(),
                )

            c = Case(
                classname=classname,
                name=case.name,
                duration=case.time or 0.0,
                outcome=outcome,
                short=short,
                stdout=stdout,
                ttsim_category=ttsim_cat,
                ttsim_func=ttsim_func,
                ttsim_detail=ttsim_detail,
            )
            stats.total += 1
            stats.cases.append(c)
            all_cases.append(c)

    cases_time = sum(c.duration for c in all_cases)
    wall_time: Optional[float] = None
    if earliest_start is not None:
        try:
            end_time = xml_path.stat().st_mtime
            if end_time > earliest_start:
                wall_time = end_time - earliest_start
        except OSError:
            pass

    meta = RunMeta(
        cases_time=cases_time,
        wall_time=wall_time,
        started_at=started_at_iso,
    )
    return sorted(by_file.values(), key=lambda s: s.file), all_cases, meta


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def _fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {int(s)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


def render(
    files: list[FileStats],
    all_cases: list[Case],
    xml_path: Path,
    meta: RunMeta,
) -> str:
    total = len(all_cases)
    passed = sum(1 for c in all_cases if c.outcome == "passed")
    failed = sum(1 for c in all_cases if c.outcome == "failed")
    skipped = sum(1 for c in all_cases if c.outcome == "skipped")
    errored = sum(1 for c in all_cases if c.outcome == "errored")

    by_category: Counter[str] = Counter()
    by_func: Counter[tuple[str, str]] = Counter()
    for c in all_cases:
        if c.ttsim_category:
            by_category[c.ttsim_category] += 1
            by_func[(c.ttsim_category, c.ttsim_func or "?")] += 1

    # Group failed tests by their exact failure signature. Numeric /hex
    # operands matter (different format codes => different bugs), so we
    # do NOT normalize. For ttsim crashes the signature is
    # "[Category] func: detail"; for other failures it's the test's
    # short failure headline.
    failed_for_sig = [c for c in all_cases if c.outcome in ("failed", "errored")]
    sig_count: Counter[str] = Counter()
    sig_meta: dict[str, dict] = {}
    for c in failed_for_sig:
        if c.ttsim_category:
            sig = f"[{c.ttsim_category}] {c.ttsim_func or '?'}: {c.ttsim_detail or ''}"
            cat = c.ttsim_category
        else:
            sig = c.short.strip().splitlines()[0] if c.short.strip() else "<no message>"
            cat = None
        sig_count[sig] += 1
        bucket = sig_meta.setdefault(sig, {"category": cat, "tests": []})
        bucket["tests"].append(f"{c.classname}::{c.name}")

    cases_time = meta.cases_time
    wall_time = meta.wall_time
    cases_time_str = _fmt_dur(cases_time)
    wall_time_str = _fmt_dur(wall_time) if wall_time is not None else None
    if wall_time is not None and cases_time > 0:
        concurrency = cases_time / wall_time
        wall_time_str_full = f"{wall_time_str} wall · avg concurrency {concurrency:.1f}"
    else:
        wall_time_str_full = wall_time_str

    pass_pct = pct(passed, total)
    fail_pct = pct(failed + errored, total)
    skip_pct = pct(skipped, total)

    # Donut: build CSS conic-gradient
    def conic_stops() -> str:
        cur = 0.0
        parts = []
        for color, n in [
            ("var(--ok)", passed),
            ("var(--bad)", failed + errored),
            ("var(--warn)", skipped),
        ]:
            if n == 0:
                continue
            end = cur + pct(n, total)
            parts.append(f"{color} {cur:.2f}% {end:.2f}%")
            cur = end
        if not parts:
            parts.append("var(--neutral) 0% 100%")
        return ", ".join(parts)

    rows_files = []
    for s in files:
        rows_files.append(
            f"<tr>"
            f"<td>{html.escape(s.file)}</td>"
            f"<td class='num'>{s.total}</td>"
            f"<td class='num ok'>{s.passed}</td>"
            f"<td class='num bad'>{s.failed + s.errored}</td>"
            f"<td class='num warn'>{s.skipped}</td>"
            f"<td class='barcell'>"
            f"  <div class='filebar'>"
            f"    <span class='seg ok'   style='width:{pct(s.passed, s.total):.1f}%'></span>"
            f"    <span class='seg bad'  style='width:{pct(s.failed + s.errored, s.total):.1f}%'></span>"
            f"    <span class='seg warn' style='width:{pct(s.skipped, s.total):.1f}%'></span>"
            f"  </div>"
            f"</td>"
            f"</tr>"
        )

    rows_cat = []
    cat_total = sum(by_category.values()) or 1
    cat_max = max(by_category.values(), default=1)
    for cat, n in by_category.most_common():
        rows_cat.append(
            f"<tr>"
            f"<td><span class='pill bad'>{html.escape(cat)}</span></td>"
            f"<td class='num'>{n}</td>"
            f"<td class='num'>{pct(n, cat_total):.1f}%</td>"
            f"<td class='barcell'><span class='minibar' style='width:{pct(n, cat_max):.1f}%'></span></td>"
            f"</tr>"
        )

    rows_func = []
    func_max = max(by_func.values(), default=1)
    for (cat, func), n in by_func.most_common(20):
        rows_func.append(
            f"<tr>"
            f"<td><span class='pill bad'>{html.escape(cat)}</span></td>"
            f"<td><code>{html.escape(func)}</code></td>"
            f"<td class='num'>{n}</td>"
            f"<td class='barcell'><span class='minibar' style='width:{pct(n, func_max):.1f}%'></span></td>"
            f"</tr>"
        )

    fail_total = len(failed_for_sig) or 1
    rows_sig = []
    for rank, (sig, n) in enumerate(sig_count.most_common(), start=1):
        bucket = sig_meta[sig]
        cat = bucket["category"]
        tests = bucket["tests"]
        # Split signature so the category renders as a pill and the rest as code.
        if cat:
            body = sig[len(f"[{cat}] ") :]
            sig_html = (
                f"<span class='pill bad'>{html.escape(cat)}</span> "
                f"<code>{html.escape(body)}</code>"
            )
        else:
            sig_html = f"<code>{html.escape(sig)}</code>"
        shown = tests[:200]
        more = len(tests) - len(shown)
        tests_html = (
            "<ul class='siglist'>"
            + "".join(f"<li><code>{html.escape(t)}</code></li>" for t in shown)
            + (f"<li><i>… {more} more</i></li>" if more > 0 else "")
            + "</ul>"
        )
        share_pct = pct(n, fail_total)
        rows_sig.append(
            f"<tr class='sigrow'>"
            f"<td class='num'>{rank}</td>"
            f"<td class='num'>{n}</td>"
            f"<td class='num'>{share_pct:.1f}%</td>"
            f"<td class='barcell'><span class='minibar' style='width:{share_pct:.1f}%'></span></td>"
            f"<td class='sigcell'>"
            f"  <details><summary>{sig_html}</summary>{tests_html}</details>"
            f"</td>"
            f"</tr>"
        )

    failed_cases = [c for c in all_cases if c.outcome in ("failed", "errored")]
    failed_cases.sort(key=lambda c: (c.ttsim_category or "ZZZ", c.classname, c.name))
    rows_fail = []
    for c in failed_cases:
        chip = ""
        if c.ttsim_category:
            chip = f"<span class='pill bad'>{html.escape(c.ttsim_category)}</span> "
        headline = html.escape(c.short)
        if c.ttsim_category:
            headline = f"{chip}<code>{html.escape(c.ttsim_func or '')}</code>: {html.escape(c.ttsim_detail or '')}"
        rows_fail.append(
            f"<details class='case'>"
            f"<summary>"
            f"  <span class='outcome bad'>FAIL</span>"
            f"  <span class='name'><code>{html.escape(c.classname)}::{html.escape(c.name)}</code></span>"
            f"  <span class='headline'>{headline}</span>"
            f"  <span class='dur'>{c.duration:.2f}s</span>"
            f"</summary>"
            f"<pre>{html.escape(c.stdout) or '<i>no captured stdout</i>'}</pre>"
            f"</details>"
        )

    cat_section = (
        (
            "<table class='datatable'>"
            "<thead><tr><th>Category</th><th class='num'>Count</th><th class='num'>% of failures</th><th class='sharecol'>Share</th></tr></thead>"
            "<tbody>" + "".join(rows_cat) + "</tbody></table>"
        )
        if rows_cat
        else (
            "<div class='card empty'>No categorized ttsim failures in this run.</div>"
        )
    )
    func_section = (
        (
            "<h2 id='functions'>Top failing functions</h2>"
            "<table class='datatable'>"
            "<thead><tr><th>Category</th><th>Function</th><th class='num'>Count</th><th class='sharecol'>Share</th></tr></thead>"
            "<tbody>" + "".join(rows_func) + "</tbody></table>"
        )
        if rows_func
        else ""
    )
    sig_section = (
        (
            f"<h2 id='signatures'>Unique failure signatures "
            f"<span class='hcount'>{len(sig_count)} distinct · {len(failed_for_sig)} failures</span></h2>"
            "<table class='datatable sigtable'>"
            "<thead><tr><th class='num'>#</th><th class='num'>Count</th><th class='num'>% of failures</th>"
            "<th class='sharecol'>Share</th><th>Signature <span class='hint'>(click to list affected tests)</span></th></tr></thead>"
            "<tbody>" + "".join(rows_sig) + "</tbody></table>"
        )
        if rows_sig
        else ""
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>ttsim regression — {html.escape(xml_path.name)}</title>
<style>
  :root {{
    --bg: #f5f7fb;          --bg-elev: #ffffff;
    --card: #ffffff;        --card-2: #f9fafc;
    --muted: #6b7180;       --fg: #1f2330;
    --ok: #16a34a;          --bad: #dc2626;     --warn: #d97706;
    --ok-bg: rgba(22,163,74,.1);   --bad-bg: rgba(220,38,38,.08);   --warn-bg: rgba(217,119,6,.1);
    --neutral: #e3e7ef;     --neutral-2: #eef1f7;
    --accent: #4f6cff;      --accent-2: #6b85ff;
    --shadow: 0 1px 2px rgba(15,23,42,.06), 0 4px 12px rgba(15,23,42,.04);
    --radius: 12px;
    --hero-grad: linear-gradient(135deg, #4f6cff 0%, #7c3aed 50%, #ec4899 100%);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0d1117;        --bg-elev: #161b22;
      --card: #161b22;      --card-2: #1c222b;
      --muted: #7d8590;     --fg: #e6edf3;
      --ok: #3fb950;        --bad: #f85149;     --warn: #d29922;
      --ok-bg: rgba(63,185,80,.16);  --bad-bg: rgba(248,81,73,.14);  --warn-bg: rgba(210,153,34,.16);
      --neutral: #30363d;   --neutral-2: #21262d;
      --accent: #2f81f7;    --accent-2: #58a6ff;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.25);
      --hero-grad: linear-gradient(135deg, #1f6feb 0%, #8957e5 50%, #db61a2 100%);
    }}
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{ margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", system-ui, sans-serif;
         background: var(--bg); color: var(--fg);
         font-feature-settings: "ss01", "cv11"; }}
  code, pre {{ font-family: "SF Mono", "JetBrains Mono", ui-monospace, "Menlo", "Consolas", monospace; }}
  .num {{ font-variant-numeric: tabular-nums; }}

  /* ── Hero ─────────────────────────────────────────── */
  .hero {{ background: var(--hero-grad); color: #fff; padding: 36px 32px 0; position: relative; overflow: hidden; }}
  .hero::after {{ content: ''; position: absolute; inset: 0;
                  background: radial-gradient(1200px 400px at 90% -50%, rgba(255,255,255,.18), transparent 60%); }}
  .hero-inner {{ position: relative; max-width: 1280px; margin: 0 auto;
                 display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: end; }}
  .brand h1 {{ margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -.01em; }}
  .brand .meta {{ margin-top: 6px; font-size: 13px; opacity: .85; }}
  .passrate {{ text-align: right; }}
  .passrate-num {{ font-size: 56px; font-weight: 700; line-height: 1; letter-spacing: -.02em;
                   text-shadow: 0 2px 24px rgba(0,0,0,.15); }}
  .passrate-label {{ font-size: 12px; opacity: .85; text-transform: uppercase; letter-spacing: .1em; margin-top: 4px; }}
  .heroprogress {{ margin-top: 24px; height: 6px; border-radius: 999px; background: rgba(255,255,255,.18);
                   overflow: hidden; display: flex; box-shadow: inset 0 1px 0 rgba(0,0,0,.08); }}
  .heroprogress span {{ display: block; height: 100%; }}
  .heroprogress .ok   {{ background: #34d399; }}
  .heroprogress .bad  {{ background: #fb7185; }}
  .heroprogress .warn {{ background: #fbbf24; }}

  /* ── Sticky section nav ───────────────────────────── */
  nav.toc {{ position: sticky; top: 0; z-index: 10; background: var(--bg-elev);
             border-bottom: 1px solid var(--neutral); padding: 0 32px;
             box-shadow: 0 1px 0 var(--neutral); margin-top: 24px; }}
  nav.toc .toc-inner {{ max-width: 1280px; margin: 0 auto; display: flex; gap: 4px; overflow-x: auto; }}
  nav.toc a {{ padding: 14px 14px; color: var(--muted); text-decoration: none; font-size: 13px;
               font-weight: 500; border-bottom: 2px solid transparent; white-space: nowrap;
               transition: color .15s, border-color .15s; }}
  nav.toc a:hover {{ color: var(--fg); }}
  nav.toc a.active {{ color: var(--accent); border-color: var(--accent); }}

  /* ── Layout ───────────────────────────────────────── */
  main {{ max-width: 1280px; margin: 0 auto; padding: 32px; }}
  section {{ animation: fadeUp .45s ease both; }}
  section + section {{ margin-top: 40px; }}
  @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(8px); }}
                       to {{ opacity: 1; transform: translateY(0); }} }}
  h2 {{ margin: 0 0 14px 0; font-size: 13px; color: var(--muted); text-transform: uppercase;
        letter-spacing: .08em; font-weight: 600; }}
  h2 .hcount {{ margin-left: 8px; padding: 2px 8px; border-radius: 999px; background: var(--neutral-2);
                color: var(--muted); font-size: 11px; letter-spacing: .04em; }}
  .hint {{ font-weight: 400; text-transform: none; letter-spacing: 0; color: var(--muted); }}

  /* ── KPI cards ────────────────────────────────────── */
  .row {{ display: grid; gap: 16px; }}
  .summary {{ grid-template-columns: 240px repeat(4, 1fr); align-items: stretch; }}
  .card {{ background: var(--card); border-radius: var(--radius); padding: 20px;
           border: 1px solid var(--neutral); box-shadow: var(--shadow);
           transition: transform .15s ease, box-shadow .15s ease; position: relative; overflow: hidden; }}
  .card.kpi {{ padding-left: 24px; }}
  .card.kpi::before {{ content: ''; position: absolute; left: 0; top: 14px; bottom: 14px;
                       width: 3px; border-radius: 0 3px 3px 0; background: var(--neutral); }}
  .card.kpi.ok::before   {{ background: var(--ok);   }}
  .card.kpi.bad::before  {{ background: var(--bad);  }}
  .card.kpi.warn::before {{ background: var(--warn); }}
  .card:hover {{ transform: translateY(-1px);
                 box-shadow: 0 1px 2px rgba(15,23,42,.08), 0 12px 28px rgba(15,23,42,.08); }}
  .card .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
                  font-weight: 600; }}
  .card .big {{ font-size: 32px; font-weight: 700; margin-top: 8px; line-height: 1.05;
                font-variant-numeric: tabular-nums; letter-spacing: -.01em; }}
  .card .sub {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}
  .donut-wrap {{ display: grid; place-items: center; padding: 14px; }}
  .donut {{ position: relative; width: 196px; height: 196px; border-radius: 50%;
            background: conic-gradient({conic_stops()}); box-shadow: inset 0 0 0 1px rgba(0,0,0,.04); }}
  .donut::after {{ content: ''; position: absolute; inset: 18px; background: var(--card);
                   border-radius: 50%; box-shadow: inset 0 0 0 1px var(--neutral); }}
  .donut .center {{ position: absolute; inset: 0; display: grid; place-items: center; text-align: center; z-index: 1; }}
  .donut .pct {{ font-size: 30px; font-weight: 700; letter-spacing: -.02em;
                 font-variant-numeric: tabular-nums; }}
  .donut .ctxt {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em;
                  margin-top: 2px; }}

  /* ── Tables ───────────────────────────────────────── */
  .ok   {{ color: var(--ok); }}
  .bad  {{ color: var(--bad); }}
  .warn {{ color: var(--warn); }}
  .pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px;
           font-weight: 600; line-height: 1.5; white-space: nowrap; }}
  .pill.bad  {{ background: var(--bad-bg);  color: var(--bad);  }}
  .pill.ok   {{ background: var(--ok-bg);   color: var(--ok);   }}
  .pill.warn {{ background: var(--warn-bg); color: var(--warn); }}
  table.datatable {{ width: 100%; border-collapse: separate; border-spacing: 0;
                     background: var(--card); border-radius: var(--radius); overflow: hidden;
                     border: 1px solid var(--neutral); box-shadow: var(--shadow); }}
  .datatable th, .datatable td {{ padding: 10px 14px; text-align: left;
                                  border-bottom: 1px solid var(--neutral-2); vertical-align: middle; }}
  .datatable tr:last-child td {{ border-bottom: none; }}
  .datatable tbody tr {{ transition: background .12s; }}
  .datatable tbody tr:hover {{ background: var(--card-2); }}
  .datatable th {{ font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
                   color: var(--muted); background: var(--card-2); font-weight: 600; }}
  .datatable td.num {{ text-align: right; }}
  .sharecol {{ width: 200px; }}
  .barcell {{ width: 200px; padding: 0 14px; }}
  .minibar {{ display: block; height: 6px; background: linear-gradient(90deg, var(--bad), #fb7185);
              border-radius: 999px; min-width: 2px; opacity: .85;
              box-shadow: 0 0 0 1px rgba(220,38,38,.06); }}

  .filebar {{ width: 220px; height: 8px; border-radius: 999px; overflow: hidden;
              background: var(--neutral-2); display: flex; }}
  .filebar .seg {{ height: 100%; }}
  .filebar .seg.ok   {{ background: var(--ok); }}
  .filebar .seg.bad  {{ background: var(--bad); }}
  .filebar .seg.warn {{ background: var(--warn); }}

  .sigtable td.sigcell {{ max-width: 720px; }}
  .sigtable td.sigcell code {{ font-size: 12px; word-break: break-word; }}
  .sigtable details summary {{ cursor: pointer; padding: 2px 0; list-style: none; user-select: none; }}
  .sigtable details summary::-webkit-details-marker {{ display: none; }}
  .sigtable details summary::before {{ content: '▸'; display: inline-block; width: 14px;
                                       color: var(--muted); transition: transform .15s; }}
  .sigtable details[open] summary::before {{ transform: rotate(90deg); }}
  .sigtable details[open] summary {{ margin-bottom: 6px; }}
  .siglist {{ margin: 0 0 4px 18px; padding-left: 8px; max-height: 240px; overflow-y: auto;
              color: var(--muted); font-size: 12px; border-left: 2px solid var(--neutral-2); }}
  .siglist li {{ margin: 2px 0; list-style: none; }}
  .siglist li code {{ font-size: 11px; }}

  /* ── Failures list ────────────────────────────────── */
  .filterbar {{ display: flex; gap: 8px; margin-bottom: 12px; position: sticky; top: 49px; z-index: 5;
                background: var(--bg); padding: 8px 0; }}
  .filterbar input {{ flex: 1; padding: 10px 14px; border-radius: 10px; border: 1px solid var(--neutral);
                      background: var(--card); color: var(--fg); font: inherit;
                      box-shadow: var(--shadow); transition: border-color .15s, box-shadow .15s; }}
  .filterbar input:focus {{ outline: none; border-color: var(--accent);
                            box-shadow: 0 0 0 3px rgba(79,108,255,.15), var(--shadow); }}
  details.case {{ background: var(--card); border: 1px solid var(--neutral); border-radius: 10px;
                  margin-bottom: 6px; transition: border-color .15s, box-shadow .15s; }}
  details.case:hover {{ border-color: var(--neutral); box-shadow: var(--shadow); }}
  details.case[open] {{ border-color: var(--bad); box-shadow: 0 0 0 1px var(--bad-bg), var(--shadow); }}
  details.case summary {{ display: grid; grid-template-columns: 60px minmax(0, 1fr) minmax(0, 1.5fr) 60px;
                          gap: 16px; align-items: baseline; padding: 10px 14px; cursor: pointer;
                          list-style: none; }}
  details.case summary::-webkit-details-marker {{ display: none; }}
  .outcome {{ font-size: 10px; font-weight: 700; padding: 3px 7px; border-radius: 4px; text-align: center;
              text-transform: uppercase; letter-spacing: .05em; }}
  .outcome.bad {{ background: var(--bad-bg); color: var(--bad); }}
  .name code {{ font-size: 12px; }}
  .name, .headline {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .headline {{ color: var(--muted); font-size: 13px; }}
  .dur {{ text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }}
  details.case pre {{ margin: 0; padding: 14px 18px; background: var(--card-2);
                      border-top: 1px solid var(--neutral); overflow-x: auto;
                      font-size: 12px; line-height: 1.6; max-height: 480px; }}

  .empty {{ color: var(--muted); text-align: center; padding: 28px; font-size: 13px; }}
</style>
</head>
<body>
<header class="hero">
  <div class="hero-inner">
    <div class="brand">
      <h1>ttsim regression</h1>
      <div class="meta">{html.escape(xml_path.name)} · {total} tests · {wall_time_str_full or (cases_time_str + " test-time")}</div>
    </div>
    <div class="passrate">
      <div class="passrate-num">{pass_pct:.1f}%</div>
      <div class="passrate-label">{passed} of {total} passing</div>
    </div>
  </div>
  <div class="hero-inner">
    <div class="heroprogress" style="grid-column: 1 / -1;">
      <span class="ok"   style="width:{pass_pct:.2f}%"></span>
      <span class="bad"  style="width:{fail_pct:.2f}%"></span>
      <span class="warn" style="width:{skip_pct:.2f}%"></span>
    </div>
  </div>
  <div class="hero-inner" style="height: 24px;"></div>
</header>
<nav class="toc">
  <div class="toc-inner">
    <a href="#summary" class="active">Summary</a>
    <a href="#categories">Categories</a>
    {("<a href='#functions'>Top functions</a>") if rows_func else ""}
    {("<a href='#signatures'>Signatures</a>") if rows_sig else ""}
    <a href="#files">Files</a>
    <a href="#failures">Failures</a>
  </div>
</nav>
<main>
  <section id="summary">
    <div class="row summary">
      <div class="card donut-wrap">
        <div class="donut">
          <div class="center">
            <div>
              <div class="pct">{pass_pct:.0f}%</div>
              <div class="ctxt">passing</div>
            </div>
          </div>
        </div>
      </div>
      <div class="card kpi"><div class="label">Total</div><div class="big">{total}</div><div class="sub" title="Σ test-time = sum of per-test durations from JUnit. Wall-clock = end-to-end runtime estimated from the suite's timestamp and the XML mtime.">{("wall " + wall_time_str + " · ") if wall_time_str else ""}Σ test {cases_time_str}</div></div>
      <div class="card kpi ok"><div class="label">Passed</div><div class="big ok">{passed}</div><div class="sub">{pass_pct:.1f}%</div></div>
      <div class="card kpi bad"><div class="label">Failed</div><div class="big bad">{failed + errored}</div><div class="sub">{fail_pct:.1f}%</div></div>
      <div class="card kpi warn"><div class="label">Skipped</div><div class="big warn">{skipped}</div><div class="sub">{skip_pct:.1f}%</div></div>
    </div>
  </section>

  <section id="categories">
    <h2>ttsim error categories</h2>
    {cat_section}
  </section>

  {f"<section id='functions-section'>{func_section}</section>" if func_section else ""}

  {f"<section id='signatures-section'>{sig_section}</section>" if sig_section else ""}

  <section id="files">
    <h2>Per-file breakdown</h2>
    <table class="datatable">
      <thead><tr><th>File</th><th class='num'>Total</th><th class='num'>Pass</th><th class='num'>Fail</th><th class='num'>Skip</th><th class='sharecol'>Distribution</th></tr></thead>
      <tbody>{"".join(rows_files)}</tbody>
    </table>
  </section>

  <section id="failures">
    <h2>Failed tests <span class='hcount'>{len(failed_cases)}</span></h2>
    <div class="filterbar">
      <input id="filter" type="text" placeholder="filter failed tests by substring (test name, function, message)…">
    </div>
    <div id="cases">{"".join(rows_fail) if rows_fail else "<div class='card empty'>No failures.</div>"}</div>
  </section>
</main>
<script>
(() => {{
  const f = document.getElementById('filter');
  const cases = document.querySelectorAll('#cases details.case');
  if (f) {{
    let raf = 0;
    const apply = () => {{
      const q = f.value.toLowerCase();
      cases.forEach(c => {{
        c.style.display = c.textContent.toLowerCase().includes(q) ? '' : 'none';
      }});
    }};
    f.addEventListener('input', () => {{
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(apply);
    }});
  }}

  // Active nav link based on scroll position.
  const links = Array.from(document.querySelectorAll('nav.toc a'));
  const targets = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  const setActive = () => {{
    const y = window.scrollY + 80;
    let active = targets[0];
    for (const t of targets) if (t.offsetTop <= y) active = t;
    links.forEach(l => l.classList.toggle('active', l.getAttribute('href') === '#' + active.id));
  }};
  document.addEventListener('scroll', setActive, {{ passive: true }});
  setActive();
}})();
</script>
</body></html>
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "xml", type=Path, help="JUnit XML produced by run_ttsim_regression.sh"
    )
    p.add_argument("html", type=Path, help="Output HTML path")
    args = p.parse_args()

    if not args.xml.is_file():
        sys.stderr.write(f"ERROR: not a file: {args.xml}\n")
        return 2

    files, cases, meta = parse(args.xml)
    out = render(files, cases, args.xml, meta)
    args.html.write_text(out, encoding="utf-8")
    print(
        f"Wrote {args.html} ({os.path.getsize(args.html)} bytes, "
        f"{len(cases)} cases across {len(files)} files)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

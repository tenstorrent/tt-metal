"""Generate and serve the eval dashboard as an HTML page.

Usage:
    python3 -m eval.dashboard [--db /path/to/eval_runs.db] [--port 8080]
    python3 -m eval.dashboard --generate-only --output dashboard.html
"""

import argparse
import html
import http.server
import json
import os
import socketserver
import sys
from pathlib import Path

from eval import db

# ---------------------------------------------------------------------------
# Color / style helpers
# ---------------------------------------------------------------------------

GRADE_COLORS = {
    "A": "#22c55e",  # green
    "B": "#84cc16",  # lime
    "C": "#eab308",  # yellow
    "D": "#f97316",  # orange
    "F": "#ef4444",  # red
}

STATUS_COLORS = {
    "passed": "#22c55e",
    "failed": "#ef4444",
    "error": "#b91c1c",
    "skipped": "#6B6488",
}

CATEGORY_COLORS = {
    "hang": "#b91c1c",
    "OOM": "#8b5cf6",
    "numerical": "#f97316",
    "compilation": "#3b82f6",
    "other": "#6B6488",
}


def _grade_bg(grade):
    color = GRADE_COLORS.get(grade, "#6B6488")
    return f"background:{color};color:white;padding:2px 10px;border-radius:4px;font-weight:bold;font-size:0.85em"


def _status_bg(status):
    color = STATUS_COLORS.get(status, "#6B6488")
    return f"background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:600"


def _category_bg(cat):
    if not cat:
        return ""
    color = CATEGORY_COLORS.get(cat, "#6B6488")
    return f"background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:600"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


THEMES = ("dark", "light")


def generate_html(conn, theme: str = "dark") -> str:
    """Generate the full dashboard HTML from a DB connection."""
    if theme not in THEMES:
        theme = "dark"
    stats = db.get_stats(conn)
    runs = db.get_all_runs(conn)

    # Pre-fetch details for all runs
    run_details = {}
    for run in runs:
        run_details[run["id"]] = {
            "tests": db.get_test_results(conn, run["id"]),
            "criteria": db.get_score_criteria(conn, run["id"]),
            "kernels": db.get_kernels(conn, run["id"]),
            "host_code": db.get_host_code(conn, run["id"]),
            "artifacts": db.get_artifacts(conn, run["id"]),
            "tdd_state": db.get_tdd_state(conn, run["id"]),
            "kw_breadcrumbs": db.get_kw_breadcrumbs(conn, run["id"]),
        }

    parts = [
        _html_head(theme),
        _html_stats(stats),
        _html_failure_bars(stats),
        _html_runs_table(runs, run_details),
        _html_foot(),
    ]
    return "\n".join(parts)


def _html_head(theme: str = "dark") -> str:
    if theme == "light":
        theme_vars = """\
  :root {
    --bg-page: #f1f5f9;
    --bg-card: #ffffff;
    --bg-card-hover: #f8fafc;
    --bg-detail: #f8fafc;
    --bg-table-header: #1e293b;
    --border: #e2e8f0;
    --border-light: #cbd5e1;
    --accent: #2563eb;
    --accent-light: #3b82f6;
    --accent-subtle: rgba(37, 99, 235, 0.06);
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --text-muted: #9ca3af;
    --header-bg: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    --shadow: 0 1px 3px rgba(0,0,0,0.1);
    --msg-expanded-bg: #fef9e7;
    --code-bg: #1e293b;
    --code-color: #e2e8f0;
    --table-header-color: white;
    --row-stripe: rgba(0,0,0,0.02);
  }"""
        hljs_theme = "github.min.css"
    else:
        theme_vars = """\
  :root {
    --bg-page: #0D0B1A;
    --bg-card: #16132A;
    --bg-card-hover: #1E1A38;
    --bg-detail: #12101F;
    --bg-table-header: #1A1630;
    --border: #2D2750;
    --border-light: #3D3668;
    --accent: #7C3AED;
    --accent-light: #A78BFA;
    --accent-subtle: rgba(124, 58, 237, 0.12);
    --text-primary: #E8E6F0;
    --text-secondary: #9993B4;
    --text-muted: #6B6488;
    --header-bg: linear-gradient(135deg, #16132A 0%, #1E1040 50%, #2D1B69 100%);
    --shadow: none;
    --msg-expanded-bg: rgba(124, 58, 237, 0.08);
    --code-bg: #0D0B1A;
    --code-color: var(--text-primary);
    --table-header-color: var(--text-secondary);
    --row-stripe: rgba(255,255,255,0.015);
  }"""
        hljs_theme = "github-dark.min.css"

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval Dashboard</title>
<style>
{theme_vars}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg-page); color: var(--text-primary); min-height: 100vh; }}

  /* Header */
  .dashboard-header {{
    background: var(--header-bg);
    border-bottom: 1px solid var(--border);
    padding: 28px 32px 24px;
    margin-bottom: 24px;
  }}
  .dashboard-header h1 {{
    font-size: 1.6rem; font-weight: 700; color: white;
    display: flex; align-items: center; gap: 10px;
  }}
  .dashboard-header h1 .accent {{ color: var(--accent-light); }}
  .dashboard-header .subtitle {{
    font-size: 0.85rem; color: rgba(255,255,255,0.5); margin-top: 4px;
  }}
  .content {{ padding: 0 32px 32px; }}

  h2 {{ font-size: 1rem; margin: 24px 0 12px; color: var(--text-secondary);
       text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}

  /* Stat cards */
  .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .stat-card {{
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
    padding: 20px 28px; min-width: 160px; position: relative; overflow: hidden;
    box-shadow: var(--shadow);
  }}
  .stat-card::before {{
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; background: var(--accent); border-radius: 3px 0 0 3px;
  }}
  .stat-value {{ font-size: 2rem; font-weight: 700; color: var(--text-primary); line-height: 1.2; }}
  .stat-label {{
    font-size: 0.8rem; color: var(--text-muted); margin-top: 6px;
    text-transform: uppercase; letter-spacing: 0.04em; font-weight: 500;
  }}

  /* Failure bar */
  .bar-container {{
    display: flex; gap: 3px; height: 32px; margin-bottom: 24px;
    border-radius: 8px; overflow: hidden;
    background: var(--bg-card); border: 1px solid var(--border);
  }}
  .bar-segment {{
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 0.75rem; font-weight: 600;
    white-space: nowrap; padding: 0 10px; min-width: 60px;
  }}

  /* Runs table */
  table.runs {{
    width: 100%; border-collapse: collapse; background: var(--bg-card);
    border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
    box-shadow: var(--shadow);
  }}
  table.runs th {{
    background: var(--bg-table-header); color: var(--table-header-color);
    padding: 12px 14px; text-align: left; font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 1px solid var(--border);
  }}
  table.runs td {{
    padding: 10px 14px; border-bottom: 1px solid var(--border);
    font-size: 0.88rem; color: var(--text-primary);
  }}
  tr.run-row {{ cursor: pointer; transition: background 0.15s; }}
  tr.run-row:hover {{ background: var(--accent-subtle); }}
  tr.run-row:nth-child(4n+1) {{ background: var(--row-stripe); }}
  tr.run-row:nth-child(4n+1):hover {{ background: var(--accent-subtle); }}
  tr.detail-row {{ display: none; }}
  tr.detail-row.open {{ display: table-row; }}
  td.detail-cell {{
    padding: 20px; background: var(--bg-detail);
    border-bottom: 2px solid var(--accent);
  }}

  .detail-content {{ max-height: 600px; overflow-y: auto; }}
  .cat-summary {{ display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }}
  .cat-badge {{ padding: 3px 12px; border-radius: 6px; color: white; font-size: 0.78rem; font-weight: 600; }}

  /* Criteria table */
  table.criteria {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  table.criteria th, table.criteria td {{
    padding: 6px 10px; text-align: left; font-size: 0.82rem;
    border-bottom: 1px solid var(--border); color: var(--text-primary);
  }}
  table.criteria th {{ background: var(--bg-card); font-weight: 600; color: var(--text-secondary); }}

  /* Tests table */
  table.tests {{ width: 100%; border-collapse: collapse; }}
  table.tests th, table.tests td {{
    padding: 6px 10px; text-align: left; font-size: 0.82rem;
    border-bottom: 1px solid var(--border); color: var(--text-primary);
  }}
  table.tests th {{ background: var(--bg-card); font-weight: 600; color: var(--text-secondary); }}

  td.msg {{
    max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.72rem;
    color: var(--text-muted); cursor: pointer;
  }}
  td.msg:hover {{ color: var(--accent-light); }}
  td.msg.expanded {{
    white-space: pre-wrap; word-break: break-word; overflow: visible; max-width: none;
    background: var(--msg-expanded-bg); padding: 10px; border-radius: 4px;
    color: var(--text-secondary);
  }}

  .annotation {{ display: inline-block; }}
  .stars {{ color: #f59e0b; font-size: 1rem; }}
  .no-data {{ color: var(--text-muted); font-style: italic; padding: 24px; text-align: center; }}

  /* Section tabs */
  .section-tabs {{ display: flex; gap: 0; margin-top: 16px; border-bottom: 2px solid var(--border); }}
  .section-tab {{
    padding: 8px 18px; cursor: pointer; font-size: 0.82rem; font-weight: 600;
    color: var(--text-muted); border-bottom: 2px solid transparent; margin-bottom: -2px;
    background: none; border-top: none; border-left: none; border-right: none;
    transition: color 0.15s, border-color 0.15s;
  }}
  .section-tab.active {{ color: var(--accent-light); border-bottom-color: var(--accent); }}
  .section-tab:hover {{ color: var(--text-primary); }}
  .section-panel {{ display: none; padding-top: 14px; }}
  .section-panel.active {{ display: block; }}

  /* Kernel/code tabs */
  .kernel-tabs {{ display: flex; gap: 4px; margin-bottom: 8px; flex-wrap: wrap; }}
  .kernel-tab {{
    padding: 4px 14px; cursor: pointer; font-size: 0.78rem; font-weight: 500;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 6px 6px 0 0; color: var(--text-muted); transition: all 0.15s;
  }}
  .kernel-tab.active {{ background: var(--accent); color: white; border-color: var(--accent); }}
  .kernel-tab:hover:not(.active) {{ color: var(--text-primary); border-color: var(--border-light); }}
  .kernel-panel {{ display: none; }}
  .kernel-panel.active {{ display: block; }}
  .kernel-code {{
    background: var(--code-bg); color: var(--code-color); padding: 18px;
    border: 1px solid var(--border); border-radius: 0 8px 8px 8px;
    overflow-x: auto; font-size: 0.8rem; line-height: 1.6;
    max-height: 500px; overflow-y: auto;
  }}
  .kernel-code pre {{ margin: 0; }}
  .kernel-code code {{ font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace; }}

  .artifact-content {{
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
    padding: 18px; font-size: 0.85rem; line-height: 1.7;
    max-height: 500px; overflow-y: auto; white-space: pre-wrap;
    color: var(--text-secondary);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }}

  /* Markdown rendered content */
  .markdown-content {{
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
    padding: 24px 28px; font-size: 0.85rem; line-height: 1.7;
    max-height: 600px; overflow-y: auto; color: var(--text-secondary);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }}
  .markdown-content h1 {{ font-size: 1.3rem; font-weight: 700; color: var(--text-primary); margin: 20px 0 10px; }}
  .markdown-content h1:first-child {{ margin-top: 0; }}
  .markdown-content h2 {{ font-size: 1.05rem; font-weight: 600; color: var(--text-primary);
    margin: 18px 0 8px; text-transform: none; letter-spacing: normal; }}
  .markdown-content h3 {{ font-size: 0.95rem; font-weight: 600; color: var(--text-primary); margin: 14px 0 6px; }}
  .markdown-content p {{ margin: 8px 0; }}
  .markdown-content ul, .markdown-content ol {{ margin: 8px 0 8px 20px; }}
  .markdown-content li {{ margin: 3px 0; }}
  .markdown-content table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.82rem; }}
  .markdown-content th, .markdown-content td {{
    padding: 6px 10px; text-align: left; border: 1px solid var(--border); color: var(--text-primary);
  }}
  .markdown-content th {{ background: var(--bg-detail); font-weight: 600; color: var(--text-secondary); }}
  .markdown-content code {{
    font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.82em;
    background: var(--bg-detail); padding: 1px 5px; border-radius: 3px; color: var(--accent-light);
  }}
  .markdown-content pre {{ margin: 10px 0; }}
  .markdown-content pre code {{
    display: block; background: var(--code-bg); color: var(--code-color);
    padding: 14px; border-radius: 6px; border: 1px solid var(--border);
    overflow-x: auto; font-size: 0.8rem; line-height: 1.5;
  }}
  .markdown-content hr {{ border: none; border-top: 1px solid var(--border); margin: 16px 0; }}
  .markdown-content strong {{ color: var(--text-primary); }}
  .md-source {{ display: none; }}

  /* Breadcrumb timeline */
  .bc-timeline {{ display: flex; flex-direction: column; gap: 0; }}
  .bc-entry {{
    display: grid; grid-template-columns: 70px auto 1fr; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid var(--border); align-items: start;
  }}
  .bc-entry:last-child {{ border-bottom: none; }}
  .bc-time {{
    font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.78rem;
    color: var(--text-muted); white-space: nowrap; padding-top: 2px;
  }}
  .bc-event {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    color: white; font-size: 0.75rem; font-weight: 600; white-space: nowrap;
  }}
  .bc-fields {{ display: flex; flex-wrap: wrap; gap: 2px 16px; font-size: 0.8rem; }}
  .bc-field {{ display: flex; gap: 4px; align-items: baseline; }}
  .bc-key {{ color: var(--text-muted); font-size: 0.75rem; white-space: nowrap; }}
  .bc-val {{ color: var(--text-primary); word-break: break-word; }}
  .bc-val-long {{
    color: var(--text-primary); word-break: break-word;
    flex-basis: 100%; padding-left: 0;
  }}
  .bc-list {{ color: var(--accent-light); }}

  /* Scrollbar styling */
  ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg-page); border-radius: 4px; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--border-light); }}
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/{hljs_theme}">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/cpp.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.0/marked.min.js"></script>
</head>
<body>
<div class="dashboard-header">
  <h1><span class="accent">&#9632;</span> Eval Dashboard</h1>
  <div class="subtitle">TTNN Operation Generation Pipeline &mdash; Test Results &amp; Scoring</div>
</div>
<div class="content">
"""


def _html_stats(stats: dict) -> str:
    return f"""\
<div class="stats">
  <div class="stat-card">
    <div class="stat-value">{stats['total_runs']}</div>
    <div class="stat-label">Total Runs</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{stats['avg_score']}</div>
    <div class="stat-label">Avg Score</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{stats['pass_rate']}%</div>
    <div class="stat-label">Golden Pass Rate</div>
  </div>
</div>
"""


def _html_failure_bars(stats: dict) -> str:
    summary = stats.get("failure_summary", {})
    if not summary:
        return ""

    total = sum(summary.values())
    if total == 0:
        return ""

    segments = []
    for cat in ["numerical", "OOM", "hang", "compilation", "other"]:
        count = summary.get(cat, 0)
        if count == 0:
            continue
        pct = count / total * 100
        color = CATEGORY_COLORS.get(cat, "#6B6488")
        segments.append(f'  <div class="bar-segment" style="width:{pct:.1f}%;background:{color}">{cat}: {count}</div>')

    return f"""\
<h2>Failure Breakdown</h2>
<div class="bar-container">
{chr(10).join(segments)}
</div>
"""


def _html_runs_table(runs: list, run_details: dict) -> str:
    if not runs:
        return '<div class="no-data">No runs recorded yet.</div>'

    header = """\
<h2>Runs</h2>
<table class="runs">
<thead>
<tr>
  <th>ID</th><th>Date</th><th>Prompt</th><th>Base Branch</th><th>Run Branch</th>
  <th>Score</th><th>Golden</th><th>Duration</th><th>Rating</th>
</tr>
</thead>
<tbody>
"""

    rows = []
    for run in runs:
        rid = run["id"]
        ts = run["timestamp"][:16].replace("T", " ") if run["timestamp"] else ""
        prompt = html.escape(run["prompt_name"])
        golden_name = html.escape(run["golden_name"]) if run.get("golden_name") else None
        base_branch = html.escape(run["starting_branch"])
        commit_short = html.escape(run["starting_commit"][:7]) if run["starting_commit"] else ""
        created_branch = html.escape(run["created_branch"])

        # Score + grade
        if run["score_total"] is not None and run["score_grade"]:
            score_html = (
                f'<span style="{_grade_bg(run["score_grade"])}">{run["score_total"]:.1f} ({run["score_grade"]})</span>'
            )
        else:
            score_html = '<span class="muted">--</span>'

        # Golden
        if run["golden_total"] is not None and run["golden_total"] > 0:
            gp = run["golden_passed"] or 0
            gt = run["golden_total"]
            pct = gp / gt * 100
            if pct == 100:
                g_color = "#22c55e"
            elif pct >= 70:
                g_color = "#eab308"
            else:
                g_color = "#ef4444"
            golden_html = f'<span style="color:{g_color};font-weight:600">{gp}/{gt}</span>'
        else:
            golden_html = '<span class="muted">--</span>'

        # Duration
        dur = run.get("duration_seconds")
        if dur is not None:
            dur_min = dur // 60
            dur_sec = dur % 60
            duration_html = f"{dur_min}m {dur_sec:02d}s"
        else:
            duration_html = '<span class="muted">--</span>'

        # Annotation
        if run["annotation_score"]:
            stars = "&#9733;" * run["annotation_score"] + "&#9734;" * (5 - run["annotation_score"])
            ann_html = f'<span class="stars">{stars}</span>'
        else:
            ann_html = '<span class="muted">--</span>'

        golden_badge = (
            f' <span style="background:var(--accent);color:white;padding:1px 6px;border-radius:3px;'
            f'font-size:0.75em;font-weight:600;margin-left:6px">golden: {golden_name}</span>'
            if golden_name
            else ""
        )

        rows.append(
            f'<tr class="run-row" onclick="toggleDetail({rid})">'
            f"  <td>{rid}</td><td>{ts}</td><td>{prompt}{golden_badge}</td>"
            f'  <td>{base_branch} <span style="color:var(--text-muted);font-size:0.8em">({commit_short})</span></td>'
            f"  <td>{created_branch}</td>"
            f"  <td>{score_html}</td><td>{golden_html}</td><td>{duration_html}</td><td>{ann_html}</td>"
            f"</tr>"
        )

        # Detail row
        detail = _html_run_detail(rid, run_details.get(rid, {}))
        rows.append(
            f'<tr class="detail-row" id="detail-{rid}">' f'  <td class="detail-cell" colspan="9">{detail}</td>' f"</tr>"
        )

    return header + "\n".join(rows) + "\n</tbody>\n</table>"


def _html_code_tabs(parts: list, group_id: str, files: list, lang: str):
    """Render a tabbed code viewer for a list of source files."""
    parts.append('<div class="kernel-tabs">')
    for i, f in enumerate(files):
        active = " active" if i == 0 else ""
        fname = html.escape(f["filename"])
        parts.append(f'  <button class="kernel-tab{active}" onclick="showKernel(\'{group_id}\',{i})">{fname}</button>')
    parts.append("</div>")
    for i, f in enumerate(files):
        active = " active" if i == 0 else ""
        escaped_code = html.escape(f["source_code"])
        parts.append(f'<div class="kernel-panel{active}" id="{group_id}-{i}">')
        parts.append(f'  <div class="kernel-code"><pre><code class="language-{lang}">{escaped_code}</code></pre></div>')
        parts.append("</div>")


TDD_STATUS_COLORS = {
    "passed": "#22c55e",
    "in_progress": "#3b82f6",
    "pending": "#6B6488",
    "failed_permanent": "#ef4444",
}


def _html_tdd_state(raw_json: str) -> str:
    """Render the .tdd_state.json as a readable stage table."""
    try:
        state = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return f'<div class="artifact-content">{html.escape(raw_json)}</div>'

    parts = []

    # Summary line
    op_name = html.escape(state.get("op_name", "unknown"))
    layout = html.escape(str(state.get("layout", "--")))
    cur_idx = state.get("current_stage_index", 0)
    stages = state.get("stages", [])
    total = len(stages)
    passed = sum(1 for s in stages if s.get("status") == "passed")

    parts.append(
        f'<div style="margin-bottom:12px;font-size:0.9rem;color:var(--text-primary)">'
        f"<strong>{op_name}</strong> &mdash; layout: <code>{layout}</code> &mdash; "
        f"{passed}/{total} stages passed"
        f"</div>"
    )

    if not stages:
        parts.append('<div class="no-data">No TDD stages registered.</div>')
        return "\n".join(parts)

    # Stage table
    parts.append('<table class="tests">')
    parts.append(
        "<thead><tr><th>#</th><th>Stage</th><th>Status</th>"
        "<th>Attempts</th><th>Free Retries</th><th>Failures</th></tr></thead>"
    )
    parts.append("<tbody>")
    for i, s in enumerate(stages):
        name = html.escape(s.get("name", f"stage_{i}"))
        status = s.get("status", "pending")
        status_color = TDD_STATUS_COLORS.get(status, "#6B6488")
        status_html = (
            f'<span style="background:{status_color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.8em;font-weight:600">{status.upper()}</span>'
        )
        attempts = s.get("attempts", 0)
        max_att = s.get("max_attempts", 6)
        free_ret = s.get("free_retries", 0)

        # Failure summary
        failures = s.get("failure_history", [])
        if failures:
            cats = {}
            for f in failures:
                c = f.get("classification", "unknown")
                cats[c] = cats.get(c, 0) + 1
            fail_parts = [f"{c}({n})" for c, n in sorted(cats.items(), key=lambda x: -x[1])]
            fail_html = html.escape(", ".join(fail_parts))
        else:
            fail_html = '<span style="color:var(--text-muted)">--</span>'

        parts.append(
            f"<tr><td>{i}</td><td><strong>{name}</strong></td><td>{status_html}</td>"
            f"<td>{attempts}/{max_att}</td><td>{free_ret}</td><td>{fail_html}</td></tr>"
        )
    parts.append("</tbody></table>")

    return "\n".join(parts)


def _html_breadcrumbs(parts: list, group_id: str, breadcrumbs: list):
    """Render breadcrumb JSONL files as tabbed, formatted event logs."""
    if len(breadcrumbs) > 1:
        # Multi-agent: use sub-tabs
        parts.append('<div class="kernel-tabs">')
        for i, b in enumerate(breadcrumbs):
            active = " active" if i == 0 else ""
            agent = html.escape(b["agent_name"])
            parts.append(
                f'  <button class="kernel-tab{active}" ' f"onclick=\"showKernel('{group_id}',{i})\">{agent}</button>"
            )
        parts.append("</div>")
        for i, b in enumerate(breadcrumbs):
            active = " active" if i == 0 else ""
            parts.append(f'<div class="kernel-panel{active}" id="{group_id}-{i}">')
            parts.append(_render_breadcrumb_events(b["content"]))
            parts.append("</div>")
    elif len(breadcrumbs) == 1:
        parts.append(_render_breadcrumb_events(breadcrumbs[0]["content"]))


EVENT_COLORS = {
    "start": "#3b82f6",
    "complete": "#22c55e",
    "stage_complete": "#22c55e",
    "stage_start": "#2563eb",
    "stage_pass": "#22c55e",
    "test_run": "#8b5cf6",
    "kernel_implemented": "#7c3aed",
    "design_parsed": "#6366f1",
    "upstream_fix": "#f97316",
    "fix_applied": "#eab308",
    "hypothesis": "#ec4899",
    "cb_sync_check": "#06b6d4",
    "commit": "#eab308",
    "action": "#6B6488",
    "result": "#6B6488",
    "design_decision": "#f97316",
    "reference_read": "#6B6488",
}

# Fields to skip in breadcrumb detail rendering (shown separately in header)
_BC_SKIP_FIELDS = {"ts", "event"}

# Fields whose values are typically long and should get their own line
_BC_LONG_FIELDS = {
    "approach",
    "reason",
    "change",
    "description",
    "evidence",
    "details",
    "error",
    "cb_summary",
    "rationale",
}


def _render_bc_value(key: str, val) -> str:
    """Render a single breadcrumb field value as HTML."""
    if isinstance(val, list):
        items = [html.escape(str(v)) for v in val]
        return f'<span class="bc-list">{", ".join(items)}</span>'
    if isinstance(val, bool):
        color = "#22c55e" if val else "#ef4444"
        label = "yes" if val else "no"
        return f'<span style="color:{color};font-weight:600">{label}</span>'
    return html.escape(str(val))


def _render_breadcrumb_events(content: str) -> str:
    """Parse JSONL breadcrumbs into a timeline with full field display."""
    events = []
    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not events:
        return '<div class="no-data">No breadcrumb events.</div>'

    parts = ['<div class="bc-timeline">']
    for ev in events:
        ts_raw = ev.get("ts", "--")
        ts = html.escape(ts_raw[-8:] if len(ts_raw) >= 8 else ts_raw)
        event_type = ev.get("event", "?")
        color = EVENT_COLORS.get(event_type, "#6B6488")

        parts.append('<div class="bc-entry">')
        parts.append(f'  <span class="bc-time">{ts}</span>')
        parts.append(f'  <span class="bc-event" style="background:{color}">' f"{html.escape(event_type)}</span>")

        # Render all remaining fields
        fields = {k: v for k, v in ev.items() if k not in _BC_SKIP_FIELDS and v is not None}
        if fields:
            parts.append('  <div class="bc-fields">')
            # Short fields first, then long fields
            short_fields = {k: v for k, v in fields.items() if k not in _BC_LONG_FIELDS}
            long_fields = {k: v for k, v in fields.items() if k in _BC_LONG_FIELDS}

            for k, v in short_fields.items():
                rendered = _render_bc_value(k, v)
                parts.append(
                    f'    <div class="bc-field">'
                    f'<span class="bc-key">{html.escape(k)}:</span> '
                    f'<span class="bc-val">{rendered}</span></div>'
                )
            for k, v in long_fields.items():
                rendered = _render_bc_value(k, v)
                parts.append(
                    f'    <div class="bc-field" style="flex-basis:100%">'
                    f'<span class="bc-key">{html.escape(k)}:</span> '
                    f'<span class="bc-val-long">{rendered}</span></div>'
                )
            parts.append("  </div>")
        else:
            parts.append('  <div class="bc-fields">--</div>')

        parts.append("</div>")
    parts.append("</div>")

    return "\n".join(parts)


def _html_run_detail(rid: int, details: dict) -> str:
    tests = details.get("tests", [])
    criteria = details.get("criteria", [])
    kernels = details.get("kernels", [])
    host_code = details.get("host_code", [])
    artifacts = details.get("artifacts", [])
    tdd_state_raw = details.get("tdd_state")
    kw_breadcrumbs = details.get("kw_breadcrumbs", [])

    parts = ['<div class="detail-content">']

    # Category summary badges
    if tests:
        categories = {}
        for t in tests:
            cat = t.get("failure_category")
            if cat:
                categories[cat] = categories.get(cat, 0) + 1
        if categories:
            parts.append('<div class="cat-summary">')
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                color = CATEGORY_COLORS.get(cat, "#6B6488")
                parts.append(f'  <span class="cat-badge" style="background:{color}">{cat}: {count}</span>')
            parts.append("</div>")

    # Section tabs
    has_kernels = len(kernels) > 0
    has_host_code = len(host_code) > 0
    has_artifacts = len(artifacts) > 0
    has_tdd_state = tdd_state_raw is not None
    has_breadcrumbs = len(kw_breadcrumbs) > 0
    tab_id = f"run{rid}"

    parts.append('<div class="section-tabs">')
    parts.append(f"  <button class=\"section-tab active\" onclick=\"showSection('{tab_id}','tests')\">Tests</button>")
    if criteria:
        parts.append(f"  <button class=\"section-tab\" onclick=\"showSection('{tab_id}','criteria')\">Score</button>")
    if has_kernels:
        parts.append(
            f"  <button class=\"section-tab\" onclick=\"showSection('{tab_id}','kernels')\">"
            f"Kernels ({len(kernels)})</button>"
        )
    if has_host_code:
        parts.append(
            f"  <button class=\"section-tab\" onclick=\"showSection('{tab_id}','host')\">"
            f"Host-Side ({len(host_code)})</button>"
        )
    if has_tdd_state:
        parts.append(
            f"  <button class=\"section-tab\" onclick=\"showSection('{tab_id}','tddstate')\">TDD State</button>"
        )
    if has_breadcrumbs:
        parts.append(
            f"  <button class=\"section-tab\" onclick=\"showSection('{tab_id}','breadcrumbs')\">"
            f"KW Breadcrumbs ({len(kw_breadcrumbs)})</button>"
        )
    if has_artifacts:
        parts.append(
            f"  <button class=\"section-tab\" onclick=\"showSection('{tab_id}','artifacts')\">Self-Reflection</button>"
        )
    parts.append("</div>")

    # --- Tests panel ---
    parts.append(f'<div class="section-panel active" id="{tab_id}-tests">')
    if tests:
        parts.append('<table class="tests">')
        parts.append(
            "<thead><tr><th>Test</th><th>Shape</th><th>Status</th>" "<th>Category</th><th>Message</th></tr></thead>"
        )
        parts.append("<tbody>")
        for t in tests:
            tname = html.escape(t["test_name"])
            shape = html.escape(t["shape"] or "--")
            status = t["status"]
            status_html = f'<span style="{_status_bg(status)}">{status.upper()}</span>'
            cat = t.get("failure_category")
            cat_html = f'<span style="{_category_bg(cat)}">{cat}</span>' if cat else "--"
            full_msg = html.escape(t.get("failure_message") or "")
            short_msg = full_msg[:120]
            parts.append(
                f"<tr><td>{tname}</td><td>{shape}</td><td>{status_html}</td>"
                f'<td>{cat_html}</td><td class="msg" onclick="toggleMsg(this)" data-full="{full_msg}" data-short="{short_msg}">{short_msg}</td></tr>'
            )
        parts.append("</tbody></table>")
    else:
        parts.append('<div class="no-data">No test results.</div>')
    parts.append("</div>")

    # --- Score criteria panel ---
    if criteria:
        parts.append(f'<div class="section-panel" id="{tab_id}-criteria">')
        parts.append('<table class="criteria">')
        parts.append("<thead><tr><th>Criterion</th><th>Raw</th><th>Weight</th><th>Weighted</th></tr></thead>")
        parts.append("<tbody>")
        for c in criteria:
            name = html.escape(c["criterion"].replace("_", " ").title())
            parts.append(
                f"<tr><td>{name}</td><td>{c['raw_score']:.1f}</td>"
                f"<td>x{c['weight']:.2f}</td><td>{c['weighted_score']:.1f}</td></tr>"
            )
        parts.append("</tbody></table>")
        parts.append("</div>")

    # --- Kernels panel (C++) ---
    if has_kernels:
        parts.append(f'<div class="section-panel" id="{tab_id}-kernels">')
        _html_code_tabs(parts, f"k{rid}", kernels, "cpp")
        parts.append("</div>")

    # --- Host-side panel (Python) ---
    if has_host_code:
        parts.append(f'<div class="section-panel" id="{tab_id}-host">')
        _html_code_tabs(parts, f"h{rid}", host_code, "python")
        parts.append("</div>")

    # --- TDD State panel ---
    if has_tdd_state:
        parts.append(f'<div class="section-panel" id="{tab_id}-tddstate">')
        parts.append(_html_tdd_state(tdd_state_raw))
        parts.append("</div>")

    # --- KW Breadcrumbs panel ---
    if has_breadcrumbs:
        parts.append(f'<div class="section-panel" id="{tab_id}-breadcrumbs">')
        _html_breadcrumbs(parts, f"bc{rid}", kw_breadcrumbs)
        parts.append("</div>")

    # --- Artifacts panel (self-reflection) ---
    if has_artifacts:
        parts.append(f'<div class="section-panel" id="{tab_id}-artifacts">')
        for a in artifacts:
            escaped = html.escape(a["content"])
            parts.append(f'<div class="markdown-content">')
            parts.append(f'<pre class="md-source">{escaped}</pre>')
            parts.append(f"</div>")
        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


def _html_foot() -> str:
    return """\
</div>
<script>
function renderMarkdown(container) {
  // Render markdown in .markdown-content elements that haven't been processed
  container.querySelectorAll('.markdown-content').forEach(function(el) {
    if (el.dataset.rendered) return;
    var source = el.querySelector('.md-source');
    if (source && typeof marked !== 'undefined') {
      el.innerHTML = marked.parse(source.textContent);
      el.dataset.rendered = 'true';
    }
  });
}
function toggleDetail(id) {
  var row = document.getElementById('detail-' + id);
  if (row) {
    row.classList.toggle('open');
    // Highlight code blocks and render markdown on first open
    if (row.classList.contains('open')) {
      renderMarkdown(row);
      row.querySelectorAll('pre code').forEach(function(block) {
        if (!block.dataset.highlighted) {
          hljs.highlightElement(block);
          block.dataset.highlighted = 'true';
        }
      });
    }
  }
}
function showSection(runId, section) {
  // Toggle section tabs
  var container = document.getElementById(runId + '-' + section);
  if (!container) return;
  var parent = container.parentElement;
  parent.querySelectorAll('.section-panel').forEach(function(p) { p.classList.remove('active'); });
  parent.querySelectorAll('.section-tab').forEach(function(t) { t.classList.remove('active'); });
  container.classList.add('active');
  // Find and activate the clicked tab
  event.target.classList.add('active');
  // Highlight code blocks and render markdown in newly shown section
  renderMarkdown(container);
  container.querySelectorAll('pre code').forEach(function(block) {
    if (!block.dataset.highlighted) {
      hljs.highlightElement(block);
      block.dataset.highlighted = 'true';
    }
  });
}
function toggleMsg(td) {
  if (td.classList.contains('expanded')) {
    td.classList.remove('expanded');
    td.textContent = td.dataset.short;
  } else {
    td.classList.add('expanded');
    td.textContent = td.dataset.full;
  }
}
function showKernel(kernelGroupId, index) {
  var parent = document.getElementById(kernelGroupId + '-' + index).parentElement;
  parent.querySelectorAll('.kernel-panel').forEach(function(p) { p.classList.remove('active'); });
  parent.querySelectorAll('.kernel-tab').forEach(function(t) { t.classList.remove('active'); });
  document.getElementById(kernelGroupId + '-' + index).classList.add('active');
  event.target.classList.add('active');
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def serve(db_path: Path, port: int = 8080, theme: str = "dark"):
    """Generate the dashboard and serve it via HTTP."""
    conn = db.connect(db_path)
    state = {"html": generate_html(conn, theme)}
    conn.close()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(state["html"].encode())
            elif self.path == "/refresh":
                c = db.connect(db_path)
                state["html"] = generate_html(c, theme)
                c.close()
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # quiet

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Dashboard: http://localhost:{port}  (refresh: http://localhost:{port}/refresh)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Eval dashboard")
    parser.add_argument("--db", default=str(db.DEFAULT_SQLITE_PATH), help="Database path")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--generate-only", action="store_true", help="Generate HTML file without serving")
    parser.add_argument("--output", "-o", default="dashboard.html", help="Output HTML path (with --generate-only)")
    parser.add_argument("--theme", choices=THEMES, default="dark", help="Color theme (default: dark)")
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.generate_only:
        conn = db.connect(db_path)
        html_content = generate_html(conn, args.theme)
        conn.close()
        Path(args.output).write_text(html_content)
        print(f"Dashboard written to {args.output}")
    else:
        serve(db_path, args.port, args.theme)


if __name__ == "__main__":
    main()

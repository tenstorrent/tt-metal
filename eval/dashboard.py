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
    "A": "#16a34a",  # green
    "B": "#65a30d",  # lime
    "C": "#ca8a04",  # yellow
    "D": "#ea580c",  # orange
    "F": "#dc2626",  # red
}

STATUS_COLORS = {
    "passed": "#16a34a",
    "failed": "#dc2626",
    "error": "#991b1b",
    "skipped": "#9ca3af",
}

CATEGORY_COLORS = {
    "hang": "#991b1b",
    "OOM": "#7c3aed",
    "numerical": "#ea580c",
    "compilation": "#2563eb",
    "other": "#6b7280",
}


def _grade_bg(grade):
    color = GRADE_COLORS.get(grade, "#6b7280")
    return f"background:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold"


def _status_bg(status):
    color = STATUS_COLORS.get(status, "#6b7280")
    return f"background:{color};color:white;padding:1px 6px;border-radius:3px;font-size:0.85em"


def _cat_bg(cat):
    if not cat:
        return ""
    color = CATEGORY_COLORS.get(cat, "#6b7280")
    return f"background:{color};color:white;padding:1px 6px;border-radius:3px;font-size:0.85em"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def generate_html(conn) -> str:
    """Generate the full dashboard HTML from a DB connection."""
    stats = db.get_stats(conn)
    runs = db.get_all_runs(conn)

    # Pre-fetch details for all runs
    run_details = {}
    for run in runs:
        run_details[run["id"]] = {
            "tests": db.get_test_results(conn, run["id"]),
            "criteria": db.get_score_criteria(conn, run["id"]),
        }

    parts = [
        _html_head(),
        _html_stats(stats),
        _html_failure_bars(stats),
        _html_runs_table(runs, run_details),
        _html_foot(),
    ]
    return "\n".join(parts)


def _html_head() -> str:
    return """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f1f5f9; color: #1e293b; padding: 24px; }
  h1 { font-size: 1.5rem; margin-bottom: 20px; }
  h2 { font-size: 1.1rem; margin: 20px 0 10px; color: #475569; }

  .stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
  .stat-card { background: white; border-radius: 8px; padding: 16px 24px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 140px; }
  .stat-value { font-size: 1.8rem; font-weight: 700; }
  .stat-label { font-size: 0.85rem; color: #64748b; margin-top: 4px; }

  .bar-container { display: flex; gap: 4px; height: 28px; margin-bottom: 20px;
                   border-radius: 6px; overflow: hidden; background: #e2e8f0; }
  .bar-segment { display: flex; align-items: center; justify-content: center;
                 color: white; font-size: 0.75rem; font-weight: 600;
                 white-space: nowrap; padding: 0 8px; min-width: 60px; }

  table.runs { width: 100%; border-collapse: collapse; background: white;
               border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  table.runs th { background: #1e293b; color: white; padding: 10px 12px;
                  text-align: left; font-size: 0.85rem; font-weight: 600; }
  table.runs td { padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-size: 0.9rem; }
  tr.run-row { cursor: pointer; }
  tr.run-row:hover { background: #f8fafc; }
  tr.detail-row { display: none; }
  tr.detail-row.open { display: table-row; }
  td.detail-cell { padding: 16px; background: #f8fafc; }

  .detail-content { max-height: 500px; overflow-y: auto; }
  .cat-summary { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
  .cat-badge { padding: 2px 10px; border-radius: 4px; color: white; font-size: 0.8rem; font-weight: 600; }

  table.criteria { width: 100%; border-collapse: collapse; margin-bottom: 16px; }
  table.criteria th, table.criteria td { padding: 4px 8px; text-align: left; font-size: 0.8rem;
                                         border-bottom: 1px solid #e2e8f0; }
  table.criteria th { background: #f1f5f9; font-weight: 600; }

  table.tests { width: 100%; border-collapse: collapse; }
  table.tests th, table.tests td { padding: 4px 8px; text-align: left; font-size: 0.8rem;
                                    border-bottom: 1px solid #e2e8f0; }
  table.tests th { background: #f1f5f9; font-weight: 600; }
  td.msg { max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
           font-family: monospace; font-size: 0.75rem; color: #64748b; }

  .annotation { display: inline-block; }
  .stars { color: #f59e0b; }
  .no-data { color: #9ca3af; font-style: italic; padding: 20px; text-align: center; }
</style>
</head>
<body>
<h1>Eval Dashboard</h1>
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
        color = CATEGORY_COLORS.get(cat, "#6b7280")
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
  <th>ID</th><th>Date</th><th>Prompt</th><th>Branch</th>
  <th>Score</th><th>Golden</th><th>Rating</th>
</tr>
</thead>
<tbody>
"""

    rows = []
    for run in runs:
        rid = run["id"]
        ts = run["timestamp"][:16].replace("T", " ") if run["timestamp"] else ""
        prompt = html.escape(run["prompt_name"])
        branch = html.escape(run["starting_branch"])

        # Score + grade
        if run["score_total"] is not None and run["score_grade"]:
            score_html = (
                f'<span style="{_grade_bg(run["score_grade"])}">{run["score_total"]:.1f} ({run["score_grade"]})</span>'
            )
        else:
            score_html = '<span style="color:#9ca3af">--</span>'

        # Golden
        if run["golden_total"] is not None and run["golden_total"] > 0:
            gp = run["golden_passed"] or 0
            gt = run["golden_total"]
            pct = gp / gt * 100
            if pct == 100:
                g_color = "#16a34a"
            elif pct >= 70:
                g_color = "#ca8a04"
            else:
                g_color = "#dc2626"
            golden_html = f'<span style="color:{g_color};font-weight:600">{gp}/{gt}</span>'
        else:
            golden_html = '<span style="color:#9ca3af">--</span>'

        # Annotation
        if run["annotation_score"]:
            stars = "&#9733;" * run["annotation_score"] + "&#9734;" * (5 - run["annotation_score"])
            ann_html = f'<span class="stars">{stars}</span>'
        else:
            ann_html = '<span style="color:#9ca3af">--</span>'

        rows.append(
            f'<tr class="run-row" onclick="toggleDetail({rid})">'
            f"  <td>{rid}</td><td>{ts}</td><td>{prompt}</td><td>{branch}</td>"
            f"  <td>{score_html}</td><td>{golden_html}</td><td>{ann_html}</td>"
            f"</tr>"
        )

        # Detail row
        detail = _html_run_detail(rid, run_details.get(rid, {}))
        rows.append(
            f'<tr class="detail-row" id="detail-{rid}">' f'  <td class="detail-cell" colspan="7">{detail}</td>' f"</tr>"
        )

    return header + "\n".join(rows) + "\n</tbody>\n</table>"


def _html_run_detail(rid: int, details: dict) -> str:
    tests = details.get("tests", [])
    criteria = details.get("criteria", [])

    parts = ['<div class="detail-content">']

    # Category summary badges
    if tests:
        cats = {}
        for t in tests:
            cat = t.get("failure_category")
            if cat:
                cats[cat] = cats.get(cat, 0) + 1
        if cats:
            parts.append('<div class="cat-summary">')
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                color = CATEGORY_COLORS.get(cat, "#6b7280")
                parts.append(f'  <span class="cat-badge" style="background:{color}">{cat}: {count}</span>')
            parts.append("</div>")

    # Score criteria table
    if criteria:
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

    # Test results table
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
            cat_html = f'<span style="{_cat_bg(cat)}">{cat}</span>' if cat else "--"
            msg = html.escape(t.get("failure_message") or "")[:120]
            parts.append(
                f"<tr><td>{tname}</td><td>{shape}</td><td>{status_html}</td>"
                f'<td>{cat_html}</td><td class="msg" title="{html.escape(t.get("failure_message") or "")}">{msg}</td></tr>'
            )
        parts.append("</tbody></table>")
    else:
        parts.append('<div class="no-data">No test results.</div>')

    parts.append("</div>")
    return "\n".join(parts)


def _html_foot() -> str:
    return """\
<script>
function toggleDetail(id) {
  var row = document.getElementById('detail-' + id);
  if (row) row.classList.toggle('open');
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def serve(db_path: Path, port: int = 8080):
    """Generate the dashboard and serve it via HTTP."""
    conn = db.connect(db_path)
    state = {"html": generate_html(conn)}
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
                state["html"] = generate_html(c)
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
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--generate-only", action="store_true", help="Generate HTML file without serving")
    parser.add_argument("--output", "-o", default="dashboard.html", help="Output HTML path (with --generate-only)")
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.generate_only:
        conn = db.connect(db_path)
        html_content = generate_html(conn)
        conn.close()
        Path(args.output).write_text(html_content)
        print(f"Dashboard written to {args.output}")
    else:
        serve(db_path, args.port)


if __name__ == "__main__":
    main()

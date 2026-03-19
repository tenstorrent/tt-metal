"""Flask server for per-run detail views.

Serves the deep-dive view (tests, score, kernels, host code, TDD state,
breadcrumbs) that Superset can't replicate. Superset's runs table links
here via the "Detail" column.

Usage:
    python3 -m eval.detail_server [--db /path/to/eval_runs.db] [--port 8090]

    With PostgreSQL:
    EVAL_DATABASE_URL="postgresql://user:pass@host/eval" python3 -m eval.detail_server
"""

import argparse
import html
import json
from pathlib import Path

from flask import Flask, abort

from eval import db
from eval.dashboard import (
    _html_run_detail,
    CATEGORY_COLORS,
    STATUS_COLORS,
    TDD_STATUS_COLORS,
)

app = Flask(__name__)
_db_path = None  # set in main() for SQLite fallback


def _get_conn():
    """Get a DB connection (Postgres if EVAL_DATABASE_URL set, else SQLite)."""
    return db.connect(_db_path)


@app.route("/")
def index():
    """List all runs with links to detail pages."""
    conn = _get_conn()
    runs = db.get_all_runs(conn)
    stats = db.get_stats(conn)
    conn.close()

    rows = []
    for run in runs:
        rid = run["id"]
        ts = run["timestamp"][:16].replace("T", " ") if run["timestamp"] else ""
        prompt = html.escape(run["prompt_name"])
        status = run.get("status", "complete")
        phase = html.escape(run.get("phase") or "--")

        # Score
        if run["score_total"] is not None and run["score_grade"]:
            score_html = f'{run["score_total"]:.1f} ({run["score_grade"]})'
        else:
            score_html = "--"

        # Golden
        if run["golden_total"] is not None and run["golden_total"] > 0:
            gp = run["golden_passed"] or 0
            golden_html = f"{gp}/{run['golden_total']}"
        else:
            golden_html = "--"

        # Status badge
        status_colors = {
            "queued": "#9ca3af",
            "cloning": "#6b7280",
            "building": "#2563eb",
            "running": "#7c3aed",
            "testing": "#ca8a04",
            "scoring": "#ea580c",
            "complete": "#16a34a",
            "failed": "#dc2626",
        }
        s_color = status_colors.get(status, "#6b7280")
        status_badge = (
            f'<span style="background:{s_color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.85em">{html.escape(status.upper())}</span>'
        )

        rows.append(
            f"<tr>"
            f'<td><a href="/run/{rid}" style="color:#2563eb;font-weight:600">{rid}</a></td>'
            f"<td>{ts}</td><td>{prompt}</td><td>{status_badge}</td>"
            f"<td>{phase}</td><td>{score_html}</td><td>{golden_html}</td>"
            f"</tr>"
        )

    active = stats.get("active_runs", 0)
    active_badge = (
        f' <span style="background:#7c3aed;color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em">{active} active</span>'
        if active > 0
        else ""
    )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval Detail Server</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f1f5f9; color: #1e293b; padding: 24px; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 20px; }}
  .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 16px 24px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 140px; }}
  .stat-value {{ font-size: 1.8rem; font-weight: 700; }}
  .stat-label {{ font-size: 0.85rem; color: #64748b; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #1e293b; color: white; padding: 10px 12px;
       text-align: left; font-size: 0.85rem; font-weight: 600; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-size: 0.9rem; }}
  tr:hover {{ background: #f8fafc; }}
  a {{ text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .refresh {{ color: #64748b; font-size: 0.85rem; margin-left: 12px; }}
</style>
<meta http-equiv="refresh" content="30">
</head>
<body>
<h1>Eval Runs{active_badge} <a href="/" class="refresh">auto-refresh 30s</a></h1>
<div class="stats">
  <div class="stat-card"><div class="stat-value">{stats['total_runs']}</div><div class="stat-label">Total Runs</div></div>
  <div class="stat-card"><div class="stat-value">{stats['avg_score']}</div><div class="stat-label">Avg Score</div></div>
  <div class="stat-card"><div class="stat-value">{stats['pass_rate']}%</div><div class="stat-label">Golden Pass Rate</div></div>
</div>
<table>
<thead><tr><th>ID</th><th>Date</th><th>Prompt</th><th>Status</th><th>Phase</th><th>Score</th><th>Golden</th></tr></thead>
<tbody>
{"".join(rows) if rows else '<tr><td colspan="7" style="text-align:center;color:#9ca3af;padding:20px">No runs yet.</td></tr>'}
</tbody>
</table>
</body>
</html>"""


@app.route("/run/<int:run_id>")
def run_detail(run_id):
    """Render the full detail view for a single run."""
    conn = _get_conn()
    run = db.get_run(conn, run_id)
    if not run:
        conn.close()
        abort(404)

    details = {
        "tests": db.get_test_results(conn, run_id),
        "criteria": db.get_score_criteria(conn, run_id),
        "kernels": db.get_kernels(conn, run_id),
        "host_code": db.get_host_code(conn, run_id),
        "artifacts": db.get_artifacts(conn, run_id),
        "tdd_state": db.get_tdd_state(conn, run_id),
        "kw_breadcrumbs": db.get_kw_breadcrumbs(conn, run_id),
    }
    conn.close()

    detail_html = _html_run_detail(run_id, details)

    # Run header
    ts = run["timestamp"][:16].replace("T", " ") if run["timestamp"] else ""
    prompt = html.escape(run["prompt_name"])
    status = run.get("status", "complete")
    phase = html.escape(run.get("phase") or "--")
    branch = html.escape(run["created_branch"])
    score = f'{run["score_total"]:.1f} ({run["score_grade"]})' if run["score_total"] else "--"

    status_colors = {
        "queued": "#9ca3af",
        "cloning": "#6b7280",
        "building": "#2563eb",
        "running": "#7c3aed",
        "testing": "#ca8a04",
        "scoring": "#ea580c",
        "complete": "#16a34a",
        "failed": "#dc2626",
    }
    s_color = status_colors.get(status, "#6b7280")
    status_badge = (
        f'<span style="background:{s_color};color:white;padding:4px 12px;'
        f'border-radius:4px;font-weight:600">{html.escape(status.upper())}</span>'
    )

    # Auto-refresh if run is still active
    refresh_meta = ""
    if status not in ("complete", "failed"):
        refresh_meta = '<meta http-equiv="refresh" content="10">'

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh_meta}
<title>Run #{run_id} — {prompt}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f1f5f9; color: #1e293b; padding: 24px; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 16px; }}
  .meta {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 20px;
           font-size: 0.9rem; color: #475569; }}
  .meta span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .meta strong {{ color: #1e293b; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  .detail-content {{ max-height: none; overflow-y: visible; }}
  .cat-summary {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }}
  .cat-badge {{ padding: 2px 10px; border-radius: 4px; color: white; font-size: 0.8rem; font-weight: 600; }}

  .section-tabs {{ display: flex; gap: 0; margin-top: 16px; border-bottom: 2px solid #e2e8f0; }}
  .section-tab {{ padding: 6px 16px; cursor: pointer; font-size: 0.85rem; font-weight: 600;
                 color: #64748b; border-bottom: 2px solid transparent; margin-bottom: -2px;
                 background: none; border-top: none; border-left: none; border-right: none; }}
  .section-tab.active {{ color: #1e293b; border-bottom-color: #2563eb; }}
  .section-tab:hover {{ color: #1e293b; }}
  .section-panel {{ display: none; padding-top: 12px; }}
  .section-panel.active {{ display: block; }}

  table.criteria {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  table.criteria th, table.criteria td {{ padding: 4px 8px; text-align: left; font-size: 0.8rem;
                                         border-bottom: 1px solid #e2e8f0; }}
  table.criteria th {{ background: #f1f5f9; font-weight: 600; }}

  table.tests {{ width: 100%; border-collapse: collapse; }}
  table.tests th, table.tests td {{ padding: 4px 8px; text-align: left; font-size: 0.8rem;
                                    border-bottom: 1px solid #e2e8f0; }}
  table.tests th {{ background: #f1f5f9; font-weight: 600; }}
  td.msg {{ max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
           font-family: monospace; font-size: 0.75rem; color: #64748b; cursor: pointer; }}
  td.msg.expanded {{ white-space: pre-wrap; word-break: break-word; overflow: visible;
                    max-width: none; background: #fef9e7; padding: 8px; }}

  .kernel-tabs {{ display: flex; gap: 4px; margin-bottom: 8px; flex-wrap: wrap; }}
  .kernel-tab {{ padding: 3px 12px; cursor: pointer; font-size: 0.8rem; font-weight: 500;
                background: #e2e8f0; border-radius: 4px 4px 0 0; border: none; color: #475569; }}
  .kernel-tab.active {{ background: #1e293b; color: white; }}
  .kernel-panel {{ display: none; }}
  .kernel-panel.active {{ display: block; }}
  .kernel-code {{ background: #1e293b; color: #e2e8f0; padding: 16px; border-radius: 0 6px 6px 6px;
                 overflow-x: auto; font-size: 0.8rem; line-height: 1.5; max-height: 600px;
                 overflow-y: auto; }}
  .kernel-code pre {{ margin: 0; }}
  .kernel-code code {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; }}

  .artifact-content {{ background: white; border: 1px solid #e2e8f0; border-radius: 6px;
                      padding: 16px; font-size: 0.85rem; line-height: 1.6;
                      max-height: 600px; overflow-y: auto; white-space: pre-wrap; }}
  .no-data {{ color: #9ca3af; font-style: italic; padding: 20px; text-align: center; }}
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/cpp.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
</head>
<body>
<p><a href="/">&larr; All Runs</a></p>
<h1>Run #{run_id} &mdash; {prompt} {status_badge}</h1>
<div class="meta">
  <span><strong>Date:</strong> {ts}</span>
  <span><strong>Branch:</strong> {branch}</span>
  <span><strong>Phase:</strong> {phase}</span>
  <span><strong>Score:</strong> {score}</span>
</div>
{detail_html}
<script>
function showSection(runId, section) {{
  var container = document.getElementById(runId + '-' + section);
  if (!container) return;
  var parent = container.parentElement;
  parent.querySelectorAll('.section-panel').forEach(function(p) {{ p.classList.remove('active'); }});
  parent.querySelectorAll('.section-tab').forEach(function(t) {{ t.classList.remove('active'); }});
  container.classList.add('active');
  event.target.classList.add('active');
  container.querySelectorAll('pre code').forEach(function(block) {{
    if (!block.dataset.highlighted) {{ hljs.highlightElement(block); block.dataset.highlighted = 'true'; }}
  }});
}}
function toggleMsg(td) {{
  if (td.classList.contains('expanded')) {{
    td.classList.remove('expanded'); td.textContent = td.dataset.short;
  }} else {{
    td.classList.add('expanded'); td.textContent = td.dataset.full;
  }}
}}
function showKernel(kernelGroupId, index) {{
  var parent = document.getElementById(kernelGroupId + '-' + index).parentElement;
  parent.querySelectorAll('.kernel-panel').forEach(function(p) {{ p.classList.remove('active'); }});
  parent.querySelectorAll('.kernel-tab').forEach(function(t) {{ t.classList.remove('active'); }});
  document.getElementById(kernelGroupId + '-' + index).classList.add('active');
  event.target.classList.add('active');
}}
// Highlight visible code blocks on load
document.addEventListener('DOMContentLoaded', function() {{
  document.querySelectorAll('.section-panel.active pre code').forEach(function(block) {{
    hljs.highlightElement(block); block.dataset.highlighted = 'true';
  }});
}});
</script>
</body>
</html>"""


@app.route("/api/runs")
def api_runs():
    """JSON API for runs — consumable by external tools."""
    conn = _get_conn()
    runs = db.get_all_runs(conn)
    stats = db.get_stats(conn)
    conn.close()
    return {"stats": stats, "runs": runs}


@app.route("/api/run/<int:run_id>")
def api_run(run_id):
    """JSON API for a single run with all details."""
    conn = _get_conn()
    run = db.get_run(conn, run_id)
    if not run:
        conn.close()
        abort(404)
    result = {
        "run": run,
        "tests": db.get_test_results(conn, run_id),
        "criteria": db.get_score_criteria(conn, run_id),
        "kernels": db.get_kernels(conn, run_id),
        "host_code": db.get_host_code(conn, run_id),
        "artifacts": db.get_artifacts(conn, run_id),
        "tdd_state": db.get_tdd_state(conn, run_id),
        "kw_breadcrumbs": db.get_kw_breadcrumbs(conn, run_id),
    }
    conn.close()
    return result


def main():
    global _db_path
    parser = argparse.ArgumentParser(description="Eval detail server")
    parser.add_argument(
        "--db", default=str(db.DEFAULT_SQLITE_PATH), help="SQLite database path (ignored if EVAL_DATABASE_URL is set)"
    )
    parser.add_argument("--port", type=int, default=8090, help="HTTP server port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    args = parser.parse_args()

    _db_path = Path(args.db)
    print(f"Detail server: http://{args.host}:{args.port}")
    print(f"  Database: {_db_path if not db._is_postgres() else 'PostgreSQL (EVAL_DATABASE_URL)'}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

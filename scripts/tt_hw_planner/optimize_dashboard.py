# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Live web dashboard for the optimize loop.

``optimize`` writes everything it does to files already: the run directory gets ``state.json``,
``events.jsonl``, ``manifest.json`` and ``profiles/``; the state dir gets the lever-attempt log
(``cc_kernlog_*``), the measurement ledger (``perf_measurements_*``), per-stage timings
(``perf_mcp_stage_ms_*``) and the full-pipeline baseline. This module reads those files — and ONLY
reads them — and serves them as one JSON snapshot plus a single-page dashboard that polls it, so an
operator can watch levers land live instead of scrolling the run log afterwards.

Read-only is a hard requirement: a dashboard that consumed ``hitl_proposal.json`` the way the
orchestrator does (read + unlink) would steal proposals from the terminal pause screen, so the
proposal is peeked at, never unlinked, and decisions stay in the terminal.

Nothing here names a model stage, op, or lever: every label rendered comes out of the files the run
wrote, so a model that reports different stages tomorrow still renders.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The perf tree location has ONE owner (the optimize command); the dashboard reads the same tree.
from .commands.optimize import PERF_DIR

# A run counts as LIVE when one of its files moved this recently. Comfortably wider than the UI poll
# interval so an active run never flickers, narrower than a lever measurement (minutes), so a dead
# run does not look alive.
_LIVE_WINDOW_S = 45.0
_EVENTS_TAIL = 60
_RUN_ID_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-(.+)$")

# HITL handshake file names, mirrored from cc_optimize/hitl.py (loaded by path, not imported — the
# engine lives outside this package). Read-only use only: the orchestrator's read CONSUMES.
_HITL_PROPOSAL = "hitl_proposal.json"


# --------------------------------------------------------------------------- tolerant file reads


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 -- a live run rewrites these files mid-read; partial = absent
        return None


def _read_jsonl_tail(path: Path, limit: int) -> list:
    try:
        lines = path.read_text().splitlines()
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for raw in lines[-limit:]:
        try:
            rows.append(json.loads(raw))
        except ValueError:
            continue
    return rows


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


# --------------------------------------------------------------------------- run / state discovery


def run_slug(run_dir: Path) -> str | None:
    """The model slug a run directory is named for (``<ts>-<slug>``), else None."""
    m = _RUN_ID_RE.match(run_dir.name)
    return m.group(2) if m else None


def find_run_dir(repo_root: Path, slug: str | None = None, run_ref: str | None = None) -> Path | None:
    """Resolve the run to show: an explicit id/path, else the newest run (for ``slug`` if given)."""
    runs_root = Path(repo_root) / PERF_DIR / "runs"
    if run_ref:
        p = Path(run_ref)
        if p.is_dir():
            return p.resolve()
        cand = runs_root / run_ref
        return cand.resolve() if cand.is_dir() else None
    if not runs_root.is_dir():
        return None
    cands = []
    for d in runs_root.iterdir():
        if not d.is_dir() or not _RUN_ID_RE.match(d.name):
            continue
        if slug and run_slug(d) != slug:
            continue
        mt = _mtime(d)
        cands.append((mt or 0.0, d))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def repo_root_for_run(run_dir: Path, default: Path) -> Path:
    """The checkout a run directory belongs to, derived from the path itself. A run lives at
    ``<checkout>/models/experimental/perf_automation/runs/<id>`` — when the operator points the
    dashboard at another checkout's run (or a leftover worktree's), the state files must come from
    THAT checkout's .state, not from wherever the command happens to run."""
    anchor = (Path(PERF_DIR) / "runs").parts
    parts = Path(run_dir).resolve().parts
    for i in range(len(parts) - len(anchor), -1, -1):
        if parts[i : i + len(anchor)] == anchor:
            return Path(*parts[:i])
    return Path(default)


def state_dir_candidates(repo_root: Path, slug: str | None) -> list:
    """Where run memory can live, in priority order: $PERF_MCP_STATE_DIR (what a --persist run sets),
    the repo's durable .state/<slug>, then the /tmp default. Resolved per call so a dashboard started
    before the run exports its state dir still picks it up on the next poll."""
    out = []
    env = os.environ.get("PERF_MCP_STATE_DIR")
    if env:
        out.append(Path(env))
    state_root = Path(repo_root) / PERF_DIR / ".state"
    if slug:
        out.append(state_root / slug)
    out.append(state_root)
    out.append(Path(tempfile.gettempdir()))
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def _glob_first(dirs: list, pattern: str) -> list:
    """All matches across candidate dirs, highest-priority dir first, no duplicates by name."""
    seen, out = set(), []
    for d in dirs:
        try:
            matches = sorted(d.glob(pattern))
        except OSError:
            continue
        for m in matches:
            if m.name not in seen:
                seen.add(m.name)
                out.append(m)
    return out


# --------------------------------------------------------------------------- attempt normalization


def _attempt_status(rec: dict) -> str:
    """One word the UI can colour. beat_baseline is written ONLY by the commit (perf_mcp.py), so it
    is the kept/reverted line; claimed_beat_baseline is what the agent believed before committing."""
    if rec.get("wedged") or rec.get("measurement_failed"):
        return "wedged"
    if rec.get("beat_baseline"):
        return "kept"
    if rec.get("claimed_beat_baseline"):
        return "reverted"
    return "no-gain"


def _load_attempts(dirs: list, slug: str | None) -> list:
    """Lever attempts for this model across tasks, archive (.cumulative) union live log — the same
    union _load_attempts_all reads, so the dashboard agrees with the engine about what was tried."""
    pattern = "cc_kernlog_%s_*.json" % slug if slug else "cc_kernlog_*.json"
    logs = _glob_first(dirs, pattern + ".cumulative") + _glob_first(dirs, pattern)
    out = []
    for path in logs:
        data = _read_json(path)
        if not isinstance(data, list):
            continue
        task = ""
        m = re.match(r"cc_kernlog_(.+?)_([^_]+)\.json", path.name.removesuffix(".cumulative"))
        if m and slug and m.group(1) == slug:
            task = m.group(2)
        for rec in data:
            if not isinstance(rec, dict):
                continue
            before = rec.get("fullpipe_best_ms")
            after = rec.get("fullpipe_ms")
            delta_pct = None
            if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before:
                delta_pct = (after - before) / before * 100.0
            out.append(
                {
                    "op": rec.get("op_signature") or "?",
                    "lever": rec.get("kernel_kind") or "?",
                    "task": task,
                    "status": _attempt_status(rec),
                    "measured_ms": rec.get("measured_ms"),
                    "fullpipe_ms": after,
                    "fullpipe_delta_ms": rec.get("fullpipe_delta_ms"),
                    "before_ms": before,
                    "after_ms": after,
                    "delta_pct": delta_pct,
                    "commit": rec.get("commit"),
                    "note": rec.get("note") or "",
                    "stages": [s for s in (rec.get("stages") or []) if isinstance(s, dict)],
                }
            )
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _attempt_matches_bucket(op_sig: str, bucket: dict) -> bool:
    """Advisory match of an attempt's op_signature to a profile bucket (the engine's own _op_match is
    the authority for clearing rungs; the dashboard only needs 'has anyone touched this hot op')."""
    sig = _norm(op_sig)
    if not sig:
        return False
    if sig in _norm(str(bucket.get("id", ""))) or _norm(str(bucket.get("id", ""))) in sig:
        return True
    for top in bucket.get("top_ops") or []:
        code = _norm(str(top.get("op_code", "")))
        if code and (sig in code or code in sig):
            return True
    return False


# --------------------------------------------------------------------------- the snapshot


def collect_state(run_dir: Path, state_dirs: list, slug: str | None = None) -> dict:
    """Assemble the one JSON snapshot the dashboard renders. Every section is best-effort: a file
    that does not exist yet (baseline still measuring) simply omits its section, never fails."""
    run_dir = Path(run_dir)
    slug = slug or run_slug(run_dir)
    now = time.time()

    manifest = _read_json(run_dir / "manifest.json") or {}
    config = manifest.get("config") or {}
    state = _read_json(run_dir / "state.json") or {}
    events = _read_jsonl_tail(run_dir / "events.jsonl", _EVENTS_TAIL)

    watched = [run_dir / "state.json", run_dir / "events.jsonl"]
    mtimes = [m for m in (_mtime(p) for p in watched) if m is not None]
    for log in _glob_first(state_dirs, "cc_kernlog_%s_*.json" % slug if slug else "cc_kernlog_*.json"):
        mt = _mtime(log)
        if mt is not None:
            mtimes.append(mt)
    age = (now - max(mtimes)) if mtimes else None

    # Per-stage timings + the full-pipeline baseline the per-lever gate banks (both keyed by the
    # names the MODEL reported, so the UI never assumes a stage set).
    stages_cur, stage_paths, stage_bytes, stage_unit = {}, {}, {}, None
    for f in _glob_first(state_dirs, "perf_mcp_stage_ms_%s_*.json" % slug if slug else "perf_mcp_stage_ms_*.json"):
        d = _read_json(f)
        if isinstance(d, dict) and isinstance(d.get("stages"), dict):
            stages_cur.update(d["stages"])
            stage_paths.update(d.get("paths") or {})
            stage_bytes.update(d.get("bytes") or {})
    fullpipe = None
    for f in _glob_first(
        state_dirs,
        "perf_mcp_full_pipeline_baseline_1cq_%s_*.json" % slug
        if slug
        else "perf_mcp_full_pipeline_baseline_1cq_*.json",
    ):
        if f.name.endswith(".pending.json"):
            continue
        d = _read_json(f)
        if isinstance(d, dict) and d.get("full_pipeline_ms") is not None:
            fullpipe = d
            stage_unit = d.get("unit") or stage_unit
            break
    stages_base = (fullpipe or {}).get("stages") or {}

    stage_names = sorted(
        set(stages_cur) | set(stages_base), key=lambda n: -(stages_cur.get(n) or stages_base.get(n) or 0)
    )
    stages = [
        {
            "name": n,
            "ms": stages_cur.get(n),
            "baseline_ms": stages_base.get(n),
            "path": stage_paths.get(n),
            "bytes": stage_bytes.get(n),
        }
        for n in stage_names
    ]

    # The ledger, grouped by the kinds the run recorded (floors, roofs, anchors) — passed through,
    # not interpreted, so a new kind tomorrow renders without a dashboard change.
    ledger = {}
    for f in _glob_first(state_dirs, "perf_measurements_%s_*.jsonl" % slug if slug else "perf_measurements_*.jsonl"):
        for row in _read_jsonl_tail(f, 2000):
            if isinstance(row, dict) and row.get("kind"):
                ledger.setdefault(row["kind"], []).append(row)

    # Throughput only when the run itself declared a per-token unit (the full-pipeline baseline's
    # "unit" field). Current comes from the ledger's committed after-rows — earliest-reading-wins
    # durability means the before row is the TRUE original, not this run's starting point.
    throughput = None
    if stage_unit == "token":
        fp = ledger.get("fullpipe_e2e") or []
        before = next((r.get("value_ms") for r in fp if r.get("phase") == "before"), None)
        afters = [r.get("value_ms") for r in fp if r.get("phase") == "after" and r.get("value_ms")]
        base_ms = before or (fullpipe or {}).get("full_pipeline_ms")
        cur_ms = afters[-1] if afters else None
        throughput = {
            "unit": "tok/s",
            "baseline": (1000.0 / base_ms) if base_ms else None,
            "current": (1000.0 / cur_ms) if cur_ms else None,
        }

    profile = _read_json(run_dir / "profiles" / "baseline_profile.json")
    buckets = []
    if isinstance(profile, dict):
        for b in profile.get("buckets") or []:
            if isinstance(b, dict):
                buckets.append(b)

    attempts = _load_attempts(state_dirs, slug)
    opportunities = []
    for b in sorted(buckets, key=lambda x: -(x.get("device_ms") or 0)):
        tried = sorted({a["lever"] for a in attempts if _attempt_matches_bucket(a["op"], b)})
        kept = any(a["status"] == "kept" for a in attempts if _attempt_matches_bucket(a["op"], b))
        opportunities.append(
            {
                "id": b.get("id"),
                "device_ms": b.get("device_ms"),
                "pct": b.get("pct"),
                "count": b.get("count"),
                "tags": b.get("tags") or {},
                "lever_state": b.get("lever_state") or {},
                "top_ops": (b.get("top_ops") or [])[:5],
                "tried_rungs": tried,
                "status": "cleared" if kept else ("touched" if tried else "open"),
            }
        )

    # HITL: PEEK ONLY. The orchestrator's read consumes the proposal; the dashboard must not.
    proposal = _read_json(run_dir / _HITL_PROPOSAL)

    thermal = None
    for f in _glob_first(state_dirs, "perf_mcp_thermal_profile.json"):
        thermal = _read_json(f)
        if thermal is not None:
            break
    topology = None
    for f in _glob_first(state_dirs, "perf_mcp_board_topology.json"):
        topology = _read_json(f)
        if topology is not None:
            break

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "run": {
            "id": run_dir.name,
            "dir": str(run_dir),
            "live": bool(age is not None and age <= _LIVE_WINDOW_S),
            "age_s": round(age, 1) if age is not None else None,
            "state": state.get("state"),
            "iteration": state.get("iteration"),
        },
        "model": {
            "slug": slug,
            "root": config.get("model_root"),
        },
        "config": {
            k: config.get(k)
            for k in ("metric", "devices", "pcc_test", "perf_test", "max_iter")
            if config.get(k) is not None
        },
        "metric": state.get("metric"),
        "stages": stages,
        "throughput": throughput,
        "fullpipe_ms": (fullpipe or {}).get("full_pipeline_ms"),
        "attempts": attempts,
        "opportunities": opportunities,
        "hitl_proposal": proposal if isinstance(proposal, dict) else None,
        "events": list(reversed(events)),
        "ledger": ledger,
        "thermal": thermal,
        "topology": topology,
    }


# --------------------------------------------------------------------------- HTTP serving


def make_handler(collect_fn):
    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 -- BaseHTTPRequestHandler names it
            if self.path in ("/", "/index.html"):
                self._send(200, _HTML.encode(), "text/html; charset=utf-8")
            elif self.path.startswith("/api/state"):
                try:
                    payload = json.dumps(collect_fn(), default=str).encode()
                    self._send(200, payload, "application/json")
                except Exception as exc:  # noqa: BLE001 -- a bad poll must not kill the server
                    self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *_args):  # keep the run's console clean
            pass

    return DashboardHandler


def make_server(host: str, port: int, collect_fn) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(collect_fn))
    srv.daemon_threads = True
    return srv


def serve(host: str, port: int, collect_fn) -> int:
    """Blocking serve (standalone command). Returns on Ctrl+C."""
    try:
        srv = make_server(host, port, collect_fn)
    except OSError as exc:
        if port != 0:
            print(f"  [dashboard] port {port} unavailable ({exc}); using a free port instead")
            srv = make_server(host, 0, collect_fn)
        else:
            raise
    url = "http://%s:%d/" % (host if host not in ("0.0.0.0", "::") else "127.0.0.1", srv.server_address[1])
    print(f"  [dashboard] live optimize view: {url}  (Ctrl+C to stop)")
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def serve_in_thread(host: str, port: int, collect_fn):
    """Non-blocking serve for ``optimize --dashboard``: daemon thread, dies with the run process."""
    import threading

    try:
        srv = make_server(host, port, collect_fn)
    except OSError as exc:
        if port != 0:
            print(f"  [dashboard] port {port} unavailable ({exc}); using a free port instead")
            srv = make_server(host, 0, collect_fn)
        else:
            raise
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
    t.start()
    url = "http://%s:%d/" % (host if host not in ("0.0.0.0", "::") else "127.0.0.1", srv.server_address[1])
    return srv, t, url


# --------------------------------------------------------------------------- the page

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Optimize — live</title>
<style>
  :root {
    --bg: #0b1220; --panel: #111a2c; --panel2: #0e1626; --line: #22314d;
    --txt: #dbe6f5; --dim: #8ba0bd; --blue: #2f81f7; --green: #3fb950;
    --red: #f85149; --amber: #d29922; --chip: #1c2a44;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--txt);
         font: 14px/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  header { display: flex; align-items: center; gap: 12px; padding: 14px 22px;
           border-bottom: 1px solid var(--line); background: var(--panel2); flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .sub { color: var(--dim); font-size: 12px; }
  .badge { font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 20px;
           letter-spacing: .5px; }
  .badge.live { background: rgba(63,185,80,.15); color: var(--green); border: 1px solid var(--green); }
  .badge.live::before { content: "●"; margin-right: 5px; animation: pulse 1.6s infinite; }
  .badge.idle { background: rgba(139,160,189,.12); color: var(--dim); border: 1px solid var(--line); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
  .wrap { padding: 18px 22px; max-width: 1500px; margin: 0 auto; }
  #cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
           gap: 12px; margin-bottom: 16px; }
  .card { background: linear-gradient(160deg, #14335e, #0f2242); border: 1px solid #245;
          border-radius: 10px; padding: 12px 14px; }
  .card .k { font-size: 11px; color: #9cc0ff; text-transform: uppercase; letter-spacing: .6px; }
  .card .v { font-size: 22px; font-weight: 700; margin-top: 4px; }
  .card .d { font-size: 12px; margin-top: 2px; color: #9cc0ff; }
  .up { color: var(--green); } .dn { color: var(--red); }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }
  td .why { margin-top: 2px; }
  .grid { display: grid; grid-template-columns: 5fr 7fr; gap: 16px; margin-bottom: 16px; }
  @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
  .panel > h2 { font-size: 13px; margin: 0; padding: 12px 16px; border-bottom: 1px solid var(--line);
                color: var(--dim); text-transform: uppercase; letter-spacing: .8px; }
  .panel .body { padding: 12px 16px; }
  .mrow { margin: 10px 0; }
  .mrow .lab { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
  .mrow .lab .r { color: var(--dim); font-variant-numeric: tabular-nums; }
  .bar { height: 8px; background: var(--chip); border-radius: 4px; position: relative; overflow: hidden; }
  .bar > i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--blue); border-radius: 4px; }
  .bar > i.win { background: var(--green); }
  .bar > b { position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--amber); }
  .opp { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin: 8px 0;
         background: var(--panel2); }
  .opp.hot { border-color: var(--amber); box-shadow: 0 0 0 1px var(--amber) inset; }
  .opp .top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .opp .name { font-weight: 600; font-size: 13px; }
  .opp .why { color: var(--dim); font-size: 12px; margin-top: 4px; }
  .opp .nums { font-size: 12px; color: var(--dim); margin-top: 4px; font-variant-numeric: tabular-nums; }
  .chip { font-size: 10.5px; font-weight: 700; padding: 2px 8px; border-radius: 12px;
          text-transform: uppercase; letter-spacing: .4px; background: var(--chip); color: var(--dim); }
  .chip.kept { background: rgba(63,185,80,.15); color: var(--green); }
  .chip.reverted, .chip.wedged { background: rgba(248,81,73,.15); color: var(--red); }
  .chip.no-gain { background: rgba(210,153,34,.15); color: var(--amber); }
  .chip.open { background: rgba(47,129,247,.15); color: var(--blue); }
  .chip.lever { background: var(--chip); color: #9cc0ff; text-transform: none; }
  .feed { max-height: 220px; overflow-y: auto; margin-top: 10px; border-top: 1px solid var(--line);
          padding-top: 8px; font-size: 12px; }
  .feed .ev { display: flex; gap: 8px; padding: 2px 0; color: var(--dim); }
  .feed .ev .t { color: #5c7191; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .feed .ev .s { color: #9cc0ff; white-space: nowrap; }
  #tabs { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
  #tabbar { display: flex; gap: 4px; padding: 8px 12px 0; border-bottom: 1px solid var(--line);
            flex-wrap: wrap; }
  #tabbar button { background: none; border: none; color: var(--dim); font-size: 13px; padding: 8px 12px;
                   cursor: pointer; border-bottom: 2px solid transparent; }
  #tabbar button.on { color: var(--txt); border-bottom-color: var(--blue); font-weight: 600; }
  #tabbody { padding: 14px 16px; }
  table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line);
           font-variant-numeric: tabular-nums; }
  th { color: var(--dim); font-weight: 600; text-transform: uppercase; font-size: 11px;
       letter-spacing: .5px; }
  .empty { color: var(--dim); font-size: 13px; padding: 18px; text-align: center; }
  .stack { display: flex; height: 26px; border-radius: 6px; overflow: hidden; margin: 8px 0 14px; }
  .stack > div { height: 100%; }
  .legend { font-size: 12px; color: var(--dim); display: flex; gap: 14px; flex-wrap: wrap; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }
</style>
</head>
<body>
<header>
  <h1>Optimize — live</h1>
  <span id="livebadge" class="badge idle">CONNECTING</span>
  <span class="sub" id="runinfo"></span>
  <span class="sub" id="updated" style="margin-left:auto"></span>
</header>
<div class="wrap">
  <div id="cards"></div>
  <div class="grid">
    <div class="panel"><h2>Performance Metrics</h2><div class="body" id="perf"></div></div>
    <div class="panel"><h2>Optimization Opportunities</h2><div class="body" id="opps"></div></div>
  </div>
  <div class="panel" style="margin-bottom:16px"><h2>Optimization History</h2><div class="body" id="histbody"></div></div>
  <div id="tabs">
    <div id="tabbar"></div>
    <div id="tabbody"></div>
  </div>
</div>
<script>
const $ = (id) => document.getElementById(id);
const fmtMs = (v) => (v === null || v === undefined) ? "—" : (v >= 100 ? v.toFixed(1) : v.toFixed(2)) + " ms";
const fmtPct = (v) => (v === null || v === undefined) ? "" : v.toFixed(1) + "%";
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const PALETTE = ["#2f81f7","#3fb950","#d29922","#f778ba","#76e3ea","#e3b341","#ff9bce","#56d4dd"];

function deltaTxt(cur, base, dir) {
  if (cur == null || base == null || !base) return "";
  const d = (cur - base) / base * 100;
  const better = dir === "max" ? d > 0 : d < 0;
  const cls = better ? "up" : "dn";
  return `<span class="${cls}">${d > 0 ? "+" : ""}${d.toFixed(1)}%</span> vs baseline`;
}

function renderCards(S) {
  const cards = [];
  const m = S.metric || {};
  const dir = m.direction || "min";
  cards.push({k: (m.name || "metric") + " · current", v: fmtMs(m.current),
              d: deltaTxt(m.current, m.baseline, dir)});
  cards.push({k: "baseline", v: fmtMs(m.baseline), d: m.target != null ? "target " + fmtMs(m.target) : ""});
  (S.stages || []).slice(0, 2).forEach(s => {
    cards.push({k: s.name, v: fmtMs(s.ms), d: deltaTxt(s.ms, s.baseline_ms, "min")});
  });
  if (S.throughput && (S.throughput.current || S.throughput.baseline)) {
    const t = S.throughput;
    cards.push({k: "throughput", v: t.current ? t.current.toFixed(1) + " " + t.unit : "—",
                d: deltaTxt(t.current, t.baseline, "max")});
  } else if (m.target != null && m.baseline && m.current != null) {
    const span = m.baseline - m.target;
    const done = span ? Math.max(0, Math.min(1, (m.baseline - m.current) / span)) : 0;
    cards.push({k: "target progress", v: (done * 100).toFixed(0) + "%", d: "goal " + fmtMs(m.target)});
  }
  $("cards").innerHTML = cards.map(c =>
    `<div class="card"><div class="k">${esc(c.k)}</div><div class="v">${esc(c.v)}</div><div class="d">${c.d}</div></div>`).join("");
}

function barRow(label, cur, base, scale, extra) {
  const w = cur != null && scale ? Math.min(100, cur / scale * 100) : 0;
  const win = cur != null && base != null && cur < base;
  const mark = base != null && scale ? Math.min(100, base / scale * 100) : null;
  return `<div class="mrow"><div class="lab"><span>${esc(label)}</span>
    <span class="r">${fmtMs(cur)}${extra ? " · " + esc(extra) : ""}</span></div>
    <div class="bar"><i class="${win ? "win" : ""}" style="width:${w}%"></i>${mark != null ? `<b style="left:${mark}%"></b>` : ""}</div></div>`;
}

function renderPerf(S) {
  const rows = [];
  const m = S.metric || {};
  const scale = Math.max(m.baseline || 0, m.current || 0, m.target || 0,
                         ...(S.stages || []).flatMap(s => [s.ms || 0, s.baseline_ms || 0]), 1e-9);
  if (m.current != null) rows.push(barRow("overall · " + (m.name || ""), m.current, m.baseline, scale));
  (S.stages || []).forEach(s => rows.push(barRow(s.name, s.ms, s.baseline_ms, scale, s.path)));
  $("perf").innerHTML = rows.join("") ||
    `<div class="empty">no measurements yet — the baseline is still being measured</div>`;
}

function oppCard(o) {
  const tags = Object.entries(o.tags || {}).map(([k, v]) => `<span class="chip lever">${esc(k)}: ${esc(v)}</span>`).join(" ");
  const tried = (o.tried_rungs || []).map(r => `<span class="chip lever">${esc(r)}</span>`).join(" ");
  const st = o.status === "cleared" ? "kept" : (o.status === "touched" ? "no-gain" : "open");
  return `<div class="opp"><div class="top"><span class="name">${esc(o.id)}</span>
    <span class="chip ${st}">${esc(o.status)}</span>
    <span style="margin-left:auto" class="nums">${fmtMs(o.device_ms)} · ${fmtPct(o.pct)}</span></div>
    <div class="nums">${tried ? "tried: " + tried : "no lever tried yet"}</div>
    <div class="why">${tags}</div></div>`;
}

function attemptCard(a, hot) {
  const d = a.fullpipe_delta_ms;
  const dTxt = (d != null) ? ` · e2e ${d > 0 ? "+" : ""}${Number(d).toFixed(2)} ms` : "";
  return `<div class="opp ${hot ? "hot" : ""}"><div class="top">
    <span class="name">${esc(a.op)}</span><span class="chip lever">${esc(a.lever)}</span>
    <span class="chip ${a.status}">${a.status === "kept" ? "✓ applied" : esc(a.status)}</span>
    <span style="margin-left:auto" class="nums">${fmtMs(a.measured_ms)}${dTxt}</span></div>
    ${a.note ? `<div class="why">${esc(a.note)}</div>` : ""}</div>`;
}

function renderOpps(S) {
  let html = "";
  const p = S.hitl_proposal;
  if (p) {
    const t = p.tried || {}, r = p.result || {}, n = p.next || {};
    html += `<div class="opp hot"><div class="top"><span class="name">HITL — decision pending in terminal</span>
      <span class="chip no-gain">awaiting commit / revert</span></div>
      <div class="nums">tried <b>${esc(t.lever || "?")}</b> on ${esc(t.op || "?")} — ${esc(r.win ? "WIN" : "no win")}
      (${fmtMs(r.before_ms)} → ${fmtMs(r.after_ms)})</div>
      ${t.why ? `<div class="why">why: ${esc(t.why)}</div>` : ""}
      ${n.target ? `<div class="why">next: ${esc(n.target)}${n.why ? " — " + esc(n.why) : ""}</div>` : ""}</div>`;
  }
  const at = S.attempts || [];
  const latest = at.length ? at[at.length - 1] : null;
  html += at.slice().reverse().slice(0, 12).map(a => attemptCard(a, a === latest && S.run.live)).join("");
  const open = (S.opportunities || []).filter(o => o.status === "open").slice(0, 5);
  if (open.length) {
    html += `<div style="margin-top:10px;color:var(--dim);font-size:12px">UNTAPPED HOTSPOTS</div>`;
    html += open.map(oppCard).join("");
  }
  if (!html) html = `<div class="empty">no lever attempts recorded yet — opportunities appear here as the run applies them</div>`;
  const ev = (S.events || []).slice(0, 30).map(e =>
    `<div class="ev"><span class="t">${esc((e.ts || "").slice(11, 19))}</span>
     <span class="s">${esc(e.stage || "")} ${esc(e.event || "")}</span><span>${esc(e.detail || "")}</span></div>`).join("");
  html += `<div class="feed">${ev || '<div class="empty">no events yet</div>'}</div>`;
  $("opps").innerHTML = html;
}

const STATUS_LABEL = {kept: "✓ applied", reverted: "✗ reverted", wedged: "wedged", "no-gain": "no gain"};

function renderHistory(S) {
  const at = (S.attempts || []).slice().reverse();
  if (!at.length) {
    $("histbody").innerHTML = `<div class="empty">no levers applied yet — every attempt the run records lands here (same kernel log RUN_REPORT.md renders from)</div>`;
    return;
  }
  const rows = at.map(a => {
    const d = a.delta_pct;
    const dTxt = d == null ? "—" : `<span class="${d < 0 ? "up" : "dn"}">${d > 0 ? "+" : ""}${d.toFixed(1)}%</span>`;
    const commit = a.commit ? `<span class="mono">${esc(String(a.commit).slice(0, 7))}</span>` : "—";
    const note = a.note ? `<div class="why">${esc(a.note)}</div>` : "";
    return `<tr><td><span class="chip lever">${esc(a.lever)}</span>${note}</td>
      <td class="mono">${esc(a.op)}</td>
      <td>${fmtMs(a.before_ms)}</td><td>${fmtMs(a.after_ms)}</td><td>${dTxt}</td>
      <td><span class="chip ${a.status}">${esc(STATUS_LABEL[a.status] || a.status)}</span></td>
      <td>${commit}</td></tr>`;
  }).join("");
  $("histbody").innerHTML = `<table><tr><th>Lever</th><th>Op</th><th>Before</th><th>After</th>
    <th>Δ%</th><th>Status</th><th>Commit</th></tr>${rows}</table>`;
}

const TABS = ["Latency Breakdown", "Roofline", "Compute vs Memory", "Power Analysis", "Scaling"];
let curTab = TABS[0];
let LAST = null;

function renderTab(S) {
  const el = $("tabbody");
  if (curTab === "Latency Breakdown") {
    const st = (S.stages || []).filter(s => s.ms != null);
    if (!st.length) { el.innerHTML = `<div class="empty">no per-stage timing captured yet</div>`; return; }
    const tot = st.reduce((a, s) => a + s.ms, 0) || 1;
    const stack = st.map((s, i) =>
      `<div style="width:${(s.ms / tot * 100).toFixed(2)}%;background:${PALETTE[i % PALETTE.length]}" title="${esc(s.name)} ${fmtMs(s.ms)}"></div>`).join("");
    const legend = st.map((s, i) =>
      `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>${esc(s.name)} — ${fmtMs(s.ms)} (${(s.ms / tot * 100).toFixed(1)}%)</span>`).join("");
    const rows = st.map(s => `<tr><td>${esc(s.name)}</td><td>${fmtMs(s.ms)}</td>
      <td>${fmtMs(s.baseline_ms)}</td><td>${esc(s.path || "")}</td></tr>`).join("");
    el.innerHTML = `<div class="stack">${stack}</div><div class="legend">${legend}</div>
      <table style="margin-top:12px"><tr><th>stage</th><th>current</th><th>baseline</th><th>path</th></tr>${rows}</table>`;
  } else if (curTab === "Roofline") {
    const ops = S.opportunities || [];
    if (!ops.length) { el.innerHTML = `<div class="empty">no profile buckets yet</div>`; return; }
    const rows = ops.map(o => `<tr><td>${esc(o.id)}</td><td>${fmtMs(o.device_ms)}</td><td>${fmtPct(o.pct)}</td>
      <td>${esc((o.tags || {}).bound || "")}</td><td>${esc((o.tags || {}).memory || "")}</td>
      <td>${esc((o.tags || {}).fidelity || "")}</td><td>${esc((o.tags || {}).grid || "")}</td></tr>`).join("");
    const m = S.metric || {};
    el.innerHTML = `<div class="legend" style="margin-bottom:10px">current ${fmtMs(m.current)} · baseline ${fmtMs(m.baseline)} · target ${fmtMs(m.target)}</div>
      <table><tr><th>bucket</th><th>device ms</th><th>% total</th><th>bound</th><th>memory</th><th>fidelity</th><th>grid</th></tr>${rows}</table>`;
  } else if (curTab === "Compute vs Memory") {
    const tops = (S.opportunities || []).flatMap(o => (o.top_ops || []).map(t => ({...t, bucket: o.id})));
    if (!tops.length) { el.innerHTML = `<div class="empty">no per-op profile yet</div>`; return; }
    const rows = tops.slice(0, 25).map(t => `<tr><td>${esc(t.op_code || "")}</td><td>${esc(t.bucket)}</td>
      <td>${t.device_ms != null ? Number(t.device_ms).toFixed(3) : "—"}</td>
      <td>${t.bytes != null ? (t.bytes / 1e9).toFixed(2) + " GB" : "—"}</td>
      <td>${t.cores ?? "—"}</td><td>${esc(t.fidelity || "")}</td></tr>`).join("");
    el.innerHTML = `<table><tr><th>op</th><th>bucket</th><th>ms</th><th>bytes read</th><th>cores</th><th>fidelity</th></tr>${rows}</table>`;
  } else if (curTab === "Power Analysis") {
    const th = S.thermal;
    if (!th) { el.innerHTML = `<div class="empty">no thermal/power profile captured for this model yet</div>`; return; }
    el.innerHTML = `<table>${Object.entries(th).map(([k, v]) =>
      `<tr><th>${esc(k)}</th><td>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</td></tr>`).join("")}</table>`;
  } else if (curTab === "Scaling") {
    const c = S.config || {}, tp = S.topology;
    let html = `<table>${Object.entries(c).map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join("")}</table>`;
    if (tp) html += `<h3 style="color:var(--dim);font-size:12px;margin:14px 0 6px">BOARD TOPOLOGY</h3>
      <table>${Object.entries(tp).map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</td></tr>`).join("")}</table>`;
    el.innerHTML = html;
  }
}

function render(S) {
  LAST = S;
  const b = $("livebadge");
  if (S.run.live) { b.className = "badge live"; b.textContent = "LIVE"; }
  else { b.className = "badge idle"; b.textContent = S.run.age_s != null ? "IDLE" : "NO RUN"; }
  $("runinfo").textContent = [S.model && S.model.slug, S.run.id, S.run.state,
    S.run.iteration != null ? "iter " + S.run.iteration : ""].filter(Boolean).join(" · ");
  $("updated").textContent = "updated " + (S.generated_at || "").slice(11, 19) + " UTC" +
    (S.run.age_s != null ? " · last write " + S.run.age_s + "s ago" : "");
  renderCards(S); renderPerf(S); renderOpps(S); renderHistory(S); renderTab(S);
}

$("tabbar").innerHTML = TABS.map(t =>
  `<button data-t="${esc(t)}" class="${t === curTab ? "on" : ""}">${esc(t)}</button>`).join("");
$("tabbar").addEventListener("click", (e) => {
  const t = e.target && e.target.dataset ? e.target.dataset.t : null;
  if (!t) return;
  curTab = t;
  document.querySelectorAll("#tabbar button").forEach(b => b.classList.toggle("on", b.dataset.t === t));
  if (LAST) renderTab(LAST);
});

async function poll() {
  try {
    const r = await fetch("/api/state", {cache: "no-store"});
    if (r.ok) render(await r.json());
  } catch (e) { /* keep last frame; next poll retries */ }
}
poll();
setInterval(poll, 2000);
</script>
</body>
</html>
"""

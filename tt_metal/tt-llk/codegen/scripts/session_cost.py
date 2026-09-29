# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Aggregate Claude Code session tokens and cost for a codegen run.

Claude Code writes every turn to a JSONL transcript under
``~/.claude/projects/<cwd-mapped>/<sessionId>.jsonl`` and every sub-agent
to ``~/.claude/projects/<cwd-mapped>/<sessionId>/subagents/*.jsonl``.
Each ``type: assistant`` entry carries a ``message.usage`` object with
``input_tokens``, ``output_tokens``, ``cache_read_input_tokens``, and
``cache_creation_input_tokens``. The model used for that turn is in
``message.model``.

This script sums those fields across the main jsonl plus every subagent
transcript (``<sessionId>/subagents/agent-*.jsonl`` — stored flat, one file per
agent regardless of spawn depth), optionally filtered to entries after
``--since``, and applies per-model Anthropic pricing to compute ``cost_usd``.

When --log-dir contains session_registry.json, explicitly linked standalone
sessions are included, with each root using its own native OTEL cost or transcript
estimate. Missing registered sessions are reported in cost_accounting; they are
never replaced with an unrelated active session.

Interactive codegen runs (the orchestrator inside ``claude``) have no
``cli_output.json`` to read from — this script is the live source of truth
for tokens + cost. Batch runs get an authoritative ``cli_output.json`` at
end-of-run; when that file lands in ``$LOG_DIR`` the dashboard will
backfill and supersede what we wrote here.

Accuracy: ``cost_usd`` is an estimate, same quality as the ``/cost`` slash
command — both multiply token counts by a local pricing table. Anthropic
notes that ``/cost`` "may differ from your actual bill; for authoritative
billing see the Usage page in the Claude Console." Keep the ``PRICING``
table below in sync with Anthropic's published list prices.

Usage:
    # Write aggregated tokens + cost to <log_dir>/run.json (patches in place)
    python codegen/scripts/session_cost.py \
        --since "$START_TIME" \
        --model "$MODEL" \
        --log-dir "$LOG_DIR"

    # Or just print JSON to stdout (no patch)
    python codegen/scripts/session_cost.py --since "$START_TIME"

    # Print the running session's full model id (e.g. claude-opus-4-8) and exit
    python codegen/scripts/session_cost.py --print-model
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Anthropic public list prices (USD per 1M tokens), verified 2026-09-21 against
# https://platform.claude.com/docs/en/about-claude/pricing
PRICING = {
    "fable": {  # Claude Fable 5 / Mythos 5
        "input": 10.00,
        "output": 50.00,
        "cache_read": 1.00,
        "cache_creation": 12.50,
        "cache_creation_1h": 20.00,
    },
    "opus": {  # Opus 5 / 4.8 / 4.7 / 4.6 / 4.5
        "input": 5.00,
        "output": 25.00,
        "cache_read": 0.50,
        "cache_creation": 6.25,
        "cache_creation_1h": 10.00,
    },
    "sonnet5": {
        "input": 2.00,
        "output": 10.00,
        "cache_read": 0.20,
        "cache_creation": 2.50,
        "cache_creation_1h": 4.00,
    },
    "sonnet": {  # Sonnet 4.6 / 4.5 / 4
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_creation": 3.75,
        "cache_creation_1h": 6.00,
    },
    "haiku": {  # Haiku 4.5
        "input": 1.00,
        "output": 5.00,
        "cache_read": 0.10,
        "cache_creation": 1.25,
        "cache_creation_1h": 2.00,
    },
}


def _tier(model_str: str | None) -> str:
    m = (model_str or "").lower()
    if "fable" in m or "mythos" in m:
        return "fable"
    if "opus" in m:
        return "opus"
    if "sonnet-5" in m:
        return "sonnet5"
    if "sonnet" in m:
        return "sonnet"
    if "haiku" in m:
        return "haiku"
    return "opus"


def _last_model(jsonl_path: Path) -> str | None:
    """Return the raw model id of the most recent real assistant turn.

    Claude Code stamps each ``type: assistant`` entry with ``message.model``
    (e.g. ``claude-opus-4-8``). Synthetic/system turns carry ``<synthetic>`` —
    skip them. Returns None when the transcript has no model-bearing turn.
    """
    if not jsonl_path.exists():
        return None
    last: str | None = None
    with jsonl_path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") != "assistant":
                continue
            m = (d.get("message") or {}).get("model")
            if m and m != "<synthetic>":
                last = m
    return last


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _build_paths(session_id: str, cwd: str) -> tuple[Path, Path]:
    """Map (sessionId, cwd) → (main_jsonl, subagents_dir)."""
    home = Path(os.path.expanduser("~"))
    proj_name = cwd.replace("_", "-").replace("/", "-")
    proj_dir = home / ".claude" / "projects" / proj_name
    return proj_dir / f"{session_id}.jsonl", proj_dir / session_id / "subagents"


def _find_by_session_id(session_id: str) -> tuple[Path, Path] | None:
    home = Path(os.path.expanduser("~"))
    matches = sorted(
        (home / ".claude" / "projects").glob(f"*/{session_id}.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return None
    jsonl = matches[0]
    return jsonl, jsonl.parent / session_id / "subagents"


def _discover_session(preferred_pid: str | None) -> tuple[str, Path, Path] | None:
    """Find the active session by consulting ``~/.claude/sessions/<pid>.json``.

    Preference order:
      1. An entry whose ``pid`` matches ``preferred_pid`` (usually ``$PPID``
         of the bash process that invoked us — the claude CLI process).
      2. The most recently started session across all session files.
    """
    home = Path(os.path.expanduser("~"))
    sessions_dir = home / ".claude" / "sessions"
    if not sessions_dir.is_dir():
        return None

    candidates: list[tuple[int, str, str, str]] = []
    for f in sessions_dir.glob("*.json"):
        try:
            meta = json.loads(f.read_text())
        except Exception:
            continue
        sid = meta.get("sessionId")
        cwd = meta.get("cwd")
        started = int(meta.get("startedAt") or 0)
        pid = str(meta.get("pid") or "")
        if sid and cwd:
            candidates.append((started, sid, cwd, pid))

    if preferred_pid:
        for started, sid, cwd, pid in candidates:
            if pid == str(preferred_pid):
                jsonl, subs = _build_paths(sid, cwd)
                return sid, jsonl, subs

    # PID matching fails when the bash Bash-tool shell's PPID doesn't match the
    # claude CLI PID stored in the session file.  Fall back to CWD matching:
    # prefer the most recently started session whose cwd equals the current
    # working directory.  This correctly disambiguates concurrent sessions for
    # different projects.
    current_cwd = os.getcwd()
    cwd_matches = [c for c in candidates if c[2] == current_cwd]
    if cwd_matches:
        cwd_matches.sort(key=lambda x: x[0], reverse=True)
        _, sid, cwd, _ = cwd_matches[0]
        jsonl, subs = _build_paths(sid, cwd)
        return sid, jsonl, subs

    # Last resort: most recently started session across all projects.
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, sid, cwd, _ = candidates[0]
        jsonl, subs = _build_paths(sid, cwd)
        return sid, jsonl, subs

    return None


def _collect(
    jsonl_path: Path,
    since_dt: datetime | None,
    override_model: str | None,
    by_req: dict,
    noreq: list,
) -> None:
    """Read one transcript's assistant turns into the shared totals.

    Claude Code writes each response a few times as it streams, with the token
    counts growing each time — so per message.id (falling back to requestId or
    transcript uuid) we keep the largest complete write. This also deduplicates
    turns copied between transcripts. Entries lacking every identity are kept
    separately in `noreq`.

    cache_creation may be split into 5-minute and 1-hour buckets (priced
    differently); when there is no split, count it all as 5-minute.

    Each collected row is [model, input, output, cache_read, cache_5m, cache_1h].
    """
    if not jsonl_path.exists():
        return
    with jsonl_path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") != "assistant":
                continue
            msg = d.get("message") or {}
            usage = msg.get("usage")
            if msg.get("model") == "<synthetic>":
                continue
            if not usage:
                continue
            if since_dt is not None:
                ts = _parse_ts(d.get("timestamp"))
                if ts is not None and ts < since_dt:
                    continue
            split = usage.get("cache_creation")
            if not isinstance(split, dict):
                split = {}  # None, or an unexpected scalar — never let .get() throw
            c5 = int(split.get("ephemeral_5m_input_tokens") or 0)
            c1h = int(split.get("ephemeral_1h_input_tokens") or 0)
            if not (c5 or c1h):
                # No recognized TTL bucket (no split, or only unknown keys) — fall
                # back to the flat total so those tokens are never silently dropped.
                c5 = int(usage.get("cache_creation_input_tokens") or 0)
            row = [
                int(usage.get("input_tokens") or 0),
                int(usage.get("output_tokens") or 0),
                int(usage.get("cache_read_input_tokens") or 0),
                c5,
                c1h,
            ]
            # The orchestrator's model is only a fallback: delegated agents may
            # use cheaper models, and their transcript records the actual model.
            model = msg.get("model") or override_model
            req = msg.get("id") or d.get("requestId") or d.get("uuid")
            if req:
                prev = by_req.get(req)
                if prev is None:
                    by_req[req] = [model, *row]
                else:
                    prev[0] = prev[0] or model
                    for i in range(len(row)):
                        if row[i] > prev[i + 1]:
                            prev[i + 1] = row[i]
            else:
                noreq.append([model, *row])


def _authoritative_cost(log_dir: Path) -> float | None:
    """Read the exact cost from <log_dir>/cli_output.json, if a run left one.

    Headless runs (claude -p --output-format json) write this file, and its
    total_cost_usd is Claude Code's own figure — use it instead of the token
    estimate. Interactive runs don't produce the file, so return None and the
    caller keeps the estimate. The value may be at the top level or under a
    "result" key.
    """
    f = log_dir / "cli_output.json"
    if not f.exists():
        return None
    try:
        doc = json.loads(f.read_text())
    except Exception:
        return None
    for scope in (doc, doc.get("result") if isinstance(doc, dict) else None):
        if isinstance(scope, dict):
            for key in ("total_cost_usd", "cost_usd"):
                v = scope.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
    return None


def _otel_cost(
    sink_path: str | None,
    session_id: str | None,
    since_dt: datetime | None,
) -> float | None:
    """Add up this session's cost from the OTEL receiver's sink file.

    The receiver appends one line per cost datapoint ({session_id, ts, cost_usd}).
    Each line is an increment, so the session's total is simply their sum. This is
    Claude Code's own cost — it includes the subagent and background spend the
    transcripts miss — so it wins over the token estimate. Returns None when there's
    no sink, no session, no matching line, or a zero total (the caller then keeps the
    estimate). If since is given, only lines at or after it are counted.
    """
    if not sink_path or not session_id:
        return None
    p = Path(sink_path)
    if not p.exists():
        return None
    # Integer nanoseconds: ts is an integer-nanos string, so compare int-to-int and
    # avoid float64 rounding (ns values ~1.7e18 exceed float's exact-integer range).
    lo = int(since_dt.timestamp()) * 1_000_000_000 if since_dt else None
    total = 0.0
    hit = False
    with p.open(errors="ignore") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("session_id") != session_id:
                continue
            ts = r.get("ts")
            if lo is not None and ts is not None:
                try:
                    tsn = int(ts)
                except (TypeError, ValueError):
                    tsn = None
                if tsn is not None and tsn < lo:
                    continue
            c = r.get("cost_usd")
            if isinstance(c, (int, float)):
                total += float(c)
                hit = True
    # Require a positive total: matched-but-all-zero datapoints shouldn't override a
    # non-zero token estimate with $0.00 — fall back to the estimate instead.
    return total if hit and total > 0 else None


def _linked_sessions(log_dir: Path, root_session_id: str) -> list[dict]:
    """Read explicit, run-bound session links; never infer them from cwd/time."""
    path = log_dir / "session_registry.json"
    if not path.exists():
        return []
    registry = json.loads(path.read_text())
    run = json.loads((log_dir / "run.json").read_text())
    if (
        not isinstance(registry, dict)
        or registry.get("schema") != "issue-solver.session-registry"
        or registry.get("version") != 1
    ):
        raise ValueError("invalid session_registry.json schema/version")
    if (
        not isinstance(run, dict)
        or not run.get("run_id")
        or registry.get("run_id") != run["run_id"]
    ):
        raise ValueError("session registry run_id does not match run.json")
    entries = registry.get("sessions")
    if not isinstance(entries, list):
        raise ValueError("session registry sessions must be a list")
    linked = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("session registry entry must be an object")
        sid = entry.get("session_id")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sid):
            raise ValueError("invalid linked session_id")
        if sid == root_session_id:
            continue
        cwd = entry.get("project_cwd")
        if not isinstance(cwd, str) or not Path(cwd).is_absolute():
            raise ValueError(f"linked session {sid} needs an absolute project_cwd")
        if not isinstance(entry.get("parent_session_id"), str):
            raise ValueError(f"linked session {sid} needs a parent_session_id")
        normalized = dict(
            session_id=sid,
            project_cwd=cwd,
            parent_session_id=entry.get("parent_session_id"),
            kind=entry.get("kind") or "linked",
        )
        if sid in linked and linked[sid] != normalized:
            raise ValueError(f"conflicting registrations for session {sid}")
        linked[sid] = normalized
    # Reject disconnected links/cycles rather than silently pricing another run.
    for sid in linked:
        visited = set()
        current = sid
        while current != root_session_id:
            if current in visited or current not in linked:
                raise ValueError(f"session {sid} is not linked to the main session")
            visited.add(current)
            current = linked[current]["parent_session_id"]
    return list(linked.values())


def _sum_usage(rows: list) -> dict:
    totals = dict(
        input=0, output=0, cache_read=0, cache_creation=0, total=0, cost_usd=0.0
    )
    for model, u_in, u_out, u_cr, u_c5, u_c1h in rows:
        totals["input"] += u_in
        totals["output"] += u_out
        totals["cache_read"] += u_cr
        totals["cache_creation"] += u_c5 + u_c1h
        p = PRICING[_tier(model)]
        totals["cost_usd"] += (
            u_in * p["input"]
            + u_out * p["output"]
            + u_cr * p["cache_read"]
            + u_c5 * p["cache_creation"]
            + u_c1h * p["cache_creation_1h"]
        ) / 1_000_000.0
    totals["total"] = totals["input"] + totals["output"]
    return totals


def _aggregate_roots(
    roots: list,
    since_dt: datetime | None,
    fallback_model: str | None,
    sink_path: str | None,
    log_dir: Path | None,
) -> dict:
    """Account each linked root once, preferring its native cost independently.

    Root cli_output.json covers only the main CLI tree. In particular it must
    not replace the aggregate and discard standalone debug sessions.
    """
    collected = []
    seen_sessions = set()
    global_messages = {}
    global_noreq = []
    for sid, main_jsonl, subs_dir, kind in roots:
        if sid in seen_sessions:
            continue
        seen_sessions.add(sid)
        by_req, noreq = {}, []
        files = [main_jsonl, *sorted(subs_dir.glob("agent-*.jsonl"))]
        for path in files:
            _collect(path, since_dt, fallback_model, by_req, noreq)
        native = _otel_cost(sink_path, sid, since_dt)
        source = "otel" if native is not None else "transcript"
        if native is None and kind == "main" and log_dir:
            native = _authoritative_cost(log_dir)
            if native is not None:
                source = "cli_output"
        present = any(path.exists() for path in files)
        if native is None and not present:
            source = "missing"
        collected.append(
            dict(
                session_id=sid,
                kind=kind,
                source=source,
                native=native,
                messages=by_req,
                noreq=noreq,
                transcript_available=present,
            )
        )
        for key, row in by_req.items():
            if key not in global_messages:
                global_messages[key] = row.copy()
            else:
                old = global_messages[key]
                for i in range(1, len(row)):
                    old[i] = max(old[i], row[i])
        global_noreq.extend(noreq)
    totals = _sum_usage(list(global_messages.values()) + global_noreq)
    # If a transcript turn appears in more than one linked tree, assign it once.
    # Prefer a tree whose native cost already covers it over fallback estimates.
    claimed = set()
    costs = {}
    for group in sorted(collected, key=lambda x: x["native"] is None):
        unique = [global_messages[k] for k in group["messages"] if k not in claimed]
        claimed.update(group["messages"])
        estimate = _sum_usage(unique + group["noreq"])["cost_usd"]
        costs[group["session_id"]] = (
            group["native"] if group["native"] is not None else estimate
        )
    totals["cost_usd"] = round(sum(costs.values()), 6)
    totals["cost_accounting"] = {
        "schema": "issue-solver.cost-accounting",
        "version": 1,
        "complete": all(g["source"] != "missing" for g in collected),
        "tokens_complete": all(g["transcript_available"] for g in collected),
        "sessions": [
            dict(
                session_id=g["session_id"],
                kind=g["kind"],
                source=g["source"],
                cost_usd=round(costs[g["session_id"]], 6),
                transcript_available=g["transcript_available"],
            )
            for g in collected
        ],
    }
    return totals


def _patch_run_json(log_dir: Path, totals: dict) -> None:
    run_json = log_dir / "run.json"
    if not run_json.exists():
        return
    doc = json.loads(run_json.read_text())
    doc["tokens"] = {
        "input": totals["input"],
        "output": totals["output"],
        "cache_read": totals["cache_read"],
        "cache_creation": totals["cache_creation"],
        "total": totals["total"],
        "cost_usd": totals["cost_usd"],
    }
    doc["cost_usd"] = totals["cost_usd"]
    doc["cost_accounting"] = totals["cost_accounting"]
    fd, tmp = tempfile.mkstemp(prefix=".run.json.", suffix=".tmp", dir=str(log_dir))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        os.chmod(tmp, 0o664)
        os.replace(tmp, str(run_json))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--since",
        default=None,
        help="ISO 8601 start; only usage after this is counted.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Fallback model when a transcript message has no model (normally derived per message).",
    )
    ap.add_argument(
        "--session-pid",
        default=os.environ.get("CLAUDE_SESSION_PID") or os.environ.get("PPID"),
        help="PID of the claude CLI process (default: $PPID).",
    )
    ap.add_argument(
        "--session-id",
        default=None,
        help="Explicit session UUID; resolved by globbing ~/.claude/projects/ (overrides PID discovery).",
    )
    ap.add_argument(
        "--project-cwd",
        default=None,
        help="CWD that maps to the project dir under ~/.claude/projects/ (optional with --session-id).",
    )
    ap.add_argument(
        "--log-dir",
        default=None,
        help="If set, patch run.json atomically with the aggregated tokens + cost_usd.",
    )
    ap.add_argument(
        "--otel-sink",
        default=os.environ.get("CODEGEN_OTEL_SINK"),
        help="JSONL of claude_code.cost.usage datapoints from otel_cost_receiver.py "
        "(or $CODEGEN_OTEL_SINK). When it has cost for this session, that authoritative "
        "figure replaces the token estimate.",
    )
    ap.add_argument(
        "--print-session",
        action="store_true",
        default=False,
        help="Print '<session_id> <project_cwd>' to stdout and exit. Used by the orchestrator "
        "to capture the session identity at startup so refresh_cost.sh can pass it "
        "explicitly on later calls (when PID-based discovery may pick the wrong session).",
    )
    ap.add_argument(
        "--print-model",
        action="store_true",
        default=False,
        help="Print the running session's full model id (e.g. claude-opus-4-8) from its "
        "most recent turn and exit; prints nothing if undeterminable. Used by the "
        "orchestrator to record the model actually running in run.json instead of a "
        "hard-coded default.",
    )
    args = ap.parse_args(argv)

    since_dt = _parse_ts(args.since) if args.since else None

    session_id = args.session_id or os.environ.get("CLAUDE_CODE_SESSION_ID")
    found_by_id = _find_by_session_id(session_id) if session_id else None

    if found_by_id:
        main_jsonl, subs_dir = found_by_id
        discovered_sid = session_id
        discovered_cwd = args.project_cwd or os.getcwd()
    elif session_id:
        # An explicit identity must never fall back to somebody else's active
        # session when its transcript has not appeared or has been removed.
        discovered_sid = session_id
        discovered_cwd = args.project_cwd or os.getcwd()
        main_jsonl, subs_dir = _build_paths(session_id, discovered_cwd)
    else:
        found = _discover_session(args.session_pid)
        if not found:
            if args.print_session:
                print(" ")  # empty pair — caller checks for blank
            elif args.print_model:
                print("")  # undeterminable — caller falls back to its default
            else:
                print(
                    json.dumps(
                        dict(
                            input=0,
                            output=0,
                            cache_read=0,
                            cache_creation=0,
                            total=0,
                            cost_usd=0.0,
                        )
                    )
                )
            return 0
        discovered_sid, main_jsonl, subs_dir = found
        discovered_cwd = str(main_jsonl.parent.parent.name).replace("-", "/")
        home = Path(os.path.expanduser("~"))
        sessions_dir = home / ".claude" / "sessions"
        for f in sessions_dir.glob("*.json"):
            try:
                meta = json.loads(f.read_text())
                if meta.get("sessionId") == discovered_sid and meta.get("cwd"):
                    discovered_cwd = meta["cwd"]
                    break
            except Exception:
                pass

    if args.print_session:
        print(f"{discovered_sid} {discovered_cwd}")
        return 0

    if args.print_model:
        print(_last_model(main_jsonl) or "")
        return 0

    try:
        linked = (
            _linked_sessions(Path(args.log_dir), discovered_sid) if args.log_dir else []
        )
    except (ValueError, OSError) as exc:
        print(f"session_cost: {exc}", file=sys.stderr)
        return 2
    roots = [(discovered_sid, main_jsonl, subs_dir, "main")]
    for entry in linked:
        child_main, child_subs = _build_paths(entry["session_id"], entry["project_cwd"])
        roots.append((entry["session_id"], child_main, child_subs, entry["kind"]))
    totals = _aggregate_roots(
        roots,
        since_dt,
        args.model,
        args.otel_sink,
        Path(args.log_dir) if args.log_dir else None,
    )

    if args.log_dir:
        _patch_run_json(Path(args.log_dir), totals)

    print(json.dumps(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())

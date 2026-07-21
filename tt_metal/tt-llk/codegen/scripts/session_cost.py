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
jsonl, optionally filtered to entries with ``timestamp >= --since``, and
applies per-model Anthropic pricing to compute ``cost_usd``.

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
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Anthropic public pricing (USD per 1M tokens) — Claude 4.x family.
PRICING = {
    "opus": {
        "input": 5.00,
        "output": 25.00,
        "cache_read": 0.50,
        "cache_creation": 6.25,
    },
    "sonnet": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_creation": 3.75,
    },
    "haiku": {
        "input": 1.00,
        "output": 5.00,
        "cache_read": 0.10,
        "cache_creation": 1.25,
    },
}


def _tier(model_str: str | None) -> str:
    m = (model_str or "").lower()
    if "opus" in m:
        return "opus"
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


def _aggregate(
    jsonl_path: Path, since_dt: datetime | None, override_model: str | None
) -> dict:
    inp = out = cr = cc = 0
    cost = 0.0
    if not jsonl_path.exists():
        return dict(
            input=0, output=0, cache_read=0, cache_creation=0, total=0, cost_usd=0.0
        )

    # Claude Code writes each completed API response 2-4 times to the JSONL
    # (once per persistence event). Deduplicate by requestId so we count each
    # turn exactly once. Entries without a requestId are kept as-is.
    seen_req_ids: set[str] = set()

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
            if not usage:
                continue
            req_id = d.get("requestId")
            if req_id:
                if req_id in seen_req_ids:
                    continue
                seen_req_ids.add(req_id)
            if since_dt is not None:
                ts = _parse_ts(d.get("timestamp"))
                if ts is not None and ts < since_dt:
                    continue
            u_in = int(usage.get("input_tokens") or 0)
            u_out = int(usage.get("output_tokens") or 0)
            u_cr = int(usage.get("cache_read_input_tokens") or 0)
            u_cc = int(usage.get("cache_creation_input_tokens") or 0)
            inp += u_in
            out += u_out
            cr += u_cr
            cc += u_cc
            tier = _tier(override_model or msg.get("model"))
            p = PRICING[tier]
            cost += (
                u_in * p["input"]
                + u_out * p["output"]
                + u_cr * p["cache_read"]
                + u_cc * p["cache_creation"]
            ) / 1_000_000.0

    return dict(
        input=inp,
        output=out,
        cache_read=cr,
        cache_creation=cc,
        total=inp + out,
        cost_usd=round(cost, 6),
    )


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
        help="Override model tier: opus|sonnet|haiku (default: derived per message).",
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
    elif args.session_id and args.project_cwd:
        main_jsonl, subs_dir = _build_paths(args.session_id, args.project_cwd)
        discovered_sid = args.session_id
        discovered_cwd = args.project_cwd
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

    totals = dict(
        input=0, output=0, cache_read=0, cache_creation=0, total=0, cost_usd=0.0
    )

    main_t = _aggregate(main_jsonl, since_dt, args.model)
    for k in ("input", "output", "cache_read", "cache_creation"):
        totals[k] += main_t[k]
    totals["cost_usd"] += main_t["cost_usd"]

    if subs_dir.is_dir():
        for sub in subs_dir.glob("*.jsonl"):
            sub_t = _aggregate(sub, since_dt, args.model)
            for k in ("input", "output", "cache_read", "cache_creation"):
                totals[k] += sub_t[k]
            totals["cost_usd"] += sub_t["cost_usd"]

    totals["total"] = totals["input"] + totals["output"]
    totals["cost_usd"] = round(totals["cost_usd"], 6)

    if args.log_dir:
        _patch_run_json(Path(args.log_dir), totals)

    print(json.dumps(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())

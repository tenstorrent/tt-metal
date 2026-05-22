# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Extract per-subagent reasoning, tool calls, and commands into $LOG_DIR/transcripts/.

Claude Code stores every main-session turn at
``~/.claude/projects/<cwd-mapped>/<sessionId>.jsonl`` and every sub-agent at
``~/.claude/projects/<cwd-mapped>/<sessionId>/subagents/agent-<id>.jsonl``.
Sub-agent transcripts are normally consumed only by the dashboard's token/cost
aggregator (``session_cost.py``). This script re-parses the same transcripts to
produce human-readable per-agent markdown files that survive past the session:

  $LOG_DIR/transcripts/INDEX.md
  $LOG_DIR/transcripts/01_{slug}_reasoning.md
  $LOG_DIR/transcripts/01_{slug}_tools.md
  $LOG_DIR/transcripts/01_{slug}_commands.md
  $LOG_DIR/transcripts/02_{slug}_reasoning.md
  ...

``reasoning``  — chronology of assistant text + ``thinking`` blocks + tool
                 invocations + trimmed tool results, in order. Reads like a
                 transcript the next engineer can scroll through without having
                 to open the raw jsonl.

``tools``      — one-row-per-call table with sequence number, tool name,
                 summarized target (file path, bash description, Confluence page
                 id, etc.), ok/err status, and result byte count. Plus a
                 per-tool histogram at the top.

``commands``   — flat lists of Bash commands (verbatim, with description),
                 Confluence pages fetched, CQL searches, DeepWiki questions,
                 files read / written / edited, glob patterns, grep patterns.
                 Optimized for "what did this agent touch" audits.

``INDEX.md``   — one-line-per-agent summary with tool call counts and time
                 range, plus relative links to the three per-agent files.

Session discovery mirrors ``session_cost.py``: pass ``--session-id`` +
``--project-cwd`` explicitly, or let the script read
``~/.claude/sessions/<pid>.json`` using ``$CLAUDE_SESSION_PID`` / ``$PPID``
(falling back to the most-recently-started session). Run once from the
orchestrator at the end of Step 5e; all writes are idempotent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Session discovery (mirrors session_cost.py — kept independent to avoid
# circular coupling if either script moves).
# --------------------------------------------------------------------------


def _build_paths(session_id: str, cwd: str) -> tuple[Path, Path]:
    home = Path(os.path.expanduser("~"))
    proj_name = cwd.replace("_", "-").replace("/", "-")
    proj_dir = home / ".claude" / "projects" / proj_name
    return proj_dir / f"{session_id}.jsonl", proj_dir / session_id / "subagents"


def _discover_session(preferred_pid: str | None) -> tuple[str, str] | None:
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
        for _, sid, cwd, pid in candidates:
            if pid == str(preferred_pid):
                return sid, cwd
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, sid, cwd, _ = candidates[0]
        return sid, cwd
    return None


# --------------------------------------------------------------------------
# Transcript parsing
# --------------------------------------------------------------------------


def _truncate(s: Any, limit: int = 600) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = json.dumps(s, default=str)
        except Exception:
            s = str(s)
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n… [truncated, {len(s) - limit} more chars]"


def _flatten_tool_result_content(content: Any) -> str:
    """tool_result.content may be a string or a list of sub-blocks — flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for c in content:
            if not isinstance(c, dict):
                continue
            t = c.get("type")
            if t == "text":
                parts.append(c.get("text", ""))
            elif t == "tool_reference":
                parts.append(f"[tool_reference: {c.get('tool_name', '')}]")
            else:
                parts.append(json.dumps(c, default=str))
        return "\n".join(parts)
    return json.dumps(content, default=str)


def _parse_subagent(jsonl_path: Path, meta_path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    turns: list[dict] = []
    with jsonl_path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("type")
            if t not in ("user", "assistant"):
                continue
            msg = d.get("message") or {}
            content = msg.get("content")
            ts = d.get("timestamp")

            if t == "user":
                if isinstance(content, str):
                    turns.append({"kind": "user_text", "ts": ts, "text": content})
                elif isinstance(content, list):
                    results: list[dict] = []
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        if c.get("type") == "tool_result":
                            results.append(
                                {
                                    "tool_use_id": c.get("tool_use_id"),
                                    "is_error": bool(c.get("is_error")),
                                    "text": _flatten_tool_result_content(
                                        c.get("content")
                                    ),
                                }
                            )
                    if results:
                        turns.append(
                            {"kind": "tool_results", "ts": ts, "results": results}
                        )
            else:  # assistant
                blocks: list[dict] = []
                if isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        ct = c.get("type")
                        if ct == "text":
                            blocks.append({"block": "text", "text": c.get("text", "")})
                        elif ct == "thinking":
                            blocks.append(
                                {"block": "thinking", "text": c.get("thinking", "")}
                            )
                        elif ct == "tool_use":
                            blocks.append(
                                {
                                    "block": "tool_use",
                                    "id": c.get("id"),
                                    "name": c.get("name"),
                                    "input": c.get("input", {}),
                                }
                            )
                model = msg.get("model")
                turns.append(
                    {"kind": "assistant", "ts": ts, "model": model, "blocks": blocks}
                )

    return meta, turns


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def _tool_target(name: str | None, inp: Any) -> str:
    """One-line summary of the tool's intent, for table cells / reasoning headers."""
    if not isinstance(inp, dict):
        return ""
    n = name or ""
    if n == "Bash":
        cmd = inp.get("command") or ""
        first = cmd.splitlines()[0] if cmd else ""
        desc = inp.get("description") or ""
        return f"{desc}: {first}" if desc else first
    if n in ("Read", "Write"):
        return str(inp.get("file_path") or "")
    if n == "Edit":
        fp = inp.get("file_path") or ""
        os_s = (inp.get("old_string") or "")[:60].replace("\n", " ")
        return f"{fp} :: {os_s}"
    if n == "Glob":
        return str(inp.get("pattern") or "")
    if n == "Grep":
        pat = str(inp.get("pattern") or "")
        path = str(inp.get("path") or "")
        return f"{pat} in {path}" if path else pat
    if n == "ToolSearch":
        return str(inp.get("query") or "")
    if n.startswith("mcp__atlassian__getConfluencePage"):
        return f"pageId={inp.get('pageId', '')}"
    if n.startswith("mcp__atlassian__searchConfluenceUsingCql"):
        return str(inp.get("cql") or "")
    if n.startswith("mcp__atlassian__getConfluencePageDescendants"):
        return f"parentId={inp.get('parentId', '')}"
    if n.startswith("mcp__deepwiki__ask_question"):
        q = str(inp.get("question") or "")
        return f"{inp.get('repoName', '')}: {q}"
    if n.startswith("mcp__deepwiki__"):
        return str(inp.get("repoName") or "")
    if n == "TaskCreate":
        return str(inp.get("subject") or "")
    return ""


def _render_tool_input_block(name: str | None, inp: Any) -> str:
    """Pretty-print tool input for the reasoning chronology."""
    if not isinstance(inp, dict):
        return "```\n" + _truncate(inp, 400) + "\n```"
    n = name or ""
    if n == "Bash":
        cmd = inp.get("command", "")
        desc = inp.get("description", "")
        header = f"# {desc}\n" if desc else ""
        return "```bash\n" + header + _truncate(cmd, 1400) + "\n```"
    if n == "Write":
        fp = inp.get("file_path", "")
        body = inp.get("content", "")
        return f"Write `{fp}`\n\n" "```\n" + _truncate(body, 800) + "\n```"
    if n == "Edit":
        fp = inp.get("file_path", "")
        old = inp.get("old_string", "")
        new = inp.get("new_string", "")
        return (
            f"Edit `{fp}`\n\n"
            "old:\n```\n"
            + _truncate(old, 400)
            + "\n```\nnew:\n```\n"
            + _truncate(new, 400)
            + "\n```"
        )
    if n == "Read":
        fp = inp.get("file_path", "")
        off = inp.get("offset")
        lim = inp.get("limit")
        suffix = f" (offset={off}, limit={lim})" if (off or lim) else ""
        return f"Read `{fp}`{suffix}"
    # Default: dump as JSON
    return (
        "```json\n" + _truncate(json.dumps(inp, indent=2, default=str), 700) + "\n```"
    )


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name or "").strip("_").lower() or "agent"


# --------------------------------------------------------------------------
# Per-agent writers
# --------------------------------------------------------------------------


def _write_reasoning(path: Path, meta: dict, turns: list[dict]) -> None:
    lines: list[str] = []
    desc = meta.get("description", "Subagent")
    lines.append(f"# {desc} — reasoning chronology")
    lines.append("")
    lines.append(f"- Agent type: `{meta.get('agentType', 'unknown')}`")
    lines.append(f"- Transcript turns: {len(turns)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Initial user prompt
    initial = next((t for t in turns if t["kind"] == "user_text"), None)
    if initial:
        lines.append("## Initial task")
        lines.append("")
        lines.append("```")
        lines.append(_truncate(initial["text"], 4000))
        lines.append("```")
        lines.append("")

    # Index tool_use_id -> tool_result for quick lookup while rendering.
    id_to_result: dict[str, dict] = {}
    for t in turns:
        if t["kind"] == "tool_results":
            for r in t["results"]:
                if r.get("tool_use_id"):
                    id_to_result[r["tool_use_id"]] = r

    step = 0
    for turn in turns:
        if turn["kind"] != "assistant":
            continue
        ts = turn.get("ts", "") or ""
        model = turn.get("model") or ""
        for b in turn["blocks"]:
            bt = b["block"]
            if bt == "text":
                txt = (b.get("text") or "").strip()
                if not txt:
                    continue
                step += 1
                lines.append(f"### Step {step} — assistant ({model}, {ts})")
                lines.append("")
                lines.append(txt)
                lines.append("")
            elif bt == "thinking":
                txt = (b.get("text") or "").strip()
                if not txt:
                    continue
                step += 1
                lines.append(f"### Step {step} — thinking ({ts})")
                lines.append("")
                lines.append("> " + txt.replace("\n", "\n> "))
                lines.append("")
            elif bt == "tool_use":
                step += 1
                tname = b.get("name") or ""
                target = _tool_target(tname, b.get("input"))
                hdr = f"### Step {step} — tool `{tname}`"
                if target:
                    hdr += f" — {target[:140]}"
                lines.append(hdr)
                lines.append("")
                lines.append(_render_tool_input_block(tname, b.get("input")))
                lines.append("")
                # Matching result (may arrive in a later turn)
                r = id_to_result.get(b.get("id"))
                if r is not None:
                    status = "error" if r.get("is_error") else "ok"
                    lines.append(f"**Result ({status}):**")
                    lines.append("")
                    lines.append("```")
                    lines.append(_truncate(r.get("text", ""), 1200))
                    lines.append("```")
                    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def _write_tools(path: Path, meta: dict, turns: list[dict]) -> None:
    id_to_result: dict[str, dict] = {}
    for t in turns:
        if t["kind"] == "tool_results":
            for r in t["results"]:
                if r.get("tool_use_id"):
                    id_to_result[r["tool_use_id"]] = r

    rows: list[tuple[int, str, str, str, int]] = []
    idx = 0
    for t in turns:
        if t["kind"] != "assistant":
            continue
        for b in t["blocks"]:
            if b["block"] != "tool_use":
                continue
            idx += 1
            name = b.get("name") or ""
            target = _tool_target(name, b.get("input"))
            r = id_to_result.get(b.get("id"))
            if r is None:
                status = "—"
                bytes_out = 0
            else:
                status = "err" if r.get("is_error") else "ok"
                bytes_out = len(r.get("text", ""))
            rows.append((idx, name, target, status, bytes_out))

    lines: list[str] = []
    lines.append(f"# {meta.get('description', 'Subagent')} — tool invocations")
    lines.append("")
    lines.append(f"Total calls: **{len(rows)}**")
    lines.append("")

    by_tool: dict[str, int] = {}
    for _, name, _, _, _ in rows:
        by_tool[name] = by_tool.get(name, 0) + 1
    if by_tool:
        lines.append("## Tool histogram")
        lines.append("")
        lines.append("| Tool | Calls |")
        lines.append("|---|---:|")
        for tool, count in sorted(by_tool.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"| `{tool}` | {count} |")
        lines.append("")

    lines.append("## Call log")
    lines.append("")
    lines.append("| # | Tool | Target | Result | Bytes |")
    lines.append("|---:|---|---|---|---:|")
    for i, name, target, status, bytes_out in rows:
        t = target.replace("|", "\\|").replace("\n", " ")[:180]
        lines.append(f"| {i} | `{name}` | {t} | {status} | {bytes_out} |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def _write_commands(path: Path, meta: dict, turns: list[dict]) -> None:
    bash_cmds: list[dict] = []
    reads: list[str] = []
    writes: list[str] = []
    edits: list[str] = []
    globs: list[str] = []
    greps: list[dict] = []
    confluence_pages: list[str] = []
    confluence_searches: list[str] = []
    confluence_descendants: list[str] = []
    deepwiki: list[dict] = []

    for t in turns:
        if t["kind"] != "assistant":
            continue
        for b in t["blocks"]:
            if b["block"] != "tool_use":
                continue
            name = b.get("name") or ""
            inp = b.get("input") or {}
            if not isinstance(inp, dict):
                continue
            if name == "Bash":
                bash_cmds.append(
                    {
                        "desc": inp.get("description", ""),
                        "cmd": inp.get("command", ""),
                    }
                )
            elif name == "Read":
                reads.append(str(inp.get("file_path", "")))
            elif name == "Write":
                writes.append(str(inp.get("file_path", "")))
            elif name == "Edit":
                edits.append(str(inp.get("file_path", "")))
            elif name == "Glob":
                globs.append(str(inp.get("pattern", "")))
            elif name == "Grep":
                greps.append(
                    {
                        "pattern": str(inp.get("pattern", "")),
                        "path": str(inp.get("path", "")),
                        "glob": str(inp.get("glob", "")),
                    }
                )
            elif name.startswith("mcp__atlassian__getConfluencePage"):
                confluence_pages.append(str(inp.get("pageId", "")))
            elif name.startswith("mcp__atlassian__searchConfluenceUsingCql"):
                confluence_searches.append(str(inp.get("cql", "")))
            elif name.startswith("mcp__atlassian__getConfluencePageDescendants"):
                confluence_descendants.append(str(inp.get("parentId", "")))
            elif name.startswith("mcp__deepwiki__"):
                deepwiki.append(
                    {
                        "name": name,
                        "repoName": str(inp.get("repoName", "")),
                        "question": str(inp.get("question", "")),
                    }
                )

    lines: list[str] = []
    lines.append(f"# {meta.get('description', 'Subagent')} — commands & artifacts")
    lines.append("")

    def _section(title: str, items: list[Any], render) -> None:
        if not items:
            return
        lines.append(f"## {title}")
        lines.append("")
        for i, it in enumerate(items, 1):
            lines.append(render(i, it))
        lines.append("")

    if bash_cmds:
        lines.append("## Bash commands")
        lines.append("")
        for i, c in enumerate(bash_cmds, 1):
            lines.append(f"### {i}. {c['desc'] or '(no description)'}")
            lines.append("")
            lines.append("```bash")
            lines.append(_truncate(c["cmd"], 2000))
            lines.append("```")
            lines.append("")

    _section(
        "Confluence pages fetched",
        confluence_pages,
        lambda i, it: f"- page `{it}`",
    )
    _section(
        "Confluence CQL searches",
        confluence_searches,
        lambda i, it: f"- `{_truncate(it, 200)}`",
    )
    _section(
        "Confluence page descendants",
        confluence_descendants,
        lambda i, it: f"- parent `{it}`",
    )
    _section(
        "DeepWiki questions",
        deepwiki,
        lambda i, it: f"- `{it['name']}` on `{it['repoName']}`: {_truncate(it['question'], 200)}",
    )

    if reads:
        lines.append("## Files read")
        lines.append("")
        seen: set[str] = set()
        for p in reads:
            if p in seen:
                continue
            seen.add(p)
            lines.append(f"- `{p}`")
        lines.append("")
    if writes:
        lines.append("## Files written")
        lines.append("")
        for p in writes:
            lines.append(f"- `{p}`")
        lines.append("")
    if edits:
        lines.append("## Files edited")
        lines.append("")
        seen = set()
        for p in edits:
            if p in seen:
                continue
            seen.add(p)
            lines.append(f"- `{p}`")
        lines.append("")
    if globs:
        lines.append("## Glob patterns")
        lines.append("")
        for g in globs:
            lines.append(f"- `{g}`")
        lines.append("")
    if greps:
        lines.append("## Grep patterns")
        lines.append("")
        for g in greps:
            extra = []
            if g["path"]:
                extra.append(f"path=`{g['path']}`")
            if g["glob"]:
                extra.append(f"glob=`{g['glob']}`")
            extras = (" — " + ", ".join(extra)) if extra else ""
            lines.append(f"- `{_truncate(g['pattern'], 180)}`{extras}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _tool_count(turns: list[dict]) -> int:
    n = 0
    for t in turns:
        if t["kind"] == "assistant":
            for b in t["blocks"]:
                if b["block"] == "tool_use":
                    n += 1
    return n


def _first_last_ts(turns: list[dict]) -> tuple[str, str]:
    first = ""
    last = ""
    for t in turns:
        ts = t.get("ts")
        if not ts:
            continue
        if not first:
            first = ts
        last = ts
    return first, last


def run(
    log_dir: str,
    session_id: str | None,
    project_cwd: str | None,
    session_pid: str | None,
) -> int:
    log_dir_path = Path(log_dir)
    transcripts_dir = log_dir_path / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if session_id and project_cwd:
        _, subs_dir = _build_paths(session_id, project_cwd)
    else:
        found = _discover_session(session_pid)
        if not found:
            print(
                "extract_run_transcripts: no claude session discovered via ~/.claude/sessions",
                file=sys.stderr,
            )
            return 1
        sid, cwd = found
        _, subs_dir = _build_paths(sid, cwd)

    if not subs_dir.is_dir():
        print(
            f"extract_run_transcripts: no subagents dir at {subs_dir}", file=sys.stderr
        )
        return 1

    subagents: list[dict] = []
    for jsonl in sorted(subs_dir.glob("agent-*.jsonl")):
        meta_path = jsonl.with_suffix(".meta.json")
        meta, turns = _parse_subagent(jsonl, meta_path)
        first_ts, last_ts = _first_last_ts(turns)
        subagents.append(
            {
                "jsonl": jsonl,
                "meta": meta,
                "turns": turns,
                "first_ts": first_ts,
                "last_ts": last_ts,
            }
        )

    # Chronological order by first turn.
    subagents.sort(key=lambda s: s["first_ts"] or "")

    index_rows: list[str] = []
    per_agent_sections: list[str] = []

    for i, sa in enumerate(subagents, 1):
        desc = sa["meta"].get("description", f"agent {i}")
        slug = _slug(desc or f"agent{i}")
        prefix = f"{i:02d}_{slug}"
        reasoning_path = transcripts_dir / f"{prefix}_reasoning.md"
        tools_path = transcripts_dir / f"{prefix}_tools.md"
        commands_path = transcripts_dir / f"{prefix}_commands.md"

        _write_reasoning(reasoning_path, sa["meta"], sa["turns"])
        _write_tools(tools_path, sa["meta"], sa["turns"])
        _write_commands(commands_path, sa["meta"], sa["turns"])

        tc = _tool_count(sa["turns"])
        agent_type = sa["meta"].get("agentType", "")
        desc_cell = desc.replace("|", "\\|")
        index_rows.append(
            f"| {i} | `{agent_type}` | {desc_cell} | {sa['first_ts']} | {sa['last_ts']} | {tc} |"
        )

        per_agent_sections.append(
            f"### {i}. {desc}\n\n"
            f"- [reasoning]({prefix}_reasoning.md)\n"
            f"- [tools]({prefix}_tools.md)\n"
            f"- [commands]({prefix}_commands.md)\n"
        )

    index_lines: list[str] = []
    index_lines.append("# Subagent transcript index")
    index_lines.append("")
    index_lines.append(f"Source: `{subs_dir}`")
    index_lines.append("")
    index_lines.append(f"Agents captured: **{len(subagents)}**")
    index_lines.append("")
    if index_rows:
        index_lines.append("| # | Type | Description | Start | End | Tool calls |")
        index_lines.append("|---:|---|---|---|---|---:|")
        index_lines.extend(index_rows)
        index_lines.append("")
    index_lines.append("## Per-agent files")
    index_lines.append("")
    index_lines.extend(per_agent_sections)

    (transcripts_dir / "INDEX.md").write_text("\n".join(index_lines) + "\n")

    print(f"extract_run_transcripts: {len(subagents)} subagent(s) → {transcripts_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--log-dir", required=True, help="Path to the run's LOG_DIR")
    ap.add_argument(
        "--session-id",
        default=None,
        help="Explicit session UUID (overrides PID discovery). Requires --project-cwd.",
    )
    ap.add_argument(
        "--project-cwd",
        default=None,
        help="CWD mapped under ~/.claude/projects/ (required with --session-id).",
    )
    ap.add_argument(
        "--session-pid",
        default=os.environ.get("CLAUDE_SESSION_PID") or os.environ.get("PPID"),
        help="PID of the claude CLI process (default: $CLAUDE_SESSION_PID or $PPID).",
    )
    args = ap.parse_args(argv)
    return run(args.log_dir, args.session_id, args.project_cwd, args.session_pid)


if __name__ == "__main__":
    sys.exit(main())

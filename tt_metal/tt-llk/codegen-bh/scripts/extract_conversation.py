#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Extract readable conversation logs from Claude CLI JSON output.

Parses the JSON array output from `claude -p --output-format json` and produces
markdown files with the conversation, split by agent threads.

Usage:
    python scripts/extract_conversation.py /path/to/run_dir/
    python scripts/extract_conversation.py /path/to/run_dir/ --json issue_880.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


def find_cli_json(run_dir: Path, json_name: str | None = None) -> Path:
    """Find the CLI JSON output file in a run directory."""
    if json_name:
        p = run_dir / json_name
        if p.exists():
            return p
    # Auto-detect: look for issue_*.json
    candidates = sorted(
        run_dir.glob("issue_*.json"), key=lambda p: p.stat().st_size, reverse=True
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No CLI JSON found in {run_dir}")


def parse_entries(data: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """Split entries into main thread and subagent threads."""
    main = []
    subagents: dict[str, list[dict]] = {}

    # First pass: identify subagent spawns by looking for Agent tool_use
    agent_ids: dict[str, str] = {}  # parent_tool_use_id -> agent description

    for entry in data:
        content = entry.get("content") or entry.get("message", {}).get("content", [])
        if isinstance(content, str):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") == "Agent"
            ):
                inp = block.get("input", {})
                desc = inp.get("description", "subagent")
                agent_ids[block["id"]] = desc

    # Second pass: split by thread
    for entry in data:
        parent_id = entry.get("parent_tool_use_id")
        if parent_id and parent_id in agent_ids:
            name = agent_ids[parent_id]
            subagents.setdefault(name, []).append(entry)
        else:
            main.append(entry)

    return main, subagents


def format_text_block(block: dict) -> str:
    """Format a text content block."""
    return block.get("text", "")


def format_thinking_block(block: dict) -> str:
    """Format a thinking content block."""
    text = block.get("thinking", block.get("text", ""))
    if len(text) > 500:
        text = text[:500] + f"\n\n... ({len(text)} chars total)"
    return f"<details><summary>Thinking</summary>\n\n{text}\n\n</details>"


def format_tool_use(block: dict) -> str:
    """Format a tool_use block concisely."""
    name = block.get("name", "unknown")
    inp = block.get("input", {})

    if name == "Read":
        return f"`Read` {inp.get('file_path', '?')}"
    elif name == "Write":
        path = inp.get("file_path", "?")
        content = inp.get("content", "")
        return f"`Write` {path} ({len(content)} chars)"
    elif name == "Edit":
        path = inp.get("file_path", "?")
        return f"`Edit` {path}"
    elif name == "Bash":
        cmd = inp.get("command", "?")
        if len(cmd) > 120:
            cmd = cmd[:120] + "..."
        return f"`Bash` `{cmd}`"
    elif name == "Grep":
        return f"`Grep` pattern=`{inp.get('pattern', '?')}` path={inp.get('path', '.')}"
    elif name == "Glob":
        return f"`Glob` `{inp.get('pattern', '?')}`"
    elif name == "Agent":
        desc = inp.get("description", "?")
        prompt = inp.get("prompt", "")
        if len(prompt) > 200:
            prompt = prompt[:200] + "..."
        return f"`Agent` **{desc}**\n> {prompt}"
    elif name == "TaskCreate":
        return f"`TaskCreate` {inp.get('subject', '?')}"
    elif name == "TaskUpdate":
        return f"`TaskUpdate` #{inp.get('taskId', '?')} → {inp.get('status', '?')}"
    else:
        inp_str = json.dumps(inp, indent=2)
        if len(inp_str) > 200:
            inp_str = inp_str[:200] + "..."
        return f"`{name}` {inp_str}"


def format_tool_result(block: dict) -> str:
    """Format a tool_result block concisely."""
    content = block.get("content", "")
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text", str(c)))
            else:
                parts.append(str(c))
        content = "\n".join(parts)
    content = str(content)
    if len(content) > 500:
        content = content[:500] + f"\n... ({len(content)} chars total)"
    return content


def render_thread(entries: list[dict]) -> str:
    """Render a list of conversation entries as markdown."""
    lines = []

    for entry in entries:
        role = entry.get("type", entry.get("role", "unknown"))
        content = entry.get("content") or entry.get("message", {}).get("content", [])

        if isinstance(content, str):
            if content.strip():
                lines.append(f"**{role}**: {content}\n")
            continue

        if not isinstance(content, list):
            continue

        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text = format_text_block(block)
                if text.strip():
                    parts.append(text)
            elif btype == "thinking":
                parts.append(format_thinking_block(block))
            elif btype == "tool_use":
                parts.append(f"→ {format_tool_use(block)}")
            elif btype == "tool_result":
                result = format_tool_result(block)
                if result.strip():
                    parts.append(f"```\n{result}\n```")

        if parts:
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
                if isinstance(b, dict)
            )
            if has_tool_result:
                # Tool results — just show output, no heading
                lines.append("\n\n".join(parts))
                lines.append("")
            elif role == "assistant":
                lines.append("---\n")
                lines.append("\n\n".join(parts))
                lines.append("")
            else:
                # Initial prompt
                lines.append(f"### Prompt\n")
                lines.append("\n\n".join(parts))
                lines.append("")

    return "\n\n".join(lines)


def render_summary(data: list[dict], main: list[dict], subagents: dict) -> str:
    """Render a summary of the run."""
    last = data[-1] if data else {}

    num_turns = last.get("num_turns", 0)
    cost = last.get("total_cost_usd", 0)
    duration_ms = last.get("duration_ms", 0)
    duration_api_ms = last.get("duration_api_ms", 0)

    model_usage = last.get("modelUsage", {})

    # Count tool uses
    tool_counts: dict[str, int] = {}
    for entry in data:
        content = entry.get("content") or entry.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name", "unknown")
                tool_counts[name] = tool_counts.get(name, 0) + 1

    lines = [
        "# Run Summary\n",
        f"- **Entries**: {len(data)}",
        f"- **Turns**: {num_turns}",
        f"- **Duration**: {duration_ms // 1000}s wall / {duration_api_ms // 1000}s API",
        f"- **Cost**: ${cost:.4f}",
        f"- **Main thread entries**: {len(main)}",
        f"- **Subagents**: {len(subagents)}",
    ]

    for name, entries in subagents.items():
        lines.append(f"  - **{name}**: {len(entries)} entries")

    if model_usage:
        lines.append("\n## Model Usage\n")
        for model, usage in model_usage.items():
            inp = usage.get("inputTokens", 0)
            out = usage.get("outputTokens", 0)
            cache = usage.get("cacheReadInputTokens", 0)
            cost_m = usage.get("costUSD", 0)
            lines.append(
                f"- **{model}**: {inp:,} in / {out:,} out / {cache:,} cached — ${cost_m:.4f}"
            )

    if tool_counts:
        lines.append("\n## Tool Usage\n")
        for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- `{name}`: {count}")

    return "\n".join(lines) + "\n"


def safe_filename(name: str) -> str:
    """Convert agent description to safe filename."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower().strip())[:60].strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Extract conversation from CLI JSON output"
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing CLI JSON")
    parser.add_argument(
        "--json", default=None, help="JSON filename (auto-detected if omitted)"
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    cli_json = find_cli_json(run_dir, args.json)

    print(f"Parsing {cli_json} ({cli_json.stat().st_size / 1024:.0f} KB)...")

    with open(cli_json) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: expected JSON array", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(data)} entries")

    main_thread, subagents = parse_entries(data)
    print(f"  Main thread: {len(main_thread)} entries")
    print(f"  Subagents: {len(subagents)}")

    # Write summary
    summary = render_summary(data, main_thread, subagents)
    (run_dir / "summary.md").write_text(summary)
    print(f"  Wrote summary.md")

    # Write main conversation
    main_md = f"# Main Thread\n\n{render_thread(main_thread)}"
    (run_dir / "conversation.md").write_text(main_md)
    print(f"  Wrote conversation.md")

    # Write per-agent files
    for name, entries in subagents.items():
        fname = f"agent_{safe_filename(name)}.md"
        agent_md = f"# Agent: {name}\n\n{render_thread(entries)}"
        (run_dir / fname).write_text(agent_md)
        print(f"  Wrote {fname}")

    print("Done.")


if __name__ == "__main__":
    main()

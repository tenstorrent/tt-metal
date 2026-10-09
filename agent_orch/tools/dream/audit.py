"""Isolation audit: did a worker reach outside its campaign?

A worker may use its worktree, its campaign's refs, history.md and the brief. After each attempt the
driver scans the worker's transcript (stream-json) for tool calls that touch anything else:

    the user's main checkout on the machine (its python_env excepted), other campaigns under $DREAM_HOME,
    git network/remote operations, curl/wget/gh, and the web tools.

Findings are recorded in the round's decisions.jsonl ({"type": "audit"}) and shown on the node in the
report and history.md. With isolation.on_violation = invalidate the node is also overridden as invalid.
This is a tripwire for the common paths, not a sandbox.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .campaign import Campaign

GIT_REMOTE = re.compile(r"\bgit\b[^|;&\n]*?\b(fetch|clone|pull|ls-remote|remote|submodule\s+update)\b")
NETWORK = re.compile(r"(^|[\s;|&(])(curl|wget|gh)\s")
WEB_TOOLS = {"WebFetch", "WebSearch"}
ALLOWED_UNDER_MAIN = ("python_env",)
ALLOWED_UNDER_HOME = ("device.lock", ".ccache")


def tool_calls(log: Path):
    """(tool name, input dict) for every tool call in a stream-json transcript."""
    if not log.exists():
        return
    for line in log.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        msg = d.get("message") if isinstance(d, dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        if d.get("type") != "assistant" or not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                yield item.get("name", ""), item.get("input") or {}


def _outside(text: str, root: str, allowed: tuple[str, ...], own: str | None = None) -> list[str]:
    """Paths under `root` mentioned in `text` that are not root/<allowed...> (or root/<own>)."""
    hits = []
    for m in re.finditer(re.escape(root.rstrip("/")) + r"(?=[/\s'\"`;|&),:]|$)(/[^\s'\"`;|&),:]*)?", text):
        rest = (m.group(1) or "/").lstrip("/")
        first = rest.split("/", 1)[0]
        if first in allowed or (own and first == own):
            continue
        hits.append(m.group(0))
    return hits


def audit_transcript(c: Campaign, log: Path) -> list[str]:
    main = str(c.main_repo)
    home = str(c.dream_home)
    flags: list[str] = []
    for name, inp in tool_calls(log):
        if name in WEB_TOOLS:
            flags.append(f"{name}: {inp.get('url') or inp.get('query') or ''}"[:200])
            continue
        text = json.dumps(inp)
        cmd = inp.get("command", "") if isinstance(inp, dict) else ""
        if cmd and GIT_REMOTE.search(cmd):
            flags.append(f"git remote/network command: {cmd}"[:200])
        if cmd and NETWORK.search(cmd):
            flags.append(f"network command: {cmd}"[:200])
        for p in _outside(text, main, ALLOWED_UNDER_MAIN):
            flags.append(f"{name} touches the main checkout: {p}"[:200])
        for p in _outside(text, home, ALLOWED_UNDER_HOME, own=c.name):
            flags.append(f"{name} touches another campaign: {p}"[:200])
    seen, out = set(), []
    for f in flags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out[:20]

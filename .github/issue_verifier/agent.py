"""Headless Claude sessions used by the issue verifier.

Two distinct sessions exist, and the split is deliberate. The planner reads an
untrusted issue body and is given no tools, so nothing in the report can cause
side effects. The prober is given a shell but is only ever asked to produce
measurements — it never decides whether the issue is valid.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Read-only analysis of an issue body. No Bash, no Write: a report that tries to
# talk the planner into running something has nothing to reach for.
PLANNER_TOOLS = ["Read", "Grep", "Glob"]

# The prober runs probe scripts, so it needs a shell — but a scoped one. The
# probe itself is generated from an untrusted issue body, so the grant is
# limited to interpreters and read-only history commands rather than bare Bash.
PROBER_TOOLS = [
    "Read",
    "Grep",
    "Glob",
    "Write",
    "Edit",
    "Bash(python3 *)",
    "Bash(python *)",
    "Bash(git log *)",
    "Bash(git blame *)",
    "Bash(git show *)",
    "Bash(git diff *)",
    "Bash(rg *)",
]


class AgentFailed(RuntimeError):
    """The session could not be started, or exited without a usable result."""


@dataclass
class AgentResult:
    text: str
    cost_usd: float
    turns: int
    session_id: str


def _require_cli() -> str:
    cli = shutil.which("claude")
    if cli is None:
        raise AgentFailed("`claude` CLI not found on PATH. Install Claude Code, or set ANTHROPIC_API_KEY in CI.")
    return cli


def run_session(
    prompt: str,
    *,
    cwd: Path,
    tools: list[str],
    model: str | None = None,
    timeout_s: int = 1800,
) -> AgentResult:
    cmd = [_require_cli(), "-p", prompt, "--output-format", "json", "--allowedTools", *tools]
    if model:
        cmd += ["--model", model]

    logger.debug(f"launching claude session: tools={tools} cwd={cwd}")
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise AgentFailed(f"session exceeded {timeout_s}s") from exc

    # The CLI reports auth and API failures as structured JSON on stdout while
    # still exiting non-zero, so stdout has to be read before the exit code is
    # trusted — reporting only stderr yields a blank error message.
    payload: dict | None = None
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None

    if payload is not None and payload.get("is_error"):
        status = payload.get("api_error_status")
        detail = str(payload.get("result") or payload.get("terminal_reason") or "").strip()
        raise AgentFailed(f"session failed{f' (HTTP {status})' if status else ''}: {detail[:2000]}")

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "(no output on either stream)"
        raise AgentFailed(f"claude exited {completed.returncode}: {detail[:2000]}")

    if payload is None:
        raise AgentFailed(f"unparseable session output: {completed.stdout[:2000]}")

    return AgentResult(
        text=payload.get("result", ""),
        cost_usd=float(payload.get("total_cost_usd") or 0.0),
        turns=int(payload.get("num_turns") or 0),
        session_id=payload.get("session_id", ""),
    )


def load_prompt(name: str, **substitutions: str) -> str:
    """Load a prompt template and fill its ``{{placeholders}}``.

    Deliberately not ``str.format``: issue bodies routinely contain braces from
    code snippets, and a stray ``{`` should never blow up a verification run.
    """
    text = (PROMPTS_DIR / name).read_text()
    for key, value in substitutions.items():
        text = text.replace("{{" + key + "}}", value)
    return text

"""PLAN agent (PLAN 8.x) — the lead turns a chosen lever into a LOCALIZED edit spec.

Reads the lever's playbook guidance + the model map, reads the target file(s)
just-in-time (Read/Grep, no Edit), and emits content-anchored edits
{file, old_string, new_string} — exact find-and-replace the harness applies
deterministically (no editor LLM, no line numbers). If the optimization is
already present, returns edits=[] (summary "NOOP: ...") so the loop skips it.

Deterministic parts (prompt, validation) are unit-tested; the live LEAD call is
the same untested boundary as the other agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


class PlanError(Exception):
    """The plan agent returned a malformed edit spec."""


PROMPT_TEMPLATE = (
    "You are planning ONE performance optimization edit to a model.\n\n"
    "Lever: {lever}\n\n"
    "Guidance (playbook section):\n{section}\n\n"
    "Model map — where the ops live (paths are relative to the model root):\n{skeleton}\n\n"
    "Read the target file(s) to confirm the EXACT current code, then express the change "
    "as one or more precise find-and-replace edits. For each edit:\n"
    "  - old_string: copy the CURRENT code VERBATIM (exact whitespace/indentation), with "
    "enough surrounding lines that it appears EXACTLY ONCE in the file.\n"
    "  - new_string: the replacement (use an empty string to delete).\n"
    "Do NOT use line numbers. Output EXACTLY ONE JSON object and nothing else:\n"
    '  {{"summary": <one sentence>, "edits": [{{"file": <repo-relative path>, '
    '"old_string": <exact current code>, "new_string": <replacement or empty string>}}]}}\n'
    'If this optimization is ALREADY present, return {{"summary": "NOOP: already applied", "edits": []}}.'
)


def build_plan_prompt(lever: str, section: str, skeleton: str) -> str:
    return PROMPT_TEMPLATE.format(
        lever=lever or "(unspecified)",
        section=section or "(no playbook text)",
        skeleton=skeleton or "(no map)",
    )


def _validate_spec(raw: Any) -> dict:
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError) as exc:
        raise PlanError(f"plan agent returned invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise PlanError("edit spec must be a JSON object")
    raw_edits = obj.get("edits")
    if not isinstance(raw_edits, list):
        raise PlanError("edit spec.edits must be a list (empty list = NOOP)")
    edits = []
    for i, e in enumerate(raw_edits):
        if not isinstance(e, dict):
            raise PlanError(f"edit[{i}] must be a JSON object")
        f, old = e.get("file"), e.get("old_string")
        if not isinstance(f, str) or not f.strip():
            raise PlanError(f"edit[{i}].file must be a non-empty path")
        if not isinstance(old, str) or old == "":
            raise PlanError(f"edit[{i}].old_string must be a non-empty anchor")
        edits.append({"file": f, "old_string": old, "new_string": str(e.get("new_string", ""))})
    return {"summary": str(obj.get("summary", "")), "edits": edits}


def make_plan_runner(
    env_agent_path: str | Path = Path(__file__).parent.parent / ".env.agent",
    max_turns: int = 12,
) -> Callable[..., dict]:
    from .config import apply_agent_env, get_model

    resolved = apply_agent_env(env_agent_path)
    model = get_model("lead", resolved)

    def runner(*, lever: str, section: str, skeleton: str, cwd: str | None = None) -> dict:
        pass

        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        from .probes import _extract_json_object, _usage_summary

        prompt = build_plan_prompt(lever, section, skeleton)
        opts: dict = dict(
            model=model,
            system_prompt="You plan one localized code edit. Read to confirm; final message is one JSON object.",
            allowed_tools=["Read", "Grep", "Glob"],
            permission_mode="bypassPermissions",
            setting_sources=[],
            max_turns=max_turns,
        )
        if cwd:
            opts["cwd"] = cwd
        options = ClaudeAgentOptions(**opts)
        chunks: list[str] = []
        usage: dict = {}

        async def _go() -> None:
            async for msg in query(prompt=prompt, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            chunks.append(block.text)
                elif isinstance(msg, ResultMessage):
                    usage["u"] = _usage_summary(msg)

        from .sdk_retry import run_with_retry

        run_with_retry(_go, lambda: (chunks.clear(), usage.clear()))
        spec = _validate_spec(_extract_json_object("\n".join(chunks)))
        spec["model"] = model
        spec["usage"] = usage.get("u")
        spec["prompt"] = prompt
        spec["response"] = "\n".join(chunks)
        return spec

    return runner

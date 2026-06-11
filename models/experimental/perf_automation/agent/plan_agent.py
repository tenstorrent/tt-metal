"""PLAN agent (PLAN 8.x) — the lead turns a chosen lever into a LOCALIZED edit spec.

Reads the lever's playbook guidance + the model map, reads the target file(s)
just-in-time (Read/Grep, no Edit), and emits {file, location, change} — a precise,
minimal instruction the editor applies mechanically. If the optimization is
already present, returns change="NOOP: ..." so the loop skips a wasted edit.

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
    "You are planning ONE code edit to apply a performance optimization to a model.\n\n"
    "Lever: {lever}\n\n"
    "Guidance (playbook section):\n{section}\n\n"
    "Model map — where the ops live (paths are relative to the model root):\n{skeleton}\n\n"
    "Read the relevant file(s) to confirm the exact spot, then output EXACTLY ONE "
    "JSON object and nothing else:\n"
    '  {{"file": <repo-relative path to edit>, "location": <function + approx line, '
    'e.g. "BgeM3MLP.forward, the FF1 ttnn.linear (~L107)">, "change": <a precise, '
    "minimal instruction an engineer can apply directly>}}\n"
    'If this optimization is ALREADY present in the code, set "change" to '
    '"NOOP: already applied" (and still name the file/location).'
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
    file, change = obj.get("file"), obj.get("change")
    if not isinstance(file, str) or not file.strip():
        raise PlanError("edit spec.file must be a non-empty path")
    if not isinstance(change, str) or not change.strip():
        raise PlanError("edit spec.change must be a non-empty instruction")
    return {"file": file, "location": str(obj.get("location", "")), "change": change}


def make_plan_runner(
    env_agent_path: str | Path = Path(__file__).parent.parent / ".env.agent",
    max_turns: int = 12,
) -> Callable[..., dict]:
    from .config import apply_agent_env, get_model

    resolved = apply_agent_env(env_agent_path)
    model = get_model("lead", resolved)

    def runner(*, lever: str, section: str, skeleton: str, cwd: str | None = None) -> dict:
        import asyncio

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

        asyncio.run(_go())
        spec = _validate_spec(_extract_json_object("\n".join(chunks)))
        spec["model"] = model
        spec["usage"] = usage.get("u")
        return spec

    return runner

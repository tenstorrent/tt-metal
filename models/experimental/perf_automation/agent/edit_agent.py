"""edit_file sub-agent (PLAN 8.5) — applies ONE lever to the model source.

Same SDK plumbing as the discovery sub-agent (probes.sdk_model_files_runner),
but with the Edit tool, scoped to the model_files. Shared by APPLY (8.5) and
both REPAIR modes (8.5.2) — REPAIR just augments the prompt with the error.

Deterministic, unit-tested parts: the prompt builder and the result validator.
The live SDK call (make_edit_runner) is exercised on hardware, not in unit tests
(same boundary as model_files).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


class EditError(Exception):
    """The edit agent returned a malformed result."""


PROMPT_TEMPLATE = (
    "You are applying ONE performance optimization to a model, then stopping.\n\n"
    "Optimization (lever '{lever}'):\n{section}\n\n"
    "Edit ONLY these files (Read first, then Edit):\n{files}\n\n"
    "Rules:\n"
    "- Make the SMALLEST change that applies this one lever.\n"
    "- Preserve the model's numerical behavior; do not refactor unrelated code.\n"
    "- Do NOT delete or weaken the optimization just to avoid an error.\n"
    "When done, output exactly ONE JSON object and nothing else:\n"
    '  {{"files": [<repo-relative paths you changed>], "summary": <one sentence>}}'
)


def build_edit_prompt(lever: str, section: str, model_files: list) -> str:
    files = "\n".join(f"  - {f}" for f in model_files)
    return PROMPT_TEMPLATE.format(
        lever=lever or "(unspecified)",
        section=section or "(no playbook text available — apply the lever named above)",
        files=files,
    )


def _validate_edit_result(raw: Any) -> dict:
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError) as exc:
        raise EditError(f"edit agent did not return valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise EditError("edit result must be a JSON object")
    files = obj.get("files")
    if not isinstance(files, list) or not files:
        raise EditError("edit result.files must be a non-empty array of changed paths")
    return {"files": [str(f) for f in files], "summary": str(obj.get("summary", ""))}


def make_edit_runner(
    env_agent_path: str | Path = Path(__file__).parent.parent / ".env.agent",
    max_turns: int = 24,
) -> Callable[..., dict]:
    """Build the production editor: runner(lever, section, model_files) -> result.

    Fails fast (section 3.1) if .env.agent is missing, BEFORE any SDK call.
    """
    from .config import apply_agent_env, get_model

    resolved = apply_agent_env(env_agent_path)
    model = get_model("sub", resolved)

    def runner(*, lever: str, section: str, model_files: list) -> dict:
        import asyncio

        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        from .probes import _extract_json_object, _usage_summary

        prompt = build_edit_prompt(lever, section, [str(p) for p in model_files])
        options = ClaudeAgentOptions(
            model=model,
            system_prompt=(
                "You apply exactly one optimization to model source using Read and "
                "Edit. Your FINAL message must be one JSON object, no prose."
            ),
            allowed_tools=["Read", "Edit", "Glob", "Grep"],
            permission_mode="bypassPermissions",
            setting_sources=[],
            max_turns=max_turns,
        )
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
        result = _validate_edit_result(_extract_json_object("\n".join(chunks)))
        result["model"] = model
        result["usage"] = usage.get("u")
        return result

    return runner

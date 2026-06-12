"""SELECT agent (PLAN 8.3) — the LEAD picks ONE lever from ROUTE's brief.

The single reasoning step of the loop. ROUTE already assembled the decision
material (route_brief: bucket + candidate table + full section texts), so SELECT
is a no-tools judgment over that text: read it, commit one lever id from the
closed candidate list. Enum-constrained; the caller falls back to untried[0] on
an invalid pick or an API error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


class SelectError(Exception):
    """The agent's choice was malformed or not a valid untried candidate."""


PROMPT_TEMPLATE = (
    "You are the lead optimization agent choosing ONE change to try next.\n\n"
    "{brief}\n\n"
    "Already tried this run (do NOT pick these): {tried}\n"
    "Choose EXACTLY ONE lever id from this list (and no other): {candidates}\n"
    "Pick the lever most likely to cut the bottleneck's device time while keeping "
    "the model numerically correct. Prefer a single-shot lever before a search "
    "lever when both look promising.\n"
    'Respond with ONE JSON object only: {{"lever": <id from the list>, '
    '"reasoning": <one sentence: why this lever for this bottleneck>}}.'
)


def build_select_prompt(brief: str, candidates: list[str], tried: list[str]) -> str:
    return PROMPT_TEMPLATE.format(
        brief=brief or "(no brief available)",
        tried=", ".join(tried) or "(none)",
        candidates=", ".join(candidates),
    )


def _validate_choice(raw: Any, candidates: list[str], tried: list[str]) -> dict:
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError) as exc:
        raise SelectError(f"select agent returned invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise SelectError("select result must be a JSON object")
    lever = obj.get("lever")
    if lever not in candidates:
        raise SelectError(f"chosen lever {lever!r} is not in the candidate list")
    if lever in set(tried):
        raise SelectError(f"chosen lever {lever!r} was already tried")
    return {"lever": lever, "reasoning": str(obj.get("reasoning", ""))}


def make_select_runner(
    env_agent_path: str | Path = Path(__file__).parent.parent / ".env.agent",
    max_turns: int = 4,
) -> Callable[..., dict]:
    """Build the production lever-picker: runner(brief, candidates, tried) -> result.

    Uses the LEAD model, no tools (pure judgment over the brief). Fails fast
    (section 3.1) if .env.agent is missing.
    """
    from .config import apply_agent_env, get_model

    resolved = apply_agent_env(env_agent_path)
    model = get_model("lead", resolved)

    def runner(*, brief: str, candidates: list[str], tried: list[str]) -> dict:
        pass

        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        from .probes import _extract_json_object, _usage_summary

        prompt = build_select_prompt(brief, candidates, tried)
        options = ClaudeAgentOptions(
            model=model,
            system_prompt="You choose one optimization lever. Final message is one JSON object, no prose.",
            allowed_tools=[],
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

        from .sdk_retry import run_with_retry

        run_with_retry(_go, lambda: (chunks.clear(), usage.clear()))
        result = _validate_choice(_extract_json_object("\n".join(chunks)), candidates, tried)
        result["model"] = model
        result["usage"] = usage.get("u")
        result["prompt"] = prompt
        result["response"] = "\n".join(chunks)
        return result

    return runner

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

from . import states


class SelectError(Exception):
    """The agent's choice was malformed or not a valid untried candidate."""


PROMPT_TEMPLATE = (
    "You are the lead optimization agent choosing ONE change to try next.\n\n"
    "{brief}\n\n"
    "Already tried this run (do NOT pick these): {tried}\n"
    "Choose EXACTLY ONE id from this list (and no other): {candidates}\n"
    "The listed levers are PRIORS, not a requirement. Look at the bottleneck's hottest ops "
    "(`top_ops` in the brief) and pick the lever most likely to cut THAT op's device time while "
    "keeping the model numerically correct. Prefer a single-shot lever over a search lever when "
    "both fit.\n"
    "REGIME: the brief's `regime_verdict` diagnoses whether this is a KNOB problem or a KERNEL "
    "problem from the roofline `bound_by` + what's already been tried. If verdict=='kernel' "
    "(dispatch/launch-bound, or the op is at its single-op TTNN floor, or the knobs are spent), "
    "the remaining gap is structural — prefer the tt-lang kernel lever (fuse/dataflow) and do "
    "NOT grind untried knobs that can't move a kernel-level cost. If verdict=='knob', a TTNN-API "
    "config change is the cheaper bet — take the matching knob first. The verdict is strong "
    "guidance, not a hard rule; override it only if the top_ops clearly contradict it.\n"
    "IMPORTANT: if NO listed lever actually targets the dominant op (e.g. the cost is layout "
    "conversion / a tensor-format mismatch that none of the levers address), choose "
    "'{from_principles}' to reason from first principles instead of forcing a poor-fit lever.\n"
    "COVERAGE: you may also list ids you judge clearly IRRELEVANT to this bottleneck in `skip` "
    "— they'll be pruned so the loop doesn't waste device runs grinding through them. Skip only "
    "levers you're confident don't target the dominant op; when unsure, leave them unskipped.\n"
    'Respond with ONE JSON object only: {{"lever": <id to try now>, '
    '"skip": [<ids to prune, may be empty>], '
    '"reasoning": <one sentence: why this for this bottleneck>}}.'
)


def build_select_prompt(brief: str, candidates: list[str], tried: list[str]) -> str:
    return PROMPT_TEMPLATE.format(
        brief=brief or "(no brief available)",
        tried=", ".join(tried) or "(none)",
        candidates=", ".join(candidates),
        from_principles=states.FROM_PRINCIPLES,
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
    raw_skip = obj.get("skip") or []
    skip = [s for s in raw_skip if s in candidates and s != lever] if isinstance(raw_skip, list) else []
    return {"lever": lever, "reasoning": str(obj.get("reasoning", "")), "skip": skip}


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
            max_buffer_size=50 * 1024 * 1024,
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

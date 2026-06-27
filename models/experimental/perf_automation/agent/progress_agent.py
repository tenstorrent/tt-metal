"""progress-judge agent — the AGENTIC waste decision (replaces a dumb static rule).

Instead of mechanically trying every knob in a bucket, an agent reasons over the
MEASURED track record so far and decides whether the effort is being wasted:
  continue  — knobs still worth trying here
  exhaust   — this bucket is tapped (measured no-gain pattern); stop spending device
              measurements on its remaining knobs, advance to the next bucket/axis
  stop      — the whole model looks optimized; end the run

CRITICAL: the agent judges ONLY over measured evidence (the ledger's before/after/
delta/reason rows) — it never predicts whether an unmeasured edit will help (that
stays the deterministic measure). It reasons about the *pattern of results*, which
is exactly the strategic-waste call a static rule is blind to.

Same SDK plumbing as select_agent.make_select_runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

VALID = ("continue", "exhaust", "stop")

PROMPT_TEMPLATE = (
    "You are deciding whether to keep spending EXPENSIVE device measurements on a perf "
    "optimization bucket, or cut your losses. Each measurement is a slow (~85s) serial "
    "device run, so wasted attempts are costly.\n\n"
    "Current bucket: {bucket}\n"
    "Knobs still untried in this bucket: {untried}\n\n"
    "MEASURED results so far in this bucket (ground truth — these already ran on device):\n"
    "{history}\n\n"
    "Decide, based ONLY on this measured evidence (do NOT predict unmeasured edits):\n"
    "  - 'continue' : the results suggest a remaining knob could still help — keep going.\n"
    "  - 'exhaust'  : a clear no-gain / inert pattern — this bucket is tapped; stop trying its\n"
    "                 knobs and move to the next bucket/axis (saves wasted measurements).\n"
    "  - 'stop'     : every attempted bucket shows no headroom — the model looks optimized; end.\n\n"
    "Output exactly ONE JSON object, no prose:\n"
    '  {{"decision": "continue|exhaust|stop", "reasoning": "<one sentence grounded in the measured results>"}}'
)


def _fmt_history(history: list[dict]) -> str:
    if not history:
        return "  (none yet)"
    lines = []
    for r in history:
        d = r.get("delta")
        dl = f"{d:+.4f}" if isinstance(d, (int, float)) else "  --  "
        # `or '?'` (not get-default): a KEEP row carries reason=None as a PRESENT key, so
        # get('reason','?') returns None and f"{None:<12}" raises TypeError — the crash that
        # killed the nemotron run at the waste-judge. Coerce None -> '?' for all three.
        lines.append(
            f"  - {(r.get('lever') or '?'):<28} {(r.get('result') or '?'):<8} {(r.get('reason') or '?'):<12} "
            f"Δ {dl} (before {r.get('before')} -> after {r.get('after')})"
        )
    return "\n".join(lines)


def build_progress_prompt(bucket: str, untried: list[str], history: list[dict]) -> str:
    return PROMPT_TEMPLATE.format(
        bucket=bucket or "(unknown)",
        untried=", ".join(untried) if untried else "(none)",
        history=_fmt_history(history),
    )


def _validate(raw: Any) -> dict:
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return {"decision": "continue", "reasoning": "unparseable judge output; default continue"}
    dec = obj.get("decision") if isinstance(obj, dict) else None
    if dec not in VALID:
        return {"decision": "continue", "reasoning": f"invalid decision {dec!r}; default continue"}
    return {"decision": dec, "reasoning": str(obj.get("reasoning", ""))}


def make_progress_judge_runner(
    env_agent_path: str | Path = Path(__file__).parent.parent / ".env.agent",
    max_turns: int = 2,
) -> Callable[..., dict]:
    """runner(bucket, untried, history) -> {decision, reasoning, model, usage}.

    Fails open: any error / invalid output -> 'continue' (never blocks the loop on a
    judge failure; worst case it falls back to the deterministic try-all behavior)."""
    from .config import apply_agent_env, get_model

    resolved = apply_agent_env(env_agent_path)
    model = get_model("lead", resolved)  # strategy call -> lead tier

    def runner(*, bucket: str, untried: list[str], history: list[dict]) -> dict:
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

        from .probes import _extract_json_object, _usage_summary

        prompt = build_progress_prompt(bucket, untried, history)
        options = ClaudeAgentOptions(
            model=model,
            system_prompt=(
                "You judge whether a perf-optimization bucket is worth continuing, from MEASURED "
                "results only. Your FINAL message must be one JSON object, no prose."
            ),
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

        try:
            run_with_retry(_go, lambda: (chunks.clear(), usage.clear()))
            result = _validate(_extract_json_object("\n".join(chunks)))
        except Exception as exc:  # fail open
            result = {"decision": "continue", "reasoning": f"judge errored: {str(exc)[:120]}"}
        result["model"] = model
        result["usage"] = usage.get("u")
        return result

    return runner

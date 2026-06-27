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

from .structural_agent import (
    _DEVICE_MAX_TURNS,
)  # device-tool turn budget (cycle-safe: structural_agent does not import this)


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


REPAIR_TEMPLATE = (
    "Your previous attempt to apply lever '{lever}' FAILED:\n{error}\n\n"
    "{prior}"
    "Fix the problem. KEEP the optimization (below) — do NOT delete or weaken it "
    "to make the error go away.\n\n"
    "Optimization:\n{section}\n\n"
    "Edit ONLY these files (Read first, then Edit):\n{files}\n\n"
    "When done, output exactly ONE JSON object: "
    '{{"files": [<repo-relative paths you changed>], "summary": <one sentence>}}'
)


def _format_prior_attempts(prior_attempts: list | None) -> str:
    """Render the history of already-tried-and-failed approaches so the repair does not blindly
    repeat them. Same 'feed accumulated context' idea the off-menu/from-principles path uses."""
    if not prior_attempts:
        return ""
    lines = []
    for a in prior_attempts:
        approach = (a.get("approach") or "").strip() or "(approach not recorded)"
        err = (a.get("error") or "").strip()
        lines.append(f"  - attempt {a.get('attempt', '?')}: {approach} -> FAILED: {err[:200]}")
    return (
        "APPROACHES ALREADY TRIED THAT FAILED — do NOT repeat any of these. They keep failing the "
        "SAME way, so a small tweak won't help: change the APPROACH (different op / dtype / layout / "
        "kernel structure). If NO materially different approach is viable, make the minimal change "
        "and say so in your summary (so the loop can stop instead of cycling):\n" + "\n".join(lines) + "\n\n"
    )


def build_repair_prompt(
    lever: str, section: str, model_files: list, error: str, prior_attempts: list | None = None
) -> str:
    files = "\n".join(f"  - {f}" for f in model_files)
    return REPAIR_TEMPLATE.format(
        lever=lever or "(unspecified)",
        error=error or "(no detail)",
        prior=_format_prior_attempts(prior_attempts),
        section=section or "(apply the lever named above)",
        files=files,
    )


SPEC_TEMPLATE = (
    "Apply this specific change to a model:\n\n"
    "File: {file}\nLocation: {location}\nChange: {change}\n\n"
    "The File path is relative to your working directory — Read exactly that one file (do NOT search the repository), then make ONLY this change. Preserve the model's numerical "
    "behavior; do not refactor unrelated code. When done, output exactly ONE JSON "
    'object: {{"files": [<repo-relative paths you changed>], "summary": <one sentence>}}'
)


def build_spec_prompt(spec: dict, model_files: list) -> str:
    return SPEC_TEMPLATE.format(
        file=spec.get("file", "?"),
        location=spec.get("location", ""),
        change=spec.get("change", ""),
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
    from .config import agent_effort, apply_agent_env, get_edit_model

    resolved = apply_agent_env(env_agent_path)

    def runner(
        *,
        lever: str,
        section: str,
        model_files: list,
        error: str | None = None,
        spec: dict | None = None,
        cwd: str | None = None,
        attempt: int = 0,
        prior_attempts: list | None = None,
        validate: "Callable[[], dict] | None" = None,
    ) -> dict:
        # Escalation ladder: APPLY (attempt 0) -> haiku, repair 1 -> sonnet, repair 2+ -> opus.
        model = get_edit_model(attempt, resolved)

        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        from .probes import _extract_json_object, _usage_summary

        files = [str(p) for p in model_files]
        if error:
            prompt = build_repair_prompt(lever, section, files, error, prior_attempts=prior_attempts)
        elif spec:
            prompt = build_spec_prompt(spec, files)
        else:
            prompt = build_edit_prompt(lever, section, files)

        # Optional: give the agent a tool to VALIDATE its edit on-device before finishing, so it
        # fixes crashes/illegal configs itself instead of submitting blind (the verified gap).
        allowed = ["Read", "Edit", "Glob", "Grep"]
        check_server = None
        if validate is not None:
            from .edit_check import EDIT_CHECK_SERVER, EDIT_CHECK_TOOL, _PROMPT_NOTE, make_edit_check_server

            check_server = make_edit_check_server(validate)
            if check_server is not None:
                allowed = allowed + [EDIT_CHECK_TOOL]
                prompt = prompt + _PROMPT_NOTE

        opts: dict = dict(
            model=model,
            system_prompt=(
                "You apply exactly one optimization to model source using Read and "
                "Edit. Your FINAL message must be one JSON object, no prose. Stay "
                "inside the working directory; do NOT search the wider repository."
            ),
            allowed_tools=allowed,
            permission_mode="bypassPermissions",
            setting_sources=[],
            # check_candidate_edit makes the editor spend extra turns per validate cycle; bump turns
            # when it's attached so it doesn't run out mid-iteration (same fix as the structural agent).
            max_turns=(max(max_turns, _DEVICE_MAX_TURNS) if check_server is not None else max_turns),
            max_buffer_size=50 * 1024 * 1024,
            effort=agent_effort(resolved),  # cap reasoning so the editor doesn't think-for-minutes
        )
        if check_server is not None:
            opts["mcp_servers"] = {EDIT_CHECK_SERVER: check_server}
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
        from .structural_agent import _DEVICE_CALL_TIMEOUT_S

        # check_candidate_edit makes minutes-long device (PCC) calls; give the editor a generous wall
        # budget when it's attached, instead of the 300s hang default that would kill+retry it.
        call_timeout = _DEVICE_CALL_TIMEOUT_S if check_server is not None else None
        run_with_retry(_go, lambda: (chunks.clear(), usage.clear()), timeout=call_timeout)
        result = _validate_edit_result(_extract_json_object("\n".join(chunks)))
        result["model"] = model
        result["usage"] = usage.get("u")
        result["prompt"] = prompt
        result["response"] = "\n".join(chunks)
        return result

    return runner

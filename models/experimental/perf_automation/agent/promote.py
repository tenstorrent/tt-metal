# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Promote / learning loop — turn a verified from-principles win into a reusable provisional playbook lever."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from . import states

PROMPT = (
    "You are distilling a PROVEN one-off optimization into a REUSABLE, GENERAL playbook lever.\n\n"
    "A from-principles edit just passed every gate (correct + measurably faster) on this model:\n"
    "  bottleneck bucket: {bucket}\n"
    "  hottest ops: {top_ops}\n"
    "  measured: {before} -> {after} ms (PCC {pcc})\n"
    "  the edit (ground-truth diff):\n{diff}\n\n"
    "Write ONE general playbook section that captures the TECHNIQUE (not this model's specifics), so "
    "the tool can apply it to ANY future model with the same bottleneck. Output EXACTLY this markdown "
    "(no prose before/after):\n"
    "## {slug_title} {{#{slug}}}\n"
    "<!-- route\n"
    "op_class: {op_class}\n"
    "lever_type: structural\n"
    "-->\n\n"
    "**Fires when:** <one general sentence describing the bottleneck signature this targets>\n\n"
    "<2-6 lines: the general recipe — what to change and why, abstracted from the specific diff. "
    "Describe the TTNN technique, not the nemotron-specific code.>\n"
)


def should_promote(ctx) -> bool:
    """Promote only a kept, faster, from-principles (off-menu) win."""
    if ctx.state.get("selected_lever") != states.FROM_PRINCIPLES:
        return False
    d = ctx.state.get("last_decision") or {}
    before, after = d.get("before"), d.get("after")
    return d.get("result", "keep") == "keep" and (before is None or after is None or after < before)


def _slug(bucket: str, model: str) -> str:
    base = f"{bucket or 'op'}-coherence-{model or 'x'}"
    return re.sub(r"[^a-z0-9-]+", "-", base.lower()).strip("-")[:48]


def _win_from_ctx(ctx) -> dict:
    d = ctx.state.get("last_decision") or {}
    bucket = ctx.state.get("current_bucket") or "?"
    top_ops = ctx.state.get("top_ops") or []
    op_codes = ", ".join(str(o.get("op_code", o.get("shape", "?"))) for o in top_ops[:4]) or "(n/a)"
    op_class = ""
    try:
        op_class = (
            next(
                (
                    b.get("tags", {}).get("op_class")
                    for b in (ctx.current_profile().get("buckets") or [])
                    if b.get("id") == bucket
                ),
                "",
            )
            or bucket
        )
    except Exception:
        op_class = bucket
    model = getattr(ctx, "model_root", lambda: Path("."))().name
    return {
        "bucket": bucket,
        "op_class": op_class,
        "top_ops": op_codes,
        "diff": (ctx.state.get("last_diff") or "(diff unavailable)")[:4000],
        "before": d.get("before"),
        "after": d.get("after"),
        "pcc": d.get("pcc"),
        "model": model,
    }


def build_promote_prompt(win: dict) -> str:
    slug = _slug(win["bucket"], win["model"])
    return PROMPT.format(
        bucket=win["bucket"],
        top_ops=win["top_ops"],
        before=win["before"],
        after=win["after"],
        pcc=win["pcc"],
        diff=win["diff"],
        op_class=win["op_class"],
        slug=slug,
        slug_title=f"Learned: {win['bucket']} coherence",
    )


def write_provisional_lever(section_text: str, slug: str, guidelines_dir: Path, learned_on: str) -> Path:
    """Write a provisional learned lever to GUIDELINES/LEARNED_<slug>.md (router auto-indexes it next run)."""
    guidelines_dir = Path(guidelines_dir)
    path = guidelines_dir / f"LEARNED_{slug}.md"
    banner = (
        f"<!-- LEARNED LEVER — provisional: true; learned_on: {learned_on} -->\n"
        "<!-- Auto-distilled from a verified from-principles win. Graduates to trusted after it "
        "lands a gain on a DIFFERENT model (cross-model validation). -->\n\n"
    )
    path.write_text(banner + section_text.strip() + "\n", encoding="utf-8")
    return path


def graduate_lever(path: Path, confirmed_on: str) -> Path:
    """Flip a provisional learned lever to trusted (it landed a gain on a 2nd, different model)
    and rename LEARNED_<slug>.md -> GRADUATED_<slug>.md so it leaves the gitignored provisional
    set and becomes a committable, version-controlled lever. Returns the (possibly new) path."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    text = text.replace("provisional: true", f"provisional: false; graduated_on: {confirmed_on}", 1)
    path.write_text(text, encoding="utf-8")
    if path.name.startswith("LEARNED_"):
        new_path = path.with_name("GRADUATED_" + path.name[len("LEARNED_") :])
        path.rename(new_path)
        return new_path
    return path


def maybe_graduate(ctx, lever_id: str, guidelines_dir: Path | None = None) -> Path | None:
    """If a KEPT win re-used a PROVISIONAL learned lever that was learned on a DIFFERENT model,
    that is cross-model validation -> graduate it. Returns the graduated path, or None. Best-effort."""
    if not lever_id:
        return None
    from . import router

    gdir = Path(guidelines_dir) if guidelines_dir else Path(router.GUIDELINES_DIR)
    current = getattr(ctx, "model_root", lambda: Path("."))().name
    for p in sorted(gdir.glob("LEARNED_*.md")):
        text = p.read_text(encoding="utf-8")
        if "provisional: true" not in text:
            continue
        anchor = re.search(r"\{#([a-z0-9-]+)\}", text)
        if not anchor or anchor.group(1) != lever_id:
            continue
        learned_on = re.search(r"learned_on:\s*([^\s>]+)", text)
        if learned_on and learned_on.group(1) == current:
            return None
        return graduate_lever(p, current)
    return None


def make_promote_runner(
    env_agent_path: str | Path = Path(__file__).parent.parent / ".env.agent",
    max_turns: int = 4,
) -> Callable[[str], str]:
    """Live promote runner: runner(prompt) -> the generated lever-section markdown. Lead model."""
    from .config import apply_agent_env, get_model

    resolved = apply_agent_env(env_agent_path)
    model = get_model("lead", resolved)

    def runner(prompt: str) -> str:
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

        options = ClaudeAgentOptions(
            model=model,
            system_prompt="You distill a proven optimization into one general playbook section. Output only the markdown.",
            allowed_tools=[],
            permission_mode="bypassPermissions",
            setting_sources=[],
            max_turns=max_turns,
            max_buffer_size=50 * 1024 * 1024,
        )
        chunks: list[str] = []

        async def _go() -> None:
            async for msg in query(prompt=prompt, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            chunks.append(block.text)

        from .sdk_retry import run_with_retry

        run_with_retry(_go, lambda: chunks.clear())
        return "\n".join(chunks)

    return runner


def promote_win(ctx, runner: Callable[[str], str] | None = None, guidelines_dir: Path | None = None) -> Path | None:
    """Generalize a kept from-principles win into a provisional learned lever; returns the path or None."""
    if not should_promote(ctx):
        return None
    runner = runner or (ctx.deps.get("promote_runner") if hasattr(ctx, "deps") else None) or make_promote_runner()
    win = _win_from_ctx(ctx)
    section = runner(build_promote_prompt(win))
    if not section or "{#" not in section:
        return None
    from . import router

    gdir = Path(guidelines_dir) if guidelines_dir else Path(router.GUIDELINES_DIR)
    return write_provisional_lever(section, _slug(win["bucket"], win["model"]), gdir, win["model"])

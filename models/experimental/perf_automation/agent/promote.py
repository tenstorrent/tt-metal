# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Promote / learning loop — turn a verified from-principles win into a reusable provisional playbook lever."""

from __future__ import annotations

import re
from pathlib import Path


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


def _slug(bucket: str, model: str) -> str:
    base = f"{bucket or 'op'}-coherence-{model or 'x'}"
    return re.sub(r"[^a-z0-9-]+", "-", base.lower()).strip("-")[:48]


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

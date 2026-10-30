"""Render violations + matching recipes as a markdown block for the
LLM prompt.

The block is designed to slot in alongside the existing PREVIOUS
ATTEMPT FAILED / SIBLING TTNN EXAMPLE sections — same heading depth,
same style. Empty-output if no violations fire.
"""

from __future__ import annotations

from typing import Iterable

from .catalog import Violation
from .recipes import RECIPES


def format_constraint_hints(violations: Iterable[Violation]) -> str:
    """Build the block. Returns "" when no violations are provided."""
    violations = list(violations)
    if not violations:
        return ""

    lines: list = [
        "",
        "TTNN CONSTRAINT WARNINGS — APPLY THE RECIPES BEFORE GENERATING",
        "--------------------------------------------------------------",
        (
            "Static analysis of the captured inputs flagged the following ttnn "
            "shape/dtype constraints this component will hit. Each item has a "
            "concrete recipe — adapt it into your generated code rather than "
            "guessing through trial and error."
        ),
        "",
    ]
    for i, v in enumerate(violations, 1):
        recipe = RECIPES.get(v.recipe_id)
        lines.append(f"### {i}. {v.constraint_name}")
        lines.append("")
        lines.append(v.description)
        lines.append("")
        if recipe is not None:
            lines.append(f"**Recipe — {recipe.title}**")
            lines.append("")
            try:
                body = recipe.body.format(**v.details)
            except Exception:
                # Bad placeholder → fall back to unsubstituted body so the
                # recipe still appears.
                body = recipe.body
            lines.append(body)
        else:
            lines.append(f"(no recipe registered for `{v.recipe_id}` — TODO)")
        lines.append("")
    return "\n".join(lines)

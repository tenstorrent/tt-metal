"""Unit tests for the extras-enrichment refactor.

Before the refactor, parallel-extra targets in the auto-iterate loop got
a SPARSE prompt while only the primary target got the full enrichment
(captured I/O, localization, exemplar, etc.). This made extras converge
far worse than primaries on the same model run -- the v11→v14 plateau
where vision_* never graduated was a direct symptom.

After the refactor, both primary and extras call the same closure helper
`_build_enriched_component_block(comp)` so they get the same enrichment
for their OWN component (not the primary's).

These tests pin the structural shape of the refactor by source-grepping
auto_iterate.py. They don't execute the closure (it depends on heavy
HF/ttnn imports) -- they verify the wiring."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

cli = importlib.import_module("scripts.tt_hw_planner.cli")


def _auto_iterate_source() -> str:
    return (Path(cli.__file__).parent / "_cli_helpers" / "auto_iterate.py").read_text()


def test_enriched_block_helper_exists() -> None:
    """The closure helper that builds the per-component enriched block must
    be defined inside _run_auto_iterate_loop. It's the single source of
    truth for which blocks go into each agent's prompt."""
    src = _auto_iterate_source()
    assert "def _build_enriched_component_block(comp: str) -> str:" in src, (
        "_run_auto_iterate_loop must define a closure helper "
        "`_build_enriched_component_block(comp)` that builds the full "
        "per-component enriched prompt block"
    )


def test_helper_returns_block_text() -> None:
    """The helper must use `return` (not list-append) so it can be called
    from both the primary loop and the extras loop without coupling them
    through a shared mutable list."""
    src = _auto_iterate_source()
    # The old pattern was `component_blocks.append(...)`. Must be gone.
    assert "component_blocks.append(" not in src, (
        "The pre-refactor `component_blocks.append(...)` pattern must be "
        "replaced with a `return` inside _build_enriched_component_block"
    )


def test_primary_loop_uses_helper() -> None:
    """The primary's component_blocks list must be built by calling the
    helper, not by inlining the enrichment logic again. Otherwise the
    helper and the primary loop drift out of sync over time."""
    src = _auto_iterate_source()
    assert "component_blocks: List[str] = [_build_enriched_component_block(comp) for comp in agent_targets]" in src, (
        "The primary loop must consume _build_enriched_component_block " "via a list comprehension over agent_targets"
    )


def test_extras_use_per_extra_components_block() -> None:
    """Each extra's prompt must use its OWN components_block (built by
    calling the helper for the extra's component), not the primary's
    components_block. This is the bug-fix the refactor exists for."""
    src = _auto_iterate_source()
    assert "_extra_components_block = _build_enriched_component_block(_extra)" in src, (
        "Each extra must build its own enriched block by calling the " "helper with its own component name"
    )
    # And the assemble call must consume that block, not the primary's
    assert "components_block=_extra_components_block," in src, (
        "assemble_iter_prompt for extras must receive _extra_components_block, " "not the primary's components_block"
    )


def test_extras_do_not_inherit_primary_components_block() -> None:
    """No remaining call to assemble_iter_prompt(...components_block=components_block...)
    for the EXTRAS path. Only the primary may pass the joined primary block."""
    src = _auto_iterate_source()
    # The line `components_block=components_block,` only appears in the
    # primary path. The extras path uses `components_block=_extra_components_block,`
    # Count both occurrences:
    primary_count = src.count("components_block=components_block,")
    extra_count = src.count("components_block=_extra_components_block,")
    # The primary may still use it in assemble_iter_prompt for the primary.
    # The extras MUST use the new variable -- at least one occurrence required.
    assert extra_count >= 1, "Extras must consume _extra_components_block"
    # And the helper should not accidentally pass the primary's block in the
    # extras' assembly. We don't ban the variable globally, but the count
    # should NOT include any inside the `for _extra in _extra_targets:` block.
    # Source-grep heuristic: locate the extras block and verify it doesn't
    # contain `components_block=components_block`.
    extras_block_start = src.find("for _extra in _extra_targets:")
    # Skip the FIRST occurrence (which is the "snapshot" loop, not the prompt
    # assembly loop -- find the SECOND).
    extras_block_start = src.find("for _extra in _extra_targets:", extras_block_start + 1)
    assert extras_block_start != -1
    # End of extras block is the if-statement that starts spawning agents:
    extras_block_end = src.find('if os.environ.get("TT_PLANNER_DRY_RUN_PROMPTS"', extras_block_start)
    if extras_block_end == -1:
        extras_block_end = src.find("if _parallel_extra_jobs:", extras_block_start)
    assert extras_block_end != -1
    extras_body = src[extras_block_start:extras_block_end]
    assert "components_block=components_block," not in extras_body, (
        "The extras assembly block must NOT pass the primary's "
        "components_block. It must pass _extra_components_block instead."
    )


def test_activation_diff_trigger_includes_crash_classes() -> None:
    """Option A: activation_diff must fire for OTHER-class failures that
    contain a Python traceback (RuntimeError / AttributeError / IndexError).
    Pre-refactor, only PCC_ONLY/DTYPE_MISMATCH/SHAPE/TT_FATAL_OPAQUE triggered,
    so crash-class failures got no localization hint."""
    src = _auto_iterate_source()
    assert "_localize_classes = (" in src, "auto_iterate.py must define _localize_classes tuple"
    assert "_crash_in_failure" in src, "auto_iterate.py must compute _crash_in_failure for OTHER-class failures"
    assert "RuntimeError:" in src
    assert "AttributeError:" in src
    assert "IndexError:" in src


def test_localization_trigger_uses_both_class_and_crash() -> None:
    """The actual `if` check must combine the class set AND the crash flag."""
    src = _auto_iterate_source()
    expected = 'if failure_class in _localize_classes or (failure_class == "OTHER" and _crash_in_failure):'
    assert expected in src, (
        "The localization trigger must use the form "
        '`if failure_class in _localize_classes or (failure_class == "OTHER" '
        "and _crash_in_failure):`"
    )


def test_prompt_block_log_inside_helper() -> None:
    """The [prompt-block] observability log must be inside the helper so
    it fires for extras too. Pre-refactor it was inside the primary-only
    loop and we couldn't see whether extras had enriched prompts."""
    src = _auto_iterate_source()
    # Helper begins with `def _build_enriched_component_block(comp: str) -> str:`
    helper_start = src.find("def _build_enriched_component_block(comp: str) -> str:")
    assert helper_start != -1
    # Helper ends when the next non-indented function-level statement appears.
    # Use a heuristic: find the next line that starts with 8 spaces of
    # indentation but doesn't continue the helper body. Easier: find the
    # closing `component_blocks: List[str] = [` which is right after the helper.
    helper_end = src.find("component_blocks: List[str] = [", helper_start)
    assert helper_end != -1
    helper_body = src[helper_start:helper_end]
    assert "[prompt-block]" in helper_body, (
        "The [prompt-block] observability log must live inside "
        "_build_enriched_component_block so it fires for extras too"
    )

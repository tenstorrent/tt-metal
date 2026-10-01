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

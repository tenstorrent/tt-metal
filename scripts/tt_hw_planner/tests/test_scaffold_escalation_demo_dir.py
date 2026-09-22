"""Pin: the force_already_supported scaffold branch must NOT use the
backend's demo file path as the new model's demo directory.

The Phi-3.5 escalation run on 2026-06-02 failed with:

    scaffold failed: [Errno 17] File exists:
      '/tmp/.../models/tt_transformers/demo/simple_text_demo.py'

because scaffold.py:232 assigned ``demo_dir_esc_rel = Path(_be_esc.demo_path)``
— treating ``simple_text_demo.py`` (a regular file) as the directory
to drop ``BRING_UP_PLAN.md`` / ``bringup_status.json`` / ``_stubs/``
into. ``mkdir(parents=True, exist_ok=True)`` raised ``[Errno 17]``
because the path exists as a FILE, not as a DIRECTORY.

The fix derives a sibling directory via
``backend_parent / _slug(new_model_tail)`` — same pattern the
non-escalation scaffold branch uses below.
"""

from __future__ import annotations

from pathlib import Path


def test_force_already_supported_scaffold_does_not_use_demo_file_as_dir():
    """Source-level guard: the force_already_supported branch must
    NOT assign ``demo_dir_esc_rel = Path(_be_esc.demo_path)``.
    That regresses the Phi-3.5 [Errno 17] failure."""
    src = Path("scripts/tt_hw_planner/scaffold.py").read_text()
    # The exact broken assignment, on its own line
    bad_assignment = "demo_dir_esc_rel = Path(_be_esc.demo_path)\n"
    assert bad_assignment not in src, (
        "scaffold.py uses backend.demo_path (a `.py` file) as the new "
        "demo DIRECTORY in the force_already_supported branch. This "
        "causes mkdir to fail with [Errno 17] File exists. Derive a "
        "sibling dir via `Path(_be_esc.demo_path).parent / "
        "_slug(new_model_tail)` instead."
    )


def test_escalation_branch_delegates_to_shared_demo_dir_resolver():
    """The branch must not hand-roll its own dir derivation again.

    The parent/slug logic now lives in one helper shared by all three
    scaffold routes; pinning the delegation keeps the branch from
    drifting back to a private copy (which is how the Phi-3.5 bug
    reached only this branch in the first place)."""
    src = Path("scripts/tt_hw_planner/scaffold.py").read_text()
    fa_idx = src.find("force_adapt_all=True")
    assert fa_idx >= 0, "force_already_supported branch missing — test invariant out of date"
    block = src[fa_idx : fa_idx + 1500]
    assert "_resolve_demo_dir_rel" in block, (
        "force_already_supported branch must derive its demo dir via the " "shared _resolve_demo_dir_rel helper"
    )


def test_resolver_returns_sibling_dir_not_the_demo_file():
    """Behavioural guard for the Phi-3.5 [Errno 17] regression: given a
    backend whose demo_path is a regular ``.py`` file, the resolved demo
    dir must be a SIBLING directory of that file, never the file."""
    from types import SimpleNamespace

    from scripts.tt_hw_planner.scaffold import _resolve_demo_dir_rel

    demo_file = Path("models") / "tt_transformers" / "demo" / "simple_text_demo.py"
    backend = SimpleNamespace(demo_path=str(demo_file))
    # An id no scaffolded demo dir can already claim, so the resolver
    # takes the derive-a-new-dir path rather than returning an existing one.
    resolved = _resolve_demo_dir_rel("pytest-org/pytest-nonexistent-model", backend)

    assert resolved != demo_file, "resolver returned the backend's .py file as the demo DIRECTORY"
    assert resolved.suffix != ".py", f"resolved demo dir is a file path: {resolved}"
    assert resolved.parent == demo_file.parent, f"resolved dir is not a sibling of the demo file: {resolved}"

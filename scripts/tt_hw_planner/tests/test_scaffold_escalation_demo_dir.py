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


def test_force_already_supported_scaffold_uses_slug_of_new_model():
    """Positive guard: the fix must slug the new model id under the
    backend's parent. Pin the exact pattern."""
    src = Path("scripts/tt_hw_planner/scaffold.py").read_text()
    # Find force_already_supported block. Look for the build_bringup_plan
    # call with force_adapt_all=True (signature of this branch).
    fa_idx = src.find("force_adapt_all=True")
    assert fa_idx >= 0, "force_already_supported branch missing — test invariant out of date"
    block = src[fa_idx : fa_idx + 1500]
    assert "_scaffold_slug" in block or "_slug" in block, (
        "force_already_supported scaffold branch must use a slug "
        "function on new_model_id to derive a sibling demo dir"
    )
    assert "demo_path).parent" in block, (
        "force_already_supported branch must take .parent of the "
        "backend's demo_path so the new model's dir is a sibling, "
        "not a child of the backend's .py file"
    )

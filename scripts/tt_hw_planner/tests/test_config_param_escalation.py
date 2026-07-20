"""Unit tests for the CONFIG_PARAM failure-class escalation.

Background (2026-06-03): Phi-3.5-mini-instruct attention couldn't graduate
because `models/tt_transformers/tt/rope.py:462` raised
``NotImplementedError("use_qk_fused")`` for head_dim=96 — the real fix lived
in `models/tt_transformers/tt/model_config.py:645` which derived
``use_qk_fused`` without checking head_dim divisibility. The LLM's iter-prompt
ESCALATED EDIT SCOPE table didn't include any failure class for "canonical
tt_transformers code raised NotImplementedError" so the LLM was never told
it could edit the canonical source.

This test pins:
  * The CONFIG_PARAM failure class is detected when NotImplementedError
    originates in tt_transformers / common code.
  * The escalation table maps CONFIG_PARAM to the canonical model_config.py
    and rope.py paths so the LLM gets the right write-permission.
  * The escalated-scope prompt block emits actionable guidance pointing
    at the model_config.py derivation as the typical fix site.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_classifier_returns_config_param_for_tt_transformers_notimplementederror():
    from scripts.tt_hw_planner.cli import _classify_failure

    summary = "test_attention failed"
    details = (
        "  File '/home/ttuser/tt-metal/models/tt_transformers/tt/rope.py', "
        "line 462, in __init__\n"
        "    raise NotImplementedError('use_qk_fused')\n"
        "NotImplementedError: use_qk_fused"
    )
    assert _classify_failure(summary, details) == "CONFIG_PARAM"


def test_classifier_returns_config_param_for_models_common_notimplementederror():
    from scripts.tt_hw_planner.cli import _classify_failure

    summary = "test_rmsnorm failed"
    details = (
        "  File '/home/ttuser/tt-metal/models/common/rmsnorm.py', line 100, in __init__\n"
        "    raise NotImplementedError('unsupported variant')\n"
        "NotImplementedError: unsupported variant"
    )
    assert _classify_failure(summary, details) == "CONFIG_PARAM"


def test_classifier_does_not_promote_unrelated_notimplementederror_to_config_param():
    """A NotImplementedError that doesn't originate from tt_transformers /
    common code should NOT be classified as CONFIG_PARAM (would mis-route
    the escalation)."""
    from scripts.tt_hw_planner.cli import _classify_failure

    summary = "test_x failed"
    details = "  File '/some/other/code.py', line 10, in foo\n" "NotImplementedError: foo"
    assert _classify_failure(summary, details) != "CONFIG_PARAM"


def test_repo_relative_edit_scope_includes_canonical_model_config():
    from scripts.tt_hw_planner.cli import _REPO_RELATIVE_EDIT_SCOPE_FOR_FAILURE_CLASS

    paths = _REPO_RELATIVE_EDIT_SCOPE_FOR_FAILURE_CLASS.get("CONFIG_PARAM", [])
    assert "models/tt_transformers/tt/model_config.py" in paths, (
        "CONFIG_PARAM escalation must include model_config.py — that's where "
        "the use_qk_fused derivation lives and where the typical fix lands"
    )
    assert "models/tt_transformers/tt/rope.py" in paths, (
        "CONFIG_PARAM should also unlock rope.py since that's where the "
        "NotImplementedError raises (LLM may need to read it to understand "
        "the trigger condition)"
    )


def test_resolve_extra_edit_paths_returns_canonical_files_for_config_param(tmp_path: Path):
    """Verify that the resolver returns repo-relative canonical paths
    for the CONFIG_PARAM class, not just demo_dir-local ones."""
    from scripts.tt_hw_planner.cli import _resolve_extra_edit_paths

    # Use an empty demo_dir to confirm the repo-relative branch fires
    paths = _resolve_extra_edit_paths(demo_dir=tmp_path, failure_class="CONFIG_PARAM")
    path_strs = [str(p) for p in paths]
    # At least one of the canonical files must be discoverable
    has_model_config = any("model_config.py" in s for s in path_strs)
    has_rope = any("rope.py" in s for s in path_strs)
    assert has_model_config or has_rope, (
        f"_resolve_extra_edit_paths(CONFIG_PARAM) returned {path_strs} — "
        f"expected at least one canonical tt_transformers path"
    )


def test_escalated_scope_block_emits_config_param_guidance(tmp_path: Path):
    from scripts.tt_hw_planner.cli import _format_escalated_edit_scope_block

    block = _format_escalated_edit_scope_block(tmp_path, "CONFIG_PARAM")
    if not block:
        pytest.skip("repo paths not present in test env; nothing to gate guidance on")
    assert "CONFIG_PARAM" in block or "config_param" in block.lower()
    assert "NotImplementedError" in block or "use_qk_fused" in block
    assert "model_config" in block.lower() or "configuration" in block.lower()


def test_unrelated_failure_class_gets_empty_escalation(tmp_path: Path):
    """Sanity check that other failure classes don't accidentally trigger
    the CONFIG_PARAM canonical-path escalation."""
    from scripts.tt_hw_planner.cli import _resolve_extra_edit_paths

    # SHAPE / DTYPE_MISMATCH are common; they should NOT get tt_transformers paths
    paths = _resolve_extra_edit_paths(demo_dir=tmp_path, failure_class="DTYPE_MISMATCH")
    path_strs = [str(p) for p in paths]
    has_canonical = any("models/tt_transformers/tt/model_config.py" in s for s in path_strs)
    assert not has_canonical, (
        "DTYPE_MISMATCH should not unlock canonical tt_transformers paths — "
        "those should be reserved for CONFIG_PARAM specifically"
    )


def test_model_config_use_qk_fused_disabled_when_head_dim_not_divisible_by_64():
    """Pin the 1-line fix in model_config.py:645 area: when head_dim % 64 != 0,
    use_qk_fused must be force-disabled to avoid NotImplementedError at runtime.

    Source-level guard test (no model load needed)."""
    src = Path("models/tt_transformers/tt/model_config.py").read_text()
    # The head_dim refinement must appear AFTER head_dim is set
    head_dim_set = src.find("self.head_dim = text_config.get")
    refinement = src.find("self.head_dim % 64 != 0")
    assert head_dim_set >= 0, "head_dim assignment not found — test invariant out of date"
    assert refinement > head_dim_set, (
        "head_dim%64 check must come AFTER `self.head_dim = ...` is set " "(otherwise AttributeError on self.head_dim)"
    )
    # The check must force use_qk_fused = False (not = True)
    near = src[refinement : refinement + 200]
    assert "self.use_qk_fused = False" in near, "when head_dim%64 != 0, use_qk_fused must be force-disabled"

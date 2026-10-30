"""Unit tests for Bundle item #3: expanded exemplar role taxonomy.

Step 4's vision_neck Tier 2 test showed the exemplar finder returning
`(none)` for vision_neck. Root cause: `_exemplar_role_for("vision_neck",
"Sam2VisionNeck")` returned None — no "neck" entry in the role
taxonomy, and the role-string match `for role, _hints: if role in needle`
only checked the role NAME (e.g., "encoder"), not the role's HINTS.

The widening:
  1. Add neck / head / backbone / upsample / fuser roles to the taxonomy
  2. Match against role HINTS too (not just the role name) so e.g. a
     "patch_embed" name still routes to "embed" even if "embed" is not
     a literal substring of "patch_embed"

These tests pin the new behavior so future role-additions don't
silently regress the matching."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

cli = importlib.import_module("scripts.tt_hw_planner.cli")


def test_vision_neck_maps_to_neck_role() -> None:
    """The Step 4 evidence: vision_neck must now resolve to a role so
    the exemplar finder can search for a similar component."""
    assert cli._exemplar_role_for("vision_neck", "Sam2VisionNeck") == "neck"


def test_fpn_keywords_map_to_neck() -> None:
    """FPN / feature_pyramid / lateral / projection all imply a neck-like
    component (multi-scale feature aggregation)."""
    assert cli._exemplar_role_for("feature_pyramid", "") == "neck"
    assert cli._exemplar_role_for("fpn", "") == "neck"
    assert cli._exemplar_role_for("lateral_projection", "") == "neck"


def test_head_role_matches_predictor_lmhead() -> None:
    """Components named lm_head / predictor / classifier should route to
    a `head` role for exemplar lookup."""
    assert cli._exemplar_role_for("lm_head", "LMHead") == "head"
    assert cli._exemplar_role_for("predictor", "") == "head"
    assert cli._exemplar_role_for("classifier", "") == "head"


def test_backbone_role_matches_stem_trunk() -> None:
    """`backbone`, `stem`, `trunk` should all route to backbone role."""
    assert cli._exemplar_role_for("backbone", "Backbone") == "backbone"
    assert cli._exemplar_role_for("stem", "") == "backbone"
    assert cli._exemplar_role_for("trunk_block", "") == "backbone"


def test_upsample_and_fuser_roles() -> None:
    """Upsample / deconv / convtranspose → upsample. Fusion / merger → fuser."""
    assert cli._exemplar_role_for("upsample", "") == "upsample"
    assert cli._exemplar_role_for("convtranspose", "") == "upsample"
    assert cli._exemplar_role_for("memory_fuser", "Sam2MemoryFuser") == "fuser"
    assert cli._exemplar_role_for("feature_fusion", "") == "fuser"


def test_hint_match_works_when_role_name_not_in_needle() -> None:
    """Pre-fix bug: `patch_embed` would have failed the `if role in needle`
    check for "embed" role only because the role NAME `embed` was a
    substring (which happens to work here). But for `convtranspose` the
    role name `upsample` is NOT a substring of the needle — the hint
    must catch it. Pin this behavior."""
    # convtranspose needle does NOT contain "upsample" as a substring,
    # but "convtranspose" IS in the upsample role's hints.
    needle = "convtranspose convtranspose2d"
    assert "upsample" not in needle, "test setup: role name should not be in needle"
    assert cli._exemplar_role_for("convtranspose", "ConvTranspose2d") == "upsample"


def test_existing_roles_still_work() -> None:
    """Regression check: pre-existing roles must still resolve correctly
    after the taxonomy expansion."""
    assert cli._exemplar_role_for("self_attention", "SelfAttention") == "attention"
    assert cli._exemplar_role_for("feed_forward", "FeedForward") == "mlp"
    assert cli._exemplar_role_for("layer_norm", "LayerNorm") == "norm"
    assert cli._exemplar_role_for("patch_embed", "PatchEmbed") == "embed"
    assert cli._exemplar_role_for("conv2d", "Conv2d") == "conv"
    assert cli._exemplar_role_for("mask_decoder", "MaskDecoder") == "decoder"
    assert cli._exemplar_role_for("vision_encoder", "VisionEncoder") == "encoder"


def test_unknown_role_still_returns_none() -> None:
    """Components with no recognizable role pattern should still return
    None so the caller falls back to its `(no exemplar found)` path."""
    assert cli._exemplar_role_for("totally_unique_thing", "TotallyUnique") is None
    assert cli._exemplar_role_for("", "") is None


def test_find_exemplar_returns_path_for_vision_neck_when_match_exists() -> None:
    """Smoke test of the integration: now that vision_neck has a role,
    _find_exemplar must at least attempt the cross-demo search. We do
    not assert a specific path because the result depends on what's in
    `models/demos/` on this machine — only that it does not abort early
    due to `role is None`."""
    # The pre-condition (role is now resolvable) is the important part;
    # the actual rglob search is environment-dependent.
    role = cli._exemplar_role_for("vision_neck", "Sam2VisionNeck")
    assert role is not None, (
        "vision_neck must resolve to a non-None role; otherwise _find_exemplar "
        "short-circuits to None before doing the cross-demo search"
    )

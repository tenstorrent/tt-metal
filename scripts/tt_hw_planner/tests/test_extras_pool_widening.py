"""Unit tests for Bundle item #2: extras pool widening.

Before: parallel-extras were drawn from `_ungraduated_now`, which only
returns components whose stubs are STILL ON AUTOFILL (CPU fallback).
This meant components with native ttnn stubs that failed PCC could
NEVER be parallel-extras — they had to wait for their turn as the
primary, one at a time.

The Step 3 dry-run on facebook/sam2-hiera-tiny v14-end-state showed
the symptom directly: vision_config, vision_model, vision_neck were
all "ungraduated" in the candidate_pool, but only vision_config was
on autofill, so _extras_pool would have had {vision_model, vision_neck}
and the dry-run would have written 3 prompt files instead of 1.

After: the pool is `_ungraduated_now ∪ candidate_pool`, which captures
both autofill-on-CPU components AND native-stub-but-PCC-failing ones.

These tests source-grep the wiring."""

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


def test_extras_pool_unions_ungraduated_with_candidate_pool() -> None:
    """The pool source must be `_ungraduated_now ∪ candidate_pool` so
    that PCC-failing-native-stub components are eligible as extras."""
    src = _auto_iterate_source()
    assert "_extras_pool = set(_ungraduated_now) | set(candidate_pool or [])" in src, (
        "auto_iterate.py must build _extras_pool as the union of "
        "_ungraduated_now and candidate_pool so PCC-failing-native-stub "
        "components are eligible as parallel-extras"
    )


def test_extras_pool_subtracts_graduated_this_run() -> None:
    """Components that graduated DURING this run shouldn't be re-attempted
    as extras — the union could otherwise include them via candidate_pool."""
    src = _auto_iterate_source()
    assert "_extras_pool -= set(graduated_this_run)" in src, (
        "_extras_pool must subtract graduated_this_run so already-graduated "
        "components are not pulled back in as extras"
    )


def test_at_cap_check_uses_extras_pool_not_ungraduated_now() -> None:
    """The at-cap exclusion must check the NEW (wider) extras pool, not
    the OLD narrow `_ungraduated_now`. Otherwise we'd silently allow
    at-cap PCC-failing components to be picked as extras."""
    src = _auto_iterate_source()
    assert (
        "_at_cap_now = {c for c in _extras_pool if _is_at_cap(c)}" in src
    ), "The at-cap exclusion must iterate over _extras_pool, not _ungraduated_now"


def test_extras_ranking_uses_extras_pool() -> None:
    """The ranking step that orders candidates for pick_n_distinct_targets
    must rank ALL pool members, not just the narrow ungraduated set."""
    src = _auto_iterate_source()
    # Look for the sorted(...) call that builds _ungraduated_ranked. After
    # the fix it should sort _extras_pool.
    sort_idx = src.find("_ungraduated_ranked = sorted(")
    assert sort_idx != -1
    end_idx = src.find(")", sort_idx)
    sort_call = src[sort_idx:end_idx]
    assert "_extras_pool" in sort_call, (
        "_ungraduated_ranked must be `sorted(_extras_pool, ...)`, not "
        "`sorted(_ungraduated_now, ...)`. Otherwise the widening has no "
        "effect — the widened pool is computed but immediately discarded."
    )


def test_pick_n_distinct_targets_consumes_ranked_pool() -> None:
    """Plumbing check: pick_n_distinct_targets receives the ranked list
    built from _extras_pool. This is what _extra_targets becomes."""
    src = _auto_iterate_source()
    assert "_extra_targets = pick_n_distinct_targets(" in src
    # And the first arg must be _ungraduated_ranked (now sourced from _extras_pool)
    call_idx = src.find("_extra_targets = pick_n_distinct_targets(")
    call_end = src.find(")", call_idx)
    call_block = src[call_idx:call_end]
    assert "_ungraduated_ranked" in call_block


def test_only_component_filter_still_zeroes_extras() -> None:
    """The --auto-only-component sandbox flag must still take precedence:
    if set, _extra_targets is empty regardless of the wider pool."""
    src = _auto_iterate_source()
    # The check should still be present after the picker
    pick_idx = src.find("_extra_targets = pick_n_distinct_targets(")
    if_idx = src.find("if only_component:", pick_idx)
    assert if_idx != -1 and if_idx - pick_idx < 500, (
        "The `if only_component: _extra_targets = []` guard must remain "
        "immediately after the picker; otherwise the widened pool would "
        "leak extras into single-component sandbox runs."
    )

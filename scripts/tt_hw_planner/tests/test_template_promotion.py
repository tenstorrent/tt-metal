"""Unit tests for the multi-model promotion gate (Item 7)."""

from __future__ import annotations

from pathlib import Path

from scripts.tt_hw_planner._cli_helpers.family_template_registry import (
    ChainedTemplateEntry,
    load_registry,
    register_template,
)
from scripts.tt_hw_planner._cli_helpers.template_promotion import (
    DEFAULT_PROMOTION_THRESHOLD,
    auto_promote_after_register,
    is_eligible_for_promotion,
    is_template_promoted,
    mark_promoted,
    templates_eligible_for_promotion,
)


def _make_entry(confirmed: list, promoted: bool = False) -> ChainedTemplateEntry:
    return ChainedTemplateEntry(
        family_key="sam2",
        template_demo_source="x",
        source_model_id=confirmed[0] if confirmed else "m1",
        confirmed_models=list(confirmed),
        promoted=promoted,
    )


# ─── is_template_promoted ───────────────────────────────────────────


def test_is_promoted_returns_flag_value() -> None:
    assert is_template_promoted(_make_entry(["m1"], promoted=False)) is False
    assert is_template_promoted(_make_entry(["m1"], promoted=True)) is True


# ─── is_eligible_for_promotion ──────────────────────────────────────


def test_eligible_returns_false_below_threshold() -> None:
    entry = _make_entry(["m1"])  # 1 < default 2
    assert is_eligible_for_promotion(entry) is False


def test_eligible_returns_true_at_threshold() -> None:
    entry = _make_entry(["m1", "m2"])  # 2 >= default 2
    assert is_eligible_for_promotion(entry) is True


def test_eligible_returns_true_above_threshold() -> None:
    entry = _make_entry(["m1", "m2", "m3"])
    assert is_eligible_for_promotion(entry) is True


def test_eligible_returns_false_when_already_promoted() -> None:
    """Once promoted, eligibility is False — no point re-promoting."""
    entry = _make_entry(["m1", "m2"], promoted=True)
    assert is_eligible_for_promotion(entry) is False


def test_eligible_respects_custom_threshold() -> None:
    entry = _make_entry(["m1", "m2"])
    assert is_eligible_for_promotion(entry, threshold=3) is False
    assert is_eligible_for_promotion(entry, threshold=2) is True
    assert is_eligible_for_promotion(entry, threshold=1) is True


# ─── templates_eligible_for_promotion ───────────────────────────────


def test_eligible_list_empty_for_empty_registry(tmp_path: Path) -> None:
    assert templates_eligible_for_promotion(repo_root=tmp_path) == []


def test_eligible_list_filters_by_threshold(tmp_path: Path) -> None:
    """Registry has 3 entries: 1 below threshold, 1 at, 1 promoted.
    Only the at-threshold one is eligible."""
    register_template(
        family_key="below", template_demo_source="x", source_model_id="m1", repo_root=tmp_path, clock=lambda: 100.0
    )
    register_template(
        family_key="at", template_demo_source="x", source_model_id="m1", repo_root=tmp_path, clock=lambda: 100.0
    )
    register_template(
        family_key="at", template_demo_source="x", source_model_id="m2", repo_root=tmp_path, clock=lambda: 200.0
    )
    register_template(
        family_key="already_promoted",
        template_demo_source="x",
        source_model_id="m1",
        repo_root=tmp_path,
        clock=lambda: 100.0,
    )
    register_template(
        family_key="already_promoted",
        template_demo_source="x",
        source_model_id="m2",
        repo_root=tmp_path,
        clock=lambda: 200.0,
    )
    mark_promoted(family_key="already_promoted", repo_root=tmp_path, clock=lambda: 250.0)

    eligible = templates_eligible_for_promotion(repo_root=tmp_path)
    keys = {e.family_key for e in eligible}
    assert keys == {"at"}


# ─── mark_promoted ──────────────────────────────────────────────────


def test_mark_promoted_returns_none_for_unknown_family(tmp_path: Path) -> None:
    """Family not in registry → None, no error."""
    assert mark_promoted(family_key="never-registered", repo_root=tmp_path) is None


def test_mark_promoted_returns_none_below_threshold(tmp_path: Path) -> None:
    """Only 1 model confirmed → not eligible → mark refuses."""
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m1", repo_root=tmp_path, clock=lambda: 100.0
    )
    assert mark_promoted(family_key="sam2", repo_root=tmp_path) is None
    reg = load_registry(repo_root=tmp_path)
    assert reg["sam2"].promoted is False


def test_mark_promoted_flips_flag_at_threshold(tmp_path: Path) -> None:
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m1", repo_root=tmp_path, clock=lambda: 100.0
    )
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m2", repo_root=tmp_path, clock=lambda: 200.0
    )
    entry = mark_promoted(family_key="sam2", repo_root=tmp_path, clock=lambda: 300.0)
    assert entry is not None
    assert entry.promoted is True
    assert entry.promoted_at == 300.0
    # Persisted
    reg = load_registry(repo_root=tmp_path)
    assert reg["sam2"].promoted is True


def test_mark_promoted_idempotent_on_already_promoted(tmp_path: Path) -> None:
    """Re-marking just updates promoted_at; doesn't reset other fields."""
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m1", repo_root=tmp_path, clock=lambda: 100.0
    )
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m2", repo_root=tmp_path, clock=lambda: 200.0
    )
    mark_promoted(family_key="sam2", repo_root=tmp_path, clock=lambda: 300.0)
    entry = mark_promoted(family_key="sam2", repo_root=tmp_path, clock=lambda: 400.0)
    assert entry is not None
    assert entry.promoted is True
    assert entry.promoted_at == 400.0  # advanced


def test_mark_promoted_respects_custom_threshold(tmp_path: Path) -> None:
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m1", repo_root=tmp_path, clock=lambda: 100.0
    )
    # With threshold=1, eligible immediately
    entry = mark_promoted(family_key="sam2", repo_root=tmp_path, clock=lambda: 200.0, threshold=1)
    assert entry is not None
    assert entry.promoted is True


# ─── auto_promote_after_register ────────────────────────────────────


def test_auto_promote_does_nothing_below_threshold(tmp_path: Path) -> None:
    """Hook called after a single-model register: nothing happens, no
    promotion. Returns None silently."""
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m1", repo_root=tmp_path, clock=lambda: 100.0
    )
    result = auto_promote_after_register(family_key="sam2", repo_root=tmp_path, clock=lambda: 150.0)
    assert result is None
    reg = load_registry(repo_root=tmp_path)
    assert reg["sam2"].promoted is False


def test_auto_promote_promotes_when_second_model_registers(tmp_path: Path) -> None:
    """The intended call site: every register_template is followed by
    auto_promote. Second register pushes count to 2 → promote."""
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m1", repo_root=tmp_path, clock=lambda: 100.0
    )
    # After first register: not promoted
    assert auto_promote_after_register(family_key="sam2", repo_root=tmp_path, clock=lambda: 110.0) is None
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m2", repo_root=tmp_path, clock=lambda: 200.0
    )
    # After second register: promote
    entry = auto_promote_after_register(family_key="sam2", repo_root=tmp_path, clock=lambda: 210.0)
    assert entry is not None
    assert entry.promoted is True
    assert entry.promoted_at == 210.0


def test_auto_promote_returns_none_for_unknown_family(tmp_path: Path) -> None:
    """Unknown family_key → None, no error."""
    assert auto_promote_after_register(family_key="never-registered", repo_root=tmp_path) is None


def test_auto_promote_returns_none_if_already_promoted(tmp_path: Path) -> None:
    """Don't redundantly mark — no-op when already promoted."""
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m1", repo_root=tmp_path, clock=lambda: 100.0
    )
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m2", repo_root=tmp_path, clock=lambda: 200.0
    )
    mark_promoted(family_key="sam2", repo_root=tmp_path, clock=lambda: 300.0)
    assert auto_promote_after_register(family_key="sam2", repo_root=tmp_path, clock=lambda: 400.0) is None
    # promoted_at unchanged
    reg = load_registry(repo_root=tmp_path)
    assert reg["sam2"].promoted_at == 300.0


# ─── Promotion roundtrips with registry persistence ─────────────────


def test_promoted_flag_persists_across_reload(tmp_path: Path) -> None:
    """Promotion must survive a full load → register-other → load
    cycle. Pin so a registry schema refactor can't drop the flag."""
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m1", repo_root=tmp_path, clock=lambda: 100.0
    )
    register_template(
        family_key="sam2", template_demo_source="x", source_model_id="m2", repo_root=tmp_path, clock=lambda: 200.0
    )
    mark_promoted(family_key="sam2", repo_root=tmp_path, clock=lambda: 300.0)
    # Register an unrelated family — full registry rewrite
    register_template(
        family_key="phi3", template_demo_source="y", source_model_id="m3", repo_root=tmp_path, clock=lambda: 400.0
    )
    reg = load_registry(repo_root=tmp_path)
    assert reg["sam2"].promoted is True
    assert reg["sam2"].promoted_at == 300.0


# ─── Constants ──────────────────────────────────────────────────────


def test_default_threshold_is_at_least_2() -> None:
    """Pin the threshold default so a refactor doesn't drop to 1
    (which would defeat the multi-model gate's whole purpose)."""
    assert DEFAULT_PROMOTION_THRESHOLD >= 2

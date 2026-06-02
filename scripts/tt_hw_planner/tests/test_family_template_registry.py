"""Unit tests for the chained-template family registry (Item 6)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.tt_hw_planner._cli_helpers.family_template_registry import (
    REGISTRY_FILENAME,
    ChainedTemplateEntry,
    confirmation_count,
    find_template_for_model,
    list_all_templates,
    load_registry,
    register_template,
    save_registry,
)


# ─── load_registry ──────────────────────────────────────────────────


def test_load_returns_empty_dict_when_file_missing(tmp_path: Path) -> None:
    """First-ever read: no file yet → empty dict, no error."""
    assert load_registry(repo_root=tmp_path) == {}


def test_load_returns_empty_for_malformed_json(tmp_path: Path) -> None:
    (tmp_path / REGISTRY_FILENAME).write_text("{not json")
    assert load_registry(repo_root=tmp_path) == {}


def test_load_returns_empty_for_top_level_list(tmp_path: Path) -> None:
    """JSON is valid but shape wrong → empty dict, don't crash."""
    (tmp_path / REGISTRY_FILENAME).write_text(json.dumps(["entries", "not", "a", "dict"]))
    assert load_registry(repo_root=tmp_path) == {}


def test_load_parses_well_formed_entries(tmp_path: Path) -> None:
    blob = {
        "sam2": {
            "family_key": "sam2",
            "template_demo_source": "models/demos/.../demo.py",
            "source_model_id": "facebook/sam2-hiera-tiny",
            "confirmed_models": ["facebook/sam2-hiera-tiny", "facebook/sam2-hiera-large"],
            "created_at": 100.0,
            "updated_at": 200.0,
            "final_pcc": 0.99,
            "notes": "first segmentation family template",
        }
    }
    (tmp_path / REGISTRY_FILENAME).write_text(json.dumps(blob))
    reg = load_registry(repo_root=tmp_path)
    assert "sam2" in reg
    entry = reg["sam2"]
    assert isinstance(entry, ChainedTemplateEntry)
    assert entry.source_model_id == "facebook/sam2-hiera-tiny"
    assert len(entry.confirmed_models) == 2
    assert entry.final_pcc == 0.99


def test_load_skips_malformed_entries(tmp_path: Path) -> None:
    """One bad entry shouldn't poison the whole registry."""
    blob = {
        "good": {"family_key": "good", "template_demo_source": "x.py", "source_model_id": "org/m"},
        "bad": "this is a string not a dict",
    }
    (tmp_path / REGISTRY_FILENAME).write_text(json.dumps(blob))
    reg = load_registry(repo_root=tmp_path)
    assert "good" in reg
    assert "bad" not in reg


# ─── save_registry ──────────────────────────────────────────────────


def test_save_writes_to_repo_root(tmp_path: Path) -> None:
    entry = ChainedTemplateEntry(
        family_key="sam2",
        template_demo_source="x.py",
        source_model_id="org/m",
        confirmed_models=["org/m"],
        created_at=1.0,
        updated_at=1.0,
    )
    assert save_registry({"sam2": entry}, repo_root=tmp_path) is True
    blob = json.loads((tmp_path / REGISTRY_FILENAME).read_text())
    assert "sam2" in blob
    assert blob["sam2"]["source_model_id"] == "org/m"


def test_save_roundtrips_via_load(tmp_path: Path) -> None:
    """Saving then loading should reproduce the same entry."""
    entry = ChainedTemplateEntry(
        family_key="phi3",
        template_demo_source="models/.../demo.py",
        source_model_id="microsoft/Phi-3.5-mini-instruct",
        confirmed_models=["microsoft/Phi-3.5-mini-instruct"],
        created_at=10.0,
        updated_at=20.0,
        final_pcc=0.995,
        notes="from cold-start success",
    )
    save_registry({"phi3": entry}, repo_root=tmp_path)
    loaded = load_registry(repo_root=tmp_path)
    assert loaded["phi3"].source_model_id == entry.source_model_id
    assert loaded["phi3"].confirmed_models == entry.confirmed_models
    assert loaded["phi3"].final_pcc == 0.995


# ─── register_template ──────────────────────────────────────────────


def test_register_returns_none_for_empty_keys(tmp_path: Path) -> None:
    """Defensive: empty family_key or model_id should fail-fast."""
    assert (
        register_template(
            family_key="",
            template_demo_source="x.py",
            source_model_id="org/m",
            repo_root=tmp_path,
        )
        is None
    )
    assert (
        register_template(
            family_key="sam2",
            template_demo_source="x.py",
            source_model_id="",
            repo_root=tmp_path,
        )
        is None
    )


def test_register_creates_new_entry_first_time(tmp_path: Path) -> None:
    entry = register_template(
        family_key="sam2",
        template_demo_source="models/demos/sam2/demo.py",
        source_model_id="facebook/sam2-hiera-tiny",
        final_pcc=0.983,
        notes="first sam2 success",
        repo_root=tmp_path,
        clock=lambda: 1000.0,
    )
    assert entry is not None
    assert entry.family_key == "sam2"
    assert entry.created_at == 1000.0
    assert entry.confirmed_models == ["facebook/sam2-hiera-tiny"]
    # Persisted to disk
    reg = load_registry(repo_root=tmp_path)
    assert "sam2" in reg


def test_register_idempotent_for_same_model(tmp_path: Path) -> None:
    """Calling twice with the same family + model: still 1 entry, only
    1 confirmed_model. updated_at advances; created_at preserved."""
    register_template(
        family_key="sam2",
        template_demo_source="x",
        source_model_id="m1",
        repo_root=tmp_path,
        clock=lambda: 100.0,
    )
    register_template(
        family_key="sam2",
        template_demo_source="x",
        source_model_id="m1",
        repo_root=tmp_path,
        clock=lambda: 200.0,
    )
    reg = load_registry(repo_root=tmp_path)
    entry = reg["sam2"]
    assert entry.confirmed_models == ["m1"]  # no duplicate
    assert entry.created_at == 100.0
    assert entry.updated_at == 200.0


def test_register_appends_new_model_to_confirmed_list(tmp_path: Path) -> None:
    """Different sibling models register against the same family →
    confirmed_models list grows. This is what Item 7's multi-model gate
    counts."""
    register_template(
        family_key="sam2",
        template_demo_source="x",
        source_model_id="m1",
        repo_root=tmp_path,
        clock=lambda: 100.0,
    )
    register_template(
        family_key="sam2",
        template_demo_source="x",
        source_model_id="m2",
        repo_root=tmp_path,
        clock=lambda: 200.0,
    )
    register_template(
        family_key="sam2",
        template_demo_source="x",
        source_model_id="m3",
        repo_root=tmp_path,
        clock=lambda: 300.0,
    )
    reg = load_registry(repo_root=tmp_path)
    assert reg["sam2"].confirmed_models == ["m1", "m2", "m3"]


def test_register_preserves_original_template_source(tmp_path: Path) -> None:
    """Re-registration confirms the existing template, doesn't
    overwrite the source_model_id or template_demo_source. That's
    intentional: the template was authored by m1, m2 is just
    confirming it works."""
    register_template(
        family_key="sam2",
        template_demo_source="m1_demo.py",
        source_model_id="m1",
        repo_root=tmp_path,
        clock=lambda: 100.0,
    )
    register_template(
        family_key="sam2",
        template_demo_source="m2_demo.py",
        source_model_id="m2",
        repo_root=tmp_path,
        clock=lambda: 200.0,
    )
    reg = load_registry(repo_root=tmp_path)
    entry = reg["sam2"]
    assert entry.source_model_id == "m1"
    assert entry.template_demo_source == "m1_demo.py"


# ─── find_template_for_model ────────────────────────────────────────


def test_find_returns_none_when_no_template(tmp_path: Path) -> None:
    """No template registered for this family yet → None. Caller falls
    through to Step-2/3 (synthesis)."""
    assert find_template_for_model(model_type="brand_new", repo_root=tmp_path) is None


def test_find_returns_entry_when_template_exists(tmp_path: Path) -> None:
    register_template(
        family_key="phi3",
        template_demo_source="x",
        source_model_id="m1",
        repo_root=tmp_path,
        clock=lambda: 100.0,
    )
    entry = find_template_for_model(model_type="phi3", repo_root=tmp_path)
    assert entry is not None
    assert entry.family_key == "phi3"


def test_find_returns_none_for_empty_model_type(tmp_path: Path) -> None:
    assert find_template_for_model(model_type="", repo_root=tmp_path) is None


# ─── list_all_templates ─────────────────────────────────────────────


def test_list_returns_sorted_by_family_key(tmp_path: Path) -> None:
    for key in ("phi3", "sam2", "qwen2"):
        register_template(
            family_key=key,
            template_demo_source="x",
            source_model_id=f"org/{key}",
            repo_root=tmp_path,
        )
    items = list_all_templates(repo_root=tmp_path)
    assert [e.family_key for e in items] == ["phi3", "qwen2", "sam2"]


def test_list_returns_empty_for_no_templates(tmp_path: Path) -> None:
    assert list_all_templates(repo_root=tmp_path) == []


# ─── confirmation_count ─────────────────────────────────────────────


def test_confirmation_count_returns_distinct_model_count() -> None:
    entry = ChainedTemplateEntry(
        family_key="sam2",
        template_demo_source="x",
        source_model_id="m1",
        confirmed_models=["m1", "m2", "m3"],
    )
    assert confirmation_count(entry) == 3


def test_confirmation_count_handles_duplicates() -> None:
    """Defensive: even though register_template deduplicates, the count
    helper itself uses set() so manual edits don't break gate logic."""
    entry = ChainedTemplateEntry(
        family_key="sam2",
        template_demo_source="x",
        source_model_id="m1",
        confirmed_models=["m1", "m1", "m2"],
    )
    assert confirmation_count(entry) == 2


def test_confirmation_count_zero_for_empty_list() -> None:
    entry = ChainedTemplateEntry(
        family_key="sam2",
        template_demo_source="x",
        source_model_id="m1",
        confirmed_models=[],
    )
    assert confirmation_count(entry) == 0

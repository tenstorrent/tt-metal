"""Tests for E — crediting duplicate-named components that are the same module.

The credit is by RESOLVED submodule_path identity (same path = same module),
proven via reliable signals only (recorded path + explicit candidate paths),
never by fuzzy class-name match. These tests pin the pure grouping logic, the
strict (no-fuzzy) resolver, the durable store, and the placement credit.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner import alias_dedup as ad  # noqa: E402


def test_normalize_path_index_form() -> None:
    assert ad._normalize_path("text_encoder.layers[0]") == "text_encoder.layers.0"
    assert ad._normalize_path("text_encoder.layers.0") == "text_encoder.layers.0"
    assert ad._normalize_path("encoder.layer[12].attn") == "encoder.layer.12.attn"


def test_group_and_credit_credits_pending_twin() -> None:
    resolved = {
        "layer": "text_encoder.layers.0",
        "encoder_stack": "text_encoder.layers.0",
        "encoder_layer": "text_encoder.layers.0",
        "other": "text_encoder.layers.1",
    }
    graduated = {"layer", "encoder_stack"}
    credits = ad.group_and_credit(resolved, graduated)
    assert set(credits.keys()) == {"encoder_layer"}  # pending twin credited
    assert credits["encoder_layer"] in graduated  # twin is a graduated sibling
    assert "other" not in credits  # different path, not credited


def test_group_and_credit_no_credit_without_graduated_sibling() -> None:
    resolved = {"a": "enc.layers.0", "b": "enc.layers.0"}
    assert ad.group_and_credit(resolved, set()) == {}


def test_group_and_credit_singletons_ignored() -> None:
    resolved = {"a": "enc.layers.0", "b": "enc.layers.1"}
    assert ad.group_and_credit(resolved, {"a"}) == {}


def test_strict_resolve_uses_recorded_then_candidates_no_fuzzy() -> None:
    class _Model:
        pass

    valid = {"text_encoder.layers.0", "text_encoder.layers[0]"}

    def fake_resolve(_model, path):
        if path in valid:
            return object()
        raise AttributeError(path)

    assert ad._strict_resolve_path(_Model(), "text_encoder.layers.0", [], fake_resolve) == "text_encoder.layers.0"
    assert (
        ad._strict_resolve_path(_Model(), None, ["nope.path", "text_encoder.layers[0]"], fake_resolve)
        == "text_encoder.layers.0"
    )
    assert ad._strict_resolve_path(_Model(), None, ["bogus.a", "bogus.b"], fake_resolve) is None


def test_alias_credit_store_roundtrip(tmp_path, monkeypatch) -> None:
    om = importlib.import_module("scripts.tt_hw_planner.overlay_manager")
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test"
    assert om.load_alias_credits(model_id) == {}
    om.persist_alias_credit(model_id, "encoder_layer", canonical_path="text_encoder.layers.0", twin="layer")
    listing = om.load_alias_credits(model_id)
    assert listing["encoder_layer"]["canonical_path"] == "text_encoder.layers.0"
    assert listing["encoder_layer"]["twin"] == "layer"
    om.persist_alias_credit(model_id, "encoder_layer", canonical_path="x", twin="y")
    assert om.load_alias_credits(model_id)["encoder_layer"]["twin"] == "layer"
    assert om.remove_alias_credit(model_id, "encoder_layer") is True
    assert om.load_alias_credits(model_id) == {}
    assert om.remove_alias_credit(model_id, "encoder_layer") is False


def test_alias_credit_routes_component_to_on_device(tmp_path, monkeypatch) -> None:
    fc = importlib.import_module("scripts.tt_hw_planner.final_categorization")
    om = importlib.import_module("scripts.tt_hw_planner.overlay_manager")
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    demo.mkdir(parents=True)
    import json

    status = {
        "new_model_id": "test/m",
        "components": [{"name": "encoder_layer", "status": "NEW", "submodule_path": "text_encoder.layers.0"}],
    }
    (demo / "bringup_status.json").write_text(json.dumps(status))

    rep = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set=set())
    assert rep.pending == ["encoder_layer"]

    om.persist_alias_credit("test/m", "encoder_layer", canonical_path="text_encoder.layers.0", twin="layer")
    rep2 = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set=set())
    assert rep2.on_device == ["encoder_layer"]
    assert rep2.pending == []

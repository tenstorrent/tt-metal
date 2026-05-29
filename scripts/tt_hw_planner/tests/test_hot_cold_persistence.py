"""Unit tests for HOT/COLD persistence in overlay_manager."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _om():
    return importlib.import_module("scripts.tt_hw_planner.overlay_manager")


def test_load_returns_empty_when_missing(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    assert om.load_hot_cold("facebook/nothing") == {}


def test_persist_and_load_roundtrip(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test"
    om.persist_hot_cold(model_id, {"a": "HOT", "b": "COLD", "c": "UNRESOLVED"})
    out = om.load_hot_cold(model_id)
    assert out == {"a": "HOT", "b": "COLD", "c": "UNRESOLVED"}


def test_persist_overwrites_not_merges(tmp_path, monkeypatch) -> None:
    """The classifier is the source of truth -- a new persist must
    OVERWRITE the prior file, not merge. If a component was HOT in
    one workload but COLD in the new one, we want the new answer."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test"
    om.persist_hot_cold(model_id, {"a": "HOT", "b": "HOT"})
    om.persist_hot_cold(model_id, {"a": "COLD"})  # b gone, a flipped
    assert om.load_hot_cold(model_id) == {"a": "COLD"}


def test_is_cold_component_only_returns_true_for_explicit_cold(tmp_path, monkeypatch) -> None:
    """Conservative: HOT and UNRESOLVED both return False. Only
    explicit COLD classification returns True. Missing entries also
    return False (we don't have signal -> don't skip)."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test"
    om.persist_hot_cold(model_id, {"a": "HOT", "b": "COLD", "c": "UNRESOLVED"})
    assert om.is_cold_component(model_id, "a") is False
    assert om.is_cold_component(model_id, "b") is True
    assert om.is_cold_component(model_id, "c") is False
    assert om.is_cold_component(model_id, "not_in_list") is False


def test_malformed_file_returns_empty_not_raise(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test"
    p = om._hot_cold_path(model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not valid json {{{")
    assert om.load_hot_cold(model_id) == {}
    assert om.is_cold_component(model_id, "anything") is False


def test_auto_iterate_loop_loads_hot_cold_into_permanently_skipped() -> None:
    """The auto-iterate loop must load hot_cold.json at session start
    and add COLD components to permanently_skipped. Without this, the
    classifier's output is decorative -- the picker would still target
    components the workload doesn't even invoke."""
    src = (Path(_REPO_ROOT) / "scripts" / "tt_hw_planner" / "_cli_helpers" / "auto_iterate.py").read_text()
    assert "load_hot_cold" in src, (
        "_run_auto_iterate_loop must import load_hot_cold to respect " "the workload-specific HOT/COLD classification"
    )
    assert "_hot_cold = load_hot_cold(MODEL)" in src, "must load the hot_cold classification at session start"
    # Within the next ~600 chars, permanently_skipped must be extended
    hc_start = src.find("_hot_cold = load_hot_cold(MODEL)")
    assert hc_start != -1
    hc_block = src[hc_start : hc_start + 800]
    assert "permanently_skipped.extend" in hc_block, (
        "loaded COLD entries must be added to permanently_skipped, "
        "otherwise the candidate pool keeps them and the loop wastes "
        "LLM budget on components that never execute in the workload"
    )


def test_classify_hot_cold_cli_subcommand_wired() -> None:
    """Source-grep check that `classify-hot-cold` is registered in the
    CLI parser. Without this, the command isn't usable from the CLI
    even though the implementation exists."""
    cli_mod = importlib.import_module("scripts.tt_hw_planner.cli")
    src = (Path(cli_mod.__file__)).read_text()
    assert '"classify-hot-cold"' in src
    assert "cmd_classify_hot_cold" in src
    assert "from .commands.classify_hot_cold import cmd_classify_hot_cold" in src

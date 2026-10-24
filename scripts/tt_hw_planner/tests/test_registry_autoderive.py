"""Offline tests for the upstream auto-derivation hop (fixes-plan Point 2a).

The synced-tree walk turns already-implemented upstream modules into REUSE/ADAPT
targets and new upstream demos into ranked sibling candidates, so bring-up wraps
them instead of writing from scratch. These tests exercise the pure-derivation
functions + the overlay wiring without any network.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _rs():
    return importlib.import_module("scripts.tt_hw_planner.registry_sync")


def test_derive_reuse_concepts_from_module_tree(tmp_path) -> None:
    rs = _rs()
    d = tmp_path / "models" / "tt_transformers" / "tt"
    d.mkdir(parents=True)
    (d / "fancy_norm.py").write_text("import ttnn\n\n\nclass FancyNorm:\n    pass\n")
    (d / "__init__.py").write_text("")
    (d / "_private.py").write_text("class X:\n    pass\n")
    concepts = rs._derive_reuse_concepts(tmp_path)
    names = {c["concept"] for c in concepts}
    assert "fancy_norm" in names, names
    assert "_private" not in names and "" not in names
    entry = next(c for c in concepts if c["concept"] == "fancy_norm")
    assert entry["tt_class"] == "FancyNorm"
    assert entry["status"] == "ADAPT"
    assert entry["tt_path"] == "models/tt_transformers/tt/fancy_norm.py"


def test_derive_demo_families_infers_category_conservatively() -> None:
    rs = _rs()
    fams = rs._derive_demo_families(["llama9", "whisper_next", "totally_unknown_thing"])
    by_name = {f["name"]: f for f in fams}
    assert "llama9 (auto-upstream)" in by_name
    assert by_name["llama9 (auto-upstream)"]["category"] == "LLM"
    assert by_name["llama9 (auto-upstream)"]["model_type_keys"] == ["llama9"]
    assert "whisper_next (auto-upstream)" in by_name
    assert by_name["whisper_next (auto-upstream)"]["category"] == "STT"
    # no category hint -> conservatively NOT registered (avoid mis-mapping)
    assert "totally_unknown_thing (auto-upstream)" not in by_name


def test_overlay_reuse_entries_are_lowest_priority_and_adapt(tmp_path, monkeypatch) -> None:
    rs = _rs()
    monkeypatch.setenv("TT_HW_PLANNER_CACHE", str(tmp_path))
    overlay = {
        "sha": "T",
        "families": [],
        "concepts": [
            {
                "concept": "some_brand_new_upstream_block",
                "tt_path": "models/tt_dit/x.py",
                "tt_class": "XBlock",
                "status": "ADAPT",
            }
        ],
    }
    (tmp_path / "registry_overlay.json").write_text(json.dumps(overlay))

    reg = importlib.import_module("scripts.tt_hw_planner.reuse_registry")
    reg._overlay_entries.cache_clear()
    hit = reg.lookup_by_concept(None, "some_brand_new_upstream_block")
    assert hit is not None and hit.tt_path == "models/tt_dit/x.py" and hit.status == "ADAPT"
    # never participates in the hf-class lookup (never-matching pattern)
    assert reg.lookup(None, "XBlock") is None
    reg._overlay_entries.cache_clear()

"""Offline tests for the broadened v2 sources + --add-source (fixes-plan Point 10).

Point 10a widens the scanned roots to the v2 sources (common/modules, tt_dit
component dirs) and turns tt_dit/pipelines subdirs into rankable sibling families;
Point 10b lets a new source be registered (persisted) or self-declared by a
``.tt_hw_planner_root`` marker. All pure-derivation — no network.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _rs():
    return importlib.import_module("scripts.tt_hw_planner.registry_sync")


def test_pipeline_subdirs_become_sibling_families_with_category(tmp_path) -> None:
    rs = _rs()
    for name in ("flux1", "wan2_1", "ltx", "stable_diffusion_35_large", "_util"):
        (tmp_path / "models" / "tt_dit" / "pipelines" / name).mkdir(parents=True)
    fams = {f["name"]: f for f in rs._derive_family_roots_candidates(tmp_path)}
    assert fams["tt_dit/flux1 (auto-upstream)"]["category"] == "Image"
    assert fams["tt_dit/flux1 (auto-upstream)"]["pipeline_tags"] == ["text-to-image"]
    assert fams["tt_dit/wan2_1 (auto-upstream)"]["category"] == "Video"
    assert fams["tt_dit/ltx (auto-upstream)"]["category"] == "Video"
    assert "tt_dit/_util (auto-upstream)" not in fams


def test_component_reuse_scans_v2_roots(tmp_path) -> None:
    rs = _rs()
    p1 = tmp_path / "models" / "common" / "modules" / "attention"
    p1.mkdir(parents=True)
    (p1 / "attn_v2.py").write_text("import ttnn\n\n\nclass AttnV2:\n    pass\n")
    p2 = tmp_path / "models" / "tt_dit" / "blocks"
    p2.mkdir(parents=True)
    (p2 / "dit_block.py").write_text("class DiTBlock:\n    pass\n")
    paths = {c["tt_path"] for c in rs._derive_reuse_concepts(tmp_path)}
    assert "models/common/modules/attention/attn_v2.py" in paths
    assert "models/tt_dit/blocks/dit_block.py" in paths


def test_add_source_persists_and_folds_into_fetch_set(tmp_path, monkeypatch) -> None:
    rs = _rs()
    monkeypatch.setenv("TT_HW_PLANNER_CACHE", str(tmp_path))
    rs.add_source("models/tt_v3", kind="component", default="adapt")
    assert any(r.path == "models/tt_v3" for r in rs.load_extra_sources())
    assert any(r.path == "models/tt_v3" for r in rs._configured_roots())
    assert "models/tt_v3" in rs._fetch_path_set()
    # dedup on re-add
    rs.add_source("models/tt_v3", kind="component", default="reuse")
    assert sum(1 for r in rs.load_extra_sources() if r.path == "models/tt_v3") == 1


def test_marker_auto_discovers_root(tmp_path) -> None:
    rs = _rs()
    (tmp_path / "models" / "tt_future" / "mymod").mkdir(parents=True)
    (tmp_path / "models" / "tt_future" / rs._ROOT_MARKER).write_text("")
    discovered = {r.path for r in rs._discover_marker_roots(tmp_path)}
    assert "models/tt_future" in discovered

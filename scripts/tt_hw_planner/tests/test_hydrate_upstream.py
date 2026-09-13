"""hydrate_upstream_into_repo copies the fetched upstream subtrees from the pinned
cache snapshot into repo_root, so new reuse/adapt siblings physically land in the
tree the bring-up ports against (across ALL synced sources, incl. the tt-v2
pattern models/common/modules and models/tt_dit).
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _rs(tmp_cache: Path):
    os.environ["TT_HW_PLANNER_CACHE"] = str(tmp_cache)
    for m in list(sys.modules):
        if m.endswith("registry_sync"):
            del sys.modules[m]
    return importlib.import_module("scripts.tt_hw_planner.registry_sync")


_NEW_SIBLINGS = {
    "models/tt_transformers/tt/new_attention.py": "class NewAttention:\n    pass\n",
    "models/tt_dit/blocks/new_dit_block.py": "class NewDitBlock:\n    pass\n",
    "models/common/modules/new_common_mod.py": "class NewCommonMod:\n    pass\n",
    "models/tt_cnn/tt/new_cnn.py": "class NewCnn:\n    pass\n",
}


def _make_cache_tree(root: Path) -> None:
    for rel, body in _NEW_SIBLINGS.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)


def test_hydrates_all_sources_including_tt_v2_pattern(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    rs = _rs(cache)

    src = tmp_path / "upstream_snapshot"
    _make_cache_tree(src)
    repo = tmp_path / "worktree"
    repo.mkdir()

    tree = rs.UpstreamTree(src, "deadbeefcafe", "remote", False)
    hydrated = rs.hydrate_upstream_into_repo(repo, tree)

    for rel, body in _NEW_SIBLINGS.items():
        assert (repo / rel).is_file(), f"{rel} was not hydrated"
        assert (repo / rel).read_text() == body
    assert set(hydrated) == set(_NEW_SIBLINGS)


def test_overwrite_refreshes_drifted_file_but_add_only_preserves(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    rs = _rs(cache)
    src = tmp_path / "snap"
    (src / "models/tt_transformers/tt").mkdir(parents=True)
    (src / "models/tt_transformers/tt/x.py").write_text("FRESH\n")

    repo = tmp_path / "wt"
    (repo / "models/tt_transformers/tt").mkdir(parents=True)
    (repo / "models/tt_transformers/tt/x.py").write_text("STALE\n")
    tree = rs.UpstreamTree(src, "sha", "remote", False)

    rs.hydrate_upstream_into_repo(repo, tree, overwrite=False)
    assert (repo / "models/tt_transformers/tt/x.py").read_text() == "STALE\n"

    rs.hydrate_upstream_into_repo(repo, tree, overwrite=True)
    assert (repo / "models/tt_transformers/tt/x.py").read_text() == "FRESH\n"


def test_noop_on_local_fallback(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    rs = _rs(cache)
    src = tmp_path / "snap"
    (src / "models/tt_dit/blocks").mkdir(parents=True)
    (src / "models/tt_dit/blocks/b.py").write_text("x\n")
    repo = tmp_path / "wt"
    repo.mkdir()
    tree = rs.UpstreamTree(src, "sha", "local", True)
    assert rs.hydrate_upstream_into_repo(repo, tree) == []
    assert not (repo / "models/tt_dit/blocks/b.py").exists()


def test_add_only_adds_new_sibling_but_never_touches_a_graduated_dep(tmp_path):
    """The wired call uses overwrite=False: a shared module a graduated stub was
    validated against must stay byte-identical, while a genuinely-new sibling is
    still added — so hydration can never regress already-graduated work."""
    cache = tmp_path / "cache"
    cache.mkdir()
    rs = _rs(cache)
    src = tmp_path / "snap"
    (src / "models/tt_dit/layers").mkdir(parents=True)
    (src / "models/tt_dit/layers/existing_dep.py").write_text("NEW_MAIN_API\n")
    (src / "models/tt_dit/layers/brand_new_sibling.py").write_text("class New:\n    pass\n")

    repo = tmp_path / "wt"
    (repo / "models/tt_dit/layers").mkdir(parents=True)
    (repo / "models/tt_dit/layers/existing_dep.py").write_text("GRADUATED_AGAINST_THIS\n")

    hydrated = rs.hydrate_upstream_into_repo(repo, rs.UpstreamTree(src, "sha", "remote", False), overwrite=False)

    assert (repo / "models/tt_dit/layers/existing_dep.py").read_text() == "GRADUATED_AGAINST_THIS\n"
    assert (repo / "models/tt_dit/layers/brand_new_sibling.py").is_file()
    assert "models/tt_dit/layers/brand_new_sibling.py" in hydrated
    assert "models/tt_dit/layers/existing_dep.py" not in hydrated


def test_noop_when_src_equals_dst(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    rs = _rs(cache)
    same = tmp_path / "repo"
    (same / "models/tt_cnn/tt").mkdir(parents=True)
    (same / "models/tt_cnn/tt/c.py").write_text("x\n")
    tree = rs.UpstreamTree(same, "sha", "remote", False)
    assert rs.hydrate_upstream_into_repo(same, tree) == []

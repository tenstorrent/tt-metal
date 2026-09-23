# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic repository map."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "repo_map.py"
spec = importlib.util.spec_from_file_location("repo_map", SCRIPT)
repo_map = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repo_map)


def _tree(tmp_path: Path, modules: dict[str, str], *, git: bool = True) -> Path:
    """A minimal worktree with tt_metal/tt-llk/tests/python_tests modules."""
    llk = tmp_path / "tt_metal/tt-llk"
    tests = llk / "tests/python_tests"
    tests.mkdir(parents=True)
    (llk / "tests/sources").mkdir(parents=True)
    for name, body in modules.items():
        (tests / name).write_text(body)
    if git:
        for args in (["init", "-q"], ["add", "-A"]):
            subprocess.run(
                ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
            )
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-qm",
                "base",
            ],
            check=True,
            capture_output=True,
        )
    return tmp_path


def test_module_level_pytestmark_is_inherited_by_every_test(tmp_path):
    """The form test_perf_header_gate.py uses; a decorator-only scan misses it.

    Getting this wrong reports a host-routed module as unmarked, which sends it
    to silicon and rejects it for having no ELF artifacts - the exact detour
    that cost trial04 19m14s.
    """
    wt = _tree(
        tmp_path,
        {
            "test_gate.py": (
                "import pytest\n"
                "pytestmark = pytest.mark.llk_host\n"
                "def test_a(): pass\n"
                "def test_b(): pass\n"
            )
        },
    )
    doc = repo_map.build_map(wt)
    assert doc["counts"]["host_marked_nodes"] == 2
    assert doc["host_marked_nodes"] == [
        "tests/python_tests/test_gate.py::test_a",
        "tests/python_tests/test_gate.py::test_b",
    ]
    module = doc["test_modules"][0]
    assert module["module_level_markers"] == ["llk_host"]
    assert all("llk_host" in t["markers"] for t in module["tests"])


def test_pytestmark_list_form_and_unresolved_aliases(tmp_path):
    """A list mixes resolvable marks with module aliases; say which is which."""
    wt = _tree(
        tmp_path,
        {
            "test_mixed.py": (
                "import pytest\n"
                "skip_for_wormhole = pytest.mark.skip(reason='x')\n"
                "pytestmark = [pytest.mark.llk_host, skip_for_wormhole]\n"
                "def test_a(): pass\n"
            )
        },
    )
    doc = repo_map.build_map(wt)
    module = doc["test_modules"][0]
    assert "llk_host" in module["module_level_markers"]
    # The alias cannot be resolved statically, so it is named, never guessed.
    assert module["unresolved_module_marks"] == ["skip_for_wormhole"]
    assert doc["counts"]["host_marked_nodes"] == 1


def test_decorator_markers_and_both_parametrize_forms(tmp_path):
    wt = _tree(
        tmp_path,
        {
            "test_axes.py": (
                "import pytest\n"
                "from helpers.param_config import parametrize\n"
                "@pytest.mark.nightly\n"
                "@pytest.mark.parametrize('fmt,arch', [(1, 2)])\n"
                "def test_pytest_form(fmt, arch): pass\n"
                "@parametrize(mathop=['a'], tile_cnt=[1])\n"
                "def test_repo_form(mathop, tile_cnt): pass\n"
            )
        },
    )
    doc = repo_map.build_map(wt)
    tests = {t["name"]: t for t in doc["test_modules"][0]["tests"]}
    assert tests["test_pytest_form"]["markers"] == ["nightly"]
    assert tests["test_pytest_form"]["param_axes"] == ["arch", "fmt"]
    assert tests["test_repo_form"]["param_axes"] == ["mathop", "tile_cnt"]


def test_helpers_and_conftest_are_not_indexed_as_tests(tmp_path):
    wt = _tree(
        tmp_path,
        {
            "test_real.py": "def test_a(): pass\n",
            "conftest.py": "def test_not_a_test(): pass\n",
        },
    )
    (tmp_path / "tt_metal/tt-llk/tests/python_tests/helpers").mkdir()
    (tmp_path / "tt_metal/tt-llk/tests/python_tests/helpers/test_helper.py").write_text(
        "def test_helper_thing(): pass\n"
    )
    doc = repo_map.build_map(wt)
    assert [m["module"] for m in doc["test_modules"]] == ["test_real.py"]


def test_modules_without_tests_are_omitted(tmp_path):
    wt = _tree(
        tmp_path,
        {
            "test_real.py": "def test_a(): pass\n",
            "compare_things.py": "def helper(): pass\n",
        },
    )
    doc = repo_map.build_map(wt)
    assert doc["counts"]["test_modules"] == 1


def test_unparseable_module_is_skipped_not_fatal(tmp_path):
    wt = _tree(
        tmp_path,
        {
            "test_ok.py": "def test_a(): pass\n",
            "test_broken.py": "def test_a( : pass\n",
        },
    )
    doc = repo_map.build_map(wt)
    assert [m["module"] for m in doc["test_modules"]] == ["test_ok.py"]


def test_llk_symbols_are_grouped_by_arch_and_exclude_tests(tmp_path):
    wt = _tree(tmp_path, {"test_a.py": "def test_a(): pass\n"})
    llk = tmp_path / "tt_metal/tt-llk"
    (llk / "tt_llk_blackhole/llk_lib").mkdir(parents=True)
    (llk / "tt_llk_blackhole/llk_lib/llk_unpack.h").write_text(
        "inline void llk_unpack_AB(int a) {}\n"
        "template <bool X>\nvoid _llk_unpack_reduce_(int a) {}\n"
    )
    # a header under tests/ must not enter the index
    (llk / "tests/helper.h").write_text("void llk_test_only(int a) {}\n")
    doc = repo_map.build_map(wt)
    by_arch = doc["llk_symbols_by_arch"]
    assert set(by_arch) == {"tt_llk_blackhole"}
    syms = by_arch["tt_llk_blackhole"]["tt_llk_blackhole/llk_lib/llk_unpack.h"]
    assert syms == ["_llk_unpack_reduce_", "llk_unpack_AB"]


def test_map_id_is_content_addressed(tmp_path):
    wt = _tree(tmp_path, {"test_a.py": "def test_a(): pass\n"})
    first = repo_map.build_map(wt)
    again = repo_map.build_map(wt)
    assert first["map_id"] == again["map_id"]
    (wt / "tt_metal/tt-llk/tests/python_tests/test_b.py").write_text(
        "def test_b(): pass\n"
    )
    assert repo_map.build_map(wt)["map_id"] != first["map_id"]


def test_cache_is_reused_for_the_same_base_commit(tmp_path):
    # Outputs live outside the worktree: writing them inside would leave the
    # tree dirty, and the cache is deliberately keyed by base commit only.
    wt = _tree(tmp_path / "wt", {"test_a.py": "def test_a(): pass\n"})
    cache = tmp_path / "cache"
    out = tmp_path / "m1.json"
    assert (
        repo_map.main(
            ["--worktree", str(wt), "--cache-dir", str(cache), "--out", str(out)]
        )
        == 0
    )
    assert json.loads(out.read_text())["served_from_cache"] is False
    again = tmp_path / "m2.json"
    assert (
        repo_map.main(
            ["--worktree", str(wt), "--cache-dir", str(cache), "--out", str(again)]
        )
        == 0
    )
    assert json.loads(again.read_text())["served_from_cache"] is True
    assert (
        json.loads(again.read_text())["map_id"] == json.loads(out.read_text())["map_id"]
    )


def test_output_written_inside_the_worktree_defeats_the_cache(tmp_path):
    """Recorded because it bit me: the map must be written outside the tree.

    In the pipeline it goes to $LOG_DIR, which is outside the worktree. Writing
    it inside leaves the tree dirty, and a dirty tree is never served from or
    stored in a base-keyed cache.
    """
    wt = _tree(tmp_path / "wt", {"test_a.py": "def test_a(): pass\n"})
    cache = tmp_path / "cache"
    inside = wt / "map-inside.json"
    assert (
        repo_map.main(
            ["--worktree", str(wt), "--cache-dir", str(cache), "--out", str(inside)]
        )
        == 0
    )
    outside = tmp_path / "m.json"
    assert (
        repo_map.main(
            ["--worktree", str(wt), "--cache-dir", str(cache), "--out", str(outside)]
        )
        == 0
    )
    assert json.loads(outside.read_text())["worktree_dirty"] is True
    assert json.loads(outside.read_text())["served_from_cache"] is False


def test_dirty_worktree_is_never_cached_or_served(tmp_path):
    """The cache is keyed by base commit, so a candidate's edits must not enter it."""
    wt = _tree(tmp_path / "wt", {"test_a.py": "def test_a(): pass\n"})
    cache = tmp_path / "cache"
    (wt / "tt_metal/tt-llk/tests/python_tests/test_new.py").write_text(
        "def test_new(): pass\n"
    )
    out = tmp_path / "m.json"
    assert (
        repo_map.main(
            ["--worktree", str(wt), "--cache-dir", str(cache), "--out", str(out)]
        )
        == 0
    )
    doc = json.loads(out.read_text())
    assert doc["worktree_dirty"] is True
    assert doc["served_from_cache"] is False
    # the candidate's tests are visible to this run, but nothing was cached
    assert doc["counts"]["test_modules"] == 2
    assert not list(cache.glob("*.json")) if cache.exists() else True


def test_summary_names_host_nodes_and_stays_small(tmp_path):
    wt = _tree(
        tmp_path,
        {
            "test_gate.py": (
                "import pytest\npytestmark = pytest.mark.llk_host\n"
                "def test_a(): pass\n"
            ),
            "test_dev.py": "def test_b(): pass\n",
        },
    )
    doc = repo_map.build_map(wt)
    text = repo_map.render_summary(doc)
    assert "tests/python_tests/test_gate.py::test_a" in text
    assert "seal to the host wrapper" in text
    # the summary is an index, not a dump
    assert len(text) < 8000


def test_missing_llk_tree_is_rejected(tmp_path):
    (tmp_path / "unrelated").mkdir()
    with pytest.raises(ValueError, match="tt_metal/tt-llk"):
        repo_map.build_map(tmp_path)

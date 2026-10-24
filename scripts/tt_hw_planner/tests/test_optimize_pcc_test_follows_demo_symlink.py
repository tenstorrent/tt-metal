# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A --pcc-test spelled through a symlinked demo folder must reach the isolation worktree.

emit-e2e links its demo from a second folder. optimize resolved --model-dir through that link and
staged only the resolved folder into the worktree, but passed --pcc-test on unchanged, so the gate
named a path the worktree never had and discovery stopped with `--pcc-test file ... not found`
(Mistral-Small-4-119B-2603, 2026-10-13).
"""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_PA = _SCRIPTS.parent / "models" / "experimental" / "perf_automation"
if str(_PA) not in sys.path:
    sys.path.insert(0, str(_PA))

from tt_hw_planner.commands import optimize as opt  # noqa: E402

_REAL = Path("models") / "real_home" / "mymodel"
_LINK = Path("models") / "link_home" / "mymodel"
_GATE = Path("tests") / "e2e" / "test_e2e.py"
_FN = "test_e2e_pcc"
_TEST_SRC = "def %s():\n    assert_with_pcc(a, b, 0.99)\n" % _FN


def _tree(root: Path, with_link: bool = True) -> Path:
    (root / _REAL / _GATE).parent.mkdir(parents=True)
    (root / _REAL / _GATE).write_text(_TEST_SRC)
    if with_link:
        (root / _LINK).parent.mkdir(parents=True)
        (root / _LINK).symlink_to(Path("..") / ".." / _REAL)
    return root


def test_a_relative_node_through_the_link_is_respelled_to_the_real_folder(tmp_path):
    repo = _tree(tmp_path / "repo")
    node = "%s::%s" % ((_LINK / _GATE).as_posix(), _FN)
    assert opt._resolve_test_node(node, repo) == "%s::%s" % ((_REAL / _GATE).as_posix(), _FN)


def test_an_absolute_node_through_the_link_stays_absolute(tmp_path):
    repo = _tree(tmp_path / "repo")
    node = "%s::%s" % (repo / _LINK / _GATE, _FN)
    assert opt._resolve_test_node(node, repo) == "%s::%s" % ((repo / _REAL / _GATE).resolve(), _FN)


def test_a_node_without_a_test_name_keeps_having_none(tmp_path):
    repo = _tree(tmp_path / "repo")
    assert opt._resolve_test_node((_LINK / _GATE).as_posix(), repo) == (_REAL / _GATE).as_posix()


def test_a_node_with_no_link_is_unchanged(tmp_path):
    repo = _tree(tmp_path / "repo")
    node = "%s::%s" % ((_REAL / _GATE).as_posix(), _FN)
    assert opt._resolve_test_node(node, repo) == node


def test_a_missing_file_keeps_its_spelling_for_the_engine_to_report(tmp_path):
    repo = _tree(tmp_path / "repo")
    node = "%s::%s" % ((_LINK / "tests" / "nope.py").as_posix(), _FN)
    assert opt._resolve_test_node(node, repo) == node


def test_a_file_outside_the_repo_keeps_its_spelling(tmp_path):
    repo = _tree(tmp_path / "repo")
    outside = _tree(tmp_path / "elsewhere", with_link=False) / _REAL / _GATE
    node = "%s::%s" % (outside, _FN)
    assert opt._resolve_test_node(node, repo) == node


def test_the_respelled_node_resolves_in_a_worktree_that_has_only_the_real_folder(tmp_path, expect_error):
    """The failure itself: the worktree gets the resolved folder staged, never the link."""
    from agent import model_files as mf

    repo = _tree(tmp_path / "repo")
    worktree = _tree(tmp_path / "worktree", with_link=False)
    node = "%s::%s" % ((_LINK / _GATE).as_posix(), _FN)

    with expect_error(mf.ModelFilesError, "not found"):
        mf.resolve_pcc_node(worktree / _REAL, node, worktree)

    node_rel, _, pcc_abs = mf.resolve_pcc_node(worktree / _REAL, opt._resolve_test_node(node, repo), worktree)
    assert pcc_abs == (worktree / _REAL / _GATE).resolve()
    assert node_rel == "%s::%s" % (_GATE.as_posix(), _FN)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""An optimize worktree must JIT-build kernels with the main tree's toolchain.

The cc engine points TT_METAL_HOME at the worktree, and tt-metal looks for its sfpi kernel compiler
under TT_METAL_HOME/runtime before falling back to the system one. The worktree borrowed python_env
and the build dirs from the main tree but not runtime, so every optimize run compiled kernels with
whatever older sfpi the host had installed. On 2026-10-12 that hung MiniCPM5-2B's device on every
perf test, while the same test passed with the main tree as TT_METAL_HOME.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from tt_hw_planner import worktree as wt_mod  # noqa: E402
from tt_hw_planner.commands import optimize as opt  # noqa: E402


def _main_tree(tmp_path, dirs):
    repo = tmp_path / "main"
    for d in dirs:
        (repo / d).mkdir(parents=True)
        (repo / d / "marker").write_text(d)
    return repo


def test_runtime_is_one_of_the_borrowed_dirs():
    assert "runtime" in opt._HOST_BUILD_DIRS


def test_every_dir_the_main_tree_has_is_linked_to_it(tmp_path):
    repo = _main_tree(tmp_path, opt._HOST_BUILD_DIRS)
    wt = tmp_path / "wt"
    wt.mkdir()
    assert opt._borrow_host_build_dirs(repo, wt) == []
    for d in opt._HOST_BUILD_DIRS:
        assert (wt / d).is_symlink()
        assert (wt / d).resolve() == (repo / d).resolve()


def test_a_dir_the_main_tree_lacks_is_neither_linked_nor_reported(tmp_path):
    repo = _main_tree(tmp_path, ("python_env", "build"))
    wt = tmp_path / "wt"
    wt.mkdir()
    assert opt._borrow_host_build_dirs(repo, wt) == []
    assert not (wt / "runtime").exists() and not (wt / "runtime").is_symlink()


def test_an_existing_link_to_the_main_tree_is_kept(tmp_path):
    repo = _main_tree(tmp_path, ("runtime",))
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "runtime").symlink_to(repo / "runtime")
    assert opt._borrow_host_build_dirs(repo, wt) == []
    assert (wt / "runtime").resolve() == (repo / "runtime").resolve()


def test_a_worktree_dir_that_is_not_the_main_trees_is_reported_and_left_alone(tmp_path):
    repo = _main_tree(tmp_path, ("runtime", "build"))
    wt = tmp_path / "wt"
    (wt / "runtime").mkdir(parents=True)
    (wt / "runtime" / "own").write_text("x")
    assert opt._borrow_host_build_dirs(repo, wt) == ["runtime"]
    assert (wt / "runtime" / "own").read_text() == "x"
    assert (wt / "build").resolve() == (repo / "build").resolve()


def test_a_failed_link_is_reported(tmp_path, monkeypatch):
    repo = _main_tree(tmp_path, ("runtime",))
    wt = tmp_path / "wt"
    wt.mkdir()

    def _refuse(self, target, target_is_directory=False):
        raise OSError("read-only")

    monkeypatch.setattr(Path, "symlink_to", _refuse)
    assert opt._borrow_host_build_dirs(repo, wt) == ["runtime"]


@dataclass
class _Session:
    path: Path


def _isolate(tmp_path, monkeypatch, main_dirs, prepare_wt=None):
    repo = _main_tree(tmp_path, main_dirs)
    demo = repo / "models" / "demos" / "m"
    demo.mkdir(parents=True)
    wt = tmp_path / "wt"
    subprocess.run(["git", "init", "-q", str(wt)], check=True)
    subprocess.run(
        ["git", "-C", str(wt), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "i"],
        check=True,
    )
    if prepare_wt:
        prepare_wt(wt)
    destroyed = []
    monkeypatch.setattr(wt_mod, "create", lambda name: _Session(wt))
    monkeypatch.setattr(wt_mod, "destroy", lambda s, **k: destroyed.append(s.path) or True)
    monkeypatch.setattr(opt, "_stage_untracked_data", lambda *a, **k: None)
    return opt._setup_isolation(repo, demo), wt, repo, destroyed


def test_isolation_links_runtime_into_the_worktree(tmp_path, monkeypatch):
    iso, wt, repo, destroyed = _isolate(tmp_path, monkeypatch, opt._HOST_BUILD_DIRS)
    assert iso is not None and iso["wt"] == wt
    assert (wt / "runtime").resolve() == (repo / "runtime").resolve()
    assert destroyed == []


def test_isolation_still_works_when_the_main_tree_has_no_runtime(tmp_path, monkeypatch):
    iso, wt, _, destroyed = _isolate(tmp_path, monkeypatch, ("python_env", "build"))
    assert iso is not None
    assert not (wt / "runtime").exists()
    assert destroyed == []


def test_isolation_refuses_a_worktree_with_its_own_toolchain(tmp_path, monkeypatch, capsys):
    iso, wt, _, destroyed = _isolate(
        tmp_path, monkeypatch, opt._HOST_BUILD_DIRS, prepare_wt=lambda w: (w / "runtime").mkdir()
    )
    assert iso is None
    assert destroyed == [wt]
    assert "runtime" in capsys.readouterr().out

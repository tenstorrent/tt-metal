# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A hung device step is caught, and the round killed over it leaves nothing behind for the next one.

A 2B LLM, 2026-10-20. An agent single-buffered a C++ matmul's output CB and measured it; the kernel
deadlocked the chip. Three things then turned one bad edit into six hours of nothing:

  * the stall check never fired: the box runs Yama ptrace_scope=1, py-spy (the supervisor's child, a
    sibling of the test) was refused every read, so no stack was ever seen and the test's background
    threads' syscalls counted as progress;
  * the round watchdog killed the round but left the edit in the tree, so the gate check and the next
    round profiled it and hung again;
  * the hung gate check, killed at its 4x ceiling, was booked as a 1891 s profile, which grew every
    later budget (p95 x 6) to the 3 h ceiling.

Paths, file names and contents here are this file's own.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))
sys.path.insert(0, str(_PA / "cc_optimize"))

import agent.probes as P  # noqa: E402

# --- the stack read survives a ptrace refusal ------------------------------------------------------


class _R:
    def __init__(self, rc, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


_FRAMES = 'Thread 1 (idle): "MainThread"\n    wait (hung.py:3)\n        step: 1\n'
_DENIED = _R(1, "", "Permission Denied: Try running again with elevated permissions")


@pytest.fixture
def spy(monkeypatch):
    monkeypatch.setattr(P, "_py_spy_exe", lambda: "/x/py-spy")
    monkeypatch.setattr(P, "_PY_SPY_MODE", ["plain"])
    calls = []

    def install(answer):
        def _run(cmd, **kw):
            calls.append(list(cmd))
            return answer(list(cmd))

        monkeypatch.setattr(subprocess, "run", _run)
        return calls

    return install


def test_a_readable_process_is_read_plainly(spy):
    calls = spy(lambda cmd: _R(0, _FRAMES))
    assert P._stack_fingerprint(7)
    assert calls[0][0] == "/x/py-spy", "a read that works must not be elevated"


def test_a_refused_read_is_retried_elevated_and_the_answer_is_kept(spy):
    calls = spy(lambda cmd: _R(0, _FRAMES) if cmd[0] == "sudo" else _DENIED)
    assert P._stack_fingerprint(7), "ptrace_scope=1 refuses the plain read; the elevated one must be used"
    assert calls[-1][:3] == ["sudo", "-n", "/x/py-spy"]
    n = len(calls)
    assert P._stack_fingerprint(8)
    assert len(calls) == n + 1 and calls[-1][0] == "sudo", "the refusal is learned once, not paid per read"


def test_where_nothing_may_read_it_says_so_once_and_stops_trying(spy, capsys):
    calls = spy(lambda cmd: _R(1, "", "sudo: a password is required") if cmd[0] == "sudo" else _DENIED)
    assert P._stack_fingerprint(7) == ""
    assert "ptrace refused" in capsys.readouterr().err
    n = len(calls)
    assert P._stack_fingerprint(8) == ""
    assert len(calls) == n, "a box that cannot read stacks must not spawn two processes per sample"
    assert "ptrace refused" not in capsys.readouterr().err, "the warning is printed once"


def test_an_unrelated_failure_is_not_taken_for_a_refusal(spy):
    calls = spy(lambda cmd: _R(1, "", "Error: No such process"))
    assert P._stack_fingerprint(7) == ""
    assert all(c[0] != "sudo" for c in calls), "an exited process is not a reason to elevate"
    assert P._PY_SPY_MODE[0] == "plain"


# --- a stall is not booked as the cost of the work ------------------------------------------------


def _device_proc_src():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _run_device_proc(")
    return src[i : src.index("\ndef ", i + 1)]


def test_the_stall_kill_marks_itself_and_the_recording_skips_it():
    body = _device_proc_src()
    stall = body.index("_stall_killed[0] = True")
    assert (
        body.index("raise subprocess.TimeoutExpired(cmd, limit)", stall) - stall < 80
    ), "the mark must sit on the stall kill itself, not on the ceiling or the thermal abort"
    rec = body.index("record_observed(observe_root, observe_op")
    guard = body.rindex("if observe_op and observe_root is not None", 0, rec)
    assert "not _stall_killed[0]" in body[guard:rec]
    assert body.count("_stall_killed[0] = True") == 1, "a ceiling kill is still evidence and still booked"


# --- a killed round's edit is put back -------------------------------------------------------------


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    m = r / "models" / "demo_x"
    (m / "tests").mkdir(parents=True)
    (m / "kernel.py").write_text("BUFFERS = 2\n")
    (m / "tests" / "test_perf.py").write_text("def test(): pass\n")
    (r / "elsewhere.py").write_text("x = 1\n")
    _git(r, "init", "-q")
    _git(r, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    # what was already uncommitted when the round began: instrumentation and a sidecar the tool wrote
    (m / "tests" / "test_perf.py").write_text("def test(): pass  # stage marks\n")
    (m / "tests" / "test_perf.py.caps.json").write_text("{}\n")
    return r


def _run():
    import run as _r

    return _r


def _round_edits(r):
    m = r / "models" / "demo_x"
    (m / "kernel.py").write_text("BUFFERS = 1\n")  # the deadlocking edit
    (m / "new_kernel.cpp").write_text("// written by the killed round\n")
    (m / "tests" / "test_perf.py").write_text("def test(): pass  # stage marks + agent tweak\n")
    (r / "elsewhere.py").write_text("x = 2\n")  # outside the model dir


def test_a_killed_round_s_edit_is_undone_and_what_the_round_found_is_kept(repo, tmp_path):
    run = _run()
    model = repo / "models" / "demo_x"
    snap = run._model_tree_snapshot(repo, model)
    _round_edits(repo)
    patch = tmp_path / "out" / "killed_round_1.patch"
    done = run._restore_model_tree(repo, snap, patch)
    assert (model / "kernel.py").read_text() == "BUFFERS = 2\n", "the hung edit is still in the tree"
    assert not (model / "new_kernel.cpp").exists(), "a file the killed round created survived"
    assert (
        (model / "tests" / "test_perf.py").read_text().endswith("# stage marks\n")
    ), "the restore must return to the round's start, not to HEAD: instrumentation was already there"
    assert (model / "tests" / "test_perf.py.caps.json").exists(), "a file that predates the round was removed"
    assert (repo / "elsewhere.py").read_text() == "x = 2\n", "nothing outside the model dir is touched"
    assert set(done) == {
        "models/demo_x/kernel.py",
        "models/demo_x/new_kernel.cpp",
        "models/demo_x/tests/test_perf.py",
    }
    kept = patch.read_text()
    assert "BUFFERS = 1" in kept and "new_kernel.cpp" in kept, "the discarded edit must stay readable"


def test_a_win_committed_during_the_round_is_not_undone(repo):
    run = _run()
    model = repo / "models" / "demo_x"
    snap = run._model_tree_snapshot(repo, model)
    (model / "kernel.py").write_text("BUFFERS = 3  # banked win\n")
    _git(repo, "add", str(model / "kernel.py"))
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "win")
    (model / "new_kernel.cpp").write_text("// next attempt, killed\n")
    run._restore_model_tree(repo, snap, None)
    assert (model / "kernel.py").read_text() == "BUFFERS = 3  # banked win\n"
    assert not (model / "new_kernel.cpp").exists()
    assert (model / "tests" / "test_perf.py").read_text().endswith("# stage marks\n")


def test_a_round_that_changed_nothing_restores_nothing(repo, tmp_path):
    run = _run()
    snap = run._model_tree_snapshot(repo, repo / "models" / "demo_x")
    patch = tmp_path / "p.patch"
    assert run._restore_model_tree(repo, snap, patch) == []
    assert not patch.exists(), "no patch is written when nothing is discarded"


def test_no_snapshot_means_no_restore(repo):
    run = _run()
    _round_edits(repo)
    assert run._model_tree_snapshot(repo, repo.parent / "not_in_repo") is None
    assert run._restore_model_tree(repo, None, None) == []
    assert (repo / "models" / "demo_x" / "kernel.py").read_text() == "BUFFERS = 1\n"


def test_the_loop_snapshots_each_round_and_restores_only_a_killed_one():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def optimize_pipeline(")
    seg = src[i : src.index("\ndef ", i + 1)]
    snap = seg.index("_model_tree_snapshot(")
    launch = seg.index("_run_round_with_watchdog(")
    assert snap < launch, "the snapshot must be taken before the round runs"
    killed = seg.index("if wedged:", launch)
    restore = seg.index("_restore_model_tree(", launch)
    assert killed < restore < seg.index("else:", killed), "the restore belongs to the killed-round branch"


def test_a_killed_round_is_not_booked_as_the_cost_of_a_round():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _run_round_with_watchdog(")
    body = src[i : src.index("\ndef ", i + 1)]
    rec = body.index('record_observed(repo_root, "round"')
    guard = body.rindex("if ", 0, rec)
    assert "not wedge_reason" in body[guard:rec], "a killed round's length is its cap, and booking it grows the cap"


def test_the_kept_diff_comes_from_the_shared_git_helper(repo, tmp_path):
    from agent import gitio

    (repo / "models" / "demo_x" / "kernel.py").write_text("BUFFERS = 1\n")
    assert "BUFFERS = 1" in gitio.diff_text(repo, "HEAD", "models/demo_x")
    assert "x = 1" not in gitio.diff_text(repo, "HEAD", "models/demo_x"), "the diff is scoped to the model dir"
    assert gitio.diff_text(tmp_path, "HEAD") == "", "outside a repo it says nothing rather than raising"
    seg = _PA.joinpath("cc_optimize", "run.py").read_text()
    seg = seg[seg.index("def _restore_model_tree(") :]
    seg = seg[: seg.index("\ndef ", 1)]
    assert '["git", "diff"' not in seg, "the restore must use gitio, not its own git call"

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.31 (owner 2026-11-13: "bookkeeping eats the Galaxy"), items 1 and 3, CPU only.

(1) every scripts/run_safe_pytest.sh run files its result by what it measured (core/runrec.py), so a gate reuses it
however it was launched; only the newest record of the same runs, only a PASS. A gate record of another task whose
command differs only in launch flags counts too; a record from before FIX.31 is found under today's keys (aliases).
(3) gate.py / batch.py / the safe runner are in the code key by their measuring part only (stamps.BOOKKEEPING): a
bookkeeping-only edit keeps every stamp, a measuring edit (or a new, unlisted definition) invalidates them."""

import os
import shutil
import stat
from pathlib import Path

import pytest

from models.demos.common.bringup.core import batch as B
from models.demos.common.bringup.core import gate as G
from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core import runrec, stamps
from models.demos.common.bringup.core.gate import run_gate
from models.demos.common.bringup.selftest.conftest import git
from models.demos.common.bringup.selftest.test_core import task
from models.demos.common.bringup.selftest.test_gate_one_process import _setup

REPO = Path(__file__).resolve().parents[5]
LADDER_T = "models/demos/common/bringup/tests/test_ladder.py"
LADDER = f"DS41_ENGRAM_TABLE=resident BRINGUP_RUNG=s1 scripts/run_safe_pytest.sh --run-all {LADDER_T}"


def _ladder_task(tid="A.1", cmd=LADDER, step="overseer_fix", **kw):
    return task(tid, cmd, step=step, device=True, paths=["src"], metrics={"ladder_failed": "== 0"}, **kw)


def _ladders(calls) -> list[str]:
    return [c for c in (calls.read_text().splitlines() if calls.exists() else []) if "test_ladder" in c]


def _runner_run(sandbox, monkeypatch, args, rc, env=None, metrics=None):
    """One safe-runner run as scripts/lib/tt_safe_run.sh files it: runrec begin, the test, runrec end."""
    led = sandbox.ledger
    monkeypatch.setenv("BRINGUP_SPEC", str(sandbox.spec_path))
    monkeypatch.setenv(M.RESULTS_ENV, str(led.results_dir))
    monkeypatch.setenv(M.TASK_ENV, "HAND")
    monkeypatch.delenv(M.RECORD_ENV, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    st = sandbox.repo.parent / "runrec.json"
    assert runrec.main(["begin", "--state", str(st), "--", *args]) == 0
    for k, v in (metrics or {}).items():
        M.record(k, v)
    assert runrec.main(["end", "--state", str(st), "--rc", str(rc)]) == 0
    for k in env or {}:
        monkeypatch.delenv(k)
    for k in ("BRINGUP_SPEC", M.TASK_ENV):
        monkeypatch.delenv(k)


def test_the_run_key_ignores_how_a_run_was_launched_not_what_it_measures():
    keys = ("code", "goldens")
    a = runrec.identity(["--run-all", LADDER_T], {"DS41_ENGRAM_TABLE": "resident", "BRINGUP_RUNG": "s1"}, keys)
    b = runrec.identity(
        ["--no-precompile", "--precompile-workers", "8", LADDER_T],
        {"DS41_ENGRAM_TABLE": "resident", "BRINGUP_RUNG": "s1", "BRINGUP_TASK": "X", "TT_METAL_CACHE": "/c",
         "BRINGUP_BATCH_KEY": "k",
         "BRINGUP_HOST_MEM_NEED_GB": "3", "TT_METAL_HOME": str(REPO)},
        keys,
    )  # fmt: skip
    assert runrec.run_key(a) == runrec.run_key(b)
    for other in (
        runrec.identity([LADDER_T], {"DS41_ENGRAM_TABLE": "mmap", "BRINGUP_RUNG": "s1"}, keys),  # another switch
        runrec.identity(["--dev", LADDER_T], dict(a["env"]), keys),  # the watcher
        runrec.identity([LADDER_T], dict(a["env"], TT_METAL_HOME="/other/checkout"), keys),
        runrec.identity([LADDER_T], dict(a["env"]), ("code2", "goldens")),
    ):
        assert runrec.run_key(other) != runrec.run_key(a)
    # a gate unit's runs: the prefix variables are the run's environment
    assert runrec.unit_keys(LADDER, {}, keys) == [runrec.run_key(a)]
    assert runrec.unit_keys(f"echo x && {LADDER}", {}, keys) is None  # a setup step: found by its stamp only


def test_the_env_key_counts_what_measures_only():
    base = {"DS41_ENGRAM_TABLE": "resident"}
    plumbing = dict(base, TT_METAL_CACHE="/c", TT_METAL_HOME=str(REPO), BRINGUP_DEVJOB_DIR="/d")
    plumbing["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"] = f"triage > /tmp/tt-safe-run-triage-{os.getpid()}.log"
    assert stamps.env_key(plumbing) == stamps.env_key(base)  # a shell's gate == the orchestrator's gate
    assert stamps.legacy_env_key(plumbing) != stamps.env_key(plumbing)  # (what a shell's record carried before)
    assert stamps.env_key(dict(base, TT_METAL_HOME="/elsewhere")) != stamps.env_key(base)
    assert stamps.env_key(dict(base, BRINGUP_RUNG="last")) != stamps.env_key(base)


def test_a_runner_record_is_reused_however_the_run_was_launched(sandbox, monkeypatch, tmp_path):
    calls = _setup(sandbox, monkeypatch, tmp_path)
    led = sandbox.tasks(_ladder_task())
    # a devjob / hand run of the same tests: other flags, the switches exported instead of prefixed
    env = {"DS41_ENGRAM_TABLE": "resident", "BRINGUP_RUNG": "s1"}
    _runner_run(sandbox, monkeypatch, ["--no-precompile", LADDER_T], 0, env, {"ladder_failed": 0})
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS", res.summary()
    assert _ladders(calls) == []  # not run again
    assert "filed by the safe runner (hand run of" in res.log.read_text()
    assert M.load("A.1", led.results_dir)["ladder_failed"]["value"] == 0


def test_never_a_failed_run_nor_an_older_pass_behind_it(sandbox, monkeypatch, tmp_path):
    calls = _setup(sandbox, monkeypatch, tmp_path)
    led = sandbox.tasks(_ladder_task())
    env = {"DS41_ENGRAM_TABLE": "resident", "BRINGUP_RUNG": "s1"}
    _runner_run(sandbox, monkeypatch, [LADDER_T], 0, env, {"ladder_failed": 0})
    _runner_run(sandbox, monkeypatch, [LADDER_T], 1, env, {"ladder_failed": 3})  # newer, failed
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS"
    assert _ladders(calls) == ["single " + LADDER_T]  # it ran
    _runner_run(sandbox, monkeypatch, [LADDER_T], 2, env, {})  # a hang is never reused either
    store = runrec.RunStore(sandbox.spec.run_dir("default"))
    key = runrec.unit_keys(LADDER, dict(os.environ), (stamps.code_key(sandbox.repo, sandbox.spec), ""), sandbox.spec)
    assert runrec.find_unit(store, key) is None


def test_another_tasks_gate_of_the_same_runs_is_reused_not_another_switch(sandbox, monkeypatch, tmp_path):
    """X.3 (perf: --no-precompile added) passed its ladder; FIX.30's gate names the same ladder without the flag."""
    calls = _setup(sandbox, monkeypatch, tmp_path)
    contract = f"scripts/run_safe_pytest.sh --run-all {B.TESTS_DIR}test_contract.py"
    two = f"{LADDER} && {contract}"
    x1 = task("X.1", two, step="perf", device=True, paths=["src"], metrics={"ladder_failed": "== 0"})
    f1 = task("F.1", two, step="overseer_fix", device=True, paths=["src"], metrics={"ladder_failed": "== 0"})
    other = f"{LADDER.replace('resident', 'mmap')} && {contract}"
    f2 = task("F.2", other, step="overseer_fix", device=True, paths=["src"], metrics={"ladder_failed": "== 0"})
    led = sandbox.tasks(x1, f1, f2)
    assert run_gate(sandbox.spec, led, "X.1").verdict == "PASS"
    n = len(calls.read_text().splitlines())
    assert len(_ladders(calls)) == 1 and "--no-precompile" in G.gate_command(x1, led.tasks())
    res = run_gate(sandbox.spec, led, "F.1")
    assert res.verdict == "PASS" and "the same runs as X.1#test_ladder" in res.log.read_text()
    assert len(calls.read_text().splitlines()) == n  # F.1: every unit reused
    assert run_gate(sandbox.spec, led, "F.2").verdict == "PASS"
    assert len(_ladders(calls)) == 2 and len(calls.read_text().splitlines()) == n + 1  # another switch: its ladder


def test_a_unit_stamp_depends_only_on_its_own_command():
    """P.51: a determinism unit joined the ladder unit's command and so changed its stamp."""
    t = _ladder_task("P.1", f"{LADDER} && scripts/run_safe_pytest.sh --run-all {B.TESTS_DIR}test_contract.py")
    t["step"] = "perf"
    units = dict(G.gate_units(t, {"P.1": t}))
    assert list(units)[0] == "P.1#test_determinism" and units["P.1#test_determinism"] == B.DETERMINISM_CMD
    assert "test_determinism" not in units["P.1#test_ladder"]


# ---------------------------------------------------------------- (3) bookkeeping out of the code key

FILES = (stamps.GATE_PY, stamps.BATCH_PY, stamps.SAFE_PYTEST_SH, stamps.SAFE_RUN_SH)


def _tree(tmp_path) -> Path:
    repo = tmp_path / "tree"
    for f in FILES:
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, repo / f)
    git(repo.parent, "init", "-q", "-b", "main", str(repo))
    git(repo, "add", "-A")
    git(repo, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "-m", "x")
    return repo


def _edit(repo: Path, path: str, old: str, new: str):
    p = repo / path
    s = p.read_text()
    assert old in s, (path, old)
    p.write_text(s.replace(old, new, 1))


BOOKKEEPING_EDITS = [
    (stamps.GATE_PY, "  units passed: ", "  units that passed: "),  # progress_lines (reporting)
    (stamps.GATE_PY, 'f"# {u}: reused: {how}, not run\\n"', 'f"# {u}: reused ({how}), not run\\n"'),  # run_units log
    (stamps.BATCH_PY, "out[-1].append((u, forms[u]))", "out[-1] += [(u, forms[u])]"),  # gate_groups (launch)
    (stamps.SAFE_PYTEST_SH, 'echo "SAFE_PYTEST_RESULT: INFRA', 'echo "SAFE_PYTEST_RESULT: INFRA!'),  # memory pre-flight
    (stamps.SAFE_RUN_SH, "SAFE_PYTEST_TOTAL_RUNTIME", "SAFE_PYTEST_TOTAL_WALL"),  # the total-runtime line
    (stamps.SAFE_RUN_SH, '--state "$TTRUN_RUNREC_STATE" --rc "$ec"', '--rc "$ec" --state "$TTRUN_RUNREC_STATE"'),
]
MEASURING_EDITS = [
    (stamps.GATE_PY, "env[M.TASK_ENV] = tid", "env[M.TASK_ENV] = tid.upper()"),  # gate_env: what a run gets
    (stamps.GATE_PY, "def run_units_only(", "def a_new_helper():\n    return 1\n\n\ndef run_units_only("),  # unlisted
    (stamps.BATCH_PY, '"BRINGUP_RUNG",\n', ""),  # ITEM_VARS
    (stamps.SAFE_PYTEST_SH, "PYTEST_CMD+=(-x)", "PYTEST_CMD+=(-x -q)"),  # the pytest command
    (stamps.SAFE_RUN_SH, "ttrun_reset_cards() {", "ttrun_reset_cards() {\n    sleep 1"),  # a device reset
]


@pytest.mark.parametrize("path,old,new", BOOKKEEPING_EDITS)
def test_a_bookkeeping_edit_keeps_the_code_key(tmp_path, path, old, new):
    repo = _tree(tmp_path)
    k0 = stamps.code_key(repo)
    _edit(repo, path, old, new)
    assert stamps.code_key(repo) == k0


@pytest.mark.parametrize("path,old,new", MEASURING_EDITS)
def test_a_measuring_edit_changes_the_code_key(tmp_path, path, old, new):
    repo = _tree(tmp_path)
    k0 = stamps.code_key(repo)
    _edit(repo, path, old, new)
    assert stamps.code_key(repo) != k0


def test_every_stamp_survives_a_bookkeeping_edit_and_none_a_measuring_one(sandbox, monkeypatch, tmp_path):
    """End to end: a gate's unit passes; a bookkeeping edit of gate.py keeps it (reused, not run); a measuring edit
    runs it again."""
    calls = _setup(sandbox, monkeypatch, tmp_path)
    for f in (stamps.GATE_PY, stamps.BATCH_PY, stamps.SAFE_RUN_SH):  # (the sandbox's safe runner is a stand-in)
        (sandbox.repo / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, sandbox.repo / f)
    led = sandbox.tasks(_ladder_task())
    delta = sandbox.spec.run_dir("default") / "delta"  # (the task's own delta would keep it anyway: stamps only)
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS"
    n = len(calls.read_text().splitlines())
    _edit(sandbox.repo, stamps.GATE_PY, "  units passed: ", "  units that passed: ")
    _edit(sandbox.repo, stamps.SAFE_RUN_SH, "SAFE_PYTEST_TOTAL_RUNTIME", "SAFE_PYTEST_TOTAL_WALL")
    shutil.rmtree(delta, ignore_errors=True)
    res = run_gate(sandbox.spec, led, "A.1", fresh=False)
    assert res.verdict == "PASS" and len(calls.read_text().splitlines()) == n and res.reused
    _edit(sandbox.repo, stamps.GATE_PY, "env[M.TASK_ENV] = tid", "env[M.TASK_ENV] = tid.upper()")
    shutil.rmtree(delta, ignore_errors=True)
    res = run_gate(sandbox.spec, led, "A.1")
    assert len(calls.read_text().splitlines()) > n and not res.reused


def test_a_record_stamped_before_the_split_is_found_under_todays_keys(sandbox, monkeypatch, tmp_path):
    """The same tree under the key before FIX.31 and today's (stamps.keys_at): its records stay valid; a shell's
    environment key (TT_METAL_HOME / TT_METAL_CACHE) aliases to the orchestrator's."""
    calls = _setup(sandbox, monkeypatch, tmp_path)
    (sandbox.repo / stamps.GATE_PY).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / stamps.GATE_PY, sandbox.repo / stamps.GATE_PY)
    led = sandbox.tasks(_ladder_task())
    sandbox.git("add", "-A")
    sandbox.git("commit", "-q", "-m", "tree")
    legacy, new = stamps.keys_at(sandbox.repo, "HEAD", sandbox.spec)
    assert legacy != new and new == stamps.code_key(sandbox.repo, sandbox.spec)
    run_dir = sandbox.spec.run_dir("default")
    shell = {"TT_METAL_HOME": str(REPO), "TT_METAL_CACHE": "/c"}
    env = G.gate_env(sandbox.spec, led, "A.1")
    [(u, cmd, st, sid)] = [
        x for x in G.unit_stamps(sandbox.spec, led.task("A.1"), env, led.tasks()) if "ladder" in x[1]
    ]
    old = dict(st, code=legacy, env=stamps.legacy_env_key(dict(env, **shell, **{M.TASK_ENV: "A.1"})))
    store = stamps.Store(run_dir)
    rec = {"stamp": old, "verdict": "PASS", "rc": 0, "metrics": {"ladder_failed": {"value": 0, "t": "x"}}, "by": "A.1"}
    store.put(u, stamps.stamp_id(old), rec, [], "A.1")
    assert stamps.legacy_env_key(dict(env, **shell)) != stamps.env_key(env)
    stamps.ensure_aliases(run_dir, sandbox.repo, sandbox.spec, dict(env, **shell))  # a shell's gate saw it once
    res = run_gate(sandbox.spec, led, "A.1")
    assert "stamped before FIX.31" in res.log.read_text()
    assert _ladders(calls) == []


def test_the_safe_runner_files_its_runs_through_its_exit_hook():
    """scripts/run_safe_pytest.sh calls ttrun_runrec_begin with the measuring arguments; tt_safe_run.sh files the
    result on every exit path (an EXIT hook) and turns BRINGUP_BATCH_CMD into a plan file (each run reported)."""
    sh = (REPO / stamps.SAFE_PYTEST_SH).read_text()
    lib = (REPO / stamps.SAFE_RUN_SH).read_text()
    assert 'ttrun_runrec_begin "${_runrec_args[@]}"' in sh
    assert "ttrun_on_exit ttrun_runrec_end" in lib and "runrec plan" in lib and "BRINGUP_RUNREC" in lib
    assert os.access(REPO / stamps.SAFE_PYTEST_SH, os.X_OK) and stat.S_ISREG(
        (REPO / stamps.SAFE_PYTEST_SH).stat().st_mode
    )


def test_a_batch_files_each_run_on_the_environment_it_was_given(sandbox, monkeypatch, tmp_path):
    """--one-process / a gate's shared process: each run of the batch is filed, keyed on the environment the run was
    given (the safe runner's own exports after the start, e.g. its per-pid dispatch-timeout command, never count)."""
    calls = _setup(sandbox, monkeypatch, tmp_path)
    led = sandbox.tasks(_ladder_task())
    entry = {"name": "run0", "group": "u", "args": [LADDER_T], "run_all": True, "log": None, "key": "k",
             "env": {"DS41_ENGRAM_TABLE": "resident", "BRINGUP_RUNG": "s1"}}  # fmt: skip
    plan = B.write_plan(tmp_path / "plan.json", [entry])
    monkeypatch.setenv("BRINGUP_SPEC", str(sandbox.spec_path))
    monkeypatch.setenv(M.RESULTS_ENV, str(led.results_dir))
    monkeypatch.setenv("BRINGUP_BATCH", str(plan))
    st = tmp_path / "rr.json"
    assert runrec.main(["begin", "--state", str(st), "--", B.BATCH_TEST]) == 0
    monkeypatch.setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "7")  # (a runner export after the start)
    metric = {"ladder_failed": {"value": 0, "t": "2026-11-13T20:00:00"}}
    B.results_path(plan).write_text(
        '{"running": null, "runs": {"run0": {"rc": 0, "s": 1.0, "metrics": %s, "files": []}}}'
        % __import__("json").dumps(metric)
    )
    assert runrec.main(["end", "--state", str(st), "--rc", "0"]) == 0
    for k in ("BRINGUP_SPEC", "BRINGUP_BATCH"):
        monkeypatch.delenv(k)
    monkeypatch.undo()  # (the gate's environment as before the run)
    calls = _setup(sandbox, monkeypatch, tmp_path)
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS" and _ladders(calls) == [], res.log.read_text()

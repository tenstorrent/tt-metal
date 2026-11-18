# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F270 (owner 2026-11-18: determinism outranks everything; a gate unit is reused only if it passed on exactly the
code, settings and test inputs it exercises). X.6's gate ran `SKIP_POS=32768,... test_skip_positions.py` and reused a
run of it at its default points: SKIP_POS was in no reuse key. A unit's key now has every variable of its command-line
prefix, the knobs its tests read (stamps.run_env), and a safe-runner record whose command line is unknown is never
reused. The gate log names what a reused unit is reused from and the keys it matched. CPU only, fake units."""

import re
import shutil

from models.demos.common.bringup.core import runrec, stamps
from models.demos.common.bringup.core.gate import run_gate
from models.demos.common.bringup.selftest.test_core import task
from models.demos.common.bringup.selftest.test_count_runs_once import LADDER, LADDER_T, _ladders, _runner_run
from models.demos.common.bringup.selftest.test_gate_one_process import _setup

SWITCHES = {"DS41_ENGRAM_TABLE": "resident", "BRINGUP_RUNG": "s1"}
KNOB_TEST = 'import os\nPOS = os.environ.get("SKIP_POS", "51200")\n'


def _sandbox(sandbox, monkeypatch, tmp_path):
    """The fake safe runner, the ladder test reading a SKIP_POS knob, the model's code (src/) and its settings."""
    calls = _setup(sandbox, monkeypatch, tmp_path)
    (sandbox.repo / LADDER_T).parent.mkdir(parents=True, exist_ok=True)
    (sandbox.repo / LADDER_T).write_text(KNOB_TEST)
    (sandbox.repo / "src/tt").mkdir(parents=True)
    (sandbox.repo / "src/model.py").write_text("def f(x):\n    return x + 1\n")
    (sandbox.repo / "src/tt/settings.py").write_text(
        'TABLE = {"POS": Setting("POS", "51200", "the default history position")}\nsettings = Settings("TOY_", TABLE)\n'
    )
    monkeypatch.delenv("SKIP_POS", raising=False)
    return calls


def _ladder(tid="A.1", cmd=LADDER):
    return task(tid, cmd, step="overseer_fix", device=True, paths=["src"], metrics={"ladder_failed": "== 0"})


def _ladder_reused(res) -> bool:
    """The ladder unit was reused (the gate's determinism unit, FIX.25, is the same run every time)."""
    return any("determinism" not in u for u in res.reused)


def _no_delta(sandbox):
    shutil.rmtree(sandbox.spec.run_dir("default") / "delta", ignore_errors=True)


def test_the_run_env_has_the_knobs_the_tests_read_and_every_prefix_variable(sandbox, monkeypatch, tmp_path):
    _sandbox(sandbox, monkeypatch, tmp_path)
    repo = sandbox.repo
    assert "SKIP_POS" in stamps.knob_names([LADDER_T], repo)
    base = dict(SWITCHES, PATH="/bin")
    assert stamps.run_env(base, [LADDER_T], {}, repo) == SWITCHES
    assert stamps.run_env(dict(base, SKIP_POS="1"), [LADDER_T], {}, repo) == dict(SWITCHES, SKIP_POS="1")
    assert stamps.run_env(dict(base, FOO="1"), [LADDER_T], {"FOO": "1"}, repo) == dict(SWITCHES, FOO="1")
    assert stamps.run_env(base, [LADDER_T], {"BRINGUP_TASK": "X.6"}, repo) == SWITCHES  # where results go
    # the stamp's environment key: a knob set in the gate's environment changes it, an unrelated variable does not
    k0 = stamps.unit_env_key(LADDER, base, repo)
    assert stamps.unit_env_key(LADDER, dict(base, SKIP_POS="32768"), repo) != k0
    assert stamps.unit_env_key(LADDER, dict(base, OTHER_VAR="1"), repo) == k0
    # a safe-runner run's command-line prefix, from the command its launcher names (gate, devjob)
    args = ["--run-all", LADDER_T]
    assert runrec.launch_prefix({}, args) is None  # a plain shell: unknown
    cmd = f"BRINGUP_TASK=X.6 SKIP_POS=1,2 scripts/run_safe_pytest.sh --run-all --no-precompile {LADDER_T}"
    assert runrec.launch_prefix({stamps.UNIT_CMD_ENV: cmd}, args) == {"BRINGUP_TASK": "X.6", "SKIP_POS": "1,2"}
    assert runrec.launch_prefix({stamps.UNIT_CMD_ENV: f"cd /x && {cmd}"}, args) is not None
    assert runrec.launch_prefix({stamps.UNIT_CMD_ENV: f"export FOO=1 && {cmd}"}, args) is None
    assert runrec.launch_prefix({stamps.UNIT_CMD_ENV: f"{cmd} && FOO=1 {cmd[16:]}"}, args) is None  # two prefixes
    keys = ("c", "g")
    assert runrec.unit_keys(f"SKIP_POS=1 {LADDER}", {}, keys, sandbox.spec) != runrec.unit_keys(
        LADDER, {}, keys, sandbox.spec
    )


def test_a_and_d_same_command_another_knob_in_the_environment_runs_the_same_one_is_reused(
    sandbox, monkeypatch, tmp_path
):
    """(a) the same command, SKIP_POS exported (not on the command line) with another value: run again (the task's own
    delta and the stamp both); (d) the same code and environment again: reused, the log names the source and keys."""
    calls = _sandbox(sandbox, monkeypatch, tmp_path)
    led = sandbox.tasks(_ladder())
    monkeypatch.setenv("SKIP_POS", "51200")
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS"
    assert len(_ladders(calls)) == 1
    monkeypatch.setenv("SKIP_POS", "32768")
    res = run_gate(sandbox.spec, led, "A.1")  # (the delta kept: F169 must not keep it either)
    assert res.verdict == "PASS" and not _ladder_reused(res) and len(_ladders(calls)) == 2
    monkeypatch.setenv("SKIP_POS", "51200")
    _no_delta(sandbox)
    res = run_gate(sandbox.spec, led, "A.1")
    assert _ladder_reused(res) and len(_ladders(calls)) == 2
    log = res.log.read_text()
    assert re.search(
        r"reused: same code key [0-9a-f]{8} and env [0-9a-f]{8}, same command \(stamp [0-9a-f]{8}\): "
        r"PASS at \S+ in the A\.1 gate, not run",
        log,
    ), log
    code = stamps.code_key(sandbox.repo, sandbox.spec)
    assert f"same code key {code[:8]} " in log


def test_a_x6_a_run_at_other_points_is_never_reused(sandbox, monkeypatch, tmp_path):
    """X.6: the gate unit names SKIP_POS on its command line; a hand run (devjob) at the default points, another
    task's gate at the default points and a run from a plain shell are not it; a hand run with the same prefix is."""
    calls = _sandbox(sandbox, monkeypatch, tmp_path)
    x6_cmd = f"SKIP_POS=32768,65536 {LADDER}"
    led = sandbox.tasks(_ladder("X.4"), _ladder("X.6", x6_cmd))
    _runner_run(sandbox, monkeypatch, ["--no-precompile", LADDER_T], 0, SWITCHES, {"ladder_failed": 0})
    _runner_run(sandbox, monkeypatch, [LADDER_T], 0, dict(SWITCHES, SKIP_POS="32768,65536"), {"ladder_failed": 0},
                unit_cmd=False)  # fmt: skip
    assert run_gate(sandbox.spec, led, "X.4").verdict == "PASS"  # (reuses the default-point hand run)
    assert _ladders(calls) == []
    res = run_gate(sandbox.spec, led, "X.6")
    assert res.verdict == "PASS" and not _ladder_reused(res) and len(_ladders(calls)) == 1, res.log.read_text()
    # a devjob hand run with the gate unit's prefix: reused, and the log names it
    _no_delta(sandbox)
    shutil.rmtree(sandbox.spec.run_dir("default") / "stamps", ignore_errors=True)
    hand = f"BRINGUP_TASK=X.6 SKIP_POS=32768,65536 scripts/run_safe_pytest.sh --run-all {LADDER_T}"
    _runner_run(sandbox, monkeypatch, ["--run-all", LADDER_T], 0, SWITCHES, {"ladder_failed": 0}, unit_cmd=hand)
    res = run_gate(sandbox.spec, led, "X.6")
    assert _ladder_reused(res) and len(_ladders(calls)) == 1
    log = res.log.read_text()
    assert "filed by the safe runner (hand run of" in log and "in task HAND; same code key" in log, log


def test_a_a_prefix_variable_no_test_names_still_counts(sandbox, monkeypatch, tmp_path):
    """FOO=1 on a hand run's command line (read by no test file): never the gate unit without it."""
    calls = _sandbox(sandbox, monkeypatch, tmp_path)
    led = sandbox.tasks(_ladder())
    foo = f"FOO=1 scripts/run_safe_pytest.sh --run-all {LADDER_T}"
    _runner_run(sandbox, monkeypatch, ["--run-all", LADDER_T], 0, SWITCHES, {"ladder_failed": 0}, unit_cmd=foo)
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS" and not _ladder_reused(res) and len(_ladders(calls)) == 1


def test_b_a_settings_default_change_is_not_reused(sandbox, monkeypatch, tmp_path):
    """(b) tt/settings.py is code: a default's change changes the code key and the code under test; a gate runs."""
    calls = _sandbox(sandbox, monkeypatch, tmp_path)
    led = sandbox.tasks(_ladder())
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS"
    code, under = stamps.code_key(sandbox.repo, sandbox.spec), stamps.under_test_key(
        sandbox.repo, sandbox.spec, ["src"]
    )
    s = sandbox.repo / "src/tt/settings.py"
    s.write_text(s.read_text().replace('"51200"', '"32768"'))
    assert stamps.code_key(sandbox.repo, sandbox.spec) != code
    assert stamps.under_test_key(sandbox.repo, sandbox.spec, ["src"]) != under
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS" and not _ladder_reused(res) and len(_ladders(calls)) == 2


def test_c_a_model_file_change_is_not_reused(sandbox, monkeypatch, tmp_path):
    """(c) a change of the model's code: run again; the same tree again: reused."""
    calls = _sandbox(sandbox, monkeypatch, tmp_path)
    led = sandbox.tasks(_ladder())
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS"
    (sandbox.repo / "src/model.py").write_text("def f(x):\n    return x + 2\n")
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS" and not _ladder_reused(res) and len(_ladders(calls)) == 2
    assert _ladder_reused(run_gate(sandbox.spec, led, "A.1")) and len(_ladders(calls)) == 2

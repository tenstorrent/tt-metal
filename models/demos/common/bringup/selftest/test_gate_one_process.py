# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.29 (OWNER 2026-11-13, "LOAD THE MODEL ONLY ONCE"): a gate runs every unit that may share a process in ONE
batch process (core/batch.gate_groups / gate_entries, core/gate._run_group), each unit keeping its own verdict,
metrics and log; gate_model_loads counts the model loads of the gate's processes. CPU only: a stand-in safe runner
plays the batch process (testing/batch.run_plan's results file) or a single test process."""

import json
import os
import stat

from models.demos.common.bringup.core import batch as B
from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core.gate import run_gate
from models.demos.common.bringup.selftest.conftest import PY
from models.demos.common.bringup.selftest.test_core import task

R = "scripts/run_safe_pytest.sh --run-all --no-precompile"
LADDER = f"BRINGUP_RUNG=last {R} models/demos/common/bringup/tests/test_ladder.py"
CONTRACT = f"{R} models/demos/common/bringup/tests/test_contract.py"
OWN = f"{R} models/demos/deepseek_v41_flash_d_p/tests/test_decoder_skip_decode.py"  # declares BATCH_SHARED
NOT_SHARED = f"{R} models/demos/deepseek_v41_flash_d_p/tests/test_ckv_codec.py"  # does not

FAKE = """import json, os, sys
from pathlib import Path
from models.demos.common.bringup.core import metrics as M
args = [a for a in sys.argv[1:] if a.endswith(".py")]
calls = Path(os.environ["FAKE_CALLS"])
METRIC = {"test_ladder.py": "ladder_failed", "test_contract.py": "contract_checks_failed",
          "test_decoder_skip_decode.py": "served_bytes_differ", "test_determinism.py": "determinism_max_abs_diff",
          "test_ckv_codec.py": "codec_ok"}
VALUE = {}
def total():
    return int((M.load(M.record_task()).get("model_loads") or {}).get("total") or 0)
def one(test):
    name = METRIC[test.rsplit("/", 1)[-1]]
    if os.environ.get("FAKE_FAIL") == name:
        print("E   AssertionError: injected failure of " + name)
        return 1
    M.record(name, VALUE.get(name, 0))
    return 0
if args[-1].endswith("test_batch.py"):
    plan_path = Path(os.environ["BRINGUP_BATCH"])
    plan = json.loads(plan_path.read_text())
    calls.open("a").write("batch " + " ".join(e["name"] for e in plan["entries"]) + "\\n")
    out, saved, rc_all = {"running": None, "runs": {}}, dict(os.environ), 0
    loads = 0
    for e in plan["entries"]:
        os.environ.update(e["env"])
        before = M.load(M.record_task())
        if loads == 0:
            loads = 1
            M.record("model_loads", 1, pid=os.getpid(), total=total() + 1)
        rc = one(e["args"][0])
        after = M.load(M.record_task())
        Path(e["log"]).open("a").write(f"ran {e['name']}\\n" + ("E   AssertionError: injected\\n" if rc else ""))
        os.environ.clear(); os.environ.update(saved)
        out["runs"][e["name"]] = {"rc": rc, "s": 1.0, "loads": 1 if e is plan["entries"][0] else 0,
                                  "metrics": {k: v for k, v in after.items() if before.get(k) != v}, "files": []}
        if rc:
            rc_all = 1
            break
    Path(str(plan_path) + ".results.json").write_text(json.dumps(out))
    sys.exit(rc_all)
calls.open("a").write("single " + args[-1] + "\\n")
M.record("model_loads", 1, pid=os.getpid(), total=total() + 1)
sys.exit(one(args[-1]))
"""


def _setup(sandbox, monkeypatch, tmp_path):
    root = sandbox.repo.parent
    (root / "fake_runner.py").write_text(FAKE)
    runner = sandbox.repo / "scripts/run_safe_pytest.sh"
    runner.parent.mkdir(exist_ok=True)
    runner.write_text(f'#!/bin/bash\nexec {PY} {root / "fake_runner.py"} "$@"\n')
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    monkeypatch.setenv("PYTHONPATH", str(M._REPO))
    return calls


def _gate(cmd):
    return task(
        "A.1",
        cmd,
        step="overseer_fix",
        device=True,
        paths=["src"],
        metrics={
            "ladder_failed": "== 0",
            "contract_checks_failed": "== 0",
            "served_bytes_differ": "== 0",
            "gate_model_loads": "== 1",
        },
    )


def test_the_planner_groups_shareable_units_and_keeps_others_apart():
    units = [("a", LADDER), ("b", CONTRACT), ("c", NOT_SHARED), ("d", OWN), ("e", f"DS41_X=on {OWN}")]
    got = [[u for u, _ in g] for g in B.gate_groups(units, {})]
    assert got == [["a", "b"], ["c"], ["d"], ["e"]]  # a test of its own / another switch value: another process
    entries = B.gate_entries(
        [("a", LADDER), ("b", f"{R} models/demos/deepseek_v41_flash_d_p/tests/test_determinism.py")],
        {},
        {"a": {"BRINGUP_TASK": "A.1"}, "b": {"BRINGUP_TASK": "A.1"}},
        {"a": "a.log", "b": "b.log"},
    )
    assert [e["unit"] for e in entries] == ["b", "a"]  # the determinism run first (FIX.25: the cold sequence)
    assert entries[1]["env"] == {"BRINGUP_TASK": "A.1", "BRINGUP_RUNG": "last"}


def test_a_gate_of_shareable_units_is_one_process_one_load(sandbox, monkeypatch, tmp_path):
    calls = _setup(sandbox, monkeypatch, tmp_path)
    sandbox.write_spec(tests={"one_process": True})
    led = sandbox.tasks(_gate(f"{LADDER} && {CONTRACT} && {OWN}"))
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS", res.summary()
    lines = calls.read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith("batch "), lines  # one process for the gate
    m = M.load("A.1", led.results_dir)
    assert m["gate_model_loads"]["value"] == 1
    assert {"ladder_failed", "contract_checks_failed", "served_bytes_differ"} <= set(m)
    log = res.log.read_text()
    for u in ("A.1#test_ladder", "A.1#test_contract", "A.1#test_decoder_skip_decode"):
        assert f"# {u}: run in the gate's shared process (FIX.29)" in log


def test_separate_processes_count_every_load_and_fail_the_limit(sandbox, monkeypatch, tmp_path):
    calls = _setup(sandbox, monkeypatch, tmp_path)
    sandbox.write_spec(tests={"one_process": False})
    led = sandbox.tasks(_gate(f"{LADDER} && {CONTRACT} && {OWN}"))
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "FAIL"  # unit 1 (determinism && ladder: 2 processes) already loaded twice: > 1
    assert [x.split()[0] for x in calls.read_text().splitlines()] == ["single", "single"]  # stopped there
    assert "determinism_max_abs_diff" in M.load("A.1", led.results_dir)  # (FIX.25's check joined unit 1)
    assert M.load("A.1", led.results_dir)["gate_model_loads"]["value"] == 2


def test_a_failing_test_in_the_shared_process_still_fails_its_unit(sandbox, monkeypatch, tmp_path):
    calls = _setup(sandbox, monkeypatch, tmp_path)
    monkeypatch.setenv("FAKE_FAIL", "contract_checks_failed")
    sandbox.write_spec(tests={"one_process": True})
    led = sandbox.tasks(_gate(f"{LADDER} && {CONTRACT} && {OWN}"))
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "FAIL"
    log = res.log.read_text()
    assert "injected" in log
    assert "served_bytes_differ" not in M.load("A.1", led.results_dir)  # stopped there, as && would
    assert len(calls.read_text().splitlines()) == 1
    assert os.environ.get("FAKE_FAIL") == "contract_checks_failed"
    assert json.loads(json.dumps(True))

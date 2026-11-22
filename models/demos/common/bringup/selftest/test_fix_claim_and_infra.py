# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.31 (OWNER 2026-11-13), items 2 and 4, CPU only.

(2) an overseer fix's gate is its claim: its selftests plus only the device units named with it (overseer fix --unit /
--selftest), not every unit of its --gate tasks chained (FIX.30 got X.3's whole gate and a 442 s determinism unit).
(4) an agent ended by an API / auth error (Cloudflare 403, 529, overloaded) is INFRA: no attempt charged, no gate (on a
changed tree either), a back-off and the same work again; a run of such errors stops the task for a person."""

import pytest

from models.demos.common.bringup import orchestrator as O
from models.demos.common.bringup import overseer as OV
from models.demos.common.bringup.core.gate import gate_units
from models.demos.common.bringup.orchestrator import DONE, STOPPED
from models.demos.common.bringup.selftest.conftest import PY
from models.demos.common.bringup.selftest.test_orchestrator import impl_task, orch  # noqa: F401 (fixture)

R = "scripts/run_safe_pytest.sh --run-all"
LADDER = f"DS41_ENGRAM_TABLE=resident BRINGUP_RUNG=s56320 {R} models/demos/common/bringup/tests/test_ladder.py"
PROFILE = f"DS41_ENGRAM_TABLE=resident BRINGUP_FULL_PREFILL=1 {R} models/demos/common/bringup/tests/test_profile.py"
POSITIONS = f"DS41_ENGRAM_TABLE=resident {R} models/demos/common/bringup/tests/test_positions.py"
X3 = {
    "id": "X.3",
    "title": "final",
    "step": "perf",
    "device": True,
    "gate": {
        "cmd": f"{LADDER} && {PROFILE} && {POSITIONS}",
        "metrics": {"err_layer_L*": "<= limit", "op_rows": ">= 1", "pos_chunk": ">= 1", "prefill_ms_full": "> 0"},
    },
}
SELFTEST = "models/demos/common/bringup/selftest/test_one_process.py"


def test_without_units_a_fix_chains_its_gate_tasks_as_before():
    fix = OV.fix_task({"id": "FIX.9", "title": "t", "gate": ["X.3"], "text": "x"}, {"X.3": X3})
    assert fix["gate"]["cmd"].count("BRINGUP_TASK=X.3") == 3 and fix["gate"]["metrics"] == X3["gate"]["metrics"]


def test_a_fix_with_units_runs_its_selftests_and_only_the_named_units(tmp_path):
    tasks = {"X.3": X3}
    job = OV.queue_fix(
        tmp_path, "the positions path", ["X.3"], tasks=tasks, units=["X.3#test_positions"], selftests=[SELFTEST]
    )
    recorded = lambda tid, label: ["pos_chunk", "pos_chunk_limit", "chunk_wall_ms_p51200"]  # noqa: E731
    fix = OV.fix_task(job, tasks, recorded)
    assert fix["gate"]["cmd"] == f"{OV.SELFTEST_RUN} {SELFTEST} && BRINGUP_TASK=X.3 {POSITIONS}"
    assert fix["gate"]["metrics"] == {"pos_chunk": ">= 1"}  # the limits the named unit records, not X.3's all
    assert fix["brief"]["claim"]["units"] == ["X.3#test_positions"]
    units = [u for u, _ in gate_units(fix, dict(tasks, **{fix["id"]: fix}))]
    # the selftest, the named unit; the model-level unit brings FIX.25's determinism (owner: no exemption)
    assert units == [f"{fix['id']}#test_determinism", fix["id"], "X.3"]  # (one segment a task: named by its task)
    assert not any("test_ladder" in c or "test_profile" in c for _, c in gate_units(fix, {"X.3": X3}))


def test_a_selftest_only_fix_has_no_device_unit_and_no_determinism(tmp_path):
    job = OV.queue_fix(tmp_path, "framework only", ["X.3"], tasks={"X.3": X3}, selftests=[SELFTEST])
    fix = OV.fix_task(job, {"X.3": X3})
    assert fix["device"] is False and fix["gate"]["metrics"] == {}
    assert [u for u, _ in gate_units(fix, {"X.3": X3})] == [fix["id"]]


def test_a_named_unit_must_exist(tmp_path):
    with pytest.raises(SystemExit, match="no such unit in X.3's gate"):
        OV.queue_fix(tmp_path, "x", ["X.3"], tasks={"X.3": X3}, units=["X.3#test_contract"])
    with pytest.raises(SystemExit, match="not one of the --gate tasks"):
        OV.queue_fix(tmp_path, "x", ["X.3"], tasks={"X.3": X3}, units=["P.1#test_ladder"])


# ---------------------------------------------------------------- (4) API errors are INFRA


def test_an_api_error_is_recognised_in_the_result_or_the_transcript_end(tmp_path):
    log = tmp_path / "a.jsonl"
    log.write_text('{"type": "result", "result": "API Error: 403 <html>Cloudflare</html>", "is_error": true}\n')
    assert "403" in O.api_error({"result": "API Error: 403 <html>Cloudflare</html>"}, log)
    assert O.api_error({"result": 'API Error: 529 {"type":"overloaded_error"}'}, log)
    log.write_text("")
    assert O.api_error({"result": "the test failed: pcc 0.97"}, log) is None


def test_api_errors_are_infra_not_attempts_and_never_gated(orch, sandbox, monkeypatch):  # noqa: F811
    """FIX.30: three Cloudflare 403s used the free retries, then a gate ran on an unchanged tree."""
    monkeypatch.setattr(O, "API_WAIT_S", 0)
    sandbox.write_spec(agents={"policy": {"implement": {"attempts": 1, "escalate": "stop"}}})
    gates = sandbox.repo.parent / "gates.txt"
    counted = f"echo run >> {gates} && {PY} tests/check.py"
    err = {"exit": 1, "result": "API Error: 403 Forbidden (Cloudflare)"}
    o = orch(
        [impl_task(tests=[], gate={"cmd": counted, "metrics": {"pcc_out": ">= 0.99"}})],
        {
            "C.1.implement.1.md": dict(err, write={"src/impl.txt": "0.5"}),  # cut off mid-work: a changed tree
            "C.1.implement.2.md": err,
            "C.1.implement.3.md": err,
            "C.1.implement.4.md": {"write": {"src/impl.txt": "1.0"}},
        },
    )
    (o.spec.repo / "src/impl.txt").write_text("0.1")
    assert o.run() == DONE  # the one charged attempt was the 4th agent run
    assert len(orch.calls()) == 4
    assert gates.read_text().count("run") == 2  # the check before the agents, and after the real attempt only
    hist = o.led.state()["C.1"]["history"]
    assert sum(h.get("status") == "INFRA" for h in hist) == 3
    assert any("not an attempt, no gate" in ln for ln in orch.lines)
    brief = (o.run_dir / "briefs" / "C.1.implement.2.md").read_text()
    assert "cut off by an API error" in brief


def test_a_run_of_api_errors_stops_the_task_for_a_person(orch, sandbox, monkeypatch):  # noqa: F811
    monkeypatch.setattr(O, "API_WAIT_S", 0)
    monkeypatch.setattr(O, "API_RETRIES", 2)
    err = {"exit": 1, "result": "API Error: 529 overloaded_error"}
    o = orch([impl_task(tests=[])], {f"C.1.implement.{i}.md": err for i in range(1, 6)})
    assert o.run() == STOPPED and len(orch.calls()) == 3
    assert "API error 3 times in a row" in " ".join(o.led.state()["C.1"]["reason"])

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F6: the orchestrator with a scripted mock agent in a throwaway git repo. CPU only, no model, no device."""

import json
import sys

import pytest

from models.demos.common.bringup.orchestrator import DONE, HUMAN, STOPPED, Orchestrator, command_violations
from models.demos.common.bringup.plan import approvals
from models.demos.common.bringup.selftest.conftest import PY, record_cmd

MOCK = f"{sys.executable} -m models.demos.common.bringup.selftest.mock_agent"

# the module under test reports pcc from impl.txt (written by the implement agent); reference 1.0, stub 0.0
CHECK = """
import os
from pathlib import Path
from models.demos.common.bringup.core import metrics as M
impl = os.environ.get("BRINGUP_IMPL", "device")
dev = float(Path("src/impl.txt").read_text()) if Path("src/impl.txt").exists() else 0.1
M.record("pcc_out", {"reference": 1.0, "stub": 0.0, "device": dev}[impl])
"""


@pytest.fixture
def orch(sandbox, monkeypatch, tmp_path):
    script = tmp_path / "script.json"
    monkeypatch.setenv("BRINGUP_AGENT_CMD", MOCK)
    monkeypatch.setenv("MOCK_AGENT_SCRIPT", str(script))
    (sandbox.repo / "tests").mkdir()
    (sandbox.repo / "tests/check.py").write_text(CHECK)
    (sandbox.repo / "src").mkdir()
    lines = []

    def make(tasks, actions):
        script.write_text(json.dumps(actions))
        sandbox.tasks(*tasks)
        return Orchestrator(sandbox.spec, echo=lines.append)

    make.lines = lines
    make.calls = (
        lambda: script.with_suffix(".calls").read_text().split("\n")[:-1]
        if script.with_suffix(".calls").exists()
        else []
    )
    return make


def impl_task(**kw):
    t = {
        "id": "C.1",
        "title": "component",
        "step": "implement",
        "deps": [],
        "tests": ["tests/check.py"],
        "paths": ["src"],
        "gate": {"cmd": f"{PY} tests/check.py", "metrics": {"pcc_out": ">= 0.99"}},
    }
    t.update(kw)
    return t


def test_scripted_step_passes_without_an_agent(orch, sandbox):
    o = orch(
        [
            {
                "id": "G.1",
                "title": "golden",
                "step": "goldens",
                "gate": {"cmd": record_cmd(n=1), "metrics": {"n": "== 1"}},
            }
        ],
        {},
    )
    assert o.run() == DONE
    assert orch.calls() == [] and sandbox.git("log", "-1", "--format=%s") == "[toy][G.1] golden"


def test_implement_freezes_first_then_retries_until_the_gate_passes(orch, sandbox):
    o = orch(
        [impl_task()],
        {
            "C.1.implement.1.md": {"write": {"src/impl.txt": "0.5"}},
            "C.1.implement.2.md": {
                "write": {"src/impl.txt": "0.995"},
                "bash": ["scripts/run_safe_pytest.sh tests/check.py"],
            },
        },
    )
    assert o.run() == DONE
    assert orch.calls() == [
        "C.1.test.1.md bringup-engineer",
        "C.1.implement.1.md bringup-engineer",
        "C.1.implement.2.md bringup-engineer",
    ]
    st = o.led.state()["C.1"]
    assert st["status"] == "PASS" and st["attempts"] == 1
    assert [r["role"] for r in st["agent_runs"]] == ["test", "implement", "implement"]
    assert st["agent"]["session_id"] == "sess-C.1.implement.2" and st["agent"]["model"] == "mock-model"
    assert st["agent"]["defs"]["models/demos/common/bringup/agents/bringup-engineer.md"]["blob"]
    assert sandbox.git("log", "-1", "--format=%s") == "[toy][C.1] component"
    assert sandbox.git("log", "-2", "--format=%s").splitlines()[1] == "[toy][C.1][freeze] component"
    brief2 = (o.run_dir / "briefs" / "C.1.implement.2.md").read_text()
    assert "Previous attempt failed" in brief2 and "pcc_out = 0.5" in brief2


def test_three_failures_hand_over_to_the_debugger_then_stop(orch, sandbox):
    o = orch([impl_task()], {f"C.1.implement.{n}.md": {"write": {"src/impl.txt": "0.5"}} for n in (1, 2, 3)})
    assert o.run() == STOPPED
    calls = orch.calls()
    assert calls[-3:] == [
        "C.1.implement.101.md ttnn-expert-debugger",
        "C.1.implement.102.md ttnn-expert-debugger",
        "C.1.implement.103.md ttnn-expert-debugger",
    ]
    st = o.led.state()["C.1"]
    assert st["status"] == "STOPPED" and st["debugger_attempts"] == 3
    assert "[toy][C.1][wip] component" in sandbox.git("log", "--format=%s")
    assert "WIP commit:" in (o.run_dir / "briefs" / "C.1.implement.101.md").read_text()


def test_debugger_can_rescue(orch):
    acts = {f"C.1.implement.{n}.md": {} for n in (1, 2, 3)}
    acts["C.1.implement.102.md"] = {"write": {"src/impl.txt": "0.999"}}
    o = orch([impl_task()], acts)
    assert o.run() == DONE and o.led.status("C.1") == "PASS"


def test_changes_outside_the_allowed_paths_fail_the_attempt(orch):
    o = orch(
        [impl_task()],
        {
            "C.1.implement.1.md": {"write": {"src/impl.txt": "0.999", "elsewhere.py": "x"}},
            "C.1.implement.2.md": {"write": {"src/impl.txt": "0.999"}},
        },
    )
    assert o.run() == DONE
    runs = o.led.state()["C.1"]["agent_runs"]
    assert any("outside the allowed paths" in p and "elsewhere.py" in p for p in runs[1]["problems"])
    assert len([r for r in runs if r["role"] == "implement"]) == 2


def test_a_test_role_that_edits_the_implementation_is_caught(orch):
    o = orch(
        [impl_task()],
        {
            "C.1.test.1.md": {"write": {"src/impl.txt": "0.999"}},
            "C.1.implement.1.md": {"write": {"src/impl.txt": "0.999"}},
        },
    )
    o.run()
    assert "outside the allowed paths" in " ".join(o.led.state()["C.1"]["agent_runs"][0]["problems"])


def test_direct_device_commands_fail_the_attempt(orch):
    o = orch(
        [impl_task()],
        {
            "C.1.implement.1.md": {"write": {"src/impl.txt": "0.999"}, "bash": ["pytest tests/check.py"]},
            "C.1.implement.2.md": {"write": {"src/impl.txt": "0.999"}},
        },
    )
    assert o.run() == DONE
    assert "direct pytest" in " ".join(o.led.state()["C.1"]["agent_runs"][1]["problems"])


def test_command_violations(sandbox):
    (sandbox.repo / "dev.py").write_text("import ttnn\n")
    (sandbox.repo / "cpu.py").write_text("import torch\n")
    bad = command_violations(
        [
            "python -m pytest x",
            "PYTHONPATH=. python dev.py",
            "python - <<EOF\nimport ttnn\nEOF",
            "python -c 'import ttnn'",
        ],
        sandbox.repo,
    )
    assert len(bad) == 4
    assert (
        command_violations(
            [
                "scripts/run_safe_pytest.sh t.py",
                "python cpu.py",
                "scripts/tt-probe.sh x <<EOF\nimport ttnn\nEOF",
                "git status && ls",
            ],
            sandbox.repo,
        )
        == []
    )


def test_plan_waits_for_approval_then_gates(orch, sandbox):
    plan_cmd = (
        f'{PY} -c "from models.demos.common.bringup.core import metrics as M; '
        "from models.demos.common.bringup.core.spec import Spec; from models.demos.common.bringup.plan import approvals; "
        f"M.record('plan_approved', int(approvals.is_approved(Spec.load('{sandbox.spec_path}'), 'plan')))\""
    )
    o = orch(
        [
            {
                "id": "PL.1",
                "title": "plan",
                "step": "plan",
                "gate": {"cmd": plan_cmd, "metrics": {"plan_approved": "== 1"}},
            },
            {"id": "N.1", "title": "next", "deps": ["PL.1"], "gate": {"cmd": "true"}},
        ],
        {
            "PL.1.plan.1.md": {
                "write": {
                    "bringup/plan.yaml": "placements: []\n",
                    "bringup/plan.md": "# plan\n",
                    "bringup/components.yaml": "components: []\n",
                }
            }
        },
    )
    assert o.run() == HUMAN
    assert o.led.status("PL.1") != "PASS" and "approve plan" in o.led.state()["PL.1"]["waiting"]
    assert o.run() == HUMAN and orch.calls() == ["PL.1.plan.1.md bringup-engineer"]  # no re-planning while waiting
    approvals.approve(sandbox.spec, "plan", by="reviewer")
    assert o.run() == DONE and o.led.status("PL.1") == "PASS" and o.led.status("N.1") == "PASS"


def test_opportunity_list_stops_for_picks(orch):
    o = orch(
        [
            {"id": "X.2", "title": "opportunities", "step": "perf", "gate": {"cmd": "true"}},
            {"id": "X.3", "title": "picked", "step": "perf", "deps": ["X.2"], "gate": {"cmd": "true"}},
        ],
        {},
    )
    assert o.run() == HUMAN and o.led.status("X.3") == "TODO"


def test_resume_after_a_fix(orch):
    o = orch([impl_task()], {})
    assert o.run() == STOPPED
    (o.spec.repo / "src/impl.txt").write_text("0.999")  # a person fixes it
    from models.demos.common.bringup.orchestrator import main

    assert main(["resume", "--spec", str(o.spec.path)]) == DONE

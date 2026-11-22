"""F204: a hang's triage report and Inspector logs survive the next unit, and no framework kill cuts off tt-triage."""

from models.demos.common.bringup import overseer as OV
from models.demos.common.bringup.core import hangdump
from models.demos.common.bringup.core.gate import run_gate
from models.demos.common.bringup.selftest.test_core import task

RUNNER = """#!/bin/bash
# unit x hangs (triage + inspector of its own); any other unit opens a mesh, which rewrites generated/inspector
if [[ "$*" == *test_x* ]]; then
  mkdir -p generated/tt-triage generated/inspector
  echo 'se3_compute cb_wait_front' > generated/tt-triage/triage.txt
  echo 'x programs' > generated/inspector/programs_log.yaml
  exit 2
fi
rm -rf generated/inspector generated/tt-triage; mkdir -p generated/inspector
echo 'y programs' > generated/inspector/programs_log.yaml
exit 0
"""


def test_a_hung_unit_keeps_triage_and_inspector_before_the_next_unit_runs(sandbox):
    runner = sandbox.repo / "scripts/run_safe_pytest.sh"
    runner.parent.mkdir()
    runner.write_text(RUNNER)
    runner.chmod(0o755)
    cmd = "scripts/run_safe_pytest.sh tests/test_y.py && scripts/run_safe_pytest.sh tests/test_x.py"
    cmd += " && scripts/run_safe_pytest.sh tests/test_z.py"
    led = sandbox.tasks(task("A.1", cmd, device=True))
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict in ("HANG", "FAIL")
    kept = sorted(res.log.parent.glob(f"{res.log.stem}.hang-*"))
    assert len(kept) == 1 and "test_x" in kept[0].name, kept
    assert (kept[0] / "tt-triage/triage.txt").read_text().startswith("se3_compute")
    assert (kept[0] / "inspector/programs_log.yaml").read_text().strip() == "x programs"
    assert any("hang dump" in line and str(kept[0]) in line for line in res.lines)


def test_keep_makes_a_new_dir_per_hang_and_nothing_without_a_dump(tmp_path):
    assert hangdump.keep(tmp_path, tmp_path / "k") is None
    (tmp_path / "generated/tt-triage").mkdir(parents=True)
    (tmp_path / "generated/tt-triage/triage.txt").write_text("a")
    log = tmp_path / "logs/K.1.units.log"
    d = hangdump.dest_for(log, "K.1#test_runner_smoke")
    assert d.parent == log.parent and "#" not in d.name and d.name.startswith("K.1.units.hang-K.1_test_runner_smoke-")
    assert (hangdump.keep(tmp_path, d) / "tt-triage/triage.txt").read_text() == "a"


def test_wait_triage_blocks_while_triage_runs():
    seen, slept = iter([True, True, False]), []
    assert hangdump.wait_triage(60, running=lambda: next(seen), sleep=slept.append, say=lambda m: None)
    assert len(slept) == 2


def test_abort_waits_for_triage_before_the_kill(tmp_path):
    bdir, rdir = tmp_path / "b", tmp_path / "r"
    bdir.mkdir()
    rdir.mkdir()
    OV.agent_started(rdir, 4242, "K.1", "contract", 6)
    order = []
    OV.abort(
        bdir, rdir, "x", kill=lambda pg, sig: order.append("kill"), wait_triage=lambda: order.append("wait") or True
    )
    assert order == ["wait", "kill"]

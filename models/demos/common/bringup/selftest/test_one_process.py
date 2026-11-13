# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F259: an A/B side's / a gate unit's model runs in one pytest process, the model loaded once (core/batch.py,
testing/batch.py, tests/test_batch.py). CPU only: the planner, the per-run isolation (env, log, results), the shared
model and the shared mesh fixture, and the orchestrator's A/B through a fake batch process."""

import json
import os
import shlex
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.common.bringup.core import batch as B
from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.orchestrator import HUMAN, InfraStop
from models.demos.common.bringup.selftest.test_perf_ab import CHANGE, LADDER_SPEC, mk, pick  # noqa: F401
from models.demos.common.bringup.testing import batch as TB

T = "models/demos/common/bringup/tests/"
R = "scripts/run_safe_pytest.sh --no-precompile --run-all"
WALL = f"BRINGUP_PROFILE_WALL=1 BRINGUP_POS=64 {R} {T}test_profile.py"
PROF = f"BRINGUP_WALL_CLEAN=1 BRINGUP_POS=64 {R} {T}test_profile.py"


# ---------------------------------------------------------------- the planner (core/batch.py)
def test_only_safe_runs_of_shared_model_tests_may_join():
    assert B.parse(WALL) == {
        "env": {"BRINGUP_PROFILE_WALL": "1", "BRINGUP_POS": "64"},
        "args": [f"{T}test_profile.py"],
        "run_all": True,
    }
    assert B.parse(f"BRINGUP_RUNG=last {R} {T}test_ladder.py")["env"] == {"BRINGUP_RUNG": "last"}
    # FIX.29: the contract test shares too (the adapter's runtime takes the batch's model: the model's loader goes
    # through testing/batch.shared_model), and a model test that declares BATCH_SHARED = True
    assert B.parse(f"{R} {T}test_contract.py")["args"] == [f"{T}test_contract.py"]
    mine = "models/demos/deepseek_v41_flash_d_p/tests/test_decoder_skip_decode.py"
    assert B.parse(f"{R} {mine}")["args"] == [mine]
    no = [
        f"{R} models/demos/x/tests/bringup/contract/test_kv_layout.py",  # a model's own test, not declared: own load
        f"{R} models/demos/deepseek_v41_flash_d_p/tests/test_ckv_codec.py",  # (exists, not declared)
        f"scripts/run_safe_pytest.sh --run-all {T}test_profile.py",  # precompile pass: a process of its own
        f"TT_METAL_DEVICE_PROFILER=1 {R} {T}test_profile.py",  # read at process start
        f"BRINGUP_PROFILER=device {R} {T}test_profile.py",
        f"{R} --dev {T}test_profile.py",
        f"python -m x {T}test_profile.py",
    ]
    assert [B.parse(c) for c in no] == [None] * len(no)
    assert B.parse_all(f"{WALL} && {PROF}") and B.parse_all(f"{WALL} && {R} {T}test_contract.py")
    assert B.parse_all(f"{WALL} && {R} models/demos/x/tests/test_y.py") is None


def test_ladders_first_clean_walls_before_profiled_runs():
    runs = [B.parse(c) for c in (PROF, f"{R} {T}test_positions.py", WALL, f"BRINGUP_RUNG=last {R} {T}test_ladder.py")]
    got = [
        r["args"][0].rsplit("/", 1)[1] + ("+wall" if r["env"].get("BRINGUP_PROFILE_WALL") else "")
        for r in sorted(runs, key=B.phase)
    ]
    assert got == ["test_ladder.py", "test_profile.py+wall", "test_positions.py", "test_profile.py"]


def test_the_model_key_ignores_per_run_variables_only():
    names = {"BRINGUP_POS", "BRINGUP_RUNG", "DS41_X", "BRINGUP_FABRIC"}
    a = B.model_key({"BRINGUP_POS": "0", "DS41_X": "on", "BRINGUP_FABRIC": "T"}, names)
    assert a == B.model_key({"BRINGUP_RUNG": "last", "DS41_X": "on", "BRINGUP_FABRIC": "T"}, names)
    assert a != B.model_key({"DS41_X": "off", "BRINGUP_FABRIC": "T"}, names)  # a switch: another process
    assert a != B.model_key({"DS41_X": "on"}, names)  # a box variable too


def test_a_gate_unit_of_a_clean_wall_and_its_profile_is_one_process(sandbox):
    s = sandbox.write_spec(target={"seq": 128, "chunk": 32})
    run = B.unit_command(f"{WALL} && {PROF}", s)
    # (FIX.29: at least target.seq, whatever loads first)
    assert run.endswith(f"{R} {B.BATCH_TEST}") and f"{B.CMD_ENV}=" in run and f"{B.MAX_SEQ_ENV}=128" in run
    assert B.LM_HEAD_ENV not in run  # no ladder in it
    assert shlex.split(run.split(f"{B.CMD_ENV}=")[1])[0] == f"{WALL} && {PROF}"  # the unit's own command, intact
    plan = B.plan_from_cmd(f"{WALL} && {PROF}")
    assert plan["stop_on_fail"] and [e["env"].get("BRINGUP_WALL_CLEAN") for e in plan["entries"]] == [None, "1"]
    assert B.unit_command(WALL, s) is None  # one run: nothing to share
    assert B.unit_command(f"{WALL} && DS41_X=on {PROF}", s) is None  # two model configurations
    assert B.unit_command(f"{WALL} && echo setup", s) is None


def test_hints_cover_the_ladders_head_and_the_furthest_position(sandbox):
    s = sandbox.write_spec(target={"seq": 128, "chunk": 32})
    runs = [
        B.parse(c)
        for c in (WALL, f"BRINGUP_POS=512 {R} {T}test_profile.py", f"BRINGUP_RUNG=last {R} {T}test_ladder.py")
    ]
    assert B.hints(runs, s) == {B.LM_HEAD_ENV: "1", B.MAX_SEQ_ENV: "544"}


# ---------------------------------------------------------------- the batch process (testing/batch.py)
def test_each_run_gets_its_own_env_log_and_result_and_the_env_comes_back(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEP", "base")
    seen = []

    def main(args, plugins):
        seen.append((args, os.environ.get("BRINGUP_POS"), os.environ.get("KEEP"), os.environ.get(B.KEY_ENV)))
        os.environ["BRINGUP_METRIC_SUFFIX"] = "_leak"  # run_profile sets it in-process: must not reach the next run
        print(f"python out {args[0]}")
        os.write(1, b"c++ log line\n")
        os.write(2, b"stderr line\n")
        return 1 if os.environ.get("FAILME") else 0

    entries = [
        {
            "name": "a.0",
            "group": "a",
            "env": {"BRINGUP_POS": "0"},
            "args": ["t1"],
            "run_all": True,
            "log": str(tmp_path / "a.log"),
            "key": "k",
        },
        {
            "name": "b.0",
            "group": "b",
            "env": {"FAILME": "1"},
            "args": ["t2"],
            "run_all": False,
            "log": str(tmp_path / "b.log"),
            "key": "k",
        },
        {
            "name": "b.1",
            "group": "b",
            "env": {},
            "args": ["t3"],
            "run_all": True,
            "log": str(tmp_path / "b.log"),
            "key": "k",
        },
        {
            "name": "c.0",
            "group": "c",
            "env": {"BRINGUP_POS": "9"},
            "args": ["t4"],
            "run_all": True,
            "log": str(tmp_path / "c.log"),
            "key": "k",
        },
    ]
    plan = B.write_plan(tmp_path / "plan.json", entries)
    failed = TB.run_plan("MESH", (1, 1), json.loads(plan.read_text()), plan, main=main)
    assert failed == ["b.0"]
    assert seen == [(["t1"], "0", "base", "k"), (["t2", "-x"], None, "base", "k"), (["t4"], "9", "base", "k")]
    assert "BRINGUP_METRIC_SUFFIX" not in os.environ and os.environ["KEEP"] == "base" and B.KEY_ENV not in os.environ
    a = (tmp_path / "a.log").read_text()
    assert "python out t1" in a and "c++ log line" in a and "stderr line" in a and "t4" not in a
    got = B.load_results(plan)
    assert got["running"] is None and got["runs"]["a.0"]["rc"] == 0 and got["runs"]["b.0"]["rc"] == 1
    assert got["runs"]["b.1"] == {"rc": None, "skipped": True} and got["runs"]["c.0"]["rc"] == 0  # its group only


def test_stop_on_fail_stops_the_unit_at_its_first_failure(tmp_path):
    entries = [
        {"name": f"run{i}", "group": "unit", "env": {}, "args": [f"t{i}"], "run_all": True, "log": None, "key": "k"}
        for i in range(3)
    ]
    ran = []
    TB.run_plan(
        "MESH",
        (1, 1),
        {"stop_on_fail": True, "entries": entries},
        None,
        main=lambda a, plugins: ran.append(a) or (a[0] == "t1"),
    )
    assert ran == [["t0"], ["t1"]]


def test_a_profiled_run_after_other_runs_of_the_batch_still_gets_its_records(monkeypatch):
    """F262: the runs before it in the process advance the device-operation id past 65535; the real-time profiler's
    records carry it modulo 2**16, and the profiled run (the last of the unit, after its clean walls) still charges
    them to its sections."""
    from models.demos.common.bringup.selftest.test_realtime_profiler import Mesh, deliver, fake_ttnn
    from models.demos.common.bringup.testing import profiler

    st, Op = fake_ttnn(monkeypatch)
    got = {}

    def main(args, plugins):
        if os.environ.get("BRINGUP_WALL_CLEAN") != "1":  # a clean wall: warm-up and timed runs, no profiler
            st.next_id += 40000
            return 0
        st.launched.clear()
        profiler.enable(Mesh(), rt=True)
        try:
            profiler.signpost("attention")
            Op("ttnn.matmul", 3)()
            profiler.signpost("end")
            deliver(st, {i: (10, 20) for i in st.launched})
            profiler.finish_rt()
            got.update(profiler.result()["kernel_ns"])
        finally:
            profiler.disable()
        return 0 if sum(got.values()) > 0 else 1

    plan = B.plan_from_cmd(f"{WALL} && {WALL} && {PROF}")
    assert TB.run_plan("MESH", (1, 1), plan, None, main=main) == []
    assert st.next_id > 65536 and got == {"attention": 60.0}


class Model:
    loads = 0

    def __init__(self, layers, lm_head, seq):
        Model.loads += 1
        self.layers, self.lm_head, self.seq, self.state, self.load_seconds = layers, lm_head, seq, None, 1.0

    def new_state(self, n):
        self.state = SimpleNamespace(freed=False, free=lambda: setattr(self.state, "freed", True))
        return self.state

    def sync(self):
        pass


def fake_spec(seq=128):
    hooks = SimpleNamespace(
        device_model=lambda mesh, s, layers, lm_head: Model(layers, lm_head, s.data["target"]["seq"])
    )
    return SimpleNamespace(data={"target": {"seq": seq, "chunk": 32}}, hooks=lambda: hooks)


@pytest.fixture
def shared(monkeypatch, tmp_path):
    monkeypatch.setenv(M.RESULTS_ENV, str(tmp_path))
    monkeypatch.setenv(M.TASK_ENV, "T")
    monkeypatch.delenv(M.RECORD_ENV, raising=False)  # (a gate's run sets it: the metric would go to the gate's task)
    monkeypatch.delenv(M.SUFFIX_ENV, raising=False)
    Model.loads = 0
    TB.release()
    yield
    TB.release()


def test_the_model_loads_once_for_every_run_of_the_batch(shared, monkeypatch):
    mesh = object()
    assert TB.device_model(fake_spec(), mesh, [0, 1], lm_head=False).lm_head is False  # outside a batch: hooks as is
    monkeypatch.setenv(B.KEY_ENV, "k")
    monkeypatch.setenv(B.LM_HEAD_ENV, "1")
    monkeypatch.setenv(B.MAX_SEQ_ENV, "544")
    m = TB.device_model(fake_spec(), mesh, [0, 1], lm_head=False)
    assert (m.lm_head, m.seq) == (True, 544)  # the head and the furthest position for the runs to come
    st = m.new_state(64)
    s2 = fake_spec()
    m2 = TB.device_model(s2, mesh, [0, 1], lm_head=True)
    assert m2 is m and Model.loads == 2 and st.freed and m.state is None  # the earlier run's state freed
    assert s2.data["target"]["seq"] == 128  # the run's own spec untouched
    assert M.load("T")["model_shared"]["value"] == 1


def test_a_run_needing_another_model_reloads_with_a_warning(shared, monkeypatch, capsys):
    monkeypatch.setenv(B.KEY_ENV, "k")
    mesh = object()
    m = TB.device_model(fake_spec(), mesh, [0, 1], lm_head=False)
    assert TB.device_model(fake_spec(), mesh, [0], lm_head=False) is not m  # other layers
    monkeypatch.setenv(B.KEY_ENV, "k2")  # other switches (the planner never does this)
    TB.device_model(fake_spec(), mesh, [0], lm_head=False)
    assert Model.loads == 3 and capsys.readouterr().out.count("WARNING") == 2


def test_the_mesh_fixture_of_a_run_is_the_batchs_mesh(tmp_path):
    """SharedMesh hands the batch's mesh to a run's mesh_device fixture, whose own open never runs."""
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.fixture
            def mesh_device(request):
                raise RuntimeError("opened a second mesh")
            """
        )
    )
    (tmp_path / "test_x.py").write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
            def test_x(mesh_device):
                assert mesh_device == "SHARED"

            @pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
            def test_other_shape(mesh_device):
                pass
            """
        )
    )
    log = tmp_path / "out.log"
    entries = [
        {
            "name": "x",
            "env": {},
            "args": [str(tmp_path / "test_x.py"), "-p", "no:cacheprovider", "-q"],
            "run_all": True,
            "log": str(log),
            "key": "k",
        }
    ]
    failed = TB.run_plan("SHARED", (1, 1), {"entries": entries}, None)
    text = log.read_text()
    assert failed == ["x"] and "1 passed" in text and "the batch holds (1, 1)" in text, text
    assert "opened a second mesh" not in text


# ---------------------------------------------------------------- the A/B (orchestrator.ab_measure)
def batch_runner(calls, hang_at=None):
    """A fake device: a batch command runs its plan's entries (each writes its metrics and log); others as one run."""

    def run_cmd(cmd, env, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        if f"{B.PLAN_ENV}=" not in cmd:
            calls.append(("single", cmd))
            out = Path(env[M.RESULTS_ENV])
            out.mkdir(parents=True, exist_ok=True)
            (out / "P.1.json").write_text(json.dumps({"metrics": {"chunk_wall_ms": {"value": 1.0}}}))
            log.write_text(f"$ {cmd}\n")
            return 0
        plan = Path(shlex.split(cmd.split(f"{B.PLAN_ENV}=")[1])[0])
        entries = json.loads(plan.read_text())["entries"]
        calls.append(("batch", [e["name"] for e in entries], env.get(B.MAX_SEQ_ENV)))
        res = {"running": None, "runs": {}}
        for e in entries:
            if e["name"] == hang_at:
                res["running"] = e["name"]
                B.results_path(plan).write_text(json.dumps(res))
                log.write_text("SAFE_PYTEST_RESULT: HANG\n")
                return 2
            on = env.get("TOY_FIDELITY") == "hifi2"
            out = Path(e["env"][M.RESULTS_ENV])
            out.mkdir(parents=True, exist_ok=True)
            p = out / "P.1.json"
            data = json.loads(p.read_text()) if p.exists() else {"metrics": {}}
            sfx = f"_p{e['env']['BRINGUP_POS']}" if e["env"].get("BRINGUP_POS") else ""
            vals = {"chunk_wall_ms": 250.0 if on else 840.0, "pcc_layer_L00": 0.999, "device_ms_total": 9.0}
            data["metrics"].update({k + sfx: {"value": v} for k, v in vals.items()})
            p.write_text(json.dumps(data))
            with open(e["log"], "a") as f:
                f.write(f"ran {e['name']} {e['env']}\n")
            res["runs"][e["name"]] = {"rc": 0, "s": 1.0}
        B.results_path(plan).write_text(json.dumps(res))
        log.write_text("batch\n")
        return 0

    return run_cmd


def test_an_ab_side_runs_in_one_process_and_keeps_each_runs_results(mk, sandbox):
    sandbox.write_spec(ladder=LADDER_SPEC, target={"seq": 128, "chunk": 32}, tests={"one_process": True})
    o = mk([pick(ab_mode="quick", ab_positions=[64])], CHANGE)
    calls = []
    o.run_cmd = batch_runner(calls)
    assert o.run() == HUMAN, mk.lines
    assert [c[0] for c in calls] == ["batch", "batch"]  # off, on: one process a side
    assert calls[0][1] == ["quick.0", "p64.0", "p64.1"] and calls[0][2] == "128"  # walls first; target.seq
    d = o.run_dir / "ab" / "P.1"
    rec = o.led.state()["P.1"]["ab"]
    assert rec["quick"]["old"]["chunk_wall_ms"] == 840.0 and rec["quick"]["new"]["chunk_wall_ms"] == 250.0
    assert rec["positions"]["p64"]["new"]["chunk_wall_ms"] == 250.0
    log = (d / "new" / "p64" / "p64.log").read_text()
    assert log.startswith("$ BRINGUP_PROFILE_WALL=1 BRINGUP_POS=64") and "ran p64.0" in log and "ran p64.1" in log
    side = json.loads((d / "new" / "side.json").read_text())["items"]
    assert side["p64"]["stamp"]["cmd"].startswith("BRINGUP_PROFILE_WALL=1")  # the stamp as without F259
    calls.clear()
    task = o.led.task("P.1")
    o.ab_measure(task, "new", rec["on"], o.ab_base(rec["env"], rec["on"], "P.1"), o.ab_code_key(task))
    assert calls == []  # kept runs are reused, not batched again


def test_a_hang_in_the_batch_keeps_the_finished_runs_and_stops(mk, sandbox):
    sandbox.write_spec(ladder=LADDER_SPEC, target={"seq": 128, "chunk": 32}, tests={"one_process": True})
    o = mk([pick(ab_mode="quick", ab_positions=[64])], CHANGE)
    calls = []
    o.run_cmd = batch_runner(calls, hang_at="p64.1")
    task = o.led.task("P.1")
    with pytest.raises(InfraStop, match="p64 hung"):
        o.ab_measure(task, "old", {}, o.ab_base({}, {}, "P.1"), o.ab_code_key(task))
    side = json.loads((o.run_dir / "ab" / "P.1" / "old" / "side.json").read_text())["items"]
    assert list(side) == ["quick"]  # the run that finished is kept; the hung one is measured again on resume


def test_segments_with_their_own_switch_go_to_another_process(mk, sandbox):
    """DeepSeek-V4.1 P.32: the gate's profile item times positions under DS41_DECODER_SKIP=replay128 and =exact: one
    process per switch value, the item's other runs with the side's own; 17 processes a side became 5."""
    sandbox.write_spec(ladder=LADDER_SPEC, target={"seq": 128, "chunk": 32}, tests={"one_process": True})
    pos = lambda v: f"DS41_SKIP={v} BRINGUP_POS=64 {R} {T}test_profile.py"  # noqa: E731
    gate = {"cmd": f"BRINGUP_RUNG=last {R} {T}test_ladder.py && {R} {T}test_profile.py && {pos('a')} && {pos('b')}"}
    o = mk([pick(ab_mode="full", gate=gate)], CHANGE)
    calls = []
    o.run_cmd = batch_runner(calls)
    task = o.led.task("P.1")
    got = o.ab_measure(task, "old", {}, o.ab_base({}, {}, "P.1"), o.ab_code_key(task))
    assert [c[0] for c in calls] == ["batch"] * 3
    assert calls[0][1] == ["last.0", "full.0", "profile.0", "profile.1"]  # ladders first, then the clean wall
    assert calls[1][1] == ["profile.2", "profile.3"] and calls[2][1] == ["profile.4", "profile.5"]
    assert got["profile"]["rc"] == 0 and got["last"]["rc"] == 0


def test_off_every_run_is_its_own_process(mk, sandbox):
    # (on by default since F259's device check, defaults.yaml tests.one_process: off only when a spec says so)
    sandbox.write_spec(ladder=LADDER_SPEC, target={"seq": 128, "chunk": 32}, tests={"one_process": False})
    o = mk([pick(ab_mode="quick", ab_positions=[64])], CHANGE)
    calls = []
    o.run_cmd = batch_runner(calls)
    task = o.led.task("P.1")
    o.ab_measure(task, "old", {}, o.ab_base({}, {}, "P.1"), o.ab_code_key(task))
    assert [c[0] for c in calls] == ["single", "single"]


def test_a_hand_run_batch_keeps_going_past_a_failure_when_asked(monkeypatch):
    monkeypatch.setenv(B.KEEP_GOING_ENV, "1")
    plan = B.plan_from_cmd(f"{WALL} && {PROF}")
    assert not plan["stop_on_fail"] and len({e["group"] for e in plan["entries"]}) == 2  # FIX.29


def test_the_device_test_reads_a_plan_or_a_units_command(monkeypatch, tmp_path):
    monkeypatch.delenv(B.PLAN_ENV, raising=False)
    monkeypatch.delenv("BRINGUP_SPEC", raising=False)
    monkeypatch.setenv(B.CMD_ENV, f"{WALL} && {PROF}")
    plan, path = TB.load_plan()
    assert path is None and len(plan["entries"]) == 2 and plan["hints"] == {}  # (no spec in this test)
    p = B.write_plan(tmp_path / "p.json", [{"name": "a"}])
    monkeypatch.setenv(B.PLAN_ENV, str(p))
    assert TB.load_plan() == ({"version": 1, "stop_on_fail": False, "entries": [{"name": "a"}]}, p)


# ---------------------------------------------------------------- FIX.30: a shared process killed mid-run
def test_a_shared_process_killed_mid_run_reports_every_unit_as_it_really_ended(sandbox, monkeypatch, tmp_path):
    """FIX.30 (X.3, 2026-11-13): the shared process was SIGKILLed (OOM) in its first run, and the gate printed the
    units it never reached as passed (their older PASS records). Every unit's status, everywhere: the summary, the
    ledger, the delta, the stamp store, and no reuse."""
    import signal

    from models.demos.common.bringup.core import gate as G
    from models.demos.common.bringup.core import stamps
    from models.demos.common.bringup.selftest import test_gate_one_process as TG

    calls = TG._setup(sandbox, monkeypatch, tmp_path)
    fake = sandbox.repo.parent / "fake_runner.py"
    kill = (
        "        if os.environ.get('FAKE_KILL') == e['args'][0].rsplit('/', 1)[-1]:\n"
        "            out['running'] = e['name']\n"
        "            Path(str(plan_path) + '.results.json').write_text(json.dumps(out))\n"
        "            os.kill(os.getpid(), 9)\n"
    )
    text = TG.FAKE.replace('        rc = one(e["args"][0])\n', kill + '        rc = one(e["args"][0])\n')
    assert text != TG.FAKE
    fake.write_text(text)
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    sandbox.write_spec(tests={"one_process": True})
    led = sandbox.tasks(TG._gate(f"{TG.LADDER} && {TG.CONTRACT} && {TG.OWN}"))
    assert run_gate_ok(sandbox, led)  # every unit passed once, on older code
    units = [u for u, _ in G.gate_units(led.tasks()["A.1"], led.tasks())]
    assert len(units) == 3  # the ladder (with FIX.25's determinism run), the contract, the model's own test
    (sandbox.repo / "src/impl.py").write_text("x = 2\n")  # the code under test changes: those passes are stale
    monkeypatch.setenv("FAKE_KILL", "test_contract.py")
    calls.write_text("")
    res = G.run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "FAIL" and calls.read_text().startswith("batch ")
    ladder, contract, own = units
    log = res.log.read_text()
    assert f"while running {contract}.0" in log and f"# {own}: {G.NOT_RUN}" in log
    passed = next(ln for ln in res.lines if ln.startswith("  units passed:"))
    assert passed == f"  units passed: {ladder}", res.lines  # ran and passed in the process before the kill
    want = {ladder: "PASS", contract: "FAIL", own: G.NOT_RUN}
    assert G.unit_states(sandbox.spec, led, "A.1") == want
    assert led.state()["A.1"]["units"] == want
    delta = stamps.Delta(sandbox.spec.run_dir(G.run_name(led)), "A.1").all()
    assert {u: delta[u]["verdict"] for u in units} == want
    store = stamps.Store(sandbox.spec.run_dir(G.run_name(led)))
    genv = G.gate_env(sandbox.spec, led, "A.1")
    sids = {u: sid for u, _, _, sid in G.unit_stamps(sandbox.spec, led.tasks()["A.1"], genv, led.tasks())}
    assert store.get(own, sids[own])["verdict"] == G.NOT_RUN and not G.reusable(store.get(own, sids[own]), {}, False)
    # the next gate keeps what passed on this code and runs the rest: never the never-run unit's older pass
    monkeypatch.delenv("FAKE_KILL")
    calls.write_text("")
    res = G.run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS", res.summary()
    assert calls.read_text().split()[1:] == [f"{contract}.0", f"{own}.0"], calls.read_text()
    assert signal.SIGKILL == 9


def run_gate_ok(sandbox, led) -> bool:
    from models.demos.common.bringup.core.gate import run_gate

    return run_gate(sandbox.spec, led, "A.1").verdict == "PASS"


# ---------------------------------------------------------------- FIX.30: host memory (core/hostmem.py)
MEMINFO = {  # bytes, as /proc/meminfo after the leak (DeepSeek-V4.1 X.3, 2026-11-13 09:45, rounded)
    "MemTotal": 566 * 1024**3,
    "MemFree": 3 * 1024**3,
    "MemAvailable": 4 * 1024**3,
    "Buffers": 0,
    "Cached": 2 * 1024**3,
    "SReclaimable": 1 * 1024**3,
    "SUnreclaim": 1 * 1024**3,
    "PageTables": 0,
    "KernelStack": 0,
    "Hugetlb": 36 * 1024**3,
    "SwapTotal": 8 * 1024**3,
    "SwapFree": 0,
}


def test_host_memory_leaked_is_what_no_process_holds():
    from models.demos.common.bringup.core import hostmem as H

    procs = {1: {"ppid": 0, "rss": 200 * 1024**3, "anon": 210 * 1024**3, "hwm": 0}}
    s = H.snapshot(MEMINFO, procs)
    assert s["used"] == 560.0 and s["anon"] == 210.0 and s["leaked"] == 313.0  # 560 - 210 - 36 - 1
    ok, line = H.preflight(100.0, s)
    assert not ok and line.startswith(H.INFRA_TAG) and "100.0 GB" in line and "313.0 GB leaked" in line
    assert H.preflight(3.0, s)[0]
    assert H.need_gb({}) == H.FLOOR_GB and H.need_gb({H.NEED_ENV: "120"}) == 120.0
    assert H.tree(1, {1: {"ppid": 0}, 2: {"ppid": 1}, 3: {"ppid": 2}, 4: {"ppid": 9}}) == [1, 2, 3]


def test_a_runs_record_gives_its_configuration_a_need_and_a_group_takes_its_units(tmp_path, monkeypatch):
    from models.demos.common.bringup.core import gate as G
    from models.demos.common.bringup.core import hostmem as H

    log = tmp_path / "logs" / "X.log"
    log.parent.mkdir()
    a, b = "A=1 scripts/run_safe_pytest.sh x.py", "B=1 scripts/run_safe_pytest.sh y.py"
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"start": H.snapshot(), "peak_rss_gb": 90.0, "peak_anon_gb": 60.0, "min_avail_gb": 0.0}))
    env = dict(os.environ, **G.hostmem_env(log, [a]))
    assert H.NEED_ENV not in env  # never measured
    lines, rec = H.report(w, 137, env)
    assert rec["key"] == G.hostmem_key([a]) and rec["need_gb"] >= 60.0 and "peak RSS 90.0 GB" in lines[0]
    w.write_text(json.dumps({"start": H.snapshot(), "peak_rss_gb": 30.0, "peak_anon_gb": 20.0, "min_avail_gb": 1e9}))
    H.report(w, 0, dict(os.environ, **G.hostmem_env(log, [b])))
    need_a = H.recorded_need(H.records(G.hostmem_file(log)) and G.hostmem_file(log), [G.hostmem_key([a])])
    assert need_a == round(rec["need_gb"] * H.MARGIN + H.MARGIN_GB, 1)
    with open(tmp_path / "gate.log", "w") as f:
        genv = G.hostmem_env(log, [a, b], f)
    assert genv[H.NEED_ENV] == str(need_a) and genv[H.KEY_ENV].startswith("group:")  # its largest unit's need
    assert "host memory (FIX.30): available" in (tmp_path / "gate.log").read_text()

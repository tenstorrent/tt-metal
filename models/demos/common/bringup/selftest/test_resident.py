# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F281 / F282: the resident runner (resident.py, testing/resident.py), the device-weight registry
(testing/weights.py) and the layers a measurement loads (testing/batch.run_layers). CPU only: fake runs, fake
fingerprints, fake device processes."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.common.bringup import resident as R
from models.demos.common.bringup.core import batch as B
from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.testing import batch as TB
from models.demos.common.bringup.testing import resident as TR
from models.demos.common.bringup.testing import weights as W


@pytest.fixture(autouse=True)
def _clean():
    W.enable(False)
    W.clear()
    TB.release()
    TB._SUBSETS.clear()
    yield
    W.enable(False)
    W.clear()
    TB.release()
    TB._SUBSETS.clear()


class FP:
    def __init__(self):
        self.next = {"restart": [], "model": [], "framework": []}
        self.marks = 0

    def changes(self):
        out, self.next = self.next, {"restart": [], "model": [], "framework": []}
        return out

    def mark(self):
        self.marks += 1


def fake_run(calls, rc=0, deferred=(), first=1.5, hang=False):
    def run(mesh, shape, plan, path, keep=False, after_run=None):
        calls.append({"env": dict(os.environ), "names": [e["name"] for e in plan["entries"]], "keep": keep})
        out = {"running": None, "runs": {}}
        TB.DEFERRED.clear()
        failed = []
        for e in plan["entries"]:
            if e["name"] in deferred:
                TB.DEFERRED.append(e["name"])
                continue
            rec = {"rc": rc, "s": 2.0, "metrics": {}, "files": [],
                   "timing": {"model_ready": 1.0, "first_chunk": first, "loaded": True, "t0": time.time()}}  # fmt: skip
            if after_run is not None and after_run(e["name"], rec):
                rec["rc"] = 2
            out["runs"][e["name"]] = rec
            if rc:
                failed.append(e["name"])
        Path(B.results_path(Path(path))).write_text(json.dumps(out))
        return failed

    return run


def mkplan(tmp_path, names=("run0",), name="plan.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"entries": [{"name": n, "env": {}, "args": ["t.py"]} for n in names]}))
    return p


def queue(d, plan, env=None, fresh=False, rid="r1"):
    R._write(d / "queue" / f"{rid}.json",
             {"id": rid, "plan": str(plan), "env": env or {}, "fresh": fresh, "t_submit": time.time()})  # fmt: skip


def server(tmp_path, calls, fp=None, mem=None, **kw):
    d = tmp_path / "res"
    s = TR.Server(
        "MESH", (1, 1), d, run=fake_run(calls, **kw), fingerprint=fp or FP(), mesh_memory=mem or (lambda m: {})
    )
    s.launch_env = {}  # (hermetic: not this test process's TT_* variables)
    return s, d


def test_a_request_runs_its_plan_with_its_env_and_gets_a_timed_reply(tmp_path, monkeypatch):
    calls = []
    s, d = server(tmp_path, calls)
    plan = mkplan(tmp_path, ("a.0", "a.1"))
    queue(d, plan, env={"BRINGUP_TASK": "X.9", "TT_METAL_X": "1", "DS41_DECODER_SKIP": "decode"})
    s.launch_env["TT_METAL_X"] = "1"
    assert TR.tt_env_diff({}, {"TT_METAL_HOME": "/r"}) == {}  # a variable the request does not set: the process's
    assert s.serve(poll=0.01, max_idle=0.2) == "idle"
    r = json.loads((d / "done" / "r1.json").read_text())
    assert r["rc"] == 0 and set(r["runs"]) == {"a.0", "a.1"}
    assert r["timing"]["first_chunk_s"] is not None and r["timing"]["model_loaded"] is True
    assert (
        calls[0]["keep"]
        and calls[0]["env"]["BRINGUP_TASK"] == "X.9"
        and calls[0]["env"]["DS41_DECODER_SKIP"] == "decode"
    )
    assert "BRINGUP_TASK" not in os.environ or os.environ["BRINGUP_TASK"] != "X.9"  # the process env came back
    assert json.loads((d / "status.json").read_text())["phase"] == "ready"
    assert (d / "requests.jsonl").read_text().count('"id"') == 1 and not list((d / "active").iterdir())


@pytest.mark.parametrize(
    "why",
    [
        "build/lib/_ttnncpp.so changed: a shared library of this process (a C++ build)",
        "kernel sources changed: ttnn/cpp/x/kernels/k.cpp (built once per process; a new process rebuilds)",
    ],
)
def test_a_cpp_build_or_kernel_change_restarts_and_the_request_waits_for_the_new_process(tmp_path, why):
    calls, fp = [], FP()
    s, d = server(tmp_path, calls, fp=fp)
    queue(d, mkplan(tmp_path))
    fp.next["restart"] = [why]
    assert s.serve(poll=0.01) == "restart"
    req = json.loads((d / "queue" / "r1.json").read_text())  # still queued: the new process serves it first
    assert req["restarts"][0]["why"] == [why] and not calls
    assert json.loads((d / "restart.json").read_text())["why"] == [why]
    s2, _ = server(tmp_path, calls)  # the new process
    assert s2.serve(poll=0.01, max_idle=0.2) == "idle"
    r = json.loads((d / "done" / "r1.json").read_text())
    assert r["rc"] == 0 and r["restarts"][0]["why"] == [why]


def test_a_fresh_request_needs_a_process_that_ran_nothing(tmp_path):
    calls = []
    s, d = server(tmp_path, calls)
    queue(d, mkplan(tmp_path), fresh=True, rid="r1")
    assert s.serve(poll=0.01, max_idle=0.2) == "idle"  # a pristine process: served here
    queue(d, mkplan(tmp_path, name="p2.json"), fresh=True, rid="r2")
    assert s.serve(poll=0.01) == "restart"
    assert "fresh process" in json.loads((d / "restart.json").read_text())["why"][0]


def test_other_tt_process_variables_restart_with_them(tmp_path):
    calls = []
    s, d = server(tmp_path, calls)
    s.launch_env = {"TT_METAL_HOME": "/a", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "5"}
    queue(d, mkplan(tmp_path), env={"TT_METAL_HOME": "/b"})
    assert s.serve(poll=0.01) == "restart"
    assert json.loads((d / "restart.json").read_text())["env"] == {"TT_METAL_HOME": "/b"}


def test_a_model_change_rebuilds_in_place_only_when_the_old_models_memory_came_back(tmp_path, monkeypatch):
    calls, fp = [], FP()
    mem = {"now": {"DRAM": 100, "L1": 10, "L1_SMALL": 0}}
    s, d = server(tmp_path, calls, fp=fp, mem=lambda m: dict(mem["now"]))
    purged = []
    monkeypatch.setattr(TR, "purge", lambda kinds: purged.append(kinds) or [])
    TB._MODELS["model"] = SimpleNamespace(state=None, sync=lambda: None)
    fp.next["model"] = ["models/demos/x/tt/model.py"]
    queue(d, mkplan(tmp_path), rid="r1")
    assert s.serve(poll=0.01, max_idle=0.2) == "idle"
    r = json.loads((d / "done" / "r1.json").read_text())
    assert r["reloaded"]["rebuild"] is True and purged == [("model", "framework")] and "model" not in TB._MODELS
    TB._MODELS["model"] = SimpleNamespace(state=None, sync=lambda: None)
    mem["now"] = {"DRAM": 100, "L1": 10 + 8192, "L1_SMALL": 0}  # the released model's L1 still held
    fp.next["model"] = ["models/demos/x/tt/attention.py"]
    queue(d, mkplan(tmp_path, name="p2.json"), rid="r2")
    assert s.serve(poll=0.01) == "restart"
    assert "did not come back (L1 8 KiB a chip)" in json.loads((d / "restart.json").read_text())["why"][0]


def test_runs_needing_another_model_continue_in_a_new_process_and_their_results_merge(tmp_path):
    calls = []
    s, d = server(tmp_path, calls, deferred=("b.0",))
    plan = mkplan(tmp_path, ("a.0", "b.0"))
    queue(d, plan)
    assert s.serve(poll=0.01) == "restart"
    req = json.loads((d / "queue" / "r1.json").read_text())
    assert req["only"] == ["b.0"]
    s2, _ = server(tmp_path, calls)
    assert s2.serve(poll=0.01, max_idle=0.2) == "idle"
    assert calls[-1]["names"] == ["b.0"]
    assert set(B.load_results(plan)["runs"]) == {"a.0", "b.0"}
    assert set(json.loads((d / "done" / "r1.json").read_text())["runs"]) == {"a.0", "b.0"}


def test_a_hang_ends_the_process_without_a_reply(tmp_path, monkeypatch):
    log = tmp_path / "triage.log"
    log.write_text("HANG_NO_TRIAGE\n")
    monkeypatch.setenv("TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE", f"echo HANG > {log} 2>&1")
    calls = []
    s, d = server(tmp_path, calls)
    queue(d, mkplan(tmp_path))
    with pytest.raises(RuntimeError, match="device hang"):
        s.serve(poll=0.01)
    assert not (d / "done" / "r1.json").exists()
    assert json.loads((d / "status.json").read_text())["phase"] == "hang"


def test_stop_and_yield_release_the_model(tmp_path):
    s, d = server(tmp_path, [])
    TB._MODELS["model"] = SimpleNamespace(state=None, sync=lambda: None)
    (d / "yield").write_text("1")
    assert s.serve(poll=0.01) == "yield" and "model" not in TB._MODELS
    assert json.loads((d / "status.json").read_text())["phase"] == "yielded"


def test_the_supervisor_restarts_on_request_and_answers_after_a_hang(tmp_path, monkeypatch):
    d = tmp_path / "res"
    monkeypatch.setattr(R, "REPO", tmp_path)
    (d / "active").mkdir(parents=True)
    seen = []

    def run(env, log):
        seen.append(env[R.DIR_ENV])
        if len(seen) == 1:  # the first process asks for a new one (with a variable)
            R._write(d / "done" / "x.json", {"id": "x"})
            R._write(d / "restart.json", {"why": ["a .so changed"], "env": {"TT_METAL_CACHE": "/b"}})
            return 0
        assert env["TT_METAL_CACHE"] == "/b"
        R._write(d / "active" / "r1.json", {"id": "r1"})  # the second hangs in r1
        R._write(d / "queue" / "r2.json", {"id": "r2"})
        Path(log).write_text("SAFE_PYTEST_RESULT: HANG\n")
        return 2

    assert R.supervise(d, "spec.yaml", run=run) == "dead"
    r1, r2 = (json.loads((d / "done" / f"{x}.json").read_text()) for x in ("r1", "r2"))
    assert r1["rc"] == 2 and "HANG" in r1["error"] and r2["rc"] == 1 and "not run" in r2["error"]
    assert json.loads((d / "status.json").read_text())["phase"] == "dead"
    assert not (tmp_path / R.HOLDER).exists() and not (d / "supervisor.json").exists()


def test_a_restart_loop_that_serves_nothing_stops(tmp_path, monkeypatch):
    d = tmp_path / "res"
    monkeypatch.setattr(R, "REPO", tmp_path)
    (d / "queue").mkdir(parents=True)
    R._write(d / "queue" / "r1.json", {"id": "r1"})

    def run(env, log):
        R._write(d / "restart.json", {"why": ["kernel sources changed"]})
        return 0

    assert R.supervise(d, "s", run=run, max_restarts=2) == "dead"
    assert "served nothing" in json.loads((d / "done" / "r1.json").read_text())["error"]


def test_submit_waits_for_the_reply_and_says_when_the_runner_is_gone(tmp_path, monkeypatch):
    d = tmp_path / "res"
    (d / "queue").mkdir(parents=True)
    R._write(d / "supervisor.json", {"pid": os.getpid()})
    R._write(d / "status.json", {"phase": "ready"})
    assert R.healthy(d)

    def answer():
        q = next((d / "queue").glob("*.json"))
        R._write(d / "done" / q.name, {"id": q.stem, "rc": 0, "runs": {}, "timing": {"first_chunk_s": 3.0}})
        q.unlink()  # (the server moved it to active/ and answered)

    import threading

    threading.Timer(0.2, answer).start()
    r = R.submit_plan(d, mkplan(tmp_path), {"A": "1"}, poll=0.05)
    assert r["rc"] == 0 and r["timing"]["first_chunk_s"] == 3.0
    R._write(d / "supervisor.json", {"pid": 2**22 + 12345})  # not a live process
    R._write(d / "status.json", {"phase": "dead", "why": "HANG in r9"})
    r = R.submit_plan(d, mkplan(tmp_path, name="p2.json"), {}, poll=0.05)
    assert r["rc"] == 1 and "HANG in r9" in r["error"] and not list((d / "queue").glob("*.json"))


def test_yield_is_a_no_op_without_a_runner_and_stops_a_live_one(tmp_path, monkeypatch):
    assert R.yield_device(tmp_path) == 0
    d = tmp_path / "res"
    R._write(tmp_path / R.HOLDER, {"dir": str(d), "supervisor": os.getpid()})
    R._write(d / "supervisor.json", {"pid": os.getpid()})
    gone = {"n": 0}

    def fake_alive(pid):
        gone["n"] += 1
        return gone["n"] < 4 and pid == os.getpid()

    monkeypatch.setattr(R, "alive", fake_alive)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    assert R.yield_device(tmp_path) == 0 and (d / "yield").exists()


def test_fingerprint_categories(tmp_path, monkeypatch):
    monkeypatch.setattr(TR, "REPO", tmp_path)
    files = {
        "models/demos/m/tt/model.py": "model",
        "models/demos/common/bringup/testing/harness.py": "framework",
        "models/demos/common/bringup/testing/batch.py": "server",
        "ttnn/ttnn/bringup/x/op.py": "ttnn",
        "models/demos/m/tests/test_a.py": "test",
    }
    mods = {}
    for i, (rel, kind) in enumerate(files.items()):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n")
        assert TR.kind_of(rel) == kind
        mods[f"m{i}"] = SimpleNamespace(__file__=str(p))
    k = tmp_path / "ttnn/cpp/op/device/kernels/k.cpp"
    k.parent.mkdir(parents=True)
    k.write_text("//")
    so = tmp_path / "build/lib/_ttnncpp.so"
    so.parent.mkdir(parents=True)
    so.write_text("elf")
    fp = TR.Fingerprint(tmp_path, modules=lambda: mods, libs=lambda: [str(so)])
    assert fp.changes() == {"restart": [], "model": [], "framework": []}
    time.sleep(0.01)
    for rel in files:
        (tmp_path / rel).write_text("x = 2  # changed\n")
    ch = fp.changes()
    assert ch["model"] == ["models/demos/m/tt/model.py"]
    assert ch["framework"] == ["models/demos/common/bringup/testing/harness.py"]
    assert len(ch["restart"]) == 2 and any("ttnn python" in w for w in ch["restart"])
    assert fp.changes()["model"] == []  # forgotten until imported again
    k.write_text("// changed kernel")
    so.write_text("elf2")
    why = " ".join(fp.changes()["restart"])
    assert "_ttnncpp.so changed" in why and "kernel sources changed: ttnn/cpp/op/device/kernels/k.cpp" in why


def test_request_env_and_tt_differences():
    start = {"PATH": "/bin", "TT_METAL_HOME": "/r", "BRINGUP_TASK": "old", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "5"}
    env = TR.run_env(
        start, {"BRINGUP_TASK": "X.9", "DS41_DECODER_SKIP": "decode", "PATH": "/other", "TT_METAL_HOME": "/x"}
    )
    assert env == {"PATH": "/bin", "TT_METAL_HOME": "/r", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "5",
                   "BRINGUP_TASK": "X.9", "DS41_DECODER_SKIP": "decode"}  # fmt: skip
    assert TR.tt_env_diff({"TT_METAL_HOME": "/r"}, start) == {}
    assert TR.tt_env_diff({"TT_METAL_HOME": "/x"}, start) == {"TT_METAL_HOME": ("/r", "/x")}
    assert TR.triage_log({"TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE": "python3 t.py --x > /a/b.log 2>&1"}) == Path(
        "/a/b.log"
    )
    assert TR.memory_back({"DRAM": 0, "L1": 0}, {"DRAM": 10 << 20, "L1": 0}, kept_bytes=0) is None
    assert "DRAM" in TR.memory_back({"DRAM": 0}, {"DRAM": 1 << 30}, kept_bytes=0)


# ----------------------------------------------------------------------------------------- the weight registry
def test_registry_keeps_file_backed_weights_across_builds_and_reloads_changed_files(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"x" * 100)
    b.write_bytes(b"y" * 50)
    loads = []

    class T:
        def __init__(self, p):
            self.p, self.ok = p, True

        def is_allocated(self):
            return self.ok

    def load(p, dev):
        loads.append(Path(p).name)
        return T(p)

    dev = object()
    assert W.load_tensor(a, dev, load=load).p == a and not W.held()  # off: as is, nothing kept
    W.enable()
    W.begin_build()
    ta1, ta2, tb = W.load_tensor(a, dev, load), W.load_tensor(a, dev, load), W.load_tensor(b, dev, load)
    assert ta1 is not ta2 and W.end_build()["loaded"] == 3
    W.begin_build()  # a rebuild: the same files
    got = [W.load_tensor(a, dev, load), W.load_tensor(a, dev, load)]
    assert {id(x) for x in got} == {id(ta1), id(ta2)}  # each copy serves one load
    time.sleep(0.01)
    b.write_bytes(b"z" * 60)  # a new layout / format: another file content
    tb2 = W.load_tensor(b, dev, load)
    st = W.end_build()
    assert tb2 is not tb and (st["reused"], st["loaded"], st["dropped"]) == (2, 1, 1)
    got[0].ok = False  # a weight the model freed is never handed out again
    W.begin_build()
    assert W.load_tensor(a, dev, load) is not got[0] or W.load_tensor(a, dev, load) is not got[0]


# ----------------------------------------------------------------------------------------- the layers a run loads
class Model:
    loads = 0

    def __init__(self, layers):
        Model.loads += 1
        self.layers, self.state = list(layers), None

    def sync(self):
        pass

    def begin_chunk(self, *a):
        return "chunk"


def fake_spec(hook):
    hooks = SimpleNamespace(device_model=lambda mesh, s, layers, lm_head: Model(layers), run_layers=hook)
    return SimpleNamespace(data={"target": {"seq": 128, "chunk": 32}}, hooks=lambda: hooks)


@pytest.fixture
def shared(monkeypatch, tmp_path):
    monkeypatch.setenv(M.RESULTS_ENV, str(tmp_path))
    monkeypatch.setenv(M.TASK_ENV, "T")
    monkeypatch.delenv(M.RECORD_ENV, raising=False)
    monkeypatch.delenv(M.SUFFIX_ENV, raising=False)
    Model.loads = 0


def test_run_layers_takes_the_models_subset_and_a_held_superset_serves_it(shared, monkeypatch):
    def hook(spec, layers, last_chunk):
        return None if last_chunk else [i for i in layers if i <= 2]

    s, full = fake_spec(hook), [0, 1, 2, 3, 4]
    assert TB.run_layers(s, full, last_chunk=True) == full
    monkeypatch.setenv("BRINGUP_CHUNK_LAST", "0")
    assert TB.run_layers(s, full) == [0, 1, 2]
    monkeypatch.setenv(TB.RUN_LAYERS_ENV, "all")
    assert TB.run_layers(s, full) == full
    monkeypatch.delenv(TB.RUN_LAYERS_ENV)
    monkeypatch.setenv(B.KEY_ENV, "k")
    mesh = object()
    m = TB.device_model(s, mesh, full, lm_head=False)  # e.g. the determinism check, all layers
    assert TB.device_model(s, mesh, TB.run_layers(s, full, last_chunk=False), lm_head=False) is m and Model.loads == 1
    TB.release()
    sub = TB.device_model(s, mesh, [0, 1, 2], lm_head=False)  # a subset held: the full list is another model
    assert sub.layers == [0, 1, 2]
    with pytest.raises(TB.NeedsFreshProcess):
        TB.device_model(s, mesh, full, lm_head=False)


def test_in_the_resident_runner_the_first_chunk_is_timed_and_the_hook_removed(shared, monkeypatch):
    monkeypatch.setenv(TB.RESIDENT_ENV, "/tmp/x")
    monkeypatch.setenv(B.KEY_ENV, "k")
    TB.TIMING.clear()
    m = TB.device_model(fake_spec(lambda *a: None), object(), [0], lm_head=False)
    assert "begin_chunk" in vars(m) and TB.TIMING["loaded"] and "first_chunk" not in TB.TIMING
    assert m.begin_chunk() == "chunk" and TB.TIMING["first_chunk"] >= TB.TIMING["model_ready"]
    TB._untimed(m)
    assert "begin_chunk" not in vars(m)


# ----------------------------------------------------------------------------------------- F284: host RAM accounting
def test_unaccounted_ram_and_the_reserve_before_a_device_open():
    from models.demos.common.bringup.core import hostmem as H

    gib = 1 << 30
    mi = {k: 0 for k in H.UNACCOUNTED} | {"MemTotal": 566 * gib, "MemFree": 200 * gib, "AnonPages": 40 * gib}
    assert H.unaccounted(mi) == 326.0
    ok, line = H.reserve_check({}, dict(mi, MemAvailable=230 * gib))  # 230 - 200.2 (215 GB) < 40 GiB (owner)
    assert not ok and line.startswith(H.INFRA_TAG) and "REBOOT" in line and "326.0 GiB" in line
    assert H.reserve_check({}, dict(mi, MemAvailable=350 * gib))[0]  # the node at 163 GiB unaccounted: fine
    assert H.reserve_check({H.LOAD_ENV: "10"}, dict(mi, MemAvailable=170 * gib))[0]
    assert H.reserve_check({H.RESERVE_ENV: "0"}, dict(mi, MemAvailable=1))[0]


def test_a_reply_carries_the_requests_accounting(tmp_path, monkeypatch):
    from models.demos.common.bringup.core import hostmem as H

    ram = iter([163.0, 164.5])
    monkeypatch.setattr(H, "unaccounted", lambda mi=None: next(ram))
    s, d = server(tmp_path, [])
    queue(d, mkplan(tmp_path))
    assert s.serve(poll=0.01, max_idle=0.2) == "idle"
    a = json.loads((d / "done" / "r1.json").read_text())["accounting"]
    assert a["ram_unaccounted_before_gib"] == 163.0 and a["ram_unaccounted_after_gib"] == 164.5
    assert a["device_opens"] == 0 and a["model_loads"] == 0 and "accounting" in R.describe({"accounting": a})

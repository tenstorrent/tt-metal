# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F281 (OWNER 2026-11-19, "iteration time"): the resident runner. One long-lived device process keeps the mesh open
and the model loaded, and runs measurement requests (the runs of a batch plan: core/batch.py) as they come; a gate
unit or an A/B side submits to it instead of starting a process that spends ~4 min loading for seconds of work.

    python -m models.demos.common.bringup.resident start  --spec S      start it (returns when its mesh is open)
    python -m models.demos.common.bringup.resident status --spec S
    python -m models.demos.common.bringup.resident submit --spec S --cmd "<safe-runner command(s)>" [--task T]
           [--results DIR] [--fresh]                                    run them there; prints the reply and timing
    python -m models.demos.common.bringup.resident stop   --spec S

Pieces: this file is the CPU side (CLI, client, supervisor); the device process is tests/test_resident.py running
testing/resident.Server under scripts/run_safe_pytest.sh (chip locks, dispatch-timeout triage, board reset: the safe
runner's protections, nothing of its own). Its state lives in <art>/runs/<ledger>/resident/ (spool: queue/, active/,
done/; status.json; server.log: the safe runner's output; requests.jsonl: every request's timing).

The supervisor (``_supervise``, detached) starts the device process and starts it again when the process asks for it
(restart.json: a C++ build, a kernel source or ttnn python change, another model configuration, a fresh-process
request; testing/resident.py); the request that needed it stays queued and the new process serves it first. When the
process ends otherwise (a hang: the safe runner's exit 2 after its triage and reset; a crash), the supervisor answers
the request it was serving and every queued one with that, writes status "dead", and stops: the next submit says so.

The device is the box's: every other device run through scripts/run_safe_pytest.sh first asks a resident runner to
step aside (``yield``: it releases the model and the mesh and ends; status "yielded"); core/batch.drive starts a
yielded runner again when a gate next submits (a stopped one stays stopped).

Fresh-process determinism: a request marked ``fresh`` (core/batch.drive marks a plan with the model determinism check)
runs in a process that has run nothing before it (a new one when this one has), so the check's first sequence is the
cold one after the load, once per gate that has it (the accept gate), not per measurement.

The yield path (``python3 -I resident.py yield``, from the safe runner) uses the standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TEST = "models/demos/common/bringup/tests/test_resident.py"
DIR_ENV = "BRINGUP_RESIDENT_DIR"  # the device process's state directory (testing/batch.RESIDENT_ENV)
OFF_ENV = "BRINGUP_RESIDENT"  # "off": core/batch.drive never submits (the old path, a process per plan)
HOLDER = "generated/resident/holder.json"  # the box's resident runner (the safe runner's yield reads it)
DEAD = ("dead", "hang", "stopped", "yielded")
MAX_RESTARTS = 4  # new processes in a row that served nothing: a restart loop, the supervisor stops


def _read(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def _write(p: Path, data: dict) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, p)


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def state_dir(spec) -> Path:
    """<art>/runs/<ledger>/resident (spec: a core/spec.Spec or its path)."""
    if isinstance(spec, (str, Path)):
        from models.demos.common.bringup.core.spec import Spec

        spec = Spec.load(spec)
    return spec.run_dir("resident")


def supervisor_pid(d: Path):
    pid = (_read(Path(d) / "supervisor.json") or {}).get("pid")
    return pid if pid and alive(pid) else None


def info(d: Path) -> dict:
    d = Path(d)
    st = _read(d / "status.json") or {}
    return {
        "dir": str(d),
        "supervisor": supervisor_pid(d),
        "phase": st.get("phase", "none"),
        "status": st,
        "queued": sorted(p.stem for p in (d / "queue").glob("*.json")) if (d / "queue").is_dir() else [],
        "active": sorted(p.stem for p in (d / "active").glob("*.json")) if (d / "active").is_dir() else [],
    }


def healthy(d: Path) -> bool:
    i = info(d)
    return bool(i["supervisor"]) and i["phase"] not in DEAD


# ------------------------------------------------------------------------------------------------------- supervisor
def _answer(d: Path, req_ids, rc: int, why: str) -> None:
    for rid in req_ids:
        if not (d / "done" / f"{rid}.json").exists():
            _write(d / "done" / f"{rid}.json", {"id": rid, "rc": rc, "error": why, "runs": {}, "timing": {}})
        for sub in ("active", "queue"):
            (d / sub / f"{rid}.json").unlink(missing_ok=True)


def _tail(p: Path, n: int = 40) -> str:
    try:
        return "".join(Path(p).read_text(errors="replace").splitlines(True)[-n:])
    except OSError:
        return ""


def supervise(d: Path, spec_path: str, run=None, max_restarts: int = MAX_RESTARTS) -> str:
    """The device process, again on each restart it asks for, until it stops, yields, hangs or dies. ``run(env,
    log)``: the device process (tests: a fake); its exit code back."""
    d = Path(d)
    for sub in ("queue", "active", "done"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    for f in ("stop", "yield", "restart.json"):
        (d / f).unlink(missing_ok=True)
    _write(d / "supervisor.json", {"pid": os.getpid(), "spec": spec_path, "t": time.time()})
    holder = REPO / HOLDER
    _write(holder, {"dir": str(d), "supervisor": os.getpid(), "spec": spec_path})

    def device(env, log):
        with open(log, "a") as f:
            f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} resident (F281): a device process\n")
            f.flush()
            return subprocess.run(
                ["scripts/run_safe_pytest.sh", "--no-precompile", "--run-all", TEST],
                cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
            ).returncode  # fmt: skip

    run = run or device
    over: dict = {}
    idle_restarts, n, end = 0, 0, "dead"
    try:
        while True:
            if (d / "stop").exists() or (d / "yield").exists():
                end = "stopped" if (d / "stop").exists() else "yielded"
                break
            env = dict(os.environ, **over, **{DIR_ENV: str(d), "BRINGUP_SPEC": spec_path, "PYTHONPATH": str(REPO)})
            env["BRINGUP_RESIDENT_PROCESS"] = str(n + 1)  # F284: the device opens of this runner so far, this one too
            _write(d / "launch_env.json", env)
            done0 = len(list((d / "done").glob("*.json")))
            n += 1
            _write(d / "status.json", {"phase": "starting", "t": time.time(), "process": n})
            rc = run(env, d / "server.log")
            rs = _read(d / "restart.json")
            (d / "restart.json").unlink(missing_ok=True)
            if rs is not None and rc == 0:
                over.update(rs.get("env") or {})
                served = len(list((d / "done").glob("*.json"))) > done0
                idle_restarts = 0 if served else idle_restarts + 1
                with open(d / "restarts.jsonl", "a") as f:
                    f.write(json.dumps(dict(rs, process=n)) + "\n")
                if idle_restarts > max_restarts:
                    why = f"{idle_restarts} new processes in a row served nothing (last: {rs.get('why')})"
                    _answer(d, [p.stem for p in (d / "queue").glob("*.json")], 1, why)
                    _write(d / "status.json", {"phase": "dead", "why": why, "t": time.time()})
                    break
                continue
            st = _read(d / "status.json") or {}
            if rc == 0 and st.get("phase") in ("stopped", "yielded"):
                end = st["phase"]
                break
            what = "HANG (the safe runner triaged it and reset the board)" if rc == 2 else f"ended with exit code {rc}"
            why = f"the device process {what}; its output: {d / 'server.log'}"
            active = [p.stem for p in (d / "active").glob("*.json")]
            _answer(d, active, 2 if rc == 2 else 1, why + "\n" + _tail(d / "server.log"))
            _answer(d, [p.stem for p in (d / "queue").glob("*.json")], 1, "not run: " + why)
            _write(d / "status.json", dict(st, phase="dead", why=why, rc=rc, t=time.time()))
            break
    finally:
        cur = _read(holder) or {}
        if cur.get("supervisor") == os.getpid():
            holder.unlink(missing_ok=True)
        (d / "supervisor.json").unlink(missing_ok=True)
        if end in ("stopped", "yielded"):
            st = _read(d / "status.json") or {}
            _write(d / "status.json", dict(st, phase=end, t=time.time()))
    return end


# ------------------------------------------------------------------------------------------------------- client
def start(d: Path, spec_path: str, wait: float = 900.0, echo=print) -> dict:
    """Start the runner (detached) unless one runs; with ``wait``, until its mesh is open (status ready)."""
    d = Path(d)
    if supervisor_pid(d):
        echo(f"resident: already running (supervisor {supervisor_pid(d)}, {info(d)['phase']})")
        return info(d)
    d.mkdir(parents=True, exist_ok=True)
    for f in ("stop", "yield"):
        (d / f).unlink(missing_ok=True)
    _write(d / "status.json", {"phase": "starting", "t": time.time()})
    log = open(d / "supervisor.log", "a")
    p = subprocess.Popen(
        [sys.executable, "-m", "models.demos.common.bringup.resident", "_supervise", "--dir", str(d), "--spec",
         str(spec_path)],
        cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO)), stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )  # fmt: skip
    echo(f"resident: started (supervisor {p.pid}); state {d}")
    t0 = time.time()
    while wait and time.time() - t0 < wait:
        i = info(d)
        if i["phase"] in ("ready", "busy") or (i["phase"] in DEAD and i["phase"] != "stopped"):
            break
        if p.poll() is not None:
            break
        time.sleep(1.0)
    i = info(d)
    echo(f"resident: {i['phase']} after {time.time() - t0:.0f} s")
    return i


def stop(d: Path, flag: str = "stop", wait: float = 600.0) -> bool:
    """Ask the runner to end after its current request (``yield``: for another device run); True when it ended."""
    d = Path(d)
    pid = supervisor_pid(d)
    if not pid:
        return True
    (d / flag).write_text(str(time.time()))
    t0 = time.time()
    while time.time() - t0 < wait:
        if not alive(pid):
            return True
        time.sleep(0.5)
    return False


def yield_device(repo: Path = REPO, why: str = "", wait: float = 3600.0) -> int:
    """Called by scripts/run_safe_pytest.sh before it takes the cards (standard library only): the box's resident
    runner, if one runs, releases the device first. 0 = the device is free of it."""
    h = _read(Path(repo) / HOLDER)
    if not h or not alive(h.get("supervisor")):
        return 0
    d = Path(h["dir"])
    print(f"SAFE_PYTEST: the resident runner ({d}) steps aside for this run{': ' + why if why else ''}", flush=True)
    ok = stop(d, "yield", wait)
    return 0 if ok else 1


def submit_plan(d: Path, plan: Path, env: dict, fresh: bool = False, timeout: float | None = None, poll=0.5) -> dict:
    """Queue a plan file (core/batch.write_plan) and wait for its reply. The reply's rc: 0 / 1 / 2 (a hang); a
    runner that died before answering: rc 1 and ``error``."""
    d = Path(d)
    rid = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    req = {
        "id": rid,
        "plan": str(Path(plan).resolve()),
        "env": dict(env),
        "fresh": bool(fresh),
        "t_submit": time.time(),
    }
    _write(d / "queue" / f"{rid}.json", req)
    t0 = time.time()
    while True:
        r = _read(d / "done" / f"{rid}.json")
        if r is not None:
            return r
        if not supervisor_pid(d):
            time.sleep(1.0)  # (the supervisor's last answer may still be on its way)
            r = _read(d / "done" / f"{rid}.json")
            if r is not None:
                return r
            (d / "queue" / f"{rid}.json").unlink(missing_ok=True)
            st = _read(d / "status.json") or {}
            why = f"the resident runner is not running ({st.get('phase')}: {st.get('why', '')})"
            return {"id": rid, "rc": 1, "runs": {}, "timing": {}, "error": why}
        if timeout is not None and time.time() - t0 > timeout:
            return {"id": rid, "rc": 1, "runs": {}, "timing": {}, "error": f"no reply in {timeout:.0f} s"}
        time.sleep(poll)


WANTED = ("ready", "busy", "starting", "restarting", "yielded")  # a runner that serves the spec (or will again)


def wanted_dir(env: dict | None = None) -> Path | None:
    """F286: the state dir of a runner that is meant to serve this environment's spec (BRINGUP_SPEC): up, starting,
    or stepped aside for another device run; None when there is none (stopped, dead, never started, or off)."""
    env = env if env is not None else os.environ
    if env.get(OFF_ENV) == "off" or env.get(DIR_ENV) or not env.get("BRINGUP_SPEC"):
        return None
    try:
        d = state_dir(env["BRINGUP_SPEC"])
    except Exception:  # noqa: BLE001 -- no spec, no runner
        return None
    return d if supervisor_pid(d) or info(d)["phase"] in WANTED and info(d)["phase"] != "starting" else None


class Refused(RuntimeError):
    """F286 (owner 2026-11-19, one open): the runner is meant to serve this run but cannot; never a process of its own."""


def for_env(env: dict | None = None, echo=print) -> Path | None:
    """core/batch.drive: the state dir of a runner that serves this environment's spec (BRINGUP_SPEC), or None (no
    runner is meant to: the old path). A runner that stepped aside for another device run (yielded) is started again
    (one device open). F286: a runner meant to serve that cannot (it did not come back) raises ``Refused``: the run
    never falls back to a fresh process of its own."""
    env = env if env is not None else os.environ
    d = wanted_dir(env)
    if d is None:
        return None
    if healthy(d):
        return d
    if info(d)["phase"] == "yielded" and not supervisor_pid(d):
        echo("resident (F281): the runner stepped aside for another device run; starting it again (one device open)")
        start(d, env["BRINGUP_SPEC"], echo=echo)
    if healthy(d):
        return d
    st = info(d)
    raise Refused(
        f"the resident runner ({d}) is meant to serve this run but is {st['phase']} "
        f"({(st.get('status') or {}).get('why', 'see its server.log')}); refused instead of a process of its own "
        "(owner rule 2026-11-19: one device open)"
    )


def task_accounting(d: Path, hostmem_jsonl: Path, tid: str) -> dict:
    """F286 (owner 2026-11-19): a task's device opens and loads: the safe-runner processes recorded for it (core/
    hostmem's records, one a device process; the runner's own processes carry the runner's task, not this one), the
    runner's new processes and model loads for its requests (requests.jsonl ``task``)."""
    procs = 0
    try:
        for line in Path(hostmem_jsonl).read_text().splitlines():
            try:
                procs += json.loads(line).get("task") == tid
            except ValueError:
                pass
    except OSError:
        pass
    r_opens = r_loads = 0
    try:
        for line in (Path(d) / "requests.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("task") == tid:
                a = r.get("accounting") or {}
                r_opens += int(a.get("device_opens") or 0)
                r_loads += int(a.get("model_loads") or 0)
    except OSError:
        pass
    return {"opens": procs + r_opens, "process_opens": procs, "runner_opens": r_opens, "runner_loads": r_loads}


def describe(reply: dict) -> str:
    t = reply.get("timing") or {}
    lines = [
        f"resident request {reply.get('id')}: rc {reply.get('rc')}"
        + (f" ({reply['error'].splitlines()[0]})" if reply.get("error") else ""),
        f"  received -> model ready {t.get('model_ready_s')} s ({'loaded' if t.get('model_loaded') else 'held'}), "
        f"-> first chunk {t.get('first_chunk_s')} s, -> done {t.get('done_s')} s",
    ]
    for r in reply.get("restarts") or []:
        lines.append(f"  new process: {'; '.join(r.get('why') or [])}")
    rl = {k: v for k, v in (reply.get("reloaded") or {}).items() if v and k != "partial"}
    if rl:
        lines.append(f"  reloaded: {json.dumps(rl, default=str)[:600]}")
    a = reply.get("accounting")
    if a:  # F284
        lines.append(
            f"  accounting: {a['device_opens']} new device opens (runner: {a['process_opens']} so far), "
            f"{a['model_loads']} model loads, {a['layers_loaded']} layers held, unaccounted RAM "
            f"{a['ram_unaccounted_before_gib']} -> {a['ram_unaccounted_after_gib']} GiB"
        )
    if reply.get("weights"):
        w = reply["weights"]
        lines.append(f"  weights: {w.get('reused')} kept, {w.get('loaded')} loaded, {w.get('dropped')} dropped")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- CLI
def _cli_submit(a) -> int:
    from models.demos.common.bringup.core import batch as B
    from models.demos.common.bringup.core import metrics as M
    from models.demos.common.bringup.core.spec import Spec

    spec = Spec.load(a.spec)
    d = state_dir(spec)
    if not healthy(d):
        print(f"resident: not running ({info(d)['phase']}); start it first", file=sys.stderr)
        return 3
    rid = time.strftime("%Y%m%d-%H%M%S")
    env = dict(os.environ, BRINGUP_SPEC=str(spec.path), PYTHONPATH=str(REPO))
    if a.task:
        env[M.TASK_ENV] = a.task
        env[M.RECORD_ENV] = a.task
    if a.results:
        env[M.RESULTS_ENV] = str(Path(a.results).resolve())
    os.environ.update({k: v for k, v in env.items() if k.startswith("BRINGUP_")})
    plan = B.plan_from_cmd(a.cmd, spec)
    for e in plan["entries"]:
        e["log"] = str(d / "logs" / f"{rid}.{e['name']}.log")
    path = d / "plans" / f"{rid}.json"
    _write(path, plan)
    B.results_path(path).unlink(missing_ok=True)
    r = submit_plan(d, path, {**env, **plan.get("hints", {})}, fresh=a.fresh)
    print(describe(r))
    for name, run in (r.get("runs") or {}).items():
        mets = {k: (v or {}).get("value") for k, v in (run or {}).get("metrics", {}).items()}
        print(
            f"  {name}: rc {run.get('rc')} in {run.get('s')} s, log "
            f"{next((e['log'] for e in plan['entries'] if e['name'] == name), '')}"
        )
        if a.metrics:
            for k in sorted(mets):
                if any(k.startswith(p) for p in a.metrics.split(",")):
                    print(f"    {k} = {mets[k]}")
    return int(r.get("rc") or 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="resident", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd_", choices=("start", "status", "submit", "stop", "yield", "_supervise"))
    ap.add_argument("--spec")
    ap.add_argument("--dir")
    ap.add_argument("--cmd", help="submit: the safe-runner command(s), as a gate unit writes them (&& between runs)")
    ap.add_argument("--task", help="submit: BRINGUP_TASK / BRINGUP_RECORD_TASK of the runs")
    ap.add_argument("--results", help="submit: BRINGUP_RESULTS_DIR of the runs")
    ap.add_argument("--fresh", action="store_true", help="submit: in a process that ran nothing before")
    ap.add_argument("--metrics", default="", help="submit: print the runs' metrics with these prefixes (a,b)")
    ap.add_argument("--why", default="")
    a = ap.parse_args(argv)
    if a.cmd_ == "yield":
        return yield_device(why=a.why)
    if a.cmd_ == "_supervise":
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        supervise(Path(a.dir), a.spec)
        return 0
    d = Path(a.dir) if a.dir else state_dir(a.spec)
    if a.cmd_ == "start":
        i = start(d, str(Path(a.spec).resolve()))
        return 0 if i["phase"] in ("ready", "busy") else 1
    if a.cmd_ == "status":
        i = info(d)
        print(json.dumps(i, indent=1, default=str))
        rl = Path(d) / "requests.jsonl"
        if rl.exists():
            print("last requests:\n" + "".join(rl.read_text().splitlines(True)[-5:]))
        return 0
    if a.cmd_ == "stop":
        ok = stop(d)
        print(f"resident: {'stopped' if ok else 'still running after the wait'}")
        return 0 if ok else 1
    return _cli_submit(a)


if __name__ == "__main__":
    sys.exit(main())

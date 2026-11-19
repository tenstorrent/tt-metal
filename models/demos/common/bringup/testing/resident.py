# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F281 (OWNER 2026-11-19, "iteration time"): the device side of the resident runner (the CPU side, the CLI and the
supervisor: models/demos/common/bringup/resident.py). ONE long-lived pytest process (tests/test_resident.py, under
scripts/run_safe_pytest.sh: its chip locks, dispatch-timeout hang triage and board reset) opens the mesh once, keeps
the model loaded (testing/batch.shared_model) and serves requests from a spool directory:

    <dir>/queue/<id>.json    a request: {"id", "plan": a batch plan file (core/batch.write_plan), "env": the
                             environment the batch process would have had, "fresh": a cold process first (the model
                             determinism check), "t_submit"}
    <dir>/active/<id>.json   the request being served;  <dir>/done/<id>.json: its reply
    <dir>/status.json        this process: phase (ready / busy / restarting / hang / stopped), pid, held model, served
    <dir>/restart.json       this process asks its supervisor for a new process (why, the environment it needs)

A request runs like a batch process runs its plan (testing/batch.run_plan: per run its env, log, results dir; the
plan's results file as a batch process writes it), the model kept between requests. Its reply: rc, the runs' results,
and its timing: received -> model ready (0 when the held model served it) -> first chunk (the first begin_chunk /
embed call after the request) -> done, also recorded per run as metrics (resident_*).

Before each request the process checks what changed since it last looked (``Fingerprint``):
    test files                                  imported again for every run already (testing/batch)
    framework python (common/bringup, not       its modules dropped: imported again by the run
    the server's own modules)
    model python (models/**, not the framework  the model released, its modules (and the framework's) dropped, the
    and not tests)                              model rebuilt by the run's load: file-backed device weights kept
                                                (testing/weights.py); a rebuild needs the old model's device memory
                                                back (DRAM / L1 / L1_SMALL at the mesh-open level plus the kept
                                                weights), else a new process
    ttnn python (ttnn/ttnn/**)                  a new process (its Python ops register with ttnn once a process)
    the server's own modules                    a new process
    a shared library mapped by this process     a new process
    a kernel source (*/kernels/*, tt_metal/hw)  a new process: a process builds a kernel once per kernel hash (the
                                                hash holds its defines and arguments, not the source text:
                                                tt_metal/impl/program/program.cpp jit_build_once), and the program
                                                cache keeps the built program; a new process's JIT checks the sources
A request whose model this process cannot hold (F269: other layers, seq, switches), a fresh request on a process that
has run anything, or a request with other TT_* process variables also gets a new process: the request stays queued
and the new process serves it first (``restart.json``; the supervisor starts it). A device hang (the safe runner's
dispatch-timeout triage log is written) ends the process with no reply; the supervisor answers the request (rc 2)
after the safe runner's triage and reset.
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
FRAMEWORK = "models/demos/common/bringup/"
# the server's own modules: the process state lives in them (the held model, the registry, the plan code)
SERVER_FILES = {
    FRAMEWORK + "testing/batch.py",
    FRAMEWORK + "testing/resident.py",
    FRAMEWORK + "testing/weights.py",
    FRAMEWORK + "core/batch.py",
    FRAMEWORK + "resident.py",
    FRAMEWORK + "tests/test_resident.py",
}
KERNEL_ROOTS = ("ttnn/cpp", "ttnn/ttnn/bringup", "tt_metal", "models")
HW_ROOT = "tt_metal/hw"
# variables of the process, not of a run: fixed at its start (the device runtime reads them once) or the shell's own
PROCESS_PREFIXES = ("TT_", "ARCH_NAME", "LD_", "PYTHON", "TTPOOL_", "TTRUN_", "SAFE_PYTEST", "OMP_", "MKL_")
PROCESS_KEYS = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "PWD", "TMPDIR", "LANG", "TERM", "HOSTNAME"}
# exported by scripts/run_safe_pytest.sh into its child (a client never has them), or where a run's records / kernel
# binaries go (core/stamps.ENV_IGNORED: not what it measures)
SAFE_RUNNER_TT = (
    "TT_METAL_OPERATION_TIMEOUT_SECONDS",
    "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE",
    "TT_METAL_CACHE",
    "TT_DEVICE_TIMING_LOG",
)
HEARTBEAT_S = 5.0


def process_key(k: str) -> bool:
    return k in PROCESS_KEYS or k.startswith(PROCESS_PREFIXES) or k == "BRINGUP_RESIDENT_DIR"


def _rel(path: str) -> str | None:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except (ValueError, OSError):
        return None


def kind_of(rel: str) -> str | None:
    """The category of a repo python file a process imported: test / server / framework / model / ttnn, or None."""
    name = rel.rsplit("/", 1)[-1]
    if rel in SERVER_FILES:
        return "server"
    if "/tests/" in f"/{rel}" and name.startswith(("test_", "conftest")):
        return "test"
    if rel.startswith("ttnn/"):
        return "ttnn"
    if rel.startswith(FRAMEWORK):
        return "framework"
    if rel.startswith("models/"):
        return "model"
    return None


def _stat(p) -> tuple | None:
    try:
        st = os.stat(p)
    except OSError:
        return None
    return (int(st.st_mtime_ns), int(st.st_size), int(st.st_ino))


def mapped_libraries(maps: str = "/proc/self/maps") -> list[str]:
    """The shared libraries of this process that live in the repo (ttnn's, tt-metal's)."""
    try:
        text = Path(maps).read_text()
    except OSError:
        return []
    out = set()
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and ".so" in parts[5] and _rel(parts[5].strip()) is not None:
            out.add(parts[5].strip())
    return sorted(out)


def kernel_sources(repo: Path = REPO) -> dict:
    """{path: stat} of every kernel source a process may build: files under a kernels/ directory, and tt_metal/hw."""
    out = {}
    for root in KERNEL_ROOTS:
        for d, dirs, files in os.walk(repo / root):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", ".git", "build")]
            if "/kernels" not in d.replace("\\", "/") + "/" and not d.startswith(str(repo / HW_ROOT)):
                continue
            for f in files:
                p = os.path.join(d, f)
                out[p] = _stat(p)
    return out


class Fingerprint:
    """What this process runs, as it was when it took it: its repo python modules (first seen), its shared libraries
    and the kernel sources (at its start). ``changes()``: what differs now, by category."""

    def __init__(self, repo: Path = REPO, modules=None, libs=None, kernels=None):
        self.repo = repo
        self._modules = modules if modules is not None else (lambda: sys.modules)
        self._libs = libs or mapped_libraries
        self._kernels = kernels or (lambda: kernel_sources(repo))
        self.py: dict = {}  # path -> (kind, stat)
        self.so = {p: _stat(p) for p in self._libs()}
        self.kern = self._kernels()
        self.mark()

    def mark(self) -> None:
        """Note every repo module loaded now that is not noted yet (its file as it is now)."""
        for mod in list(self._modules().values()):
            f = getattr(mod, "__file__", None)
            if not f or not f.endswith(".py") or f in self.py:
                continue
            rel = _rel(f)
            kind = kind_of(rel) if rel else None
            if kind and kind != "test":
                pkg = Path(f).name == "__init__.py"
                self.py[f] = ("package" if pkg and kind in ("framework", "model") else kind, _stat(f))

    def changes(self) -> dict:
        """{"restart": [why], "model": [files], "framework": [files]}; the changed python files are forgotten (noted
        again when imported again)."""
        out = {"restart": [], "model": [], "framework": []}
        for f, (kind, st) in list(self.py.items()):
            if _stat(f) == st:
                continue
            rel = _rel(f) or f
            if kind in ("model", "framework"):
                out[kind].append(rel)
                del self.py[f]
            else:
                why = {"ttnn": "ttnn python (its ops register once a process)", "server": "the runner's own code",
                       "package": "a package __init__"}[kind]  # fmt: skip
                out["restart"].append(f"{rel} changed: {why}")
        for p, st in self.so.items():
            if _stat(p) != st:
                out["restart"].append(f"{_rel(p) or p} changed: a shared library of this process (a C++ build)")
        now = self._kernels()
        diff = sorted(p for p in set(now) | set(self.kern) if now.get(p) != self.kern.get(p))
        if diff:
            names = ", ".join(_rel(p) or p for p in diff[:3]) + (f" (+{len(diff) - 3})" if len(diff) > 3 else "")
            out["restart"].append(f"kernel sources changed: {names} (built once per process; a new process rebuilds)")
        return out


def purge(kinds=("model", "framework")) -> list[str]:
    """Drop the loaded leaf modules of those categories (never a package, a test module or the server's own): the next
    import loads them from their files."""
    gone = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None) or ""
        rel = _rel(f) if f.endswith(".py") else None
        if rel and kind_of(rel) in kinds and Path(f).name != "__init__.py":
            sys.modules.pop(name, None)
            gone.append(name)
    return gone


def triage_log(env=None) -> Path | None:
    """The safe runner's triage log (written by the dispatch-timeout hook: the hang signal), from its hook command."""
    cmd = (env or os.environ).get("TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE") or ""
    m = re.search(r">\s*(\S+)\s+2>&1", cmd)
    return Path(m.group(1)) if m else None


def hung(env=None) -> bool:
    p = triage_log(env)
    return bool(p and p.exists() and p.stat().st_size > 0)


def tt_env_diff(req_env: dict, launch_env: dict) -> dict:
    """{key: (process value, request value)} of the TT_* process variables a request sets otherwise than this process
    was started with (the safe runner's own exports aside). One that the request does not set is the process's (a
    gate's environment need not carry the shell's TT_METAL_HOME): never a new process (owner: every open leaks)."""
    keys = {k for k in req_env if k.startswith("TT_") and k not in SAFE_RUNNER_TT}
    return {k: (launch_env.get(k), req_env.get(k)) for k in sorted(keys) if launch_env.get(k) != req_env.get(k)}


def run_env(start_env: dict, req_env: dict) -> dict:
    """The environment a request's plan runs in: this process's own variables, the request's others (what a batch
    process the client started would have had)."""
    out = {k: v for k, v in start_env.items() if process_key(k)}
    out.update({k: str(v) for k, v in req_env.items() if not process_key(k)})
    return out


def memory_view(mesh) -> dict:
    """Allocated bytes a chip per buffer type (the mesh's lockstep view), or {} without one."""
    try:
        import ttnn
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for name in ("DRAM", "L1", "L1_SMALL"):
        try:
            v = ttnn.get_memory_view(mesh, getattr(ttnn.BufferType, name))
            out[name] = int(v.total_bytes_allocated_per_bank) * int(v.num_banks)
        except Exception:  # noqa: BLE001
            pass
    return out


def memory_back(base: dict, now: dict, kept_bytes: int, slack: int = 64 << 20) -> str | None:
    """None when the device memory after a model's release is the mesh-open level plus the kept weights; else what is
    still held (a rebuild in place would run on top of it: F269)."""
    if not base or not now:
        return None
    bad = []
    if now.get("DRAM", 0) - base.get("DRAM", 0) > kept_bytes + slack:
        bad.append(f"DRAM {(now['DRAM'] - base['DRAM'] - kept_bytes) / (1 << 20):.0f} MiB a chip")
    for t in ("L1", "L1_SMALL"):
        if now.get(t, 0) - base.get(t, 0) > 4096:
            bad.append(f"{t} {(now[t] - base[t]) / 1024:.0f} KiB a chip")
    return ", ".join(bad) or None


def _write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, p)


def _read(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


class Server:
    """One process of the resident runner. ``run(mesh, shape, plan, path, keep, after_run)``: testing/batch.run_plan
    (tests: a fake); ``mesh_memory(mesh)``: memory_view (tests: a fake)."""

    def __init__(self, mesh, shape, d: Path, run=None, fingerprint=None, mesh_memory=None, clock=time.time):
        from models.demos.common.bringup.testing import batch, weights

        self.batch, self.weights = batch, weights
        self.mesh, self.shape, self.d = mesh, tuple(shape), Path(d)
        self.run = run or batch.run_plan
        self.mem = mesh_memory or memory_view
        self.clock = clock
        for sub in ("queue", "active", "done"):
            (self.d / sub).mkdir(parents=True, exist_ok=True)
        self.start_env = dict(os.environ)
        self.launch_env = _read(self.d / "launch_env.json", {}) or {k: v for k, v in os.environ.items()}
        self.fp = fingerprint or Fingerprint()
        self.base_mem = self.mem(mesh)  # the mesh open, before any model
        self.served = 0
        self.pristine = True  # nothing run in this process yet: a fresh-process request needs no new one
        self.status = {"pid": os.getpid(), "since": self.clock(), "served": 0}
        weights.enable(True)

    # -------------------------------------------------------------------------------------------------- state files
    def set_status(self, phase: str, **kw) -> None:
        held = self.batch._MODELS
        model = {k: held.get(k) for k in ("layers", "seq", "lm_head", "config", "by", "load_s")} if held else None
        self.status.update(kw, phase=phase, t=self.clock(), served=self.served, model=model)
        _write(self.d / "status.json", self.status)

    def flag(self) -> str | None:
        for f in ("stop", "yield"):
            if (self.d / f).exists():
                return f
        return None

    def claim(self) -> dict | None:
        """The oldest queued request, moved to active/ (a restarted process takes the request that restarted it
        first: it is still queued)."""
        q = sorted((self.d / "queue").glob("*.json"), key=lambda p: (_read(p, {}).get("t_submit", 0), p.name))
        for p in q:
            req = _read(p)
            if req is None:
                continue
            dst = self.d / "active" / p.name
            os.replace(p, dst)
            return req
        return None

    def requeue(self, req: dict) -> None:
        _write(self.d / "queue" / f"{req['id']}.json", req)
        (self.d / "active" / f"{req['id']}.json").unlink(missing_ok=True)

    def restart(self, req: dict | None, why: list[str], env: dict | None = None) -> str:
        """Leave the request queued (with why) and ask the supervisor for a new process."""
        if req is not None:
            req.setdefault("restarts", []).append({"why": why, "pid": os.getpid(), "t": self.clock()})
            req.setdefault("t_received_first", req.get("t_received"))
            self.requeue(req)
        _write(self.d / "restart.json", {"why": why, "env": env or {}, "pid": os.getpid(), "t": self.clock()})
        self.set_status("restarting", why=why)
        print(f"RESIDENT (F281): a new process: {'; '.join(why)}", flush=True)
        return "restart"

    # -------------------------------------------------------------------------------------------------- the loop
    def serve(self, poll: float = 0.2, max_idle: float | None = None) -> str:
        """Serve until stop / yield (returns it), a restart ("restart") or a hang (raises). ``max_idle``: tests."""
        self.set_status("ready")
        idle0, beat = self.clock(), 0.0
        while True:
            f = self.flag()
            if f:
                self.batch.release()
                self.weights.clear()
                self.set_status("stopped" if f == "stop" else "yielded")
                return f
            req = self.claim()
            if req is None:
                if max_idle is not None and self.clock() - idle0 > max_idle:
                    return "idle"
                if self.clock() - beat > HEARTBEAT_S:
                    beat = self.clock()
                    self.set_status(self.status.get("phase", "ready"))
                time.sleep(poll)
                continue
            out = self.handle(req)
            if out != "served":
                return out
            idle0 = self.clock()

    def check(self, req: dict) -> tuple[list[str], dict, dict]:
        """(why a new process, the TT_* variables it needs, what to reload here)."""
        ch = self.fp.changes()
        why = list(ch["restart"])
        diff = tt_env_diff(req.get("env") or {}, self.launch_env)
        env = {}
        if diff:
            why.append("other process variables: " + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in diff.items()))
            env = {k: b for k, (_, b) in diff.items()}
        if req.get("fresh") and not self.pristine:
            why.append("a request for a fresh process (the model determinism check's cold first run)")
        return why, env, ch

    def reload(self, ch: dict) -> dict:
        """Model python changed: release the model (its device weights stay in the registry), drop the model's and
        the framework's modules; framework only: drop those. What was done, or {"restart": why}."""
        out = {}
        if ch["model"]:
            had = self.batch._MODELS.get("model") is not None
            self.batch.release()
            purge(("model", "framework"))
            gc.collect()
            gc.collect()
            out = {"model": ch["model"], "framework": ch["framework"], "rebuild": had}
            if had:
                n_dev = max(1, int(getattr(self.mesh, "get_num_devices", lambda: 1)()))
                kept = int(sum(int(k[2]) for k, v in self.weights._BUILT.items() for _ in v) / n_dev)
                now = self.mem(self.mesh)
                held = memory_back(self.base_mem, now, kept)
                out["memory"] = {"base": self.base_mem, "after_release": now, "kept_weights": kept}
                if held:
                    self.weights.clear()
                    return dict(out, restart=f"the released model's device memory did not come back ({held})")
        elif ch["framework"]:
            purge(("framework",))
            out = {"framework": ch["framework"]}
        return out

    def handle(self, req: dict) -> str:
        t_recv = self.clock()
        from models.demos.common.bringup.core import hostmem

        req.setdefault("ram_before", hostmem.unaccounted())  # F284: the first process's view, kept over restarts
        loads0 = self.batch._LOADS["n"]
        req["t_received"] = t_recv
        self.set_status("busy", request=req["id"])
        why, env, ch = self.check(req)
        if why:
            return self.restart(req, why, env)
        reloaded = self.reload(ch)
        if reloaded.get("restart"):
            return self.restart(req, [reloaded["restart"]])
        plan_path = Path(req["plan"])
        plan = json.loads(plan_path.read_text())
        only = req.get("only")
        run_path = plan_path
        if only:  # F269: the runs an earlier process left for this one, in a plan of their own; results merged
            plan = dict(plan, entries=[e for e in plan["entries"] if e["name"] in set(only)])
            k = len(req.get("restarts") or [])
            run_path = Path(f"{plan_path}.resident{k}.json")
            run_path.write_text(json.dumps(plan, indent=1) + "\n")
            from models.demos.common.bringup.core import batch as B

            B.results_path(run_path).unlink(missing_ok=True)
        self.batch.REQUEST.update(received=t_recv, first=None)
        saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(run_env(self.start_env, req.get("env") or {}))
        hang = {"run": None}

        def after_run(name, rec):
            if hung():
                hang["run"] = name
                return True
            return False

        try:
            failed = self.run(self.mesh, self.shape, plan, run_path, keep=True, after_run=after_run)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        self.pristine = False
        self.fp.mark()
        if hang["run"]:  # no reply: the supervisor answers after the safe runner's triage and reset
            self.set_status("hang", request=req["id"], run=hang["run"])
            raise RuntimeError(f"RESIDENT (F281): device hang in {req['id']} run {hang['run']}")
        from models.demos.common.bringup.core import batch as B

        got = B.load_results(run_path)
        if only:
            main = B.load_results(plan_path)
            main["runs"].update(got.get("runs") or {})
            main["running"], main["deferred"] = got.get("running"), got.get("deferred") or []
            _write(B.results_path(plan_path), main)
            got = main
        deferred = list(self.batch.DEFERRED)
        if deferred:  # F269: the rest needs another model: a new process serves it (this one keeps nothing)
            req["only"] = deferred
            req.setdefault("partial", []).append(dict(reloaded, loads=self.batch._LOADS["n"] - loads0))
            return self.restart(req, [f"F269: runs {', '.join(deferred)} need another model than this process held"])
        reply = self.reply(req, got, failed, reloaded, t_recv)
        held = self.batch._MODELS
        reply["accounting"] = {  # F284 (owner 2026-11-19): what the request cost the node
            "device_opens": len(req.get("restarts") or []),  # (new processes it needed; this one's open is earlier)
            "model_loads": self.batch._LOADS["n"]
            - loads0
            + sum(int(p.get("loads", 0)) for p in req.get("partial") or []),
            "layers_loaded": len(held.get("layers") or []) if held else 0,
            "ram_unaccounted_before_gib": req["ram_before"],
            "ram_unaccounted_after_gib": hostmem.unaccounted(),
            "process_opens": int(os.environ.get("BRINGUP_RESIDENT_PROCESS", "1") or 1),
        }
        _write(self.d / "done" / f"{req['id']}.json", reply)
        (self.d / "active" / f"{req['id']}.json").unlink(missing_ok=True)
        with open(self.d / "requests.jsonl", "a") as f:
            keep = ("id", "rc", "timing", "restarts", "reloaded", "accounting")
            f.write(json.dumps({k: reply.get(k) for k in keep}, default=str))
            f.write("\n")
        self.served += 1
        self.set_status("ready", request=None, last=req["id"])
        return "served"

    def reply(self, req: dict, got: dict, failed: list, reloaded: dict, t_recv: float) -> dict:
        runs = got.get("runs") or {}
        t_first_recv = req.get("t_received_first") or t_recv
        firsts = [r["timing"]["t0"] + r["timing"]["first_chunk"] for r in runs.values()
                  if (r or {}).get("timing", {}).get("first_chunk") is not None]  # fmt: skip
        readies = [r["timing"]["t0"] + r["timing"]["model_ready"] for r in runs.values()
                   if (r or {}).get("timing", {}).get("model_ready") is not None]  # fmt: skip
        loaded = any((r or {}).get("timing", {}).get("loaded") for r in runs.values())
        done = self.clock()
        timing = {
            "submitted": req.get("t_submit"),
            "received": t_first_recv,
            "received_here": t_recv,
            "model_ready_s": round(min(readies) - t_first_recv, 2) if readies else None,
            "first_chunk_s": round(min(firsts) - t_first_recv, 2) if firsts else None,
            "done_s": round(done - t_first_recv, 2),
            "model_loaded": loaded,
        }
        rc = 0 if not failed else 1
        return {
            "id": req["id"], "rc": rc, "failed": failed, "runs": runs, "timing": timing, "pid": os.getpid(),
            "restarts": req.get("restarts") or [], "reloaded": dict(reloaded, partial=req.get("partial") or []),
            "weights": self.batch._MODELS.get("weights"), "plan": req["plan"],
        }  # fmt: skip


def serve(mesh, shape, d=None) -> str:
    """tests/test_resident.py: this process's loop."""
    return Server(mesh, shape, Path(d or os.environ["BRINGUP_RESIDENT_DIR"])).serve()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.31 (owner 2026-11-13: "count every device run once"): every device run through scripts/run_safe_pytest.sh files
its result in the run's stamp store, keyed by what it measured, so a gate reuses it however it was launched (a gate,
gate --check, a devjob, --one-process, by hand).

A run's identity (``identity``) is its tests and the pytest arguments that change what runs, the environment that
changes what it measures (stamps.measuring_env: BRINGUP_* / TT_* / model switches, without where results are filed
and how it was launched), the code key, the goldens key and the owner waivers its tests read. Not in it: the
precompile flags (where kernels compile), --run-all (a PASS ran every test either way), --one-process, the task the
run was filed under. Records: runs/<run>/stamps/_runs/<key>.json, the newest first, each {verdict, rc, metrics,
outputs, s, t, cmd, env, task, launched}; the result files the run wrote are kept beside it. A gate reuses a unit
whose every run's NEWEST record is a PASS on the same key (never a FAIL / HANG, never an older pass behind a newer
failure) and whose metrics meet the gate's limits (core/gate._lookup).

The safe runner calls this module around pytest (scripts/lib/tt_safe_run.sh ttrun_runrec_begin / _end), only when
BRINGUP_SPEC names a spec; any failure here is reported on one line and never changes the run or its exit code:

    python -m models.demos.common.bringup.core.runrec begin --state F -- <run_safe_pytest args...>
    python -m models.demos.common.bringup.core.runrec end --state F --rc N
    python -m models.demos.common.bringup.core.runrec plan --cmd CMD --out PLAN   (BRINGUP_BATCH_CMD as a plan file,
                                                                                    so a batch reports each run)
While a run is open, runs/<run>/live/<pid>.json names it (the supervisor's watch counts it as a live run); its end
appends one line to runs/<run>/gates.log.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path

from models.demos.common.bringup.core import stamps

KEEP = 8  # records kept per run key
LAUNCH_FLAGS = {"--precompile", "--no-precompile", "--no-jit-server", "--run-all", "--one-process", "--mesh"}
LAUNCH_FLAGS_WITH_VALUE = {"--precompile-workers", "--farm-min-programs"}
SAFE_RUN = re.compile(r"^((?:\w+=\S*\s+)*)scripts/run_safe_pytest\.sh\s+(.+)$", re.S)
ASSIGN = re.compile(r"^(\w+)=(\S*)$")
LIVE_DIR = "live"
GATES_LOG = "gates.log"


def canonical_args(args: list[str]) -> tuple[list[str], list[str]]:
    """(tests, other arguments) of a run_safe_pytest call, without the launch flags. --mesh is the default selector
    and dropped; --device N / --dev / --profile and every pytest argument stay."""
    tests, other, skip = [], [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in LAUNCH_FLAGS or a.startswith("--jit-server"):
            continue
        if a in LAUNCH_FLAGS_WITH_VALUE:
            skip = True
            continue
        if a.split("=", 1)[0] in LAUNCH_FLAGS_WITH_VALUE:
            continue
        (other if a.startswith("-") else tests).append(a)
    return tests, other


def identity(args: list[str], env: dict, keys: tuple[str, str], spec=None, prefix: dict | None = None) -> dict:
    """What a run measured: its canonical arguments, measuring environment, code key, goldens key and waivers.
    F270: the environment with the knobs its tests read and every variable of its command-line ``prefix``
    (stamps.run_env); ``prefix`` None: the run's command line is not known (a plain shell run): ``prefix: false``, a
    key no gate unit has, so such a run is never reused (an unseen `FOO=1` on its command line could change it)."""
    tests, other = canonical_args(args)
    return {
        "tests": tests,
        "args": other,
        "env": stamps.run_env(env, tests, prefix, spec.repo if spec is not None else None),
        "prefix": prefix is not None,
        "code": keys[0],
        "goldens": keys[1],
        "waivers": stamps.waiver_key(spec, " ".join(tests)) if spec is not None else "",
    }


def run_key(ident: dict) -> str:
    return stamps.sha16(json.dumps(ident, sort_keys=True))


def segment_runs(cmd: str) -> list[tuple[dict, list[str]]] | None:
    """A gate unit's command as its safe-runner runs [(prefix env, args)], or None when any `&&` segment is not one
    (a setup step, a loop, another tool: such a unit is only ever found by its own stamp)."""
    out = []
    for seg in (s.strip() for s in cmd.split("&&")):
        if not seg:
            continue
        m = SAFE_RUN.match(seg)
        if not m:
            return None
        pre = {}
        for w in m.group(1).split():
            a = ASSIGN.match(w)
            if not a:
                return None
            pre[a.group(1)] = a.group(2)
        try:
            args = shlex.split(m.group(2))
        except ValueError:
            return None
        out.append((pre, args))
    return out or None


def unit_keys(cmd: str, env: dict, keys: tuple[str, str], spec=None) -> list[str] | None:
    """The run keys of a gate unit's runs, in order (``env``: the unit's environment before its prefixes)."""
    runs = segment_runs(cmd)
    if runs is None:
        return None
    return [run_key(identity(args, dict(env, **pre), keys, spec, pre)) for pre, args in runs]


_CD = re.compile(r"^cd\s+\S+$")


def launch_prefix(env: dict, args: list[str]) -> dict | None:
    """F270: the `VAR=value` prefix this safe-runner run was launched with, from the command its launcher names
    (stamps.UNIT_CMD_ENV: a gate unit's command, a devjob's): the segment that runs these arguments. None when not
    known: no such command, no segment or two with different prefixes, or another step that could set the
    environment (anything but a safe-runner call or a `cd`)."""
    cmd = env.get(stamps.UNIT_CMD_ENV)
    if not cmd:
        return None
    want = sorted(sum(canonical_args(args), []))
    found = []
    for seg in (s.strip() for s in re.split(r"&&|;|\n", cmd)):
        if not seg or _CD.match(seg):
            continue
        one = segment_runs(seg)
        if one is None:
            return None
        pre, a = one[0]
        if sorted(sum(canonical_args(a), [])) == want:
            found.append(pre)
    if not found or any(p != found[0] for p in found):
        return None
    return found[0]


class RunStore:
    """runs/<run>/stamps/_runs/<key>.json: [record, ...] newest first; outputs in _runs/<key>/<n>/."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "stamps" / "_runs"

    def records(self, key: str) -> list[dict]:
        p = self.dir / f"{key}.json"
        try:
            return json.loads(p.read_text()) if p.exists() else []
        except ValueError:
            return []

    def latest(self, key: str) -> dict | None:
        recs = self.records(key)
        return recs[0] if recs else None

    def put(self, key: str, rec: dict, outputs: list[Path], record_tid: str) -> dict:
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp_t = rec.get("t") or time.strftime("%Y-%m-%dT%H:%M:%S")
        slot = f"{int(time.time() * 1000)}-{os.getpid()}"
        names = []
        for f in outputs:
            d = self.dir / key / slot
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, d / stamps.output_key(f.name, record_tid))
            names.append(stamps.output_key(f.name, record_tid))
        rec = dict(rec, t=stamp_t, outputs=names, slot=slot)
        recs = [rec] + self.records(key)
        for old in recs[KEEP:]:
            shutil.rmtree(self.dir / key / str(old.get("slot")), ignore_errors=True)
        tmp = self.dir / f"{key}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(recs[:KEEP], indent=1, sort_keys=True) + "\n")
        tmp.replace(self.dir / f"{key}.json")
        return rec

    def restore(self, key: str, rec: dict, results_dir: Path, record_tid: str) -> list[str]:
        out = []
        for name in rec.get("outputs") or []:
            src = self.dir / key / str(rec.get("slot")) / name
            if src.exists():
                dst = results_dir / (f"{record_tid}{name}" if name.startswith("_") else name)
                shutil.copy2(src, dst)
                out.append(dst.name)
        return out


def find_unit(store: RunStore, keys: list[str]) -> list[dict] | None:
    """FIX.31: the records a unit of these run keys is reused from: each run's newest record, all PASS; else None."""
    got = []
    for k in keys:
        rec = store.latest(k)
        if not rec or rec.get("verdict") != "PASS" or rec.get("rc") != 0:
            return None
        got.append(dict(rec, key=k))
    return got


# ---- the safe runner's side ----


def _spec_and_run_dir():
    path = os.environ.get("BRINGUP_SPEC")
    if not path or os.environ.get("BRINGUP_RUNREC") == "0":
        return None, None
    from models.demos.common.bringup.core.gate import run_name
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import Spec

    spec = Spec.load(path)
    return spec, spec.run_dir(run_name(Ledger(spec.ledger_dir)))


def _results_view(env: dict) -> tuple[str, Path, dict, dict]:
    from models.demos.common.bringup.core import metrics

    task = env.get(metrics.RECORD_ENV) or env.get(metrics.TASK_ENV) or "adhoc"
    root = Path(env.get(metrics.RESULTS_ENV) or metrics.results_dir())
    files = {}
    if root.is_dir():
        for pat in ("*.json", "*.json.gz"):
            files.update({f.name: f.stat().st_mtime_ns for f in root.glob(pat)})
    return task, root, metrics.load(task, root), files


def launched_by(env: dict) -> str:
    if env.get("BRINGUP_DEVJOB_DIR") or "devjobs" in os.getcwd():
        return "devjob"
    if env.get("BRINGUP_RECORD_TASK") and env.get("BRINGUP_RESULTS_DIR"):
        return "gate"
    return "hand"


def begin(state: Path, args: list[str]) -> int:
    spec, run_dir = _spec_and_run_dir()
    if spec is None:
        return 0
    env = dict(os.environ)
    keys = (stamps.code_key(spec.repo, spec), stamps.goldens_key(spec))
    stamps.ensure_aliases(run_dir, spec.repo, spec, env)
    ident = identity(args, env, keys, spec, launch_prefix(env, args))  # F270
    task, root, m0, f0 = _results_view(env)
    st = {
        "run_dir": str(run_dir),
        "args": args,
        "identity": ident,
        "key": run_key(ident),
        "keys": list(keys),
        "t0": time.time(),
        "task": env.get("BRINGUP_TASK", ""),
        "record_task": task,
        "results_dir": str(root),
        "metrics0": m0,
        "files0": f0,
        "launched": launched_by(env),
        "pid": os.getppid(),
        # the environment the run was given, before the safe runner's own exports (its dispatch-timeout command names
        # its pid): a batch's runs are keyed on it at the end
        "env0": stamps.measuring_env(env),
    }
    state.write_text(json.dumps(st))
    live = Path(run_dir) / LIVE_DIR
    live.mkdir(parents=True, exist_ok=True)
    tests = " ".join(ident["tests"])[:200]
    (live / f"{st['pid']}.json").write_text(
        json.dumps({"pid": st["pid"], "what": f"run_safe_pytest {tests}", "t": time.time(), "task": st["task"]})
    )
    return 0


def _verdict(rc: int | None) -> str:
    return "PASS" if rc == 0 else ("HANG" if rc == 2 else "FAIL")


def _file(store: RunStore, key: str, ident: dict, rc, metrics: dict, files: list[Path], secs, st: dict, task, rtask):
    rec = {
        "verdict": _verdict(rc),
        "rc": rc,
        "metrics": metrics,
        "s": round(float(secs or 0), 1),
        "cmd": " ".join(ident["tests"] + ident["args"]),
        "env": ident["env"],
        "prefix_known": ident.get("prefix", False),  # F270
        "task": task,
        "record_task": rtask,
        "launched": st["launched"],
        "by": "runner",
    }
    return store.put(key, rec, files, rtask)


def end(state: Path, rc: int) -> int:
    try:
        st = json.loads(state.read_text())
    except (OSError, ValueError):
        return 0
    run_dir = Path(st["run_dir"])
    store = RunStore(run_dir)
    secs = time.time() - st["t0"]
    plan_path = os.environ.get("BRINGUP_BATCH")
    lines = []
    if rc not in (0, 1, 2):  # 3: a setup error, or not started (host memory, FIX.30): nothing measured, no record
        lines.append(f"not run (rc {rc}), no record")
    elif plan_path and "test_batch.py" in " ".join(st["args"]):
        from models.demos.common.bringup.core import batch as B
        from models.demos.common.bringup.core.spec import Spec

        spec = Spec.load(os.environ["BRINGUP_SPEC"])
        plan = json.loads(Path(plan_path).read_text())
        runs = (B.load_results(Path(plan_path)) or {}).get("runs") or {}
        keys = tuple(st["keys"])
        for e in plan.get("entries") or []:
            r = runs.get(e["name"])
            if not r or r.get("rc") is None:
                continue  # never reached (or skipped): no record
            given = {k: str(v) for k, v in (e.get("env") or {}).items()}
            # F270: an entry's environment is its unit segment's prefix (known); a knob its tests read is taken from
            # the environment the batch was given
            base = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(stamps.ENV_PREFIXES + stamps.settings_prefixes())
            }
            ident = identity(list(e["args"]), dict(base, **st["env0"], **given), keys, spec, given)
            task, root, _, _ = _results_view(dict(os.environ, **given))
            outs = [root / n for n in r.get("files") or [] if (root / n).exists() and n != f"{task}.json"]
            # a hang ends the process inside a run that never reports: it has no rc and no record (above)
            _file(store, run_key(ident), ident, r["rc"], r.get("metrics") or {}, outs, r.get("s"), st,
                  given.get("BRINGUP_TASK", st["task"]), task)  # fmt: skip
            lines.append(f"{e['name']} {_verdict(r['rc'])}")
    else:
        env = dict(os.environ)
        task, root, m1, f1 = _results_view(env)
        m0, f0 = st["metrics0"], st["files0"]
        got = {k: v for k, v in m1.items() if m0.get(k) != v}
        outs = [root / n for n, t in f1.items() if f0.get(n) != t and n != f"{task}.json"]
        _file(store, st["key"], st["identity"], rc, got, outs, secs, st, st["task"], task)
        lines.append(_verdict(rc))
    (run_dir / LIVE_DIR / f"{st['pid']}.json").unlink(missing_ok=True)
    tests = " ".join(st["identity"]["tests"])[:200]
    with open(run_dir / GATES_LOG, "a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} runner [{st['task'] or '-'}] {tests}: rc {rc} "
            f"({', '.join(lines)}; {secs:.0f} s, launched by {st['launched']})\n"
        )
    state.unlink(missing_ok=True)
    return 0


def plan(cmd: str, out: Path) -> int:
    """BRINGUP_BATCH_CMD as a plan file (the same plan testing/batch.load_plan builds from it, hints included), so
    the batch process writes each run's result beside it."""
    from models.demos.common.bringup.core import batch as B

    spec = None
    if os.environ.get("BRINGUP_SPEC"):
        from models.demos.common.bringup.reference.golden import load_spec

        spec = load_spec()
    p = B.plan_from_cmd(cmd, spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=1) + "\n")
    B.results_path(out).unlink(missing_ok=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="runrec")
    sub = ap.add_subparsers(dest="op", required=True)
    b = sub.add_parser("begin")
    b.add_argument("--state", required=True)
    b.add_argument("args", nargs=argparse.REMAINDER)
    e = sub.add_parser("end")
    e.add_argument("--state", required=True)
    e.add_argument("--rc", type=int, required=True)
    p = sub.add_parser("plan")
    p.add_argument("--cmd", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.op == "begin":
            return begin(Path(a.state), a.args[1:] if a.args[:1] == ["--"] else a.args)
        if a.op == "end":
            return end(Path(a.state), a.rc)
        return plan(a.cmd, Path(a.out))
    except Exception as ex:  # noqa: BLE001 -- bookkeeping never fails a run
        print(f"SAFE_PYTEST: run record ({a.op}) skipped: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 0 if a.op != "plan" else 1


if __name__ == "__main__":
    sys.exit(main())

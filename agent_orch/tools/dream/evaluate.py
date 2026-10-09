"""Run one evaluation on the machine: lock, check out, build if needed, run eval.command, score.

Workers call it through `agent_orch/bin/dream eval --campaign C --node N` from their worktree. The
engine calls it for the baseline and for root re-measures. Only one evaluation runs at a time per
machine (a file lock in $DREAM_HOME), so parallel workers queue instead of colliding on the device.
"""

from __future__ import annotations

import contextlib
import fcntl
import fnmatch
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .campaign import Campaign
from .gitops import changed_files, git, snapshot
from .scoring import apply_gates, drift, log_excerpt, make_baseline, read_result, score_vs_baseline, summary_md

PRE_RUN = {"build_error", "forbidden_edit", "hang", "infra"}


@contextlib.contextmanager
def device_lock(c: Campaign):
    c.dream_home.mkdir(parents=True, exist_ok=True)
    with open(c.dream_home / "device.lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[eval] waiting for the device lock (another evaluation is running)...", flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_cmd(
    cmd: str, cwd: Path, env: dict, log: Path, timeout_s: int, sessions: Path | None = None
) -> tuple[int, float]:
    """Run a shell command in its own process group; kill the whole group on timeout. Returns (rc, seconds)."""
    from .agents import track

    t0 = time.time()
    with open(log, "w") as out:
        p = subprocess.Popen(
            ["bash", "-c", cmd], cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
        with track(p.pid, sessions):
            rc = _wait_cmd(p, timeout_s)
    return rc, time.time() - t0


def _wait_cmd(p: subprocess.Popen, timeout_s: int) -> int:
    try:
        return p.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        return 124


def needs_build(c: Campaign, ev: Path, snap: str) -> bool:
    b = c.cfg["build"]
    if not b.get("command"):
        return False
    last_f = ev / ".dream_last_build"
    if not last_f.exists():
        return True
    if b.get("check_file") and not (ev / b["check_file"]).exists():
        return True
    changed = git("diff", "--name-only", last_f.read_text().strip(), snap, cwd=ev).splitlines()
    skip = list(b.get("skip_if_only", [])) + ["agent_orch/*"]
    return any(not any(fnmatch.fnmatch(p, g) for g in skip) for p in changed if p)


def eval_env(c: Campaign, ev: Path, label: str, result: Path, out_dir: Path) -> dict:
    env = dict(os.environ)
    py = c.python_env
    if (py / "bin").exists():
        env["VIRTUAL_ENV"] = str(py)
        env["PATH"] = f"{py / 'bin'}:{env.get('PATH', '')}"
    for k, v in (c.cfg["eval"].get("env") or {}).items():
        env[k] = str(v).replace("${EVAL}", str(ev))
    env.update(
        DREAM_RESULT=str(result),
        DREAM_OUT=str(out_dir),
        DREAM_LABEL=label,
        DREAM_CAMPAIGN=c.name,
        TT_METAL_CACHE=str(c.home / "jitcache"),
    )
    return env


def run_eval(c: Campaign, snap: str, label: str) -> dict:
    """Check `snap` out in the eval checkout, build if needed, run eval.command. Holds the device lock."""
    ev = c.eval_checkout
    rep = c.report_dir(label)
    c.logs.mkdir(parents=True, exist_ok=True)
    info = {"commit_under_test": snap, "build_seconds": 0, "eval_seconds": 0, "report_dir": str(rep)}
    with device_lock(c):
        print(f"[eval] {label}: checking out {snap[:12]} in {ev}", flush=True)
        last = (ev / ".dream_last_build").read_text().strip() if (ev / ".dream_last_build").exists() else None
        git("checkout", "--detach", "-f", "-q", snap, cwd=ev)
        git("submodule", "update", "--init", "--recursive", "-q", cwd=ev)
        if needs_build(c, ev, snap):
            blog = c.logs / f"build_{label}.log"
            print(f"[eval] building (log: {blog})", flush=True)
            env = dict(os.environ, CCACHE_DIR=os.environ.get("CCACHE_DIR", str(c.dream_home / ".ccache")))
            if (c.python_env / "bin").exists():
                env["PATH"] = f"{c.python_env / 'bin'}:{env['PATH']}"
            rc, secs = run_cmd(
                c.cfg["build"]["command"], ev, env, blog, int(c.cfg["build"]["timeout_s"]), c.home / "sessions"
            )
            info["build_seconds"] = round(secs)
            if rc != 0:
                if last:
                    git("checkout", "--detach", "-f", "-q", last, cwd=ev)
                return {
                    **info,
                    "status": "build_error",
                    "log": blog,
                    "error": log_excerpt(blog.read_text(errors="replace")),
                }
            (ev / ".dream_last_build").write_text(snap + "\n")
        else:
            print("[eval] no host-side change since the last build: skipping the build", flush=True)
        if rep.exists():
            subprocess.run(["rm", "-rf", str(rep)], check=True)
        rep.mkdir(parents=True)
        subprocess.run(["rm", "-rf", str(c.home / "jitcache")], check=True)
        result = rep / "result.json"
        log = rep / "run.log"
        timeout = int(c.cfg["eval"]["timeout_s"])
        print(f"[eval] running eval.command (timeout {timeout}s, log: {log})", flush=True)
        rc, secs = run_cmd(
            c.cfg["eval"]["command"], ev, eval_env(c, ev, label, result, rep), log, timeout, c.home / "sessions"
        )
        info["eval_seconds"] = round(secs)
        text = log.read_text(errors="replace")
        if rc == 124:
            reset = c.cfg["resource"].get("reset_command")
            print(f"[eval] timed out; resetting the device ({reset})", flush=True)
            if reset:
                with open(log, "a") as f:
                    subprocess.run(["bash", "-c", reset], stdout=f, stderr=subprocess.STDOUT)
            return {**info, "status": "hang", "log": log, "error": f"timed out after {timeout}s\n" + log_excerpt(text)}
        res = read_result(result, text)
        if rc != 0 and res["valid"]:
            res.update(valid=False, fail_class="runtime_error", error=f"eval command exited {rc}\n" + log_excerpt(text))
        return {**info, "status": "ran", "log": log, "result": res}


def score_run(c: Campaign, run: dict, baseline: dict | None) -> dict:
    """score.json content for one run."""
    if run["status"] != "ran":
        res = {"valid": False, "fail_class": run["status"], "error": run.get("error"), "score": 0.0, "cases": {}}
    else:
        res = apply_gates(dict(run["result"]), c.cfg["eval"]["gates"])
        res["score"] = 0.0
        if res["valid"]:
            if baseline is None:
                res.update(valid=False, fail_class="infra", error="no baseline.json; run `dream check` first")
            else:
                s = score_vs_baseline(res, baseline, c.cfg["eval"]["direction"])
                if s is None:
                    res.update(
                        valid=False,
                        fail_class="infra",
                        error=f"cases {sorted(res['cases'])} don't match the baseline's {sorted(baseline['cases'])}",
                    )
                else:
                    res["score"] = s
    res["noise_pct"] = baseline["noise_pct"] if baseline else None
    res["score_def"] = (
        f"geomean over cases of the improvement ratio vs baseline "
        f"({c.cfg['eval']['direction']} {c.cfg['eval'].get('unit', '')}); baseline = 1.0"
    )
    for k in ("commit_under_test", "build_seconds", "eval_seconds", "report_dir"):
        res[k] = run.get(k)
    return res


def load_baseline(c: Campaign) -> dict | None:
    p = c.ledger / "baseline.json"
    return json.loads(p.read_text()) if p.exists() else None


def eval_node(c: Campaign, wt: Path, node: str) -> dict:
    """Worker-side evaluation of its worktree. Writes <node dir>/eval/."""
    out = wt / c.attempts_rel(node) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("score.json", "summary.md", "error.txt", "result.json"):
        (out / stale).unlink(missing_ok=True)
    baseline = load_baseline(c)
    bad = [p for p in changed_files(wt) if not c.allowed(p, node)]
    if bad:
        run = {"status": "forbidden_edit", "error": f"changed files outside the editable paths: {bad}"}
    else:
        snap = snapshot(wt, f"{c.attempts_rel(node)}/eval", f"dream eval snapshot {node}")
        run = run_eval(c, snap, node)
    res = score_run(c, run, baseline)
    (out / "score.json").write_text(json.dumps(res, indent=2) + "\n")
    (out / "summary.md").write_text(summary_md(node, res, c.cfg["eval"].get("unit", "")))
    if res.get("error"):
        (out / "error.txt").write_text(str(res["error"]) + "\n")
    if run.get("result") is not None:
        (out / "result.json").write_text(json.dumps(run["result"], indent=2) + "\n")
    return res


def measure_ref(c: Campaign, ref: str, label: str) -> dict:
    snap = c.git("rev-parse", f"{ref}^{{commit}}")
    run = run_eval(c, snap, label)
    if run["status"] != "ran" or not run["result"]["valid"]:
        err = run.get("error") or run.get("result", {}).get("error")
        raise RuntimeError(f"{label}: evaluation of {ref} failed ({run['status']}): {err}")
    res = apply_gates(dict(run["result"]), c.cfg["eval"]["gates"])
    if not res["valid"]:
        raise RuntimeError(f"{label}: {ref} fails the campaign gates: {res['error']}")
    return res


def make_baseline_runs(c: Campaign, runs: int | None = None) -> dict:
    n = int(runs or c.cfg["eval"]["baseline_runs"])
    ms = [measure_ref(c, c.ref_root(), f"baseline_{i + 1}") for i in range(n)]
    base = make_baseline(
        ms, float(c.cfg["eval"]["min_noise_pct"]), c.cfg["eval"]["direction"], c.cfg["eval"].get("unit", "")
    )
    (c.ledger / "baseline.json").write_text(json.dumps(base, indent=2) + "\n")
    return base


def drift_check(c: Campaign, label: str) -> tuple[bool, list]:
    m = measure_ref(c, c.ref_root(), label)
    rows = drift(m, load_baseline(c))
    return any(r[4] for r in rows), rows

# SPDX-License-Identifier: Apache-2.0
"""Claude-Code-native (cc) optimize engine, driven in-process from tt_hw_planner.

Ports the scratch bash driver into Python. For EACH discovered pipeline it drives `claude -p`
against the perf-mcp deterministic gate, re-invoking until the gate's OWN termination_check returns
can_stop (the agent's self-declared "done" is never trusted — only the deterministic gate stops it).

Single-key: authenticates off the ambient ANTHROPIC_API_KEY exactly like every other planner command
(auto-up / up / promote / emit-e2e). It does NOT read .env.agent and does NOT mint a proxy — so a
local LiteLLM mapping can't clobber the key the user exported.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PERF_DIR = "models/experimental/perf_automation"
CC_DIR = PERF_DIR + "/cc_optimize"
DEFAULT_MAX_ROUNDS = 20

_ALLOWED_TOOLS = [
    "mcp__perf-mcp__profile_model",
    "mcp__perf-mcp__measure_candidate",
    "mcp__perf-mcp__check_pcc",
    "mcp__perf-mcp__recall_knobs",
    "mcp__perf-mcp__distill_knob",
    "mcp__perf-mcp__git_head",
    "mcp__perf-mcp__git_commit",
    "mcp__perf-mcp__git_revert",
    "mcp__perf-mcp__termination_check",
    "mcp__perf-mcp__record_kernel_attempt",
    "Read",
    "Edit",
    "Write",
    "Bash",
    "Grep",
    "Glob",
]

_PROMPT = """You are optimizing the TTNN model {model} ({task} pipeline) for {metric} via the perf-mcp tools. Drive {metric} toward the roofline floor. Run CONTINUOUSLY.

termination_check() is the SOLE authority on whether more optimization is needed. It returns a DETERMINISTIC per-op CHECKLIST and a single next_target = {{op, op_class, grid, bound_by, rung}} you MUST work next. The per-op ladder ORDER is: knob:grid -> knob:dtype -> tt-lang -> cpp. An op is "nothing left" ONLY when every rung is ticked; you may STOP ONLY when can_stop=true.

LOOP:
  git_head -> termination_check -> read next_target.
  REUSE-FIRST: call recall_knobs(next_target.op_class, next_target.grid, next_target.bound_by) and APPLY/ADAPT any matching catalogued knob (heed its negative knowledge) BEFORE improvising one.
  Do EXACTLY next_target.rung on next_target.op:
    knob:grid  -> full-grid program_config. check_pcc; measure_candidate; commit a real win else revert. record_kernel_attempt(op,'grid',measured_ms,beat_baseline).
    knob:dtype -> lower that op's WEIGHT dtype (bf16->bf8_b->bf4_b). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'dtype',measured_ms,beat_baseline) EVEN IF pcc forced a revert (that marks the knob tried).
    tt-lang    -> author a tt-lang (ttl) kernel (Read GUIDELINES/11). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'tt-lang',measured_ms,beat_baseline).
    cpp        -> author a C++ Metalium kernel via ttnn.generic_op (Read GUIDELINES/12). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'cpp',measured_ms,beat_baseline).
  (IRON RULE: a real win = check_pcc ok AND verdict 'valid' AND is_real_gain. REJECTED is never a win.)
  WRITE-BACK: after you COMMIT a win you IMPROVISED (recall_knobs had no match), call distill_knob to persist the general technique; if the win RE-USED a provisional lever learned on another model, pass its id to distill_knob to graduate it.
  Re-run termination_check. Repeat. NEVER stop while can_stop=false. NEVER reason a lever "won't help" — prove it by measuring + recording the attempt.

LEAVE CLEAN (commit wins, revert in-progress edits); end with git_head. Report start->final {metric}, committed wins, and per blocking op which rungs were done + measured ms."""


def _visible_devices(devices: str) -> str | None:
    """Translate the --devices spec to a TT_VISIBLE_DEVICES value the same way before_loop does (UMD
    wants explicit ids or the var UNSET): 'single' -> '0', 'all'/'' -> None (all chips visible, var
    left unset), explicit ids ('0,1,2,3') pass through. Setting the literal 'all'/'single' breaks UMD."""
    if not devices or devices == "all":
        return None
    if devices == "single":
        return "0"
    return devices


def cc_env(repo_root: Path, devices: str) -> dict:
    """Subprocess env for cc. SINGLE-KEY: inherits the ambient environment verbatim (so the user's
    exported ANTHROPIC_API_KEY flows to `claude`), and adds only the repo paths + visible devices.
    Deliberately does NOT read .env.agent — nothing may clobber the exported key."""
    env = dict(os.environ)
    env["TT_METAL_HOME"] = str(repo_root)
    env["PYTHONPATH"] = f"{repo_root / PERF_DIR}{os.pathsep}{repo_root}"
    # MANDATORY from-scratch perf tests: discovery (_enumerate_pipelines) regenerates each pipeline's
    # WHOLE-forward perf test from its demo every run and never reuses a prior/partial one.
    env["PERF_REGEN_PERF_TEST"] = "1"
    pybin = repo_root / "python_env" / "bin"
    if pybin.is_dir():
        env["PATH"] = str(pybin) + os.pathsep + env.get("PATH", "")
    vis = _visible_devices(devices)
    if vis is not None:
        env["TT_VISIBLE_DEVICES"] = vis
        env["TT_METAL_VISIBLE_DEVICES"] = vis
    else:
        env.pop("TT_VISIBLE_DEVICES", None)
        env.pop("TT_METAL_VISIBLE_DEVICES", None)
    return env


def _python_bin(repo_root: Path) -> str:
    cand = repo_root / "python_env" / "bin" / "python"
    return str(cand) if cand.is_file() else "python"


def _latest_manifest(perf_dir: Path) -> Path | None:
    cands = sorted((perf_dir / "runs").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def discover(
    demo_dir: Path, repo_root: Path, devices: str, metric: str, perf_test=None, case=None, pcc_test=None
) -> dict | None:
    """Run before_loop (discovery + per-pipeline perf-test auto-gen) and return the manifest dict."""
    perf_dir = repo_root / PERF_DIR
    cmd = [
        _python_bin(repo_root),
        "-m",
        "agent.before_loop",
        str(demo_dir),
        "--metric",
        metric,
        "--devices",
        devices,
        "--cc-discovery",
    ]
    if perf_test:
        cmd += ["--perf-test", perf_test]
    if pcc_test:
        cmd += ["--pcc-test", pcc_test]
    if case:
        cmd += ["-k", case]
    rc = subprocess.run(cmd, cwd=str(perf_dir), env=cc_env(repo_root, devices)).returncode
    if rc != 0:
        return None
    mani = _latest_manifest(perf_dir)
    return json.loads(mani.read_text()) if mani else None


def pipelines_from_manifest(manifest: dict, model_rel: str) -> list[dict]:
    """Normalize the discovered pipeline list. Multi-modal -> one entry per task; else a single
    'main' pipeline from the top-level perf_test. Paths are made model-root-relative for the mcp env."""
    pm = manifest.get("pathmap", {})
    resolved_case = (manifest.get("perf_test_resolved") or {}).get("case")
    out = []
    for p in pm.get("pipelines", []) or []:
        if not p.get("perf_test"):
            continue
        out.append(
            {
                "task": p.get("task", "main"),
                "perf_test": f"{model_rel}/{p['perf_test']}",
                "pcc_test": f"{model_rel}/{p['pcc_test']}" if p.get("pcc_test") else "",
                "case": p.get("case") or resolved_case,
            }
        )
    if not out and pm.get("perf_test", {}).get("path"):
        out.append(
            {
                "task": "main",
                "perf_test": f"{model_rel}/{pm['perf_test']['path']}",
                "pcc_test": "",
                "case": pm["perf_test"].get("case") or resolved_case,
            }
        )
    return out


def _mcp_config(repo_root: Path, manifest_path: str, pipe: dict, devices: str, kernel_log: str) -> dict:
    env = {
        "PERF_MCP_MANIFEST": manifest_path,
        "PERF_MCP_PERF_TEST": pipe["perf_test"],
        "PERF_MCP_PCC_TEST": pipe["pcc_test"],
        "PERF_MCP_KERNEL_LOG": kernel_log,
        "TT_METAL_HOME": str(repo_root),
        "PYTHONPATH": str(repo_root),
        "PATH": f"{repo_root / 'python_env' / 'bin'}{os.pathsep}/usr/bin:/bin",
    }
    if pipe.get("case"):
        env["PERF_MCP_PERF_CASE"] = pipe["case"]
    vis = _visible_devices(devices)
    if vis is not None:
        env["TT_VISIBLE_DEVICES"] = vis
        env["TT_METAL_VISIBLE_DEVICES"] = vis
    return {
        "mcpServers": {
            "perf-mcp": {
                "command": _python_bin(repo_root),
                "args": [str(repo_root / CC_DIR / "perf_mcp.py")],
                "env": env,
            }
        }
    }


def _can_stop(repo_root: Path, mcp_env: dict, devices: str) -> bool:
    """Ask the gate ITSELF (not the agent) whether to stop — the deterministic stop authority."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import perf_mcp as P; "
        "t=P.termination_check\n"
        "for a in ('fn','func','_fn','__wrapped__'):\n"
        "    if hasattr(t,a): t=getattr(t,a); break\n"
        "print('CANSTOP=' + str(bool(t().get('can_stop'))))"
    )
    env = cc_env(repo_root, devices)
    env.update(mcp_env)  # PERF_MCP_* so the gate targets this pipeline
    try:
        r = subprocess.run(
            [_python_bin(repo_root), "-c", code, str(repo_root / CC_DIR)],
            cwd=str(repo_root / PERF_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except Exception:  # noqa: BLE001 — gate crashed/timed out -> treat as not-done, loop retries
        return False
    return "CANSTOP=True" in (r.stdout or "")


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _baseline_ms() -> float | None:
    try:
        import tempfile

        d = json.loads((Path(tempfile.gettempdir()) / "perf_mcp_baseline.json").read_text())
        return float(d["device_ms"]) if d.get("device_ms") is not None else None
    except Exception:  # noqa: BLE001
        return None


def _emit_summary(repo_root: Path, kernel_log: str, model_name: str, task: str, metric: str, start_sha: str) -> None:
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location("cc_summary", str(Path(__file__).parent / "summary.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] summary unavailable: {exc}")
        return
    wins = None
    if start_sha:
        c = _git(repo_root, "rev-list", f"{start_sha}..HEAD", "--count")
        wins = int(c) if c.isdigit() else None
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    text = mod.render_summary(
        kernel_log, _baseline_ms(), model=model_name, task=task, metric=metric, committed_wins=wins, opt_branch=branch
    )
    print("\n" + text + "\n")
    md = _latest_manifest(repo_root / PERF_DIR)
    if md:
        try:
            (md.parent / "summary.md").write_text(text)
            print(f"  [optimize/cc] summary saved: {md.parent / 'summary.md'}")
        except OSError:
            pass


def optimize_pipeline(
    repo_root: Path,
    manifest_path: str,
    pipe: dict,
    devices: str,
    metric: str,
    model_name: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> dict:
    """Drive one pipeline: claude -p re-invoked until the gate's can_stop, bounded by max_rounds."""
    task = pipe["task"]
    kernel_log = f"/tmp/cc_kernlog_{model_name}_{task}.json"
    try:
        os.path.exists(kernel_log) and os.remove(kernel_log)  # fresh ladder state per pipeline
    except OSError:
        pass
    cfg = _mcp_config(repo_root, manifest_path, pipe, devices, kernel_log)
    cfg_path = repo_root / CC_DIR / f".mcp_config_{model_name}_{task}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    prompt = _PROMPT.format(model=model_name, task=task, metric=metric)
    start_sha = _git(repo_root, "rev-parse", "HEAD")
    mcp_env = cfg["mcpServers"]["perf-mcp"]["env"]
    rounds, can_stop = 0, False
    while rounds < max_rounds:
        subprocess.run(
            [
                "claude",
                "-p",
                prompt,
                "--mcp-config",
                str(cfg_path),
                "--strict-mcp-config",
                "--allowedTools",
                *_ALLOWED_TOOLS,
                "--output-format",
                "stream-json",
                "--verbose",
            ],
            cwd=str(repo_root),
            env=cc_env(repo_root, devices),
        )
        rounds += 1
        if _can_stop(repo_root, mcp_env, devices):
            can_stop = True
            break
    _emit_summary(repo_root, kernel_log, model_name, task, metric, start_sha)
    return {"task": task, "rounds": rounds, "can_stop": can_stop}


_GL_REL = PERF_DIR + "/GUIDELINES"


def catalog_pull(repo_root: Path, remote: str, branch: str) -> None:
    """Best-effort: fetch the shared catalog branch and bring its GRADUATED_* knobs into the live
    GUIDELINES dir, so this run recalls the latest cross-model-proven knobs. Never raises."""
    try:
        f = subprocess.run(
            ["git", "fetch", remote, branch], cwd=str(repo_root), capture_output=True, text=True, timeout=180
        )
        if f.returncode != 0:
            print(f"  [catalog] pull skipped (no {remote}/{branch} yet)")
            return
        ls = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "FETCH_HEAD", _GL_REL],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        n = 0
        for path in (ls.stdout or "").splitlines():
            if "/GRADUATED_" in path and path.endswith(".md"):
                blob = subprocess.run(
                    ["git", "show", f"FETCH_HEAD:{path}"],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if blob.returncode == 0:
                    (repo_root / path).parent.mkdir(parents=True, exist_ok=True)
                    (repo_root / path).write_text(blob.stdout)
                    n += 1
        print(f"  [catalog] pulled {n} GRADUATED knob(s) from {remote}/{branch}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [catalog] pull error (ignored): {str(exc)[-120:]}")


def catalog_push(repo_root: Path, remote: str, branch: str) -> None:
    """Best-effort: commit the local GRADUATED_* knobs onto a TEMP WORKTREE of <branch> and push —
    isolated from the working branch so NO model-optimization commits are included. Never raises."""
    import shutil
    import tempfile

    try:
        gl = repo_root / _GL_REL
        grads = sorted(gl.glob("GRADUATED_*.md"))
        if not grads:
            print("  [catalog] nothing to push (no GRADUATED knobs).")
            return
        subprocess.run(
            ["git", "fetch", remote, branch], cwd=str(repo_root), capture_output=True, text=True, timeout=180
        )
        has_remote = (
            subprocess.run(
                ["git", "rev-parse", "--verify", "FETCH_HEAD"], cwd=str(repo_root), capture_output=True, text=True
            ).returncode
            == 0
        )
        wt = tempfile.mkdtemp(prefix="cc_catalog_")
        try:
            subprocess.run(
                ["git", "worktree", "add", "--detach", wt, *(["FETCH_HEAD"] if has_remote else [])],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=180,
            )
            if has_remote:
                subprocess.run(["git", "checkout", "-B", branch], cwd=wt, capture_output=True, text=True)
            else:
                subprocess.run(["git", "checkout", "--orphan", branch], cwd=wt, capture_output=True, text=True)
                subprocess.run(["git", "rm", "-rf", "."], cwd=wt, capture_output=True, text=True)
            dest = Path(wt) / _GL_REL
            dest.mkdir(parents=True, exist_ok=True)
            for g in grads:
                shutil.copy2(g, dest / g.name)
            subprocess.run(["git", "add", _GL_REL], cwd=wt, capture_output=True, text=True)
            c = subprocess.run(
                ["git", "commit", "-m", f"[perf-catalog] graduated knobs ({len(grads)})"],
                cwd=wt,
                capture_output=True,
                text=True,
            )
            if c.returncode != 0:
                print("  [catalog] no new graduated knobs to push.")
            else:
                p = subprocess.run(
                    ["git", "push", remote, f"HEAD:{branch}"], cwd=wt, capture_output=True, text=True, timeout=240
                )
                print(
                    f"  [catalog] push {'ok' if p.returncode == 0 else 'FAILED'}: "
                    f"{(p.stderr or p.stdout).strip()[-140:]}"
                )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", wt], cwd=str(repo_root), capture_output=True, text=True
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  [catalog] push error (ignored): {str(exc)[-140:]}")


def run_cc_optimize(
    demo_dir: Path,
    repo_root: Path,
    devices: str = "0",
    metric: str = "device_ms",
    perf_test=None,
    case=None,
    pcc_test=None,
    baseline_only: bool = False,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    sync_catalog: bool = False,
    catalog_remote: str = "origin",
    catalog_branch: str = "perf-catalog",
) -> dict | None:
    """Top-level cc engine: discover pipeline(s), then optimize EVERY one to the gate's can_stop.

    If sync_catalog: pull the shared GRADUATED_* knob catalog from catalog_remote/catalog_branch BEFORE
    discovery (so this run recalls the latest cross-model-proven knobs), and push any GRADUATED_* back
    at the end. Off by default — learning stays local unless opted in. Both steps are best-effort and
    never fail the run; the remote/branch is fully configurable (nothing hard-coded)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        # No exported key is FINE: `claude` may be authenticated via `claude /login` (README §5.2
        # Option A). Every claude subprocess uses those stored creds; claude surfaces its own error
        # if truly unauthenticated. So we don't hard-block here.
        print("  [optimize/cc] no ANTHROPIC_API_KEY in env — using `claude` login credentials.")
    if sync_catalog:
        catalog_pull(repo_root, catalog_remote, catalog_branch)
    manifest = discover(demo_dir, repo_root, devices, metric, perf_test, case, pcc_test)
    if not manifest:
        print("  [optimize/cc] discovery failed (before_loop produced no manifest).")
        return None
    perf_dir = repo_root / PERF_DIR
    manifest_path = str(_latest_manifest(perf_dir))
    model_rel = os.path.relpath(demo_dir, repo_root)
    model_name = Path(demo_dir).name
    pipes = pipelines_from_manifest(manifest, model_rel)
    is_mm = manifest.get("pathmap", {}).get("is_multimodal")
    print(f"  [optimize/cc] discovered pipelines: {[p['task'] for p in pipes]} (multimodal={is_mm})")
    if baseline_only or not pipes:
        return {"pipelines": pipes, "is_multimodal": is_mm, "results": []}
    results = []
    for pipe in pipes:
        print(f"  [optimize/cc] === optimizing pipeline: {pipe['task']} ===")
        results.append(optimize_pipeline(repo_root, manifest_path, pipe, devices, metric, model_name, max_rounds))
    if sync_catalog:
        catalog_push(repo_root, catalog_remote, catalog_branch)
    return {"pipelines": pipes, "is_multimodal": is_mm, "results": results}

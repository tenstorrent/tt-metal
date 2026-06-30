from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PERF_DIR = "models/experimental/perf_automation"
CC_RUN_REL = PERF_DIR + "/cc_optimize/run.py"


def _load_cc_runner(repo_root: Path):
    """Load the cc engine (perf_automation/cc_optimize/run.py) by path — it's outside this package.
    run.py imports only stdlib, so a standalone file-load is safe. Returns run_cc_optimize or None."""
    import importlib.util

    path = repo_root / CC_RUN_REL
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("cc_optimize_run", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.run_cc_optimize
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] failed to load cc engine: {exc}")
        return None


def _repo_root() -> Path:
    from ..discovery import BRINGUP_ROOT

    return BRINGUP_ROOT()


def _resolve_target(target: str, repo_root: Path):
    p = Path(target)
    if not p.is_absolute():
        rel = repo_root / target
        if rel.is_dir():
            return rel.resolve()
    if p.is_dir():
        return p.resolve()
    from ..bringup_loop import find_demo_dir

    d = find_demo_dir(target, repo_root)
    return d.resolve() if d else None


def classify_pipeline(demo_dir: Path) -> str:
    return "emitted" if (Path(demo_dir) / "bringup_status.json").is_file() else "existing"


def _setup_isolation(repo_root: Path, demo_dir: Path):
    """Isolate an EXISTING tt-metal demo's optimization in a throwaway git worktree on a fresh
    branch, so the user's working tree + current branch are never mutated. The cc engine derives
    its whole run env (TT_METAL_HOME/PYTHONPATH/PATH/python) from repo_root, so pointing it at the
    worktree (with python_env/build symlinked in from the main tree) runs the perf test there and
    commits every kept win to the new branch. Returns {wt, branch, demo_in_wt} or None on failure."""
    import time

    from .. import worktree as wt_mod

    try:
        rel = demo_dir.resolve().relative_to(repo_root.resolve())
    except ValueError:
        print("  [optimize/cc] isolation: demo dir is outside the repo; cannot worktree-isolate")
        return None
    try:
        session = wt_mod.create(demo_dir.name)  # detached worktree at HEAD + shared host-dir symlinks
        wt = session.path
        branch = f"opt/{wt_mod._slug(demo_dir.name)}-{int(time.time())}"
        co = subprocess.run(["git", "-C", str(wt), "checkout", "-b", branch], capture_output=True, text=True)
        if co.returncode != 0:
            wt_mod.destroy(session)
            print(f"  [optimize/cc] isolation: branch create failed: {co.stderr.strip()}")
            return None
        # the worktree lives in /tmp and has no build artifacts or venv — borrow them from the
        # main tree so `import ttnn` and the venv python resolve (symlinks, same binaries).
        for d in ("python_env", "build", "build_Release"):
            src = repo_root / d
            dst = wt / d
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError as exc:  # noqa: BLE001
                    print(f"  [optimize/cc] isolation: WARN could not symlink {d}: {exc}")
        return {"wt": wt, "branch": branch, "demo_in_wt": wt / rel, "session": session}
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] isolation setup failed: {exc}")
        return None


def _report_isolation(iso: dict, repo_root: Path) -> None:
    """The wins are committed on `branch` (shared .git, visible from the main repo). Leave the
    worktree in place so the run artifacts (runs/, profiles/) persist; print how to inspect/merge."""
    branch = iso["branch"]
    print(f"  [optimize/cc] done — wins committed on branch '{branch}' (your working tree was untouched)")
    print(f"      inspect:  git -C {repo_root} log --oneline HEAD..{branch}")
    print(f"      diff:     git -C {repo_root} diff HEAD..{branch}")
    print(f"      merge:    git -C {repo_root} merge {branch}    (or cherry-pick individual wins)")
    print(f"      worktree: {iso['wt']}  (remove when done: git -C {repo_root} worktree remove --force {iso['wt']})")


def _perf_env(repo_root: Path) -> dict:
    env = dict(os.environ)
    env["TT_METAL_HOME"] = str(repo_root)
    env["PYTHONPATH"] = str(repo_root)
    pybin = repo_root / "python_env" / "bin"
    if pybin.is_dir():
        env["PATH"] = str(pybin) + os.pathsep + env.get("PATH", "")
    envfile = repo_root / PERF_DIR / ".env.agent"
    if envfile.is_file():
        for raw in envfile.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                # SINGLE-KEY: never let .env.agent clobber a Claude key the user already exported
                # (e.g. a LiteLLM mapping that rejects claude-opus-4-8). Ambient creds win.
                if k in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN") and os.environ.get(k):
                    continue
                env[k] = v.strip()
    return env


def _python_bin(repo_root: Path) -> str:
    cand = repo_root / "python_env" / "bin" / "python"
    return str(cand) if cand.is_file() else sys.executable


def _latest_summary(perf_dir: Path):
    runs = sorted((perf_dir / "runs").glob("2*"), reverse=True)
    if not runs:
        return None
    latest = runs[0]
    rows = []
    ledger = latest / "ledger.jsonl"
    if ledger.is_file():
        for raw in ledger.read_text().splitlines():
            try:
                rows.append(json.loads(raw))
            except Exception:
                pass
    keeps = [r for r in rows if r.get("result") == "keep"]
    afters = [r.get("after") for r in rows if r.get("after") is not None]
    baseline_ms = None
    base = latest / "profiles" / "baseline_profile.json"
    if base.is_file():
        try:
            baseline_ms = json.loads(base.read_text()).get("device_ms")
        except Exception:
            baseline_ms = None
    return {
        "run_dir": str(latest),
        "baseline_ms": baseline_ms,
        "iters": len(rows),
        "kept": len(keeps),
        "final_ms": afters[-1] if afters else None,
        "kept_levers": [{"lever": r.get("lever"), "before": r.get("before"), "after": r.get("after")} for r in keeps],
    }


def run_perf_optimization(
    demo_dir,
    repo_root,
    devices="0",
    mesh=None,
    box=None,
    metric="device_ms",
    max_iter=1000,
    budget_usd=1_000_000_000.0,
    perf_test=None,
    case=None,
    baseline_only=False,
):
    perf_dir = repo_root / PERF_DIR
    env = _perf_env(repo_root)
    py = _python_bin(repo_root)
    before = [
        py,
        "-m",
        "agent.before_loop",
        str(demo_dir),
        "--metric",
        metric,
        "--devices",
        devices,
        "--max-iter",
        str(max_iter),
        "--budget-usd",
        str(budget_usd),
    ]
    if box:
        before += ["--box", box]
    if mesh:
        before += ["--mesh", mesh]
    if perf_test:
        before += ["--perf-test", perf_test]
    if case:
        before += ["--case", case]
    rc = subprocess.run(before, cwd=str(perf_dir), env=env).returncode
    if rc != 0:
        return None, rc
    if baseline_only:
        return _latest_summary(perf_dir), 0
    rc = subprocess.run([py, "-m", "agent.loop", "runs"], cwd=str(perf_dir), env=env).returncode
    return _latest_summary(perf_dir), rc


def cmd_optimize(args) -> int:
    try:
        from ..cli import _quiet_framework_logging

        _quiet_framework_logging()
    except Exception:
        pass
    repo_root = _repo_root()
    model_dir = getattr(args, "model_dir", None)
    pcc_test = getattr(args, "pcc_test", None)
    target = model_dir or args.target
    if not target:
        print("  [optimize] need a target (model_id / demo dir) or --model-dir.")
        return 2
    demo_dir = _resolve_target(target, repo_root)
    if demo_dir is None or not demo_dir.is_dir():
        print(
            f"  [optimize] could not resolve '{target}' to a directory "
            f"(not a path, and no planner-emitted demo with that model_id). "
            f"Pass a demo/model directory path."
        )
        return 2
    kind = "existing" if model_dir else classify_pipeline(demo_dir)
    engine = getattr(args, "engine", "cc") or "cc"
    if model_dir and engine != "cc":
        print("  [optimize] --model-dir / --pcc-test is supported only on the cc engine.")
        return 2
    print(f"  [optimize] {target} -> {demo_dir} ({kind})")
    if pcc_test:
        print(f"  [optimize] pcc gate: {pcc_test} (perf test auto-generated from it)")
    print(f"  [optimize] engine={engine} devices={args.devices} mesh={args.mesh or '-'} metric={args.metric}")
    if engine == "cc":
        run_cc = _load_cc_runner(repo_root)
        if run_cc is None:
            print(f"  [optimize/cc] could not load cc engine from {repo_root / CC_RUN_REL}")
            return 1
        # EXISTING (non-planner) demos are the user's real tt-metal source: never mutate them on the
        # current branch. Isolate in a throwaway worktree on a fresh branch unless --in-place is set.
        # Planner-emitted demos are tool-owned scaffolds, so they stay in-place.
        run_root, run_demo, iso = repo_root, demo_dir, None
        if kind == "existing" and not getattr(args, "in_place", False) and not getattr(args, "baseline_only", False):
            iso = _setup_isolation(repo_root, demo_dir)
            if iso is None:
                print(
                    "  [optimize/cc] refusing to mutate an existing demo in place. "
                    "Pass --in-place to override, or commit/branch first."
                )
                return 1
            run_root, run_demo = iso["wt"], iso["demo_in_wt"]
            print(f"  [optimize/cc] existing demo -> isolated on branch '{iso['branch']}' (working tree untouched)")
        result = run_cc(
            run_demo,
            run_root,
            devices=args.devices,
            metric=args.metric,
            perf_test=getattr(args, "perf_test", None),
            case=getattr(args, "case", None),
            pcc_test=pcc_test,
            baseline_only=getattr(args, "baseline_only", False),
            sync_catalog=getattr(args, "sync_catalog", False),
            catalog_remote=getattr(args, "catalog_remote", "origin"),
            catalog_branch=getattr(args, "catalog_branch", "perf-catalog"),
            allow_tp_latency=getattr(args, "tp_latency", False),
            model_id_hint=(None if model_dir else args.target),
        )
        if result is None:
            print("  [optimize/cc] run failed (see messages above)")
            return 1
        for r in result.get("results", []):
            print(f"      pipeline {r['task']}: {r['rounds']} round(s), can_stop={r['can_stop']}")
        if iso is not None:
            _report_isolation(iso, repo_root)
        return 0
    summary, rc = run_perf_optimization(
        demo_dir,
        repo_root,
        devices=args.devices,
        mesh=args.mesh,
        box=getattr(args, "box", None),
        metric=args.metric,
        max_iter=args.max_iter,
        budget_usd=args.budget_usd,
        perf_test=getattr(args, "perf_test", None),
        case=getattr(args, "case", None),
        baseline_only=getattr(args, "baseline_only", False),
    )
    if summary is None:
        print(f"  [optimize] perf run failed (rc={rc})")
        return rc or 1
    print(
        f"  [optimize] baseline={summary['baseline_ms']} ms, {summary['iters']} iters, "
        f"{summary['kept']} kept, final={summary['final_ms']} ms  ({summary['run_dir']})"
    )
    for k in summary["kept_levers"]:
        print(f"      keep {k['lever']}: {k['before']} -> {k['after']} ms")
    return 0 if rc == 0 else rc

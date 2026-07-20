from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
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


def _stage_untracked_data(repo_root: Path, rel: Path, wt: Path) -> None:
    scope = str(rel)
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--others", "--directory", "--", scope],
            capture_output=True,
            text=True,
        )
    except Exception:
        return
    if out.returncode != 0:
        return
    staged = []
    for line in out.stdout.splitlines():
        entry = line.strip().rstrip("/")
        if not entry or Path(entry).name == "__pycache__":
            continue
        src = repo_root / entry
        if not src.is_dir():
            continue
        dst = wt / entry
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
            staged.append(entry)
            print(f"  [optimize/cc] isolation: staged untracked data {entry} -> worktree (no re-download)")
        except OSError as exc:  # noqa: BLE001
            print(f"  [optimize/cc] isolation: WARN could not stage {entry}: {exc}")
    if staged:
        try:
            gp = subprocess.run(
                ["git", "-C", str(wt), "rev-parse", "--git-path", "info/exclude"], capture_output=True, text=True
            )
            excl = Path(gp.stdout.strip())
            if not excl.is_absolute():
                excl = wt / excl
            excl.parent.mkdir(parents=True, exist_ok=True)
            with open(excl, "a") as fh:
                fh.write("\n" + "\n".join("/" + e for e in staged) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"  [optimize/cc] isolation: WARN could not exclude staged data: {exc}")


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
        _stage_untracked_data(repo_root, rel, wt)
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


def _optimize_section_present(demo_dir) -> bool:
    p = Path(demo_dir) / "RUN_REPORT.md"
    try:
        return p.is_file() and "<!-- BEGIN optimize -->" in p.read_text()
    except Exception:
        return False


def _optimize_summary_md(label, args, summ) -> str:
    import time as _t

    engine = getattr(args, "engine", "cc") or "cc"
    md = [
        f"# Optimize (perf) — `{label}`",
        "",
        f"_Generated: {_t.strftime('%Y-%m-%d %H:%M:%S %Z')}_",
        "",
        f"- engine={engine} devices={args.devices} mesh={args.mesh or '-'} metric={args.metric}",
    ]
    if summ:
        md.append(
            f"- baseline {summ.get('baseline_ms')} ms -> final {summ.get('final_ms')} ms · "
            f"{summ.get('kept')} lever(s) kept over {summ.get('iters')} iter(s)"
        )
        if summ.get("run_dir"):
            md.append(f"- run dir: `{summ['run_dir']}`")
        kl = summ.get("kept_levers") or []
        if kl:
            md += ["", "## Kept levers", "", "| lever | before ms | after ms |", "|---|---|---|"]
            for k in kl:
                md.append(f"| `{k.get('lever')}` | {k.get('before')} | {k.get('after')} |")
    else:
        md.append("- no per-lever summary available (baseline-only or no ledger yet).")
    return "\n".join(md)


def _write_optimize_fallback(demo_dir, label, args, summ) -> None:
    if _optimize_section_present(demo_dir):
        return
    try:
        from ..run_report import upsert_report_section

        upsert_report_section(demo_dir, "optimize", _optimize_summary_md(label, args, summ))
    except Exception:
        pass


def _chip_count_from_mesh(mesh_arg) -> int:
    if not mesh_arg:
        return 0
    try:
        prod = 1
        for tok in str(mesh_arg).lower().split("x"):
            prod *= int(tok)
        return max(prod, 0)
    except Exception:  # noqa: BLE001
        return 0


def _optimize_chip_count(args):
    mesh_chips = _chip_count_from_mesh(getattr(args, "mesh", None))
    if mesh_chips >= 1:
        return mesh_chips
    dev = (getattr(args, "devices", "") or "").strip()
    if dev == "single":
        return 1
    if dev in ("", "all"):
        return None
    ids = [x for x in dev.split(",") if x.strip() != ""]
    return len(ids) or None


def _derive_topology_env(args, model_dir):
    """Reshape topology from --devices/--mesh the SAME way emit-e2e does: chip count -> shared
    plan_parallelism (kernel-viable TP x DP) -> export TT_PERF_MESH_ROWS/COLS the model's open + the
    perf skeleton read via perf_adapter.resolve_mesh_shape. Falls back to a 1D 1xN mesh when the model
    can't be probed (existing --model-dir with no HF id). No-op when chip count is unknown ('all')."""
    chips = _optimize_chip_count(args)
    if not chips:
        return
    if chips <= 1:
        os.environ["TT_PERF_MESH_ROWS"] = "1"
        os.environ["TT_PERF_MESH_COLS"] = "1"
        print("  topology : single chip -> mesh 1x1")
        return
    rows, cols, tag = 1, chips, "1D default"
    model_id = None if model_dir else getattr(args, "target", None)
    try:
        from ..parallelism import plan_parallelism

        pc = plan_parallelism(model_id, chips)
    except Exception:  # noqa: BLE001
        pc = None
    if pc is not None:
        rows, cols, tag = pc.dp, pc.tp, "kernel-viable"
    os.environ["TT_PERF_MESH_ROWS"] = str(rows)
    os.environ["TT_PERF_MESH_COLS"] = str(cols)
    print(f"  topology : {chips}-chip -> mesh {rows}x{cols} (TP={cols} DP={rows}) [{tag}]")


_MIN_FREE_BYTES = 20 * 1024**3
_STALE_TMP_AGE = 3600


def _lowest_free_bytes():
    paths = {tempfile.gettempdir()}
    try:
        paths.add(str(_repo_root()))
    except Exception:
        pass
    low = None
    culprits = []
    for p in paths:
        try:
            free = shutil.disk_usage(p).free
        except Exception:
            continue
        culprits.append((p, free))
        low = free if low is None else min(low, free)
    return low, culprits


def _disk_gate():
    low, culprits = _lowest_free_bytes()
    if low is None:
        return True, low, culprits
    return low >= _MIN_FREE_BYTES, low, culprits


def _sweep_stale_perf_mcp():
    now = time.time()
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "perf_mcp_*")):
        try:
            if os.path.isdir(d) and now - os.path.getmtime(d) > _STALE_TMP_AGE:
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


def _out_of_disk_msg(low):
    gb = (low or 0) / (1024**3)
    return (
        f"  [optimize] OUT OF DISK — only {gb:.1f} GB free (need >= {_MIN_FREE_BYTES // 1024 ** 3} GB). "
        "Free space and rerun; clear stale /tmp/perf_mcp_* dirs and old worktrees."
    )


def invalid_trace_flag_error():
    v = os.environ.get("TT_PERF_TRACE")
    if v is not None and v not in ("0", "1"):
        return (
            "TT_PERF_TRACE=%r is invalid: it is a trace on/off flag (0=eager, 1=trace+2CQ), NOT a "
            "command-queue count. Set it to 0 or 1; control command queues via TT_PERF_NUM_CQ." % v
        )
    return None


def cmd_optimize(args) -> int:
    _tf = invalid_trace_flag_error()
    if _tf:
        print("error: " + _tf)
        return 1
    if os.environ.get("PERF_MCP_SUPERVISED") != "1":
        _sweep_stale_perf_mcp()
    _ok, _low, _cul = _disk_gate()
    if not _ok:
        print(_out_of_disk_msg(_low))
        return 1
    # AUTO-RESTART SUPERVISOR: an orchestrator SIGSEGV / native tt-metal crash kills the whole Python
    # process, which no in-process recovery can catch. Run the real work in a supervised CHILD and, on
    # abnormal exit, reset the device and relaunch it -- the per-op ladder + attempt log persist on disk,
    # so a native crash becomes an automatic restart instead of a dead run. Disable with
    # PERF_MCP_SUPERVISE=0; bounded by PERF_MCP_MAX_RESTARTS (default 3).
    import os as _os, sys as _sys, subprocess as _sp, shutil as _sh, time as _t

    if _os.environ.get("PERF_MCP_SUPERVISED") != "1" and _os.environ.get("PERF_MCP_SUPERVISE", "1") == "1":
        _max = int(_os.environ.get("PERF_MCP_MAX_RESTARTS", "3") or "3")
        _ttsmi = _sh.which("tt-smi")
        for _n in range(_max + 1):
            _rc = _sp.run(
                [_sys.executable, "-m", "scripts.tt_hw_planner", *_sys.argv[1:]],
                env={**_os.environ, "PERF_MCP_SUPERVISED": "1"},
            ).returncode
            if _rc != 0:
                _dok, _dlow, _ = _disk_gate()
                if not _dok:
                    print(_out_of_disk_msg(_dlow), flush=True)
                    return _rc
            if _rc == 0 or _n >= _max:
                if _rc != 0:
                    print(f"  [optimize/supervisor] child exited rc={_rc}; {_max} restart(s) exhausted.", flush=True)
                return _rc
            print(
                f"  [optimize/supervisor] orchestrator exited rc={_rc} (likely native crash / device wedge) "
                f"-- resetting device + restarting (restart {_n + 1}/{_max}); ladder state is preserved on disk.",
                flush=True,
            )
            try:
                from models.experimental.perf_automation.cc_optimize.run import _reclaim_device as _rcl

                print("  [optimize/supervisor] " + _rcl(getattr(args, "devices", "all") or "all"), flush=True)
            except Exception as _e:  # noqa: BLE001
                if _ttsmi:
                    try:
                        _sp.run([_ttsmi, "-r"], timeout=420, capture_output=True, text=True)
                    except Exception:  # noqa: BLE001
                        pass
                print(f"  [optimize/supervisor] reclaim fell back to reset ({_e})", flush=True)
            _t.sleep(5)

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
    _sep = "=" * 78
    _hitl = " · HITL" if getattr(args, "hitl", False) else ""
    print(f"\n{_sep}\n  Optimize (perf) — {target}{_hitl}\n{_sep}")
    print(f"  model    : {demo_dir} ({kind})")
    print(f"  engine   : {engine} · devices {args.devices} · mesh {args.mesh or '-'} · metric {args.metric}")
    if pcc_test:
        print(f"  pcc gate : {pcc_test} (perf test auto-generated from it)")
    _derive_topology_env(args, model_dir)
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
        if getattr(args, "module_level", False):
            from .module_optimize import run_module_level_optimize

            return run_module_level_optimize(args, run_demo, run_root, run_cc)
        result = run_cc(
            run_demo,
            run_root,
            devices=args.devices,
            metric=args.metric,
            perf_test=getattr(args, "perf_test", None),
            case=getattr(args, "case", None),
            pcc_test=pcc_test,
            baseline_only=getattr(args, "baseline_only", False),
            e2e_only=getattr(args, "e2e_only", False),
            sync_catalog=getattr(args, "sync_catalog", False),
            catalog_remote=getattr(args, "catalog_remote", "origin"),
            catalog_branch=getattr(args, "catalog_branch", "perf-catalog"),
            max_rounds=getattr(args, "max_rounds", 3),
            model_id_hint=(None if model_dir else args.target),
            hitl=getattr(args, "hitl", False),
        )
        if result is None:
            print("  [optimize/cc] run failed (see messages above)")
            return 1
        for r in result.get("results", []):
            print(f"      pipeline {r['task']}: {r['rounds']} round(s), can_stop={r['can_stop']}")
        if iso is None:
            _write_optimize_fallback(
                demo_dir, args.target or demo_dir.name, args, _latest_summary(repo_root / Path(PERF_DIR))
            )
        if iso is not None:
            _report_isolation(iso, repo_root)
        return 0

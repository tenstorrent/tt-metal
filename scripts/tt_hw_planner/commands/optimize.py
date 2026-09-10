from __future__ import annotations

import glob
import json
import os
import re
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


def _derive_mesh_device_env(args) -> None:
    """Resolve MESH_DEVICE from --box/--mesh, as bring-up already does.

    MESH_DEVICE is tt-metal's OWN convention, not the planner's (227 files under models/, renamed
    from FAKE_DEVICE "for consistency with vLLM"): it tells the MODEL which board profile to load.

    Deliberately does NOT set TT_MESH_GRAPH_DESC_PATH. That variable OVERRIDES tt-metal's topology
    auto-discovery, and emit_e2e.py:1660 says plainly not to set it. Measured on a 4-chip p300c:
    both llama3_1_8b_p150 (PCC 0.996046) and gemma-3-12b-it (PCC 0.989811) pass with it UNSET, at
    the same speed -- the fabric timeout it was once used for came from invoking a demo node
    directly, without the harness's device params, not from anything optimize does.

    --box already carries the board and --mesh the shape, so the operator was restating the same
    hardware in the environment. bring-up resolves it (bringup.py:456) via
    find_box(name).arch -> mesh_device_for(arch, shape); this calls that SAME function rather than
    copying the table, so the two paths cannot drift.

    Never overrides an explicitly exported value -- the operator's choice is not silently replaced.
    Best-effort: an unknown box, an unparseable mesh, or an unlabelled shape leaves the environment
    exactly as it is today.
    """
    box_name = getattr(args, "box", None)
    if not box_name:
        return
    try:
        from ..bringup import mesh_device_for
        from ..hardware import HARDWARE, find_box

        # find_box is CASE-SENSITIVE ('p150' raises, 'P150' works) while the CLI help suggests
        # lowercase ("e.g. p300c, T3K, Galaxy"), so match case-insensitively before looking up.
        canon = next((b.name for b in HARDWARE if b.name.lower() == str(box_name).lower()), box_name)
    except Exception:  # noqa: BLE001 -- the table itself failed to import; nothing to validate against
        return
    try:
        box = find_box(canon)
    except Exception:  # noqa: BLE001
        # A BOX NAME THAT DOES NOT RESOLVE MUST NOT PASS QUIETLY. This used to fall into the same
        # blanket `return` as an import failure, so --box p300c -- the board series tt-smi actually
        # prints for these chips, and the example the CLI help itself used to give -- set nothing,
        # printed nothing, and left the model loading a default profile. The operator asked for a
        # specific board and got silence. The names are a closed set, so this is checkable.
        raise SystemExit(
            "unknown --box %r. Valid boxes: %s (case-insensitive). Note this is the planner's BOX "
            "name, not the board series tt-smi prints: four 'p300c' Blackhole chips are the box QB2."
            % (box_name, ", ".join(b.name for b in HARDWARE))
        )
    shape = (1, 1)
    raw = getattr(args, "mesh", None)
    if raw:
        try:
            r, c = str(raw).lower().split("x")
            shape = (int(r), int(c))
        except (ValueError, AttributeError):
            return
    if "MESH_DEVICE" not in os.environ:
        try:
            label, _note = mesh_device_for(box.arch, shape)
        except Exception:  # noqa: BLE001
            label = None
        if label:
            os.environ["MESH_DEVICE"] = label
            print(f"  mesh device : --box {box_name} + mesh {shape[0]}x{shape[1]} -> MESH_DEVICE={label}")


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
_RUNS_KEEP_FULL = int(os.environ.get("PERF_MCP_RUNS_KEEP_FULL", "3") or "3")
_RUNS_KEEP_TOTAL = int(os.environ.get("PERF_MCP_RUNS_KEEP_TOTAL", "20") or "20")


def _prune_runs(perf_dir: Path) -> None:
    runs_dir = Path(perf_dir) / "runs"
    if not runs_dir.is_dir():
        return
    try:
        dirs = sorted(
            (p for p in runs_dir.glob("2*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return
    for stale in dirs[_RUNS_KEEP_TOTAL:]:
        shutil.rmtree(stale, ignore_errors=True)
    for old in dirs[_RUNS_KEEP_FULL:_RUNS_KEEP_TOTAL]:
        tracy = old / "profiles" / "tracy_out"
        if tracy.is_dir():
            shutil.rmtree(tracy, ignore_errors=True)


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
        return "TT_PERF_TRACE=%r is invalid: it is a trace on/off flag (0=eager, 1=trace). " "Set it to 0 or 1." % v
    return None


def _run_matmul_sweep_prepass(args, run_root: Path, run_demo: Path, node: str = None, case: str = None) -> None:
    """Optional --matmul-sweep pre-pass: BEFORE the cc engine runs, sweep each distinct matmul
    (fidelity x dtype, PCC-gated) and write matmul_sweep.json as a warm-start table. Loads the
    standalone perf_automation cc_optimize/matmul_sweep.py by PATH (like the cc runner) so nothing in
    the optimize engine is imported or changed.

    ``node``/``case`` override the enumeration node: the full-pipeline path passes None and falls back
    to --perf-test; the per-module path passes each module's own PCC test node (which runs that
    module's forward, so its matmuls are what get swept). If no node can be resolved it warns and
    skips. Any failure is reported and swallowed so the optimize run still proceeds."""
    # Whether the CALLER supplied a node has to be captured BEFORE the fallback overwrites it.
    # `case = case if node else args.case` asked the question one line too late: by then `node` had
    # already been reassigned from --perf-test and was always truthy, so args.case was unreachable
    # and the full-pipeline sweep enumerated the whole perf test -> 0 matmul shapes (issue #14).
    _caller_supplied_node = node is not None
    node = node or getattr(args, "perf_test", None)
    # Fall back to --case ONLY when the caller supplied neither half of the selection. An argument
    # the caller passed explicitly must never lose to a namespace default.
    if case is None and not _caller_supplied_node:
        case = getattr(args, "case", None)
    if not node:
        print("  [optimize/matmul-sweep] no node to sweep (need --perf-test or a per-module PCC node); skipping")
        return
    out = str(Path(run_demo) / "matmul_sweep.json")
    sweep_path = Path(run_root) / PERF_DIR / "cc_optimize" / "matmul_sweep.py"
    if not sweep_path.is_file():
        print(f"  [optimize/matmul-sweep] sweep module not found at {sweep_path}; skipping")
        return
    print(f"  [optimize/matmul-sweep] pre-pass on {node}{' -k ' + case if case else ''} -> {out}")
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("cc_matmul_sweep", sweep_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        s = mod.run_prepass(
            node,
            case=case,
            out_path=out,
            pcc_threshold=getattr(args, "matmul_sweep_pcc", 0.99),
            iters=getattr(args, "matmul_sweep_iters", 5),
            max_shapes=getattr(args, "matmul_sweep_max_shapes", 0),
            repo_root=Path(run_root),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/matmul-sweep] sweep failed ({type(exc).__name__}: {str(exc)[-300:]}); optimize continues")
        return
    print(
        f"  [optimize/matmul-sweep] {s.get('shapes', 0)} matmul shapes, {s.get('seeded', 0)} seeded, "
        f"{s.get('improved', 0)} beat full-precision -> {out}"
    )


def cmd_optimize(args) -> int:
    _tf = invalid_trace_flag_error()
    if _tf:
        print("error: " + _tf)
        return 1
    os.environ["PERF_MCP_DEVICES"] = (getattr(args, "devices", "") or "").strip()
    if os.environ.get("PERF_MCP_SUPERVISED") != "1":
        _sweep_stale_perf_mcp()
        try:
            _prune_runs(_repo_root() / PERF_DIR)
        except Exception:
            pass
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
        # How often the attempt's tree is re-snapshotted. It is the only record of a grandchild once
        # the root exits, so it bounds how stale the kill list can be; a /proc scan costs microseconds
        # against a run measured in hours.
        _REAP_POLL_S = float(_os.environ.get("PERF_MCP_SUPERVISOR_REAP_POLL_S", "10"))
        # Imported from the ONE definition rather than restated: a second literal here that drifted
        # from run.py's would turn every refusal back into three device resets, silently.
        try:
            from models.experimental.perf_automation.cc_optimize.run import EXIT_REFUSED as _EXIT_REFUSED
        except Exception:  # noqa: BLE001 -- a supervisor that cannot import must still supervise
            _EXIT_REFUSED = 3
        _ttsmi = _sh.which("tt-smi")

        # THE ATTEMPT IS NOT OVER WHEN MY CHILD EXITS.
        #
        # This was `_sp.run(...)`, and its return was taken as the attempt ending. It is not: the run
        # spawns workers with start_new_session=True (run.py, perf_test_agent.py) so their groups can
        # be killed independently, and that same flag means they SURVIVE their parent -- reparented
        # to init, in their own sessions, invisible to a group kill. _reclaim_device does not see
        # them either; it kills device HOLDERS, and a worker between device operations holds nothing.
        #
        # Measured 2026-08-16: attempt 1 gave up on perf-test generation (rc=1), and its orchestrator,
        # a detached subprocess and a perf-test agent were still running 77, 70 and 37 minutes later.
        # The supervisor launched attempt 2 into the same board. Two runs driving one board took the
        # ARC cores down -- `tt-smi -r` then failed with "ARC core (8, 0) failed to start" until the
        # tree was killed by hand, after which the identical reset succeeded first try.
        #
        # So the tree is snapshotted WHILE it lives (once the root exits the PPID links are gone for
        # good) and reaped before anything else happens. The tool already had three tree-killers --
        # cli.py:_kill_process_tree, cc_harness.py:_kill_agent_tree, probes.py:_kill_tree -- and this
        # path used none of them.
        try:
            from models.experimental.perf_automation.agent.probes import _descendant_pids as _desc, _kill_tree as _reap
        except Exception:  # noqa: BLE001 -- supervising without a reaper beats not supervising
            _desc, _reap = (lambda _p: []), (lambda _p, extra=(): None)

        def _alive(pid):
            try:
                _os.kill(int(pid), 0)
                return True
            except Exception:  # noqa: BLE001
                return False

        def _run_attempt(_argv, _env):
            """Run one attempt, tracking its whole tree, and reap whatever outlives it."""
            _p = _sp.Popen(_argv, env=_env)
            _tree: set = set()
            while True:
                try:
                    _p.wait(timeout=_REAP_POLL_S)
                    break
                except _sp.TimeoutExpired:
                    _tree.update(_desc(_p.pid))
            _tree.update(_desc(_p.pid))
            _left = [q for q in _tree if _alive(q)]
            if _left:
                # REPORTED, NOT SILENT. A leaked tree is the difference between "the run crashed" and
                # "the run is still going and about to be raced", and the operator cannot see it.
                print(
                    "  [optimize/supervisor] the attempt left %d process(es) running after exiting "
                    "(%s) -- killing them before going on" % (len(_left), ", ".join(str(q) for q in sorted(_left)[:8])),
                    flush=True,
                )
                _reap(_p.pid, extra=_left)
                _t.sleep(2)
                _still = [q for q in _left if _alive(q)]
                if _still:
                    # A process SIGKILL cannot clear is in D-state on the device. Starting another
                    # attempt now is what broke the board; say so and stop instead.
                    print(
                        "  [optimize/supervisor] %d process(es) survived SIGKILL (%s) -- refusing to "
                        "start another attempt on a board they may still be holding."
                        % (len(_still), ", ".join(str(q) for q in sorted(_still)[:8])),
                        flush=True,
                    )
                    return _p.returncode, _still
            return _p.returncode, []

        for _n in range(_max + 1):
            _rc, _stuck = _run_attempt(
                [_sys.executable, "-m", "scripts.tt_hw_planner", *_sys.argv[1:]],
                {**_os.environ, "PERF_MCP_SUPERVISED": "1"},
            )
            if _stuck:
                return _rc or 1
            if _rc != 0:
                _dok, _dlow, _ = _disk_gate()
                if not _dok:
                    print(_out_of_disk_msg(_dlow), flush=True)
                    return _rc
            # A REFUSAL IS NOT A CRASH. The child refused to start on evidence it already gathered
            # (a red preflight suite, a dirty tree under PERF_MCP_REQUIRE_CLEAN). Relaunching
            # re-derives the same decision from the same evidence, so a restart can only spend three
            # device resets and ten minutes to reach the verdict that was available at once -- and
            # bury the reason under three "likely native crash" lines that misdescribe it. See
            # cc_optimize/run.py:EXIT_REFUSED.
            if _rc == _EXIT_REFUSED:
                # A REFUSAL IS RETRIED NOW, because discovery is REGENERATED on each attempt.
                #
                # It used to return here, on the reasoning that "relaunching re-derives the same
                # decision from the same evidence". That holds for a refusal grounded in something
                # fixed -- a red preflight, a dirty tree -- and not for the one that actually fires:
                # the lead review rejecting a plan that an AGENT wrote. The next attempt writes a
                # different plan, so the verdict is not re-derived, it is re-earned.
                #
                # The harm that made this non-retryable was never the retry itself. It was run 9's
                # sibling failure: a restart that left the previous attempt's process tree alive, so
                # two runs loaded the model onto one board and wedged it past what tt-smi -r could
                # restart. That is fixed at the source -- the supervisor now reaps the tree and
                # REFUSES to start again if anything survives SIGKILL -- so a retry no longer races
                # anything.
                #
                # Still bounded by the same restart limit, so a refusal that IS grounded in something
                # fixed costs three attempts and stops, rather than looping.
                print(
                    f"  [optimize/supervisor] child REFUSED (rc={_rc}) — a decision, not a crash. "
                    f"Discovery is regenerated per attempt, so retrying (restart {_n + 1}/{_max}); "
                    "the reason is above.",
                    flush=True,
                )
                if _n >= _max:
                    print(
                        f"  [optimize/supervisor] refused {_max + 1} times; the decision is not going to change.",
                        flush=True,
                    )
                    return _rc
            if _rc == 0 or _n >= _max:
                if _rc != 0:
                    print(f"  [optimize/supervisor] child exited rc={_rc}; {_max} restart(s) exhausted.", flush=True)
                return _rc
            print(
                # "likely native crash / device wedge" was printed for EVERY non-zero rc, including a
                # perf-test generation failure that never touched the device. A fixed string is not a
                # diagnosis, and it sent three separate investigations to the wrong subsystem. State
                # the code; the reason is in the child's own output above.
                f"  [optimize/supervisor] orchestrator exited rc={_rc} -- resetting device + restarting "
                f"(restart {_n + 1}/{_max}); the reason is in the output above. Ladder state is preserved on disk.",
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
    _derive_mesh_device_env(args)
    _derive_topology_env(args, model_dir)
    if getattr(args, "target_band", False):
        os.environ["PERF_MCP_TARGET_BAND"] = "1"
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
        dash_url = None
        if getattr(args, "dashboard", False):
            # The run directory does not exist until the engine creates it, so the collector
            # RE-RESOLVES it on every poll; state-dir env exported later (--persist) is likewise
            # read per poll. A dashboard failure must never fail the run.
            from ..optimize_dashboard import (
                collect_state,
                find_run_dir,
                post_hitl_decision,
                serve_in_thread,
                state_dir_candidates,
            )

            _dash_slug = run_demo.name

            def _dash_collect():
                rd = find_run_dir(run_root, slug=_dash_slug)
                if rd is None:
                    return {"run": {"id": None, "live": False}, "model": {"slug": _dash_slug}}
                return collect_state(rd, state_dir_candidates(run_root, _dash_slug), _dash_slug)

            def _dash_decision(action):
                rd = find_run_dir(run_root, slug=_dash_slug)
                if rd is None:
                    return False, "no run yet"
                return post_hitl_decision(rd, action)

            try:
                _srv, _t, dash_url = serve_in_thread(
                    getattr(args, "dashboard_host", "127.0.0.1"),
                    getattr(args, "dashboard_port", 8798),
                    _dash_collect,
                    decision_fn=_dash_decision,
                )
                print(f"  [optimize/cc] dashboard: {dash_url} (levers shown live as they land)")
            except Exception as exc:  # noqa: BLE001
                print(f"  [optimize/cc] dashboard unavailable ({exc}); the run continues without it")
        if getattr(args, "module_level", False):
            from .module_optimize import run_module_level_optimize

            return run_module_level_optimize(args, run_demo, run_root, run_cc)
        if getattr(args, "matmul_sweep", False):
            # HAND THE FLAG TO THE ENGINE instead of sweeping here. This used to call the pre-pass
            # directly, which ran BEFORE the engine's discover() generates a perf test -- so the
            # sweep's only possible node was an operator-supplied --perf-test, and `--matmul-sweep`
            # alone silently did nothing. The engine now runs it straight after discovery, where the
            # generated node exists, so the same test serves both and nobody is asked for it twice.
            os.environ["PERF_MCP_MATMUL_SWEEP"] = "1"
            os.environ["PERF_MCP_MATMUL_SWEEP_PCC"] = str(getattr(args, "matmul_sweep_pcc", 0.99))
            os.environ["PERF_MCP_MATMUL_SWEEP_ITERS"] = str(getattr(args, "matmul_sweep_iters", 5))
            os.environ["PERF_MCP_MATMUL_SWEEP_MAX_SHAPES"] = str(getattr(args, "matmul_sweep_max_shapes", 0))
        # --persist: keep the run's MEMORY somewhere a reboot does not erase.
        #
        # tmpstate.state_dir() is `PERF_MCP_STATE_DIR or tempfile.gettempdir()`, and nothing sets that
        # variable -- so by default the attempt history, the ledger and the full-pipeline bar all live
        # in /tmp. That is the right home for a WORKTREE, which is a disposable sandbox whose only
        # durable output is committed to the run's branch. It is the wrong home for the record of what
        # has already been tried: lose it and the next run re-runs every knob it had already proved
        # useless, which is exactly what the rung-closure enforcement exists to prevent.
        #
        # Keyed per model so two models never share a history, and OPT-IN so the default keeps /tmp's
        # self-cleaning. LEDGER_DIR follows it because measurements.py resolves the ledger relative to
        # the state dir; setting one without the other splits them apart and the report then silently
        # finds no anchors -- the defect measurements.py:213 documents. setdefault, so an operator who
        # exported either by hand still wins.
        if getattr(args, "persist", False):
            _slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", (run_demo.name or "model")).strip("_") or "model"
            # IN THE REPO, AND IN THE REAL ONE. This lived in ~/.perf_mcp, which is durable but
            # sits apart from everything else a run produces -- the archive, the profiles, the
            # reports are all under the repo, and the record of what was tried was somewhere else
            # entirely. `.state/` is gitignored beside `runs/`, so it is durable AND does not
            # dirty a tracked tree, which is what kept this out of the model directory.
            #
            # repo_root, NEVER run_root: run_root is the throwaway worktree under /tmp once
            # isolation is set up, and putting the run's memory there is the bug --persist exists
            # to fix. The whole point is a home the worktree's deletion cannot reach.
            _persist_dir = repo_root / "models" / "experimental" / "perf_automation" / ".state" / _slug
            _persist_dir.mkdir(parents=True, exist_ok=True)
            # CARRY FORWARD what the old location already learned, once, rather than starting a
            # model over because its memory moved. Copied and not moved: an older tool version
            # pointed at ~/.perf_mcp still finds its state where it left it.
            _legacy = Path.home() / ".perf_mcp" / _slug
            try:
                if _legacy.is_dir() and not any(_persist_dir.iterdir()):
                    import shutil as _shutil

                    for _f in _legacy.iterdir():
                        (_shutil.copytree if _f.is_dir() else _shutil.copy2)(_f, _persist_dir / _f.name)
                    print(f"  [optimize/cc] --persist: carried {_slug}'s existing memory over from {_legacy}")
            except Exception as _exc:  # noqa: BLE001 -- a failed carry-forward is a fresh start, not a failed run
                print(f"  [optimize/cc] --persist: WARN could not carry over {_legacy}: {_exc}")
            os.environ.setdefault("PERF_MCP_STATE_DIR", str(_persist_dir))
            os.environ.setdefault("PERF_MCP_LEDGER_DIR", str(_persist_dir))
            print(f"  [optimize/cc] --persist: run memory in {_persist_dir} (survives reboots; /tmp does not)")
        # --fresh: FORGET, then run. State is carried forward on purpose -- a baseline is expensive, a
        # coverage window costs device probes, and the ceiling anchor is write-once so the report and
        # the stop gate cannot score one run against two ceilings. That is right while the tool is
        # unchanged, and wrong the moment it changes: a pinned value records WHAT it is and never
        # WHICH RULE produced it, so a number from a superseded formula outlives the fix.
        #
        # Measured on Voxtral 2026-08-14: the anchor held active_bytes = 3611.48 MB
        # ("checkpoint bytes + HF config"), which is total_params x 1.0 -- the placeholder width from
        # before the ceiling learned to divide by the width the loader actually chose. compute_target
        # takes that anchor ahead of every other source, so the corrected rule never ran and the run
        # published 141.8 tok/s/u against a true ~71, making the model read as twice as close to the
        # wall as it is -- the input to can_stop. Clearing the coverage and knob caches did not touch
        # it: it lives in the persistent ledger.
        if getattr(args, "fresh", False):
            try:
                sys.path.insert(0, str(Path(run_root) / "models" / "experimental" / "perf_automation"))
                from agent.fresh_start import describe as _fresh_describe, wipe as _fresh_wipe

                _sd = os.environ.get("PERF_MCP_STATE_DIR") or tempfile.gettempdir()
                _removed = _fresh_wipe(
                    _sd,
                    tool_root=Path(run_root) / "models" / "experimental" / "perf_automation",
                    model_dir=run_demo,
                )
                print("  [optimize/cc] --fresh: %s" % _fresh_describe(_removed))
                # AND THE MODEL, back to the state it was published in. The wins are committed to the
                # model tree and survive a restart; the baseline and the ceiling they are measured
                # against live in the state just cleared above. Keeping the first while resetting the
                # other two is the combination that lies: the run re-derives its ceiling from a model
                # that already carries the optimizations, so the target moves with the work.
                #
                # voxtral, measured: a fidelity lever took the pinned peak from 175.5 TFLOPS (HiFi4,
                # pre-campaign) to 702.0 (LoFi) and prefill's ceiling from 203.82 ms to 50.95 -- a 4x
                # change in the yardstick caused by a win, while the report presented the mid-campaign
                # checkpoint as the model's starting point and said nothing about the 38 commits
                # already in the tree.
                #
                # Skipped, loudly, for a model with no published commit: there is no origin to return
                # to and inventing one would discard work nobody agreed to lose.
                from agent.fresh_start import reset_model_to_published as _fresh_reset

                _mr = _fresh_reset(run_demo)
                if _mr.get("changed"):
                    print(
                        "  [optimize/cc] --fresh: model %s (baseline and ceiling now describe the same tree)"
                        % _mr["why"]
                    )
                else:
                    print("  [optimize/cc] --fresh: model NOT reset -- %s" % _mr.get("why"))
            except Exception as _fe:  # noqa: BLE001 -- a clear that cannot run must not take the run down
                print("  [optimize/cc] --fresh skipped: %s" % str(_fe)[:160])
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
            rc = 1
        else:
            for r in result.get("results", []):
                print(f"      pipeline {r['task']}: {r['rounds']} round(s), can_stop={r['can_stop']}")
            if iso is None:
                _write_optimize_fallback(
                    demo_dir, args.target or demo_dir.name, args, _latest_summary(repo_root / Path(PERF_DIR))
                )
            if iso is not None:
                _report_isolation(iso, repo_root)
            rc = 0
        if dash_url is not None:
            # The daemon thread dies with the process; hold the process open so the final state —
            # which levers landed and what they bought — stays inspectable after the run.
            print(f"  [optimize/cc] run finished — dashboard still serving at {dash_url} (Ctrl+C to exit)")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                pass
        return rc

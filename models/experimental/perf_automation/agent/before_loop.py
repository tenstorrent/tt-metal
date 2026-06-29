"""Stage-1 driver — the Before Loop (PLAN section 7).

Invocation is FOLDER + METRIC; everything else is discovered by the sub-agent
(perf test + case, PCC entry points, components, model files) and verified by
deterministic stages (preflight pytest --collect-only before any long run).

    python -m agent.before_loop <model_root> --metric device_ms --target 12

Stages (each prints a banner to stderr and appends to runs/<id>/events.jsonl):
  [1/5] environment_check   [2/5] cache_playbook   [3/5] discover (sub-agent)
  [4/5] preflight collect   [5/5] tracy baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .checkpoint import Checkpoint
from .events import write_event
from .environment import environment_check
from .model_files import read_model_files
from .opclass import STRUCTURAL_OP_CLASSES
from .router import build_index, cache_playbook
from .probes import PerfRunFailed
from .run import Run
from .tracy_tool import profile_model, stack_report

PKG_ROOT = Path(__file__).parent.parent
DEFAULT_PLAYBOOK = PKG_ROOT / "GUIDELINES"
DEFAULT_RUNS_ROOT = PKG_ROOT / "runs"
DEFAULT_CACHE = PKG_ROOT / ".cache" / "playbook_index.json"
FIXTURES = PKG_ROOT / "tests" / "fixtures"

METRIC_UNITS = {"device_ms": "ms", "wall_ms": "ms", "fps": "fps", "throughput_tok_s": "tok/s"}
N_STAGES = 7


_SHAPE_CONFIG_CRASH_RE = re.compile(
    r"block_h|per_core_M|per_core_N|num_cores_r|ceil\(Mt|must equal ceil|program.?config",
    re.IGNORECASE,
)


def _seq_retry_candidates(err: str, current_seq: int) -> list[int]:
    cands: list[int] = []
    m_bh = re.search(r"block_h\s*\((\d+)\)", err or "")
    m_nc = re.search(r"num_cores_r=(\d+)", err or "")
    m_mt = re.search(r"Mt=(\d+)", err or "")
    if m_bh and m_nc and m_mt and current_seq > 0:
        cur_mt = int(m_mt.group(1))
        wanted_mt = int(m_bh.group(1)) * int(m_nc.group(1))
        if cur_mt > 0 and wanted_mt > cur_mt:
            scaled = int(round(current_seq * wanted_mt / cur_mt))
            if scaled > current_seq:
                cands.append(scaled)
    for s in (256, 384, 512, 768):
        if s > current_seq and s not in cands:
            cands.append(s)
    return cands


class _Stages:
    """Stage banners to stderr + machine-readable events.jsonl."""

    def __init__(self, events_path: Path | None):
        self.events_path = events_path
        self._t0 = 0.0
        self._n = 0
        self._name = ""

    def start(self, name: str, detail: str = "") -> None:
        self._n += 1
        self._name = name
        self._t0 = time.monotonic()
        print(f"[{self._n}/{N_STAGES}] {name:<18} {detail}", file=sys.stderr, flush=True)
        self._event("start", detail)

    def done(self, detail: str = "") -> None:
        dt = time.monotonic() - self._t0
        print(f"      ✔ {self._name}: {detail}  ({dt:.1f}s)", file=sys.stderr, flush=True)
        self._event("done", detail, dt)

    def _event(self, kind: str, detail: str, dt: float | None = None) -> None:
        if self.events_path is None:
            return
        write_event(
            self.events_path,
            phase="before_loop",
            stage=self._name,
            event=kind,
            detail=detail,
            seconds=round(dt, 2) if dt is not None else None,
        )


def check_dependencies() -> list[str]:
    """Verify the two hard tool dependencies BEFORE any stage runs.

    tt-perf-report is needed by every profile (stage-2 REFINE runs it even in
    mock-tracy mode); claude-agent-sdk is needed by the discovery sub-agent and
    the lead review gate. Returns actionable messages for anything missing."""
    import importlib.util
    import shutil as _shutil

    missing: list[str] = []
    if _shutil.which("tt-perf-report") is None:
        missing.append(
            "tt-perf-report not on PATH — install it in your tt-metal python env: "
            "pip install tt-perf-report (see requirements-agent.txt)"
        )
    if importlib.util.find_spec("claude_agent_sdk") is None:
        missing.append(
            "claude-agent-sdk not importable — install it in your tt-metal python "
            "env: pip install claude-agent-sdk (see requirements-agent.txt)"
        )
    return missing


# ---- mock boundaries (test ladder / no hardware) ----------------------------


def mock_env_probe() -> str:
    return (FIXTURES / "tt_smi_snapshot.json").read_text()


def make_mock_model_runner(model_root: str | Path) -> Callable[[str], str]:
    """Deterministic stand-in emitting the EXPANDED schema from real files."""
    root = Path(model_root)

    def runner(prompt: str) -> str:
        files = sorted(p for p in root.rglob("*.py") if p.is_file())[:6]
        if not files:
            files = sorted(p for p in root.rglob("*") if p.is_file())[:6]
        rel = [str(p.relative_to(root)) for p in files]
        runner.last_usage = {"tokens_in": 1200, "tokens_out": 300, "cost_usd": 0.012, "latency_s": 0.0}
        return json.dumps(
            {
                "perf_test": {"path": rel[0], "case": "mock"},
                "pcc": {"end_to_end": {"path": rel[0], "threshold": 0.99}},
                "components": {},
                "model_files": rel,
            }
        )

    runner.last_usage = None
    return runner


def mock_run_profiled(pcc_path, batch_size, seq_len, profiles_dir, i):
    dest = Path(profiles_dir) / f"run{i}_raw.csv"
    shutil.copyfile(FIXTURES / "ops_perf_sample.csv", dest)
    return dest, 20.0 + 0.1 * i


def mock_preflight(tt_metal_root, perf_test, case, env=None):
    return 1


def mock_review(pathmap):
    return {
        "decision": "continue",
        "reasoning": "mock review",
        "model": "mock",
        "usage": {"tokens_in": 800, "tokens_out": 60, "cost_usd": 0.008, "latency_s": 0.0},
    }


def mock_collect_cases(tt_root, perf_test, env=None):
    return [f"{perf_test}::test_mock[mock]"], "1 test collected"


# ---- the driver --------------------------------------------------------------


def before_loop(
    config: dict[str, Any],
    env_probe: Callable[[], str],
    model_runner: Callable[[str], str],
    run_profiled_factory: Callable[[str, str | None], Callable],
    preflight: Callable[..., int],
    review: Callable[[dict], dict],
    collect: Callable[..., list[str]],
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    playbook_dir: str | Path = DEFAULT_PLAYBOOK,
    cache_path: str | Path = DEFAULT_CACHE,
    tt_metal_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run Stage 1 end to end. run_profiled_factory(perf_test_repo_rel, case)
    is called AFTER discovery so stage 5 profiles what stage 3 found."""
    model_root = Path(config["model_root"]).resolve()
    tt_root = Path(tt_metal_root or os.environ.get("TT_METAL_HOME", PKG_ROOT.parents[2]))

    run = Run.create(runs_root, config=None)
    stages = _Stages(run.dir / "events.jsonl")
    print(f"run: {run.run_id}  ->  {run.dir}", file=sys.stderr, flush=True)

    stages.start("environment_check")
    env = environment_check(env_probe)
    box = config.get("box")
    if box:
        try:
            from .environment import box_facts

            mesh = config.get("mesh")
            env = box_facts(box, tuple(mesh) if mesh else None)
            print(
                f"      box={env['card']} mesh={env.get('mesh_shape')} -> worker_cores={env['worker_cores']}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(f"      WARN --box {box}: {exc}; using auto-detected single-chip env", file=sys.stderr, flush=True)
    stages.done(f"{env['card']} · {env['arch']} · {env['worker_cores']} cores")

    devices = str(config.get("devices") or "single")
    if devices == "single":
        visible = "0"
    elif devices == "all":
        visible = None
    else:
        visible = devices
    sub_env = dict(os.environ)
    if visible is not None:
        # TT_VISIBLE_DEVICES is the UMD-level var the fabric/topology mapper
        # honors (verified on 4xn300: the TT_METAL_* one alone does NOT gate
        # topology discovery). Set both for safety.
        sub_env["TT_VISIBLE_DEVICES"] = visible
        sub_env["TT_METAL_VISIBLE_DEVICES"] = visible
    config["visible_devices"] = visible
    print(
        f"      devices={devices} -> TT_VISIBLE_DEVICES="
        f"{visible if visible is not None else '(unset: full fabric)'}",
        file=sys.stderr,
        flush=True,
    )

    # Startup restore: a prior run killed mid-iteration leaves its edit ON DISK (the REVERT never
    # ran), so this run's baseline would profile the model WITH that leftover edit — observed: a
    # leftover kernel edit crashed tracy_baseline ("Illegal Runtime Args"), and a leftover crash-y
    # edit silently truncated the capture to a fake-fast baseline. Restore the model demo to its
    # committed HEAD before baselining. SCOPED to model_root (never touches unrelated repo changes);
    # disable with AGENT_NO_STARTUP_RESET=1.
    if os.environ.get("AGENT_NO_STARTUP_RESET", "").lower() not in ("1", "true", "yes"):
        stages.start("startup_reset", "restore model demo to clean git state")
        try:
            from . import gitio

            repo = gitio.repo_root(model_root)
            head = gitio.head_sha(repo)
            dirty = gitio.changed_files(repo, head, pathspec=str(model_root))
            if dirty:
                gitio.checkout(repo, head, pathspec=str(model_root))
                stages.done(f"restored {len(dirty)} leftover-dirty file(s) to {head[:9]} (prior interrupted run?)")
            else:
                stages.done(f"clean ({head[:9]})")
        except Exception as exc:  # never block the run on the restore
            stages.done(f"skipped: {exc}")

    stages.start("cache_playbook", str(playbook_dir))
    cache_playbook(playbook_dir, cache_path)
    index = build_index(playbook_dir)
    stages.done(f"{len(index)} sections indexed")

    # Agent-SDK health: the claude CLI auto-updates and can drift out of sync with the pinned
    # python claude-agent-sdk, after which EVERY agent call fails ("error result: success").
    # Detect that here (a trivial call in a clean subprocess) and, by default, auto-upgrade +
    # re-test BEFORE the first in-process SDK import (discover), so the run picks up the fix.
    stages.start("agent_sdk_health", "verify claude-agent-sdk <-> CLI compatibility")
    from .sdk_health import ensure_compatible

    sdk_status = ensure_compatible()
    if sdk_status.get("ok"):
        stages.done(
            f"healthy (claude-agent-sdk {sdk_status.get('version')}"
            + (f", auto-synced from {sdk_status['version_before']}" if sdk_status.get("healed") else "")
            + ")"
        )
    else:
        # Not healthy and not auto-fixable here — fail fast with the remediation instead of
        # running the whole sweep only to error/hang on the first agent call.
        raise SystemExit(
            "BEFORE-LOOP FAILED: agent SDK unhealthy "
            f"(claude-agent-sdk {sdk_status.get('version')}): {sdk_status.get('reason')}\n"
            f"  detail: {sdk_status.get('detail', '')}\n"
            "  fix: pip install -U claude-agent-sdk  (or set AGENT_SDK_AUTOSYNC=1 to auto-fix)"
        )

    stages.start("discover", f"sub-agent mapping {model_root}")
    agent_calls_path = run.dir / "agent_calls.jsonl"
    agent_totals = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}

    def record_agent_call(stage: str, role: str, model: str, usage: dict | None) -> str:
        """Append one row per query(); accumulate totals; return event suffix."""
        from .events import append_jsonl, make_agent_call_row, next_agent_call_seq

        usage = usage or {}
        row = make_agent_call_row(
            run_id=run.run_id,
            phase="before_loop",
            iteration=None,
            stage=stage,
            role=role,
            model=model,
            usage=usage,
            seq=next_agent_call_seq(agent_calls_path),
        )
        append_jsonl(agent_calls_path, row)
        for k in ("tokens_in", "tokens_out"):
            agent_totals[k] += usage.get(k) or 0
        agent_totals["cost_usd"] += usage.get("cost_usd") or 0.0
        if not usage:
            return " · usage n/a"
        return f" · tok {usage.get('tokens_in')}/{usage.get('tokens_out')} · ${usage.get('cost_usd') or 0:.4f}"

    # discover is a non-deterministic sub-agent: it intermittently returns a glob/list instead of a
    # concrete file ("...test_*.py is not a file") or exhausts its turn budget ("Reached maximum
    # number of turns"). Both kill before_loop and are fixed by a blind re-run — so RETRY here
    # (bounded) instead of dying. (#31)
    pcc_override = None
    pcc_abs = None
    if config.get("pcc_test"):
        from .model_files import resolve_pcc_node

        pcc_node_rel, pcc_thr, pcc_abs = resolve_pcc_node(model_root, config["pcc_test"], tt_root)
        pcc_override = {"path": pcc_node_rel, "threshold": pcc_thr}
        print(f"      --pcc-test gate -> {pcc_node_rel} (threshold {pcc_thr})", file=sys.stderr, flush=True)
    pathmap = None
    _last_exc = None
    for _attempt in range(3):
        try:
            pathmap = read_model_files(model_root, model_runner, pcc_override=pcc_override)
            break
        except Exception as exc:  # noqa: BLE001 — glob/max-turns/transient: retry a fresh discover
            _last_exc = exc
            print(
                f"      discover attempt {_attempt + 1}/3 failed ({str(exc)[:120]}); retrying",
                file=sys.stderr,
                flush=True,
            )
    if pathmap is None:
        raise _last_exc if _last_exc else RuntimeError("discover produced no pathmap")
    if pcc_abs is not None:
        from .perf_test_gen import generate_perf_test

        perf_node = generate_perf_test(model_root, "main", None, force=True, source_abs=pcc_abs, source_kind="pcc")
        if not perf_node:
            raise RuntimeError("could not auto-generate a perf test from --pcc-test (see messages above)")
        _pp, _, _pf = perf_node.partition("::")
        pathmap["perf_test"] = {"path": _pp, "case": _pf, "note": "auto-gen from --pcc-test"}
        pathmap["perf_tests"] = [pathmap["perf_test"]]
        pathmap["pipelines"] = [{"task": "main", "perf_test": perf_node, "pcc_test": pcc_override["path"]}]
        pathmap["is_multimodal"] = False
        print(f"      auto-gen perf from pcc -> {perf_node}", file=sys.stderr, flush=True)
    usage_suffix = record_agent_call(
        "discover",
        "discovery_sub_agent",
        getattr(model_runner, "model", "mock"),
        getattr(model_runner, "last_usage", None),
    )
    # perf test path: discovery returns model-root-relative; pytest runs from tt-metal root
    perf_rel = config.get("perf_test") or os.path.relpath(model_root / pathmap["perf_test"]["path"], tt_root)
    case = config.get("case") or pathmap["perf_test"]["case"]
    # SELF-HEAL the case: the discovery agent (or a stale config) can emit a case id that selects
    # NOTHING (e.g. 'device_params0-0' vs the real 'device_params0') -> preflight would hard-fail.
    # Validate against the test's ACTUAL collected ids and auto-correct to a collectable case (best
    # substring match, else the first). Keeps from-scratch discovery self-sufficient (no manual -k).
    if case:
        _ids, _raw = collect(tt_root, perf_rel, env=sub_env)
        if _ids and not any(case in _i for _i in _ids):
            from .probes import first_case_param

            _params = [p for p in (first_case_param(_i) for _i in _ids) if p]
            corrected = next((p for p in _params if p in case or case in p), None) or first_case_param(_ids[0])
            if corrected and corrected != case:
                msg = f"discovery case '{case}' selects 0 tests -> auto-correcting to '{corrected}' (of {len(_ids)} collected)"
                print(f"      ⚠ {msg}", file=sys.stderr, flush=True)
                stages._event("note", msg)
                case = corrected
    for w in pathmap.get("warnings", []):
        print(f"      ⚠ {w.get('code')}: {w.get('detail')}", file=sys.stderr, flush=True)
    stages.done(
        f"perf_test={perf_rel} -k {case} · pcc={list(pathmap['pcc'])} · "
        f"components={list(pathmap['components'])} · {len(pathmap['model_files'])} files" + usage_suffix
    )

    user_input = config.get("input")
    if user_input:
        ids, raw_tail = collect(tt_root, perf_rel, env=sub_env)
        from .probes import first_case_param, match_input_to_case

        params = [first_case_param(i) for i in ids]
        case = match_input_to_case(str(user_input), params)
        msg = f"input '{user_input}' -> matched case '{case}'"
        print(f"      {msg}", file=sys.stderr, flush=True)
        stages._event("note", msg)
    if not case:
        ids, raw_tail = collect(tt_root, perf_rel, env=sub_env)
        if not ids:
            raise RuntimeError(
                f"could not READ the test list for {perf_rel} — pytest may have "
                f"collected fine but the output was unparseable. pytest said:\n{raw_tail}"
            )
        from .probes import first_case_param

        case = first_case_param(ids[0])
        msg = (
            f"no case given — DEFAULTING to FIRST collected case "
            f"'{case or ids[0]}' of {len(ids)} available; pass -k to change"
        )
        print(f"      ⚠ {msg}", file=sys.stderr, flush=True)
        stages._event("note", msg)

    stages.start("lead_review", "lead agent reviewing discovery evidence")
    verdict = review(pathmap)
    usage_suffix = record_agent_call("lead_review", "lead", verdict.get("model", "?"), verdict.get("usage"))
    stages.done(f"{verdict['decision']}: {verdict['reasoning'][:90]}" + usage_suffix)

    stages.start("preflight", f"pytest --collect-only -k {case}")
    n_selected = preflight(tt_root, perf_rel, case, env=sub_env)
    stages.done(f"{n_selected} test(s) selected")

    stages.start("resolve_signposts", f"scan {model_root.name}/tests/ for tracy signposts")
    from .probes import resolve_signposts

    sp = resolve_signposts(model_root / "tests")
    config.setdefault("start_signpost", sp["start_signpost"])
    config.setdefault("end_signpost", sp["end_signpost"])
    if sp.get("warning"):
        pathmap.setdefault("warnings", []).append({"code": "signpost", "detail": sp["warning"]})
        print(f"      WARN signpost: {sp['warning']}", file=sys.stderr, flush=True)
    stages.done(f"start={config['start_signpost']!r} end={config['end_signpost']!r} found={sp['found']}")

    # Manifest BEFORE the long profile run: a failed tracy still leaves the
    # full discovery + review record for post-mortem.
    manifest = {
        "config": config,
        "env": env,
        "pathmap": pathmap,
        "discovery_review": verdict,
        "perf_test_resolved": {"path": perf_rel, "case": case},
        "playbook_sections": len(index),
    }
    run.manifest.write(manifest)

    stages.start("tracy_baseline", f"runs={config.get('runs', 1)} · tail -f {run.profiles_dir}/run0_tracy.log")

    def _run_baseline():
        return profile_model(
            perf_test=perf_rel,
            config=config,
            env=env,
            profiles_dir=run.profiles_dir,
            run_profiled=run_profiled_factory(perf_rel, case),
        )

    try:
        profile = _run_baseline()
    except PerfRunFailed as _exc:
        if not _SHAPE_CONFIG_CRASH_RE.search(_exc.error or ""):
            raise
        _cur = int(os.environ.get("TT_PERF_SEQ_LEN", "128") or "128")
        profile = None
        for _seq in _seq_retry_candidates(_exc.error, _cur):
            msg = (
                f"baseline crashed at TT_PERF_SEQ_LEN={_cur} with a shape/program-config assertion "
                f"(model program configs pinned to native shape); retrying at TT_PERF_SEQ_LEN={_seq}"
            )
            print(f"      ⚠ {msg}", file=sys.stderr, flush=True)
            stages._event("note", msg)
            os.environ["TT_PERF_SEQ_LEN"] = str(_seq)
            try:
                profile = _run_baseline()
                break
            except PerfRunFailed as _exc2:
                if not _SHAPE_CONFIG_CRASH_RE.search(_exc2.error or ""):
                    raise
                continue
        if profile is None:
            raise
    _seq_env = os.environ.get("TT_PERF_SEQ_LEN")
    if _seq_env:
        (run.dir / "perf_seq_len").write_text(_seq_env)
    # Persist the tagged buckets for the loop: ROUTE reads this, not the CSVs.
    (Path(run.profiles_dir) / "baseline_profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True))
    _bk = {b.get("id"): int(b.get("count", 0)) for b in (profile.get("buckets") or [])}
    _struct_ops = sum(c for i, c in _bk.items() if i in STRUCTURAL_OP_CLASSES)
    if profile.get("device_ms", 0) <= 0 or _struct_ops == 0:
        raise RuntimeError(
            f"baseline capture looks partial/degenerate (device_ms={profile.get('device_ms')}, "
            f"structural ops={_struct_ops}, buckets={_bk}); refusing to optimize against it. "
            f"Inspect {run.profiles_dir}/run0_tracy.log for a crash or profiler-marker overflow."
        )
    stages.done(
        f"device {profile['device_ms']:.3f} ms · wall {profile['wall_ms']:.0f} ms "
        f"· {len(profile['buckets'])} buckets"
    )

    metric_name = config.get("metric") or "device_ms"
    if metric_name == "auto":
        try:
            from .strategist import choose_axis, make_axis_runner

            metric_name = choose_axis(profile, make_axis_runner())
            print(f"      strategist chose axis -> metric={metric_name}", flush=True)
        except Exception as exc:
            metric_name = "device_ms"
            print(f"      strategist failed ({exc}); falling back to metric=device_ms", flush=True)
        config["metric"] = metric_name
    # device_ms = sum of profiled device-kernel time (the optimization target);
    # wall_ms = harness clock incl. compile/setup (reference only);
    # fps / tok_s still TBD(wall-metric-source).
    baseline = profile["device_ms"] if metric_name == "device_ms" else profile["wall_ms"]
    target = config.get("target")
    if target is None and metric_name == "device_ms":
        try:
            from . import roofline

            r = roofline.compute_rooflines(profile, env)
            gap = r.get("total_gap_ms")
            if gap is not None:
                # Achievable floor = measured - attainable gap. NOT Σideal: a per-op ideal can
                # OVERESTIMATE (e.g. an L1-resident op modeled at DRAM bandwidth when l1_bw_gbps
                # is unknown, or a dispatch floor summed as if ops ran serially), so Σideal can
                # exceed the measured total. Targeting Σideal then makes a measured<Σideal exit
                # falsely declare DONE while real per-op gaps (Σgap>0) remain on the table.
                target = round(max(0.0, baseline - gap), 4)
                print(
                    f"      roofline auto-target: achievable floor={target} ms "
                    f"= measured {round(baseline, 4)} - Σgap {round(gap, 4)} ms",
                    flush=True,
                )
        except Exception as exc:
            print(f"      roofline auto-target skipped: {exc}", flush=True)
    elif target is None and metric_name in ("wall_ms", "host_ms"):
        dev = profile.get("device_ms")
        if metric_name == "wall_ms" and dev:
            target = round(dev, 4)
            print(f"      host-axis target: wall floor = device_ms {target} ms (host fully overlapped)", flush=True)
    Checkpoint(run.state_path).save(
        {
            "run_id": run.run_id,
            "state": "BEFORE_LOOP_DONE",
            "iteration": 0,
            "metric": {
                "name": metric_name,
                "unit": METRIC_UNITS.get(metric_name, metric_name),
                "direction": config.get("direction", "min"),
                "baseline": baseline,
                "current": baseline,
                "target": target,
            },
            # uncapped by default (None) -> exit_policy stops only on roofline target-met or
            # full bucket exhaustion, never on an arbitrary iter/dollar wall. See --max-iter/--budget-usd.
            "max_iter": config.get("max_iter"),
            "budget_usd": config.get("budget_usd"),
            "cost_usd": round(agent_totals["cost_usd"], 6),
            "tokens_in": agent_totals["tokens_in"],
            "tokens_out": agent_totals["tokens_out"],
            "git_sha_clean": None,
            "candidates": [],
            "tried": [],
            "crash_retries": 0,
            "last_error": None,
        }
    )

    return {
        "run_id": run.run_id,
        "run_dir": str(run.dir),
        "env": env,
        "pathmap": pathmap,
        "profile": profile,
        "sections": len(index),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent.before_loop", description=__doc__)
    ap.add_argument("model_root", help="model directory — everything else is discovered")
    ap.add_argument("--metric", default="device_ms", choices=[*sorted(METRIC_UNITS), "auto"])
    ap.add_argument("--direction", default="min", choices=["min", "max"])
    ap.add_argument("--target", type=float)
    # No artificial wall by default: the loop stops at the roofline TARGET (target met -> DONE)
    # or true bucket exhaustion, NOT at an arbitrary iter/dollar cap. Pass these only to impose a
    # ceiling deliberately; unset (None) = uncapped (exit_policy treats None as "no limit").
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--notes", default="")
    # power-user overrides for the discovered values (optional)
    ap.add_argument(
        "--input",
        help="human input spec: sequence length like "
        "'128', or image size like '128x128'. Matched against the "
        "test's cases; NO match = hard stop (never runs the wrong case)",
    )
    ap.add_argument("--perf-test", help="override discovered perf test (tt-metal-root-relative)")
    ap.add_argument(
        "--pcc-test",
        dest="pcc_test",
        help="e2e PCC test node id 'path::fn' (tt-root-relative or absolute) to pin as the correctness "
        "gate; discovery still maps the model and the perf test is auto-generated from it",
    )
    ap.add_argument("-k", "--case", help="expert override: raw pytest -k case id")
    ap.add_argument(
        "--devices",
        default="single",
        help="single (default: TT_METAL_VISIBLE_DEVICES=0) | all | " "explicit ids like '0,1'",
    )
    ap.add_argument(
        "--box",
        help="declared TT box for roofline calibration (e.g. QB2, T3K, Galaxy) — reuses "
        "tt-hw-planner's hardware registry; sets worker_cores = mesh chips × per-chip grid",
    )
    ap.add_argument("--mesh", help="mesh shape for --box, e.g. '2x2' (default: the box's canonical mesh)")
    ap.add_argument("--mock-env", action="store_true")
    ap.add_argument("--mock-model-files", action="store_true")
    ap.add_argument(
        "--cc-discovery",
        action="store_true",
        dest="cc_discovery",
        help="map the model via the claude CLI (login, no SDK/model-tier) — used by the cc engine so "
        "discovery is claude-code like the rest of cc. Off => the FSM SDK sub-agent. Gates unchanged.",
    )
    ap.add_argument("--mock-tracy", action="store_true")
    args = ap.parse_args(argv)

    missing = check_dependencies()
    if missing:
        for m in missing:
            print(f"MISSING DEPENDENCY: {m}", file=sys.stderr)
        return 1

    config = {
        k: getattr(args, k)
        for k in (
            "model_root",
            "metric",
            "direction",
            "target",
            "max_iter",
            "budget_usd",
            "runs",
            "timeout",
            "notes",
            "perf_test",
            "pcc_test",
            "case",
            "devices",
            "input",
            "box",
        )
    }
    if args.mesh:
        try:
            config["mesh"] = tuple(int(x) for x in args.mesh.lower().replace(",", "x").split("x") if x)
        except Exception:
            ap.error(f"--mesh {args.mesh!r} must look like '2x2'")
    if args.input and args.case:
        ap.error("--input and -k are mutually exclusive (use one)")

    try:
        env_probe = mock_env_probe if args.mock_env else None
        if env_probe is None:
            from .probes import tt_smi_probe

            env_probe = tt_smi_probe

        model_runner = make_mock_model_runner(args.model_root) if args.mock_model_files else None
        if model_runner is None:
            if getattr(args, "cc_discovery", False):
                from .probes import cli_model_files_runner

                model_runner = cli_model_files_runner()
            else:
                from .probes import sdk_model_files_runner

                model_runner = sdk_model_files_runner()  # section 3.1 fail-fast

        if args.mock_model_files:
            review = mock_review  # gatherer mocked -> nothing real to review
        else:
            from .probes import lead_review_gate

            review = lead_review_gate

        if args.mock_tracy:
            factory = lambda perf, case: mock_run_profiled
            preflight = mock_preflight
            collect = mock_collect_cases
        else:
            from .probes import make_run_profiled, preflight_collect

            tt_root = os.environ.get("TT_METAL_HOME", str(PKG_ROOT.parents[2]))
            visible = "0" if args.devices == "single" else (None if args.devices == "all" else args.devices)
            xenv = {"TT_VISIBLE_DEVICES": visible, "TT_METAL_VISIBLE_DEVICES": visible} if visible is not None else {}
            factory = lambda perf, case: make_run_profiled(tt_root, perf, case, timeout_s=args.timeout, extra_env=xenv)
            preflight = preflight_collect
            from .probes import collect_cases as collect

        result = before_loop(config, env_probe, model_runner, factory, preflight, review, collect)
    except Exception as exc:
        print(f"BEFORE-LOOP FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    p = result["profile"]
    print(
        f"\nbaseline device time: {p['device_ms']:.3f} ms   "
        f"(wall incl. compile/setup: {p['wall_ms']:.0f} ms, median of {config['runs']})\n"
    )
    print(stack_report(p["buckets"]))
    print(f"\nartifacts: {result['run_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

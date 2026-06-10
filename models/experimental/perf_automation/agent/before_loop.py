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
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .checkpoint import Checkpoint
from .environment import environment_check
from .model_files import read_model_files
from .router import build_index, cache_playbook
from .run import Run
from .tracy_tool import stack_report, tracy_tool

PKG_ROOT = Path(__file__).parent.parent
DEFAULT_PLAYBOOK = PKG_ROOT / "GUIDELINES"
DEFAULT_RUNS_ROOT = PKG_ROOT / "runs"
DEFAULT_CACHE = PKG_ROOT / ".cache" / "playbook_index.json"
FIXTURES = PKG_ROOT / "tests" / "fixtures"

METRIC_UNITS = {"device_ms": "ms", "wall_ms": "ms", "fps": "fps", "throughput_tok_s": "tok/s"}
N_STAGES = 6


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
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": time.time(), "stage": self._n, "name": self._name, "event": kind, "detail": detail}
        if dt is not None:
            row["seconds"] = round(dt, 2)
        with open(self.events_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


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

    stages.start("cache_playbook", str(playbook_dir))
    cache_playbook(playbook_dir, cache_path)
    index = build_index(playbook_dir)
    stages.done(f"{len(index)} sections indexed")

    stages.start("discover", f"sub-agent mapping {model_root}")
    agent_calls_path = run.dir / "agent_calls.jsonl"
    agent_totals = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}

    def record_agent_call(stage: str, role: str, model: str, usage: dict | None) -> str:
        """Append one row per query(); accumulate totals; return event suffix."""
        usage = usage or {}
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, "role": role, "model": model, **usage}
        with open(agent_calls_path, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        for k in ("tokens_in", "tokens_out"):
            agent_totals[k] += usage.get(k) or 0
        agent_totals["cost_usd"] += usage.get("cost_usd") or 0.0
        if not usage:
            return " · usage n/a"
        return f" · tok {usage.get('tokens_in')}/{usage.get('tokens_out')} · ${usage.get('cost_usd') or 0:.4f}"

    pathmap = read_model_files(model_root, model_runner)
    usage_suffix = record_agent_call(
        "discover",
        "discovery_sub_agent",
        getattr(model_runner, "model", "mock"),
        getattr(model_runner, "last_usage", None),
    )
    # perf test path: discovery returns model-root-relative; pytest runs from tt-metal root
    perf_rel = config.get("perf_test") or os.path.relpath(model_root / pathmap["perf_test"]["path"], tt_root)
    case = config.get("case") or pathmap["perf_test"]["case"]
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
    profile = tracy_tool(
        pcc_path=perf_rel,
        batch_size=config.get("batch_size", 1),
        seq_len=config.get("seq_len", 0),
        runs=config.get("runs", 1),
        profiles_dir=run.profiles_dir,
        start_signpost=config.get("start_signpost", "start"),
        end_signpost=config.get("end_signpost", "stop"),
        arch=env["arch"],
        available_cores=env["worker_cores"],
        run_profiled=run_profiled_factory(perf_rel, case),
    )
    # Persist the tagged buckets for the loop: ROUTE reads this, not the CSVs.
    (Path(run.profiles_dir) / "baseline_profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True))
    stages.done(
        f"device {profile['device_ms']:.3f} ms · wall {profile['wall_ms']:.0f} ms "
        f"· {len(profile['buckets'])} buckets"
    )

    metric_name = config.get("metric", "device_ms")
    # device_ms = sum of profiled device-kernel time (the optimization target);
    # wall_ms = harness clock incl. compile/setup (reference only);
    # fps / tok_s still TBD(wall-metric-source).
    baseline = profile["device_ms"] if metric_name == "device_ms" else profile["wall_ms"]
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
                "target": config.get("target"),
            },
            "max_iter": config.get("max_iter", 25),
            "budget_usd": config.get("budget_usd", 5.0),
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
    ap.add_argument("--metric", default="device_ms", choices=sorted(METRIC_UNITS))
    ap.add_argument("--direction", default="min", choices=["min", "max"])
    ap.add_argument("--target", type=float)
    ap.add_argument("--max-iter", type=int, default=25)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--notes", default="")
    # power-user overrides for the discovered values (optional)
    ap.add_argument(
        "--input",
        help="human input spec: sequence length like "
        "'128', or image size like '128x128'. Matched against the "
        "test's cases; NO match = hard stop (never runs the wrong case)",
    )
    ap.add_argument("--perf-test", help="override discovered perf test (tt-metal-root-relative)")
    ap.add_argument("-k", "--case", help="expert override: raw pytest -k case id")
    ap.add_argument(
        "--devices",
        default="single",
        help="single (default: TT_METAL_VISIBLE_DEVICES=0) | all | " "explicit ids like '0,1'",
    )
    ap.add_argument("--mock-env", action="store_true")
    ap.add_argument("--mock-model-files", action="store_true")
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
            "case",
            "devices",
            "input",
        )
    }
    if args.input and args.case:
        ap.error("--input and -k are mutually exclusive (use one)")

    try:
        env_probe = mock_env_probe if args.mock_env else None
        if env_probe is None:
            from .probes import tt_smi_probe

            env_probe = tt_smi_probe

        model_runner = make_mock_model_runner(args.model_root) if args.mock_model_files else None
        if model_runner is None:
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

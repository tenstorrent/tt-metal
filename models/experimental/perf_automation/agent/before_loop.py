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
import errno
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

# ONE state directory for every durable temp artifact -- see cc_optimize/tmpstate.py.
# agent/state_dir.py loads cc_optimize/tmpstate.py by path, once, for the four modules that need it.
from .state_dir import state_dir


PKG_ROOT = Path(__file__).parent.parent
DEFAULT_PLAYBOOK = PKG_ROOT / "GUIDELINES"
DEFAULT_RUNS_ROOT = PKG_ROOT / "runs"
DEFAULT_CACHE = PKG_ROOT / ".cache" / "playbook_index.json"
FIXTURES = PKG_ROOT / "tests" / "fixtures"

METRIC_UNITS = {"device_ms": "ms", "wall_ms": "ms", "fps": "fps", "throughput_tok_s": "tok/s"}
N_STAGES = 10


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
    # DERIVED FROM THIS MODEL'S OWN SEQUENCE, NOT A LADDER BORROWED FROM ANOTHER.
    #
    # This was `for s in (256, 384, 512, 768)`, four numbers that came from whichever model was in
    # hand when it was written. On a model whose sequence is 1500 every one of them is below
    # current_seq and the loop contributes nothing; on a model at 64 it jumps straight to 4x. The
    # scaling branch above is already model-derived -- it reads block_h/num_cores_r/Mt out of the
    # error and computes what the shard actually wants -- and this fallback exists only for when the
    # error did not carry those numbers.
    #
    # So grow from what the model is actually running: the next tile boundary, then 1.5x, 2x, 3x,
    # each tile-aligned because a sequence that is not a multiple of the tile height cannot shard
    # cleanly and would only produce the same class of failure again. TILE is a hardware constant
    # (agent/tp.py), not a model one.
    from .tp import TILE

    if current_seq > 0:
        _next_tile = ((current_seq // TILE) + 1) * TILE
        for s in (_next_tile, *(int(round(current_seq * f / TILE)) * TILE for f in (1.5, 2, 3))):
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
        print(f"  Step {self._n}/{N_STAGES}  {detail or name}", file=sys.stderr, flush=True)
        self._event("start", detail)

    def done(self, detail: str = "") -> None:
        dt = time.monotonic() - self._t0
        print(f"      ✔ {detail} ({dt:.1f}s)" if detail else f"      ✔ done ({dt:.1f}s)", file=sys.stderr, flush=True)
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


def _record_baseline_anchor(profile: dict, model: str = "", task: str = "") -> None:
    """Record the baseline profile as the ledger's eager anchor, at the point it is produced.

    The baseline IS the "before" by definition, so the code that computes it has to be a ledger
    writer -- otherwise the anchor depends on an optional downstream MCP call (see the call site).
    Mirrors perf_mcp._ledger_record's phase rule so a rerun APPENDS an 'after' and can never
    overwrite the original 'before'. Best-effort: a run must still produce its baseline if the
    ledger is unavailable.
    """
    try:
        import importlib.util as _ilu

        _spec = _ilu.spec_from_file_location(
            "_cc_measurements", str(Path(__file__).resolve().parent.parent / "cc_optimize" / "measurements.py")
        )
        led = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(led)
        # The SAME predicate perf_mcp guards its anchor with -- loaded by path for the same reason
        # the ledger is (the package pulls in device deps that must not load at import time).
        _pm_spec = _ilu.spec_from_file_location(
            "_cc_perf_mcp_credible", str(Path(__file__).resolve().parent.parent / "cc_optimize" / "perf_mcp.py")
        )
        _pm = _ilu.module_from_spec(_pm_spec)
        _pm_spec.loader.exec_module(_pm)
        _is_credible_profile = _pm._is_credible_profile
        ms = (profile or {}).get("device_ms")
        # Stamp the depth this was profiled at; an unstamped number is how a 2-layer reading once
        # anchored a 16-layer run.
        depth = str((profile or {}).get("perf_layers") or "all")
        model = model or os.environ.get("PERF_MCP_MODEL_NAME") or ""
        task = task or os.environ.get("PERF_MCP_TASK", "main")
        seen = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model=model, task=task)
        phase = led.PHASE_AFTER if seen else led.PHASE_BEFORE
        # NEVER PIN A CAPTURE THE RUN IS ABOUT TO REJECT. This anchor is written ~12 lines before the
        # degeneracy guard below (`device_ms <= 0 or _struct_ops == 0` -> "refusing to optimize
        # against it"), and anchors are WRITE-ONCE -- so a partial capture claims the BEFORE slot and
        # the good measurement from the retry lands as an AFTER, unable to displace it. gemma-3-12b-it
        # on 2026-07-31 pinned device_ms=0.1004 (0 structural ops) this way, and every eager
        # "gain vs baseline" for that run was then computed against 0.1 ms.
        #
        # perf_mcp._ledger_record has always had this guard; _record_baseline_anchor was written by
        # copying that function and dropping it. Reuse the SAME predicate rather than restating it,
        # so the two cannot drift. AFTER readings are unaffected: they are not anchors, and refusing
        # them would silently discard real measurements.
        if phase == led.PHASE_BEFORE and not _is_credible_profile(profile):
            return
        led.record(led.KIND_EAGER, phase, ms, depth=depth, mode="eager", source="before_loop", model=model, task=task)
        _tr = led.trace_ms_from_profile(profile)
        if _tr:
            led.record(
                led.KIND_TRACE_PASS,
                phase,
                _tr,
                depth=depth,
                mode="tracy-trace",
                source="before_loop",
                model=model,
                task=task,
            )
    except Exception:  # noqa: BLE001
        pass


def check_dependencies() -> list[str]:
    """Verify the two hard tool dependencies BEFORE any stage runs.

    tt-perf-report is needed by every profile (stage-2 REFINE runs it even in
    mock-tracy mode); claude-agent-sdk is needed by the discovery sub-agent and
    the lead review gate. Returns actionable messages for anything missing."""
    import shutil as _shutil

    from .pkgtools import installer_hint

    hint = f"{installer_hint()} -r models/experimental/perf_automation/requirements-agent.txt"
    missing: list[str] = []
    if _shutil.which("tt-perf-report") is None:
        missing.append(f"tt-perf-report not on PATH — install the agent deps into your tt-metal venv: {hint}")
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


def _enforce_pcc_gate_policy(pcc_test, model_root, config) -> None:
    """Decide where the correctness gate comes from, and STOP when there is no ground truth.

    The gate is the only thing preventing optimize from committing a perf lever that silently
    degrades the model, so "no usable gate" has to be a stop condition. It was not: a supplied file
    that declares no threshold got the 0.99 DEFAULT, which satisfied the fatal `no_pcc_threshold`
    check, and a gate that never prints a PCC then read as a pass on every edit.
    """
    from .pcc_gate_policy import GENERATE_FROM_HF, STOP, decide

    model_id = config.get("model_id") or os.environ.get("HF_MODEL") or ""
    d = decide(pcc_test, model_id or None)
    act = d["action"]
    if act == STOP:
        raise SystemExit("CANNOT CONTINUE — no usable correctness gate.\n      " + d["reason"])
    if d.get("warning"):
        print("      correctness gate WARNING: " + d["warning"], file=sys.stderr, flush=True)
    if act == GENERATE_FROM_HF:
        print(
            "      correctness gate: no gate supplied -> generating an HF-referenced PCC gate\n"
            "        reason: " + d["reason"],
            file=sys.stderr,
            flush=True,
        )
        node = _generate_pcc_gate(model_root, model_id, config)
        if not node:
            raise SystemExit(
                "CANNOT CONTINUE — could not generate a usable PCC gate for %r.\n      "
                "PLEASE GIVE A PCC TEST TO RUN OPTIMIZE: pass --pcc-test <file>::<test>. It must "
                "compare against a reference, print `PCC: <float>`, and declare its own threshold." % (model_id,)
            )
        config["pcc_test"] = node
        print("      generated gate -> %s" % node, file=sys.stderr, flush=True)


def _generate_pcc_gate(model_root, model_id, config):
    """Author + validate an HF-referenced gate. Returns the node id, or None.

    A generated gate is only accepted if it passes on the unedited model AND fails on a perturbed
    one -- "it ran and passed" proves nothing, since a gate that always passes also passes.
    """
    try:
        from .gitio import repo_root
        from .pcc_gate_gen import generate_pcc_gate
        from .perf_test_gen import _claude

        return generate_pcc_gate(
            model_dir=model_root,
            model_id=model_id,
            repo_root=repo_root(model_root),
            runner=_claude,
            threshold=float(os.environ.get("PERF_MCP_PCC_MIN", "0.95") or "0.95"),
            perturb_env=_pcc_perturb_env(),
        )
    except Exception as exc:  # noqa: BLE001
        print("      PCC-gate generation failed: %s" % (str(exc)[-300:],), file=sys.stderr, flush=True)
        return None


def _pcc_perturb_env() -> dict:
    """Env that deliberately DEGRADES the model, so the gate can be proven able to fail.

    Model-agnostic: a low-precision weight dtype damages the numbers on any TTNN model without
    changing shapes or the call path, which is exactly the kind of damage a perf lever can do.
    """
    return {"TT_PERF_FORCE_WEIGHT_DTYPE": "bfloat4_b", "PERF_PCC_SELFTEST_PERTURB": "1"}


def _measure_baseline(run_baseline, stages):
    """Take the baseline profile, retrying at a smaller seq len on a shape/program-config assertion.

    Split out of before_loop so the reuse branch above can skip the whole measurement without
    duplicating the retry ladder or early-returning out of before_loop (which continues on to
    metric selection and target computation).
    """
    try:
        return run_baseline()
    except PerfRunFailed as _exc:
        if not _SHAPE_CONFIG_CRASH_RE.search(_exc.error or ""):
            raise
        _cur = int(os.environ.get("TT_PERF_SEQ_LEN", "128") or "128")
        for _seq in _seq_retry_candidates(_exc.error, _cur):
            msg = (
                f"baseline crashed at TT_PERF_SEQ_LEN={_cur} with a shape/program-config assertion "
                f"(model program configs pinned to native shape); retrying at TT_PERF_SEQ_LEN={_seq}"
            )
            print(f"      ⚠ {msg}", file=sys.stderr, flush=True)
            stages._event("note", msg)
            os.environ["TT_PERF_SEQ_LEN"] = str(_seq)
            try:
                return run_baseline()
            except PerfRunFailed as _exc2:
                if not _SHAPE_CONFIG_CRASH_RE.search(_exc2.error or ""):
                    raise
                continue
        raise


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

    run = Run.create(runs_root, config=None, label=model_root.name)
    # THE CONSOLE OUTPUT IS AN ARTIFACT TOO. Everything else this run produces is filed in run.dir,
    # which is named for the model and the run; the log that says WHY a run stopped was the one thing
    # left to a shell redirect, and voxtral run 41's reason was lost with it. Installed here because
    # this is the first line that knows where the run directory is, and the banner below is the first
    # thing worth keeping.
    from .console_log import install as _install_console_log

    _install_console_log(run.dir)
    stages = _Stages(run.dir / "events.jsonl")
    print(f"run: {run.run_id}  ->  {run.dir}", file=sys.stderr, flush=True)
    _sep = "=" * 78
    print(f"\n{_sep}\n  Setup & discovery — {model_root.name}\n{_sep}", file=sys.stderr, flush=True)

    stages.start("environment_check", "Checking the Tenstorrent device")
    env = environment_check(env_probe)
    physical_chips = int(env.get("device_count") or 0)
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
    stages.done(f"{env['card']} ({env['arch']}), {env['worker_cores']} cores")
    from .probes import note_board

    chips = max(physical_chips, int(env.get("mesh_chips") or env.get("device_count") or 0))
    note_board(str(env.get("card") or ""), chips, box=str(box or ""))

    devices = str(config.get("devices") or "single")
    # DEVICE VISIBILITY IS INTENTIONALLY NEVER RESTRICTED (chip-count / hardware agnostic): pinning
    # TT_VISIBLE_DEVICES to a chip SUBSET makes fabric auto-discovery classify the board as a CUSTOM
    # cluster and fatally demand a mesh-graph descriptor we don't provide — crashing device/fabric
    # init before any forward on any multi-chip board. Leave the full topology visible; the mesh
    # SHAPE (TT_PERF_MESH_ROWS/COLS) is the chip-count lever, not OS-level visibility.
    visible = None
    sub_env = dict(os.environ)
    sub_env.pop("TT_VISIBLE_DEVICES", None)
    sub_env.pop("TT_METAL_VISIBLE_DEVICES", None)
    config["visible_devices"] = None
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
        stages.start("startup_reset", "Resetting the model demo to a clean state")
        try:
            from . import gitio

            repo = gitio.repo_root(model_root)
            head = gitio.head_sha(repo)
            dirty = gitio.changed_files(repo, head, pathspec=str(model_root))
            _generated = {"RUN_REPORT.md", ".module_optimize_state.json"}
            code_dirty = [d for d in dirty if os.path.basename(d) not in _generated]
            if code_dirty:
                gitio.checkout(repo, head, pathspec=code_dirty)
                stages.done(f"restored {len(code_dirty)} leftover-dirty file(s) to {head[:9]} (prior interrupted run?)")
            else:
                stages.done(f"clean ({head[:9]})")
        except Exception as exc:  # never block the run on the restore
            stages.done(f"skipped: {exc}")

    stages.start("cache_playbook", "Loading the optimization playbook")
    cache_playbook(playbook_dir, cache_path)
    index = build_index(playbook_dir)
    stages.done(f"{len(index)} sections indexed")

    # Agent-SDK health: the claude CLI auto-updates and can drift out of sync with the pinned
    # python claude-agent-sdk, after which EVERY agent call fails ("error result: success").
    # Detect that here (a trivial call in a clean subprocess) and, by default, auto-upgrade +
    stages.start("ensure_tt_lang", "Verifying the kernel toolchain (tt-lang)")
    try:
        from .ttlang import ensure_ttl

        _ttl = ensure_ttl(cache_dir=str(Path(__file__).resolve().parent.parent))
    except Exception as exc:
        _ttl = {"available": False, "error": str(exc)}
    if _ttl.get("available"):
        stages.done(f"tt-lang available ({_ttl.get('version')})")
    else:
        stages.done(
            f"tt-lang unavailable ({_ttl.get('tried') or _ttl.get('error')}) — "
            "the run HALTS later only if a material op actually reaches the tt-lang rung"
        )

    # THE MODEL'S SHAPE, BEFORE ANYTHING IS BUILT OR RUN. Read-only, sub-second, no device: a model
    # that cannot be measured the way this tool measures should be told so here rather than forty
    # minutes later as a crash with no obvious connection to its cause. gemma-3's prefill decides its
    # own traced-vs-eager from an allow-list inside the model, so a profiled run traced anyway and
    # died with 194 x "Event Synchronization is not supported during trace capture" -- after the
    # weights had loaded and the board had been busy for minutes. That clause is visible in the
    # source.
    #
    # WARN, NOT REFUSE, for now. gemma-3 is the only model exercised end to end and it fails two
    # clauses today; gating hard would block the work that proves the clauses are right. Set
    # PERF_MCP_REQUIRE_CONTRACT=1 to make an unmet blocking clause stop the run, which is where this
    # should land once the compliance cost is known.
    stages.start("model_contract", "Checking the model against the optimize contract")
    try:
        from .model_contract import check as _contract_check, report as _contract_report

        _cf = _contract_check(model_root)
        _blocking = [f for f in _cf if f.blocking]
        print(_contract_report(_cf, model_root), file=sys.stderr, flush=True)

        # REPAIR WHAT THE MODEL GETS WRONG ABOUT THE HARNESS -- opt-in, and only the compatibility
        # clauses. A blocking clause means this run WILL fail: gemma-3's trace gate ignored the
        # harness, the profiled baseline traced anyway, and 194 fatals later there was no data and
        # no baseline, after the weights had loaded. The edit that fixes it is the same edit every
        # time, which is why it can be automated at all.
        #
        # OPT-IN, because this writes to source the tool did not author. A run that silently edits a
        # model leaves the next reader a change nobody made, in a file they own, with no record of
        # why -- worse than the bug. PERF_MCP_REPAIR_MODEL=1 is someone deciding.
        #
        # The PORTING clauses are never touched: generating PIPELINE_STAGES, the per-stage hooks and
        # the self-tests needs the model's stage decomposition and reference outputs, which is
        # emit-e2e's job.
        if _blocking:
            from .model_repair import apply as _repair_apply, plan as _repair_plan, report as _repair_report

            _edits = _repair_plan(model_root)
            print(_repair_report(_edits, model_root), file=sys.stderr, flush=True)
            if _edits and os.environ.get("PERF_MCP_REPAIR_MODEL") == "1":
                _res = _repair_apply(model_root, _edits)
                # RE-CHECKED, not assumed. A repair that does not clear its clause is a failed
                # repair, and the run must hear that now rather than discover it on the device.
                print(
                    "  [repair] wrote %d file(s); blocking clauses now: %s"
                    % (len(_res["written"]), [f.clause for f in _res["remaining"]] or "none"),
                    file=sys.stderr,
                    flush=True,
                )
                _cf = _contract_check(model_root)
                _blocking = [f for f in _cf if f.blocking]
            elif _edits:
                print(
                    "  [repair] not applied — set PERF_MCP_REPAIR_MODEL=1 to write these edits, or "
                    "make them by hand. This run will fail on the blocking clause(s) above.",
                    file=sys.stderr,
                    flush=True,
                )
        if _blocking and os.environ.get("PERF_MCP_REQUIRE_CONTRACT") == "1":
            raise SystemExit(
                "  [contract] %d blocking clause(s) unmet and PERF_MCP_REQUIRE_CONTRACT=1 — "
                "refusing to optimize a model that cannot be measured as specified." % len(_blocking)
            )
        stages.done(
            "meets all clauses" if not _cf else "%d unmet (%d blocking) — see above" % (len(_cf), len(_blocking))
        )
    except SystemExit:
        raise
    except Exception as _ce:  # noqa: BLE001 -- a contract check must never take the run down
        stages.done("skipped (%s)" % str(_ce)[:120])

    stages.start("discover", "Mapping the model's pipelines & building perf tests")
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
            return ""
        return f"  [{usage.get('tokens_in')}/{usage.get('tokens_out')} tok, ${usage.get('cost_usd') or 0:.4f}]"

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
        # Pass the RESOLVED absolute path: the raw argument is relative to the invocation directory,
        # but discovery runs inside an isolated worktree, so checking the raw string reports
        # "gate not found" for a gate that resolved perfectly well one line above.
        _enforce_pcc_gate_policy(pcc_abs or config["pcc_test"], model_root, config)
    else:
        _enforce_pcc_gate_policy(None, model_root, config)
    pathmap = None
    _last_exc = None
    for _attempt in range(3):
        try:
            pathmap = read_model_files(model_root, model_runner, pcc_override=pcc_override)
            break
        except Exception as exc:  # noqa: BLE001 — glob/max-turns/transient: retry a fresh discover
            _last_exc = exc
            if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
                print("      OUT OF DISK — free space and rerun", file=sys.stderr, flush=True)
                raise
            print(
                f"      discover attempt {_attempt + 1}/3 failed ({str(exc)[:120]}); retrying",
                file=sys.stderr,
                flush=True,
            )
    if pathmap is None:
        raise _last_exc if _last_exc else RuntimeError("discover produced no pathmap")
    if pcc_abs is not None:
        from .perf_test_gen import generate_perf_test

        _task = "main"
        if os.environ.get("TT_PERF_MODULE_LEVEL", "") not in ("", "0", "false", "False"):
            _stem = Path(str(config["pcc_test"]).partition("::")[0]).stem
            _task = (_stem[5:] if _stem.startswith("test_") else _stem) or "main"
        # WALK BEFORE WRITING. generate_perf_test has always accepted `stacks` and has always had a
        # multi-stack branch behind it -- one depth variable per stack instead of a single
        # TT_PERF_LAYERS -- and no production caller ever passed it, so that branch had never run.
        # Every perf test this tool generated was written as if the model had one stack.
        #
        # Measured on Voxtral 2026-08-13: the test read only TT_PERF_LAYERS, the bridge later set
        # TT_PERF_STACK0/1_LAYERS that nothing read, so ONE depth went to every stack and had to be
        # max(2, 32, 3) = 32 -- the encoder's full depth. Capping to full depth changes no work, and
        # the run concluded the knob never reached the builder.
        #
        # The PCC gate makes this answerable here: it is supplied by the operator, not generated, so
        # it exists before anything is written and it builds the model. An empty answer costs nothing
        # -- generation is then exactly as blind as it was before.
        _survey = []
        try:
            from .stack_survey import (
                describe as _survey_describe,
                survey as _survey_stacks,
                survey_model as _survey_build,
            )

            # BUILD THE MODEL, do not borrow a test's. Running a test and waiting for it to call
            # build_pipeline works only for tests that build it that way -- the correctness gate does
            # not, so the hook never fired and a two-stack model reported zero. The contract
            # guarantees the factory; calling it directly is both correct and seconds rather than
            # minutes.
            _survey = _survey_build(
                tt_root,
                model_root,
                env=sub_env,
                model_id=str(config.get("model_name") or config.get("config_ref") or ""),
            )
            if not _survey:
                # Fall back to walking the PCC gate: a model the factory cannot build standalone is
                # still worth one attempt through the test that is known to build it.
                # ABSOLUTE, not pcc_node_rel: that is MODEL-root relative and the probe runs from the
                # REPO root, so the relative form is a path pytest cannot find.
                _pcc_case = str(config["pcc_test"]).partition("::")[2] or pcc_node_rel.partition("::")[2]
                _survey_node = "%s::%s" % (pcc_abs, _pcc_case) if _pcc_case else str(pcc_abs)
                _survey = _survey_stacks(tt_root, _survey_node, env=sub_env)
            print("      stack survey (pre-generation): %s" % _survey_describe(_survey), file=sys.stderr, flush=True)
        except Exception as _sv_e:  # noqa: BLE001 -- never block generation on the survey
            print("      stack survey skipped: %s" % str(_sv_e)[:120], file=sys.stderr, flush=True)
        perf_node = generate_perf_test(
            model_root, _task, None, force=True, source_abs=pcc_abs, source_kind="pcc", stacks=_survey or None
        )
        if not perf_node:
            raise RuntimeError("could not auto-generate a perf test from --pcc-test (see messages above)")
        # SAY WHAT THE TEST DECIDED FOR ITSELF. The tool owns ISL/OSL/batch/depth and sends them; a
        # generated test that defines its OWN capped count is a measurement condition nobody asked
        # for, and nothing read it back. Voxtral's test defined TT_PERF_AUDIO_STREAMS=2 while the
        # pipeline was built for its declared batch of 8, so prefill measured a quarter of the real
        # workload and was printed against a full-batch roofline. Reported rather than refused: the
        # same test also defines TT_PERF_FLUSH_EVERY=32, which changes no workload, and no static rule
        # tells the two apart.
        try:
            from .perf_test_gen import invented_workload_vars as _invented

            _pp_abs = model_root / str(perf_node).partition("::")[0]
            _inv = _invented(_pp_abs.read_text(errors="ignore"), stages=_survey and [] or [])
            if _inv:
                print(
                    "      note - the generated perf test defines its own workload knob(s): %s "
                    "(the tool does not set these; a capped default measures less than the model's "
                    "declared batch)" % ", ".join("%s=%d" % (v, d) for v, d in _inv),
                    file=sys.stderr,
                    flush=True,
                )
        except Exception:  # noqa: BLE001 -- never block generation on a report line
            pass
        _pp, _, _pf = perf_node.partition("::")
        pathmap["perf_test"] = {"path": _pp, "case": _pf, "note": "auto-gen from --pcc-test"}
        pathmap["perf_tests"] = [pathmap["perf_test"]]
        pathmap["pipelines"] = [{"task": _task, "perf_test": perf_node, "pcc_test": pcc_override["path"]}]
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
    _warnings = pathmap.get("warnings", [])
    _verbose = bool(os.environ.get("TT_HW_PLANNER_VERBOSE"))
    if _warnings and _verbose:
        for w in _warnings:
            print(f"      note - {w.get('code')}: {w.get('detail')}", file=sys.stderr, flush=True)
    _gates = " and ".join(pathmap["pcc"]) or "none"
    _caveats = ""
    if _warnings and not _verbose:
        _caveats = f", plus {len(_warnings)} caveats (run with TT_HW_PLANNER_VERBOSE to read them)"
    stages.done(
        f"built {Path(perf_rel).name} for '{case}', covering {len(pathmap['components'])} "
        f"components across {len(pathmap['model_files'])} files; correctness gate(s): {_gates}{_caveats}" + usage_suffix
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

    stages.start("lead_review", "Reviewing the discovery plan")
    verdict = review(pathmap)
    usage_suffix = record_agent_call("lead_review", "lead", verdict.get("model", "?"), verdict.get("usage"))
    stages.done(f"{verdict['decision']}: {verdict['reasoning'][:90]}" + usage_suffix)

    stages.start("preflight", "Final checks before optimizing")
    n_selected = preflight(tt_root, perf_rel, case, env=sub_env)
    stages.done(f"{n_selected} test(s) selected")

    # STAGE MARKS GO IN HERE, at the one point every run passes. Injection at generation could not
    # reach this model: generate_perf_test is not called for the main pipeline on a run that
    # regenerates nothing, so run 24's test was the one written the previous evening -- unmarked, for
    # the eighth time. THIS is where the run decides which file it will profile, whatever produced
    # it, so it is the only placement that does not depend on how the file came to exist.
    #
    # BEFORE resolve_signposts, deliberately: that scan reads the test for signpost names, so the
    # start/stop pair the injected block emits is found and configured rather than defaulted.
    #
    # The injector is idempotent and refuses when it cannot place the block, so running it on every
    # run costs nothing and can never produce two marked passes.
    stages.start("stage_marks", "Marking stage boundaries for the profiler")
    try:
        from .stage_marks import inject_stage_marks as _inject_marks

        _pt = Path(tt_root) / perf_rel
        _cur = _pt.read_text()
        _new, _why = _inject_marks(_cur)
        if _new != _cur:
            _pt.write_text(_new)
        stages.done(_why)
    except Exception as _mi:  # noqa: BLE001
        stages.done("not injected (%s: %s)" % (type(_mi).__name__, str(_mi)[:90]))

    stages.start("resolve_signposts", "Locating profiler signposts")
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

    # Size the trace region from the board's DRAM (the value discovery already probed into env) BEFORE
    # the tracy baseline. Without this the profiled perf test opens at its own hardcoded 23 MB default,
    # the whole-pipeline trace capture (~29 MB) overflows (mesh_trace.cpp:80), the trace is skipped, and
    # tracy post-processing then floods ~180k "device data missing" warnings for the un-dumped trace ops
    # -- minutes of grinding that the no-output watchdog kills as a false wedge. run_profiled inherits
    # os.environ, so setting it here reaches the profiled subprocess. DRAM-derived, not hardcoded.
    try:
        from .environment import default_trace_region_bytes

        _tr_start, _ = default_trace_region_bytes(env.get("dram_capacity_bytes"))
        if _tr_start > int(os.environ.get("TT_PERF_TRACE_REGION") or 0):
            os.environ["TT_PERF_TRACE_REGION"] = str(_tr_start)
    except Exception:  # noqa: BLE001
        pass

    try:
        from models.experimental.perf_automation.cc_optimize.run import (
            _bridge_depth_env,
            _coverage_layers,
            _llm_depth_env,
            _model_root_from_node,
        )

        _bl_mr = _model_root_from_node(tt_root, perf_rel)
        _bl_knob = _llm_depth_env(_bl_mr, 2) if _bl_mr is not None else {}
        _bl_cov, _bl_facts = _coverage_layers(
            tt_root,
            sub_env,
            devices,
            perf_rel,
            case,
            model_name=str(config.get("model_name") or ""),
            config_ref=str(config.get("config_ref") or ""),
            depth_knob=_bl_knob,
        )
        # NO WINDOW MEANS NO WINDOW. _coverage_layers returns None for a REASON, and the commonest
        # reason is not failure: the signpost path returns None precisely when it has proved the depth
        # knob INERT -- the cap left the work signal unchanged, so the model builds every layer
        # whatever is asked of it (run.py: "Profiling FULL depth"). gemma3's build_pipeline takes no
        # layer count at all, so there is nothing for TT_PERF_LAYERS to attach to.
        #
        # This read that None as "the probe failed" and substituted a literal 4 -- a number nothing
        # derived, contradicting the ladder's own fallback of 2 (run.py: _cov = 2,
        # "unverified-floor") -- then EXPORTED it as TT_PERF_LAYERS and announced "profiled at a
        # SUBSTITUTED depth of 4 layers" on a run that profiled all 48. The claim was wrong, the
        # export was wrong, and only the bridge's empirical check downstream ("did not reduce work;
        # ignoring") kept it from mattering. A value that survives solely because something later
        # discards it should not be produced.
        #
        # The depth ladder OWNS this question -- signposts, then 2/4/8/16 bounded by the declared
        # depth, then 2 -- and it has already answered. When its answer is "no cap", TT_PERF_LAYERS is
        # REMOVED, which is how the rest of the tool spells full depth (layer_depth.set_depth: the cap
        # is expressed by ABSENCE, never by a sentinel, because "0" arrives as a truthy string and is
        # read as "build zero layers").
        _bl_full = int((_bl_facts or {}).get("full_signal") or 0)
        _bl_blocks = int((_bl_facts or {}).get("full_blocks") or 0)
        if _bl_cov is None or _bl_cov == 0:
            os.environ.pop("TT_PERF_LAYERS", None)
            # READ THE REASON, do not infer one. _coverage_layers now says which of these it is; a
            # deliberate "no cap" and a broken probe both profile full depth, but only one of them is
            # a problem, and a reader cannot act on a line that will not say which.
            _why = str((_bl_facts or {}).get("no_window") or "")
            _said = {
                "knob_inert": "the depth knob does not reach this model's builder (capping changed no "
                "work), so nothing can be capped",
                "sizing_disabled": "coverage sizing is off (PERF_MCP_COVERAGE_SIZING=0)",
                "no_node": "no perf-test node to probe",
                "probe_failed": "the op-signature probe found nothing and no config declares a layer "
                "pattern -- this one is NOT a decision, it is unknown",
            }.get(_why, "reason not reported by the coverage probe (%r)" % (_why or None))
            print(
                "      depth-bridge: no profiling window -- %s. The baseline profiles FULL depth "
                "(%d blocks); nothing is capped and nothing is substituted." % (_said, _bl_blocks),
                file=sys.stderr,
                flush=True,
            )
        print(
            f"      depth-bridge: node={perf_rel} case={case} cov={_bl_cov} full_signal={_bl_full} full_blocks={_bl_blocks} knob={bool(_bl_knob)}",
            file=sys.stderr,
            flush=True,
        )
        if _bl_cov:
            # A SCALAR, BECAUSE THAT IS WHAT A MODEL READS. _coverage_layers returns a per-stack
            # DICT, and str() of it wrote "{'stack3': 2, 'stack2': 2}" into the one variable the perf
            # test parses -- which fails .isdigit(), yields None, and means ALL LAYERS. The BASELINE
            # was therefore measured at FULL depth while every candidate after it was measured
            # capped, so "before" and "after" described different models and any gain computed from
            # the pair was meaningless. Measured on Voxtral 2026-08-13: baseline 3977 ms over ~32700
            # device ops against a capped model of 2965 dispatched ops.
            _bl_scalar = max(int(v) for v in _bl_cov.values()) if isinstance(_bl_cov, dict) else int(_bl_cov)
            os.environ["TT_PERF_LAYERS"] = str(_bl_scalar)
        # The bridge exists to find an env spelling that makes a cap REACH the builder. With no cap
        # to apply there is nothing for it to search for, and calling it with a None depth would only
        # rediscover -- at the price of more device probes -- what _coverage_layers already proved.
        if _bl_cov:
            _bl_depth = _bridge_depth_env(
                tt_root,
                sub_env,
                devices,
                perf_rel,
                case,
                _bl_cov,
                full_hint=_bl_full,
                full_blocks=_bl_blocks,
                knob=_bl_knob,
                stage_depths=(_bl_facts or {}).get("per_stage"),
            )
            if _bl_depth:
                os.environ["PERF_MCP_PROFILE_ENV"] = json.dumps(_bl_depth)
    except Exception as _bl_e:  # noqa: BLE001
        import traceback as _tb

        print(f"      depth-bridge skipped: {str(_bl_e)[:160]}", file=sys.stderr, flush=True)
        print(_tb.format_exc()[-600:], file=sys.stderr, flush=True)

    def _run_baseline():
        return profile_model(
            perf_test=perf_rel,
            config=config,
            env=env,
            profiles_dir=run.profiles_dir,
            run_profiled=run_profiled_factory(perf_rel, case),
        )

    # ENFORCED IN CODE, not advised in the prompt: a model that already HAS a baseline does not get
    # a new one. The bar is established ONCE, the first time a model is optimized, and afterwards
    # moves only downward on a win (perf_mcp._promote_baseline).
    #
    # Re-measuring is not merely wasteful (250 s on gemma-3-12b-it). It silently redefines what every
    # later verdict is graded against: successive runs measured 381.186 / 381.222 / 381.263 / 381.291
    # / 381.311 for IDENTICAL code, and the resume filter compares those stamps with exact equality,
    # so a different subset of attempt history survived each run -- the upstream cause of the 38%
    # repeat rate on this model.
    #
    # A note in GUIDELINES would be advice, and advice gets worked around; this is a branch the run
    # cannot take. PERF_MCP_FORCE_REBASELINE=1 is the deliberate escape hatch, for when the model has
    # changed enough that the stored bar means nothing.
    _bl_key_model = Path(model_root).name or "model"
    _bl_key_task = os.environ.get("PERF_MCP_TASK", "main")
    _stored_baseline = None
    if str(os.environ.get("PERF_MCP_FORCE_REBASELINE", "")).lower() not in ("1", "true", "yes"):
        try:
            _sp_path = state_dir() / ("perf_mcp_baseline_%s_%s.json" % (_bl_key_model, _bl_key_task))
            _sp_doc = json.loads(_sp_path.read_text())
            if float(_sp_doc.get("device_ms") or 0.0) > 0 and (_sp_doc.get("buckets") or []):
                _stored_baseline = _sp_doc
            # A BASELINE MEASURED UNDER A DIFFERENT DEFINITION IS NOT A BASELINE. device_ms used to
            # come from DEVICE KERNEL DURATION [ns] -- first start on ANY core to last end on ANY
            # core -- which on hardware whose cores do not share a clock includes the inter-core
            # offset as well as the op. It now comes from the per-core column. Both are called
            # device_ms, so a stored one and a fresh one look identical and subtracting them reports
            # a gain or a regression that never happened.
            #
            # RE-MEASURED, not flagged. Carrying the old number forward with a warning attached
            # leaves every later delta wrong and asks a human to remember why; a baseline exists to
            # be compared against, so one that cannot be is worth nothing and the honest cost is one
            # profiling run at start-up. Absent stamp = pre-stamp = old definition, because the
            # stamp was added with the change.
            if _stored_baseline is not None:
                _srcs = {
                    str(b.get("device_time_source") or "")
                    for b in (_stored_baseline.get("buckets") or [])
                    if isinstance(b, dict)
                }
                if _srcs and not _srcs <= {"per_core_max"}:
                    print(
                        "      baseline DISCARDED: its device_ms came from %s, this build measures "
                        "per_core_max -- the two are not comparable, so it is being re-measured "
                        "rather than differenced." % ("+".join(sorted(x for x in _srcs if x)) or "an older definition"),
                        file=sys.stderr,
                        flush=True,
                    )
                    _stored_baseline = None
        except Exception:  # noqa: BLE001
            _stored_baseline = None

    if _stored_baseline is not None:
        stages.start("tracy_baseline", "Reusing the baseline established on the first run")
        print(
            "      ✔ baseline REUSED: %.4f ms device, established previously and not re-measured "
            "(it moves only on a win). PERF_MCP_FORCE_REBASELINE=1 forces a fresh one."
            % float(_stored_baseline.get("device_ms") or 0.0),
            file=sys.stderr,
            flush=True,
        )
        profile = _stored_baseline
    else:
        stages.start("tracy_baseline", "Measuring the baseline latency (trace+1CQ)")
        profile = _measure_baseline(_run_baseline, stages)
    _seq_env = os.environ.get("TT_PERF_SEQ_LEN")
    if _seq_env:
        (run.dir / "perf_seq_len").write_text(_seq_env)
    # STAMP THE DEPTH ONTO THE MEASUREMENT. A device_ms with no record of the depth it was taken at
    # cannot be checked against anything: the baseline was measured at FULL depth for a whole day
    # while every candidate after it ran capped, and nothing in the artifact could reveal it --
    # _record_baseline_anchor already reads profile["perf_layers"] and had been getting None, so the
    # anchor said "all" whatever the truth was. The cap is expressed by ABSENCE, so absent means all.
    if isinstance(profile, dict) and "perf_layers" not in profile:
        from .layer_depth import depth_in_force as _depth_in_force

        profile["perf_layers"] = _depth_in_force()
    # Persist the tagged buckets for the loop: ROUTE reads this, not the CSVs.
    (Path(run.profiles_dir) / "baseline_profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True))
    # ...and record the SAME profile as the ledger's eager anchor, right here. This file and the
    # KIND_EAGER anchor are two views of one measurement, but only perf_mcp._ledger_record wrote the
    # anchor and it fires solely from the agent-invoked profile_model MCP tool. A run whose ledger
    # starts empty and never happens to make that call therefore had a complete baseline_profile.json
    # and NO anchor, so the report printed "not measured (no ledger reading)" for a number it had
    # just measured -- and each consumer reached for whichever store it knew about, which is how one
    # profile showed up as three different totals (120.59 / 152.02 / 178.85).
    # KEY IT FROM THE MODEL DIRECTORY, not from PERF_MCP_MODEL_NAME: run.py does not set that
    # variable until AFTER discover() has taken this baseline, so reading it here yields "" and
    # ledger_path falls back to the literal "model" -- which is how the gemma-3-12b-it run of
    # 2026-07-31 wrote its eager anchor (240.86 ms) to perf_measurements_model_main.jsonl while
    # every other writer used perf_measurements_gemma3_main.jsonl. Two ledgers for one run, so the
    # write-once BEFORE/AFTER rule was applied per FILE and the report said "not measured" for a
    # number it was holding. The line just below already keys the /tmp baseline copy this way.
    _record_baseline_anchor(profile, model=Path(model_root).name)
    try:
        pass

        _bl_model = Path(model_root).name or os.environ.get("PERF_MCP_MODEL_NAME") or "model"
        _bl_task = os.environ.get("PERF_MCP_TASK", "main")
        _bl_name = "perf_mcp_baseline_%s_%s.json" % (_bl_model, _bl_task)
        (state_dir() / _bl_name).write_text(json.dumps(profile))
    except Exception:  # noqa: BLE001
        pass
    _bk = {b.get("id"): int(b.get("count", 0)) for b in (profile.get("buckets") or [])}
    _struct_ops = sum(c for i, c in _bk.items() if i in STRUCTURAL_OP_CLASSES)
    if profile.get("device_ms", 0) <= 0 or _struct_ops == 0:
        raise RuntimeError(
            f"baseline capture looks partial/degenerate (device_ms={profile.get('device_ms')}, "
            f"structural ops={_struct_ops}, buckets={_bk}); refusing to optimize against it. "
            f"Inspect {run.profiles_dir}/run0_tracy.log for a crash or profiler-marker overflow."
        )
    stages.done(
        f"baseline {profile['device_ms']:.3f} ms on device "
        f"(wall {profile['wall_ms']:.0f} ms incl. compile), {len(profile['buckets'])} op-class buckets"
    )

    metric_name = config.get("metric") or "device_ms"
    if metric_name == "auto":
        try:
            from .strategist import choose_axis, make_axis_runner, make_cli_axis_runner

            _axis_runner = make_cli_axis_runner() if config.get("cc_discovery") else make_axis_runner()
            metric_name = choose_axis(profile, _axis_runner)
            print(f"      strategist chose axis -> metric={metric_name}", flush=True)
        except Exception as exc:
            metric_name = "device_ms"
            print(f"      strategist failed ({exc}); falling back to metric=device_ms", flush=True)
        config["metric"] = metric_name
    # device_ms = sum of profiled device-kernel time (the optimization target);
    # wall_ms = harness clock incl. compile/setup (reference only);
    # fps / tok_s still TBD(wall-metric-source).
    if metric_name not in ("device_ms", "wall_ms"):
        print(
            "      WARNING: --metric %s is recorded from wall_ms; no profile carries a %r key, so "
            "downstream comparisons fall back to wall_ms. Prefer device_ms or wall_ms." % (metric_name, metric_name),
            file=sys.stderr,
            flush=True,
        )
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
            # full bucket exhaustion, never on an arbitrary iter wall. See --max-iter.
            "max_iter": config.get("max_iter"),
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
    # or true bucket exhaustion, NOT at an arbitrary iter cap. Pass --max-iter only to impose a
    # ceiling deliberately; unset (None) = uncapped (exit_policy treats None as "no limit").
    ap.add_argument("--max-iter", type=int, default=None)
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
            "runs",
            "timeout",
            "notes",
            "perf_test",
            "pcc_test",
            "case",
            "devices",
            "input",
            "box",
            "cc_discovery",
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
            # ONE runner now. The `else:` branch here called probes.sdk_model_files_runner, the
            # Claude-Agent-SDK twin of this CLI runner, and only the retired FSM engine ever took it
            # -- cc always passes --cc-discovery. Both are gone with the SDK.
            from .probes import cli_model_files_runner

            model_runner = cli_model_files_runner()

        if args.mock_model_files:
            review = mock_review  # gatherer mocked -> nothing real to review
        else:
            from .probes import cli_lead_review_gate

            review = cli_lead_review_gate

        if args.mock_tracy:
            factory = lambda perf, case: mock_run_profiled
            preflight = mock_preflight
            collect = mock_collect_cases
        else:
            from .probes import make_run_profiled, preflight_collect

            tt_root = os.environ.get("TT_METAL_HOME", str(PKG_ROOT.parents[2]))
            # Visibility is NEVER restricted (see the devices block above): a chip-subset pin crashes
            # fabric auto-discovery on multi-chip boards. Chip count is driven by the mesh shape, not
            # OS visibility, so pass no visibility override here regardless of --devices.
            xenv = {}
            factory = lambda perf, case: make_run_profiled(tt_root, perf, case, timeout_s=args.timeout, extra_env=xenv)
            preflight = preflight_collect
            from .probes import collect_cases as collect

        result = before_loop(config, env_probe, model_runner, factory, preflight, review, collect)
    except Exception as exc:
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
            print("\n  ✗ OUT OF DISK — free space and rerun", file=sys.stderr)
            return 1
        print(f"\n  ✗ discovery failed ({type(exc).__name__}):", file=sys.stderr)
        for _ln in str(exc).splitlines():
            print(f"      {_ln}", file=sys.stderr)
        # A REJECTION IS NOT A CRASH. The lead agent stopping the run is a decision, and returning 1
        # made the supervisor read it as a likely native crash and RESTART the child -- which on a
        # real run meant a second optimize, still carrying the gate that had just been rejected,
        # racing the corrected run for the same board until both wedged it.
        from .probes import DiscoveryRejected

        if isinstance(exc, DiscoveryRejected):
            # ABSOLUTE, BECAUSE THIS FILE RUNS AS `python -m ...agent.before_loop`.
            #
            # The relative form raised "attempted relative import beyond top-level package" -- the
            # package is `agent`, so `..` walks off the top -- and it raised INSIDE the handler that
            # exists to return EXIT_REFUSED. So a refused discovery exited rc=1, the supervisor read
            # that as a crash, and restarted it: precisely the "racing the corrected run for the
            # same board until both wedged it" the comment above warns about, caused by the line
            # meant to prevent it. Observed run 9, 2026-08-17, on a flaky lead-review verdict.
            #
            # Same import the supervisor uses, same literal fallback, so the two cannot disagree
            # about which code means "refused" -- see optimize.py and
            # test_r5_the_exit_code_has_one_definition.
            try:
                from models.experimental.perf_automation.cc_optimize.run import EXIT_REFUSED
            except Exception:  # noqa: BLE001 -- a refusal must still be reportable without the import
                EXIT_REFUSED = 3

            return EXIT_REFUSED
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

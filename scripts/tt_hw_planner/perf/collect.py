# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""perf collect — profile a brought-up model with Tracy + model_tracer.

This wraps the demo invocation produced by `bringup.prepare_bringup` in a
single `python -m tracy -r -p -m pytest ...` command that ALSO loads the
model_tracer pytest plugin and enables --trace-params. The outputs land in
`perf-data/run_<timestamp>/`:

  ops_perf_results_*.csv        - the Tracy CSV (timing, util, kernel hashes)
  ttnn_operations_master.json   - the model_tracer master (op args by name)
  run_meta.json                 - provenance (model id, box, mesh, git sha,
                                  baseline run id if any)

Designed to be model-agnostic: any HF model that `compat` reports as
ALREADY SUPPORTED / READY / FEASIBLE WITH WORK and that `prepare` can
emit a runnable invocation for can be collected by this module.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..bringup import REPO_ROOT, BringupPlan, prepare_bringup


PERF_DATA_ROOT = REPO_ROOT / "perf-data"
TRACER_OUTPUT_SUBDIR = "ttnn_traces"


@dataclass
class RunArtifacts:
    """Paths to the artifacts that `collect_run` produced (or expects)."""

    run_id: str
    run_dir: Path
    tracy_csv: Optional[Path]
    tracer_master: Optional[Path]
    run_meta: Path
    log: Path
    exit_code: int = 0
    failed: bool = False
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "tracy_csv": str(self.tracy_csv) if self.tracy_csv else None,
            "tracer_master": str(self.tracer_master) if self.tracer_master else None,
            "run_meta": str(self.run_meta),
            "log": str(self.log),
            "exit_code": self.exit_code,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
        }


class CollectError(RuntimeError):
    """Raised when a perf collect run failed and the artifacts must not be
    treated as a usable baseline. The run directory is preserved so the
    user can inspect collect.log; the run is marked `failed=true` in
    run_meta.json so downstream tools refuse to use it.

    The `run_dir`, `run_id`, and `post_mortem` attributes are populated so
    a caller (the CLI auto-retry loop, in particular) can read the
    structured failure analysis and decide whether the failure is
    auto-recoverable without re-opening files. `post_mortem` is typed
    `Any` here to avoid an import cycle with `_postmortem`.
    """

    def __init__(
        self,
        message: str,
        *,
        run_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        post_mortem: Optional[object] = None,
    ) -> None:
        super().__init__(message)
        self.run_dir = run_dir
        self.run_id = run_id
        self.post_mortem = post_mortem


@dataclass
class RunMeta:
    """Provenance written next to the artifacts so re-analyses are reproducible."""

    run_id: str
    timestamp_utc: str
    model_id: str
    box: str
    arch: str
    mesh_shape: Tuple[int, int]
    mesh_device: str
    dtype: str
    git_sha: Optional[str]
    baseline_run_id: Optional[str]
    pytest_argv: List[str]
    env: Dict[str, str]
    test_path: str
    # Architecture metadata pulled from the HF config at collect time. Stored
    # here so downstream commands (`perf report`, `perf dashboard`) can
    # auto-infer transformer-block boundaries from the Tracy op stream
    # without re-probing HF Hub. None if the probe failed.
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["mesh_shape"] = list(self.mesh_shape)
        return d


def _git_sha() -> Optional[str]:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run(
            [git, "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def _new_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _shell_safe(token: str) -> str:
    """Pre-shell-quote an argv item if it contains characters that would be
    re-split by a shell.

    `tools/tracy/__main__.py` reconstructs its inner command with
    `" ".join(originalArgs[1:])` and then runs it through `shell=True`, so
    any argv item we hand it which contains whitespace, glob chars, or
    other shell metas will be re-tokenized incorrectly. Concretely,
    `-k "performance and batch-1"` becomes the four shell words
    `-k performance and batch-1` and pytest interprets `and` as a path.

    `shlex.quote` is a no-op for safe tokens, so it is cheap to apply
    unconditionally to every argv item we pass to tracy.
    """
    return shlex.quote(token)


def _build_tracy_argv(plan: BringupPlan, tracer_dir: Path) -> List[str]:
    """Wrap the demo's pytest argv with `python -m tracy -r -p -m pytest`
    and tack on the model_tracer plugin + --trace-params.

    Every item that originates from `inv.args` or `inv.test_path` is
    passed through `_shell_safe` because `tools/tracy/__main__.py`
    rejoins argv with a plain `" ".join(...)` before handing the string
    to a shell. See `_shell_safe` for the rationale.

    We also pass `--dump-device-data-mid-run`. The Metal device-side
    profiler keeps markers in a per-(device, core, RISC) DRAM ring of
    fixed capacity (12000 entries). With four devices and >~50 ops
    being profiled, the ring overflows mid-decode and silently drops
    markers — which then surfaces downstream as
    `AssertionError: Device data missing: Op N not present in
    cpp_device_perf_report.csv`. Tracy's mid-run dump flushes the ring
    between ops so the ring never fills. Set `TT_METAL_PROFILER_MID_RUN_DUMP=1`
    has the same effect; we set both for belt-and-braces.
    """
    if plan.invocation is None:
        raise RuntimeError("BringupPlan has no runnable invocation")

    inv = plan.invocation
    argv: List[str] = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-p",
        "--dump-device-data-mid-run",
        "-m",
        "pytest",
        _shell_safe(inv.test_path),
        "-p",
        "model_tracer.tracer_pytest_plugin",
        "--trace-params",
    ]
    argv.extend(_shell_safe(a) for a in inv.args)
    return argv


_OPS_CSV_PATTERN = re.compile(r"ops_perf_results_.*\.csv$")


def _locate_outputs(
    after_run_search_roots: List[Path],
    not_older_than: Optional[float] = None,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """After tracy finishes, find the most-recent ops_perf_results_*.csv,
    ttnn_operations_master.json, and ttnn_module_hierarchy.json from the
    search roots.

    `not_older_than` is a unix timestamp (float); any candidate whose mtime
    is older than this is rejected. This prevents the run from accidentally
    "succeeding" on a stale CSV left over from a previous, unrelated run
    when the current run's pytest crashed before producing fresh outputs.
    Generic across any model — the staleness check is by mtime, not name.

    The third return is the module-hierarchy sidecar (None if the run
    didn't produce one, e.g. legacy runs predating module_hierarchy or
    runs where every nn.Module hook failed to install).
    """
    csv_candidates: List[Path] = []
    tracer_candidates: List[Path] = []
    hierarchy_candidates: List[Path] = []
    for root in after_run_search_roots:
        if not root.exists():
            continue
        for p in root.rglob("ops_perf_results_*.csv"):
            csv_candidates.append(p)
        for p in root.rglob("ttnn_operations_master.json"):
            tracer_candidates.append(p)
        for p in root.rglob("ttnn_module_hierarchy.json"):
            hierarchy_candidates.append(p)
    if not_older_than is not None:
        csv_candidates = [p for p in csv_candidates if p.stat().st_mtime >= not_older_than]
        tracer_candidates = [p for p in tracer_candidates if p.stat().st_mtime >= not_older_than]
        hierarchy_candidates = [p for p in hierarchy_candidates if p.stat().st_mtime >= not_older_than]
    csv = max(csv_candidates, key=lambda p: p.stat().st_mtime, default=None)
    tracer = max(tracer_candidates, key=lambda p: p.stat().st_mtime, default=None)
    hierarchy = max(hierarchy_candidates, key=lambda p: p.stat().st_mtime, default=None)
    return csv, tracer, hierarchy


def _copy_into_run_dir(src: Optional[Path], dst: Path) -> Optional[Path]:
    if src is None:
        return None
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _synthesize_tracer_master_from_csv(csv_path: Optional[Path], dst: Path) -> Optional[Path]:
    """Build a minimal `ttnn_operations_master.json` from Tracy CSV.

    Some demos currently do not emit per-op model_tracer JSON files even
    when `--trace-params` is enabled. We still want downstream tooling to
    have a stable artifact contract (`ops_perf_results.csv` +
    `ttnn_operations_master.json` + hierarchy sidecar), so this creates a
    best-effort master where:

      - operation_name comes from Tracy's `OP CODE`
      - arguments are `{}` (unknown)
      - executions.counter is the number of rows for that op code

    This keeps the join/report/dashboard pipeline alive while making it
    explicit (via `run_meta.json` and `completion.json`) that arguments are
    unavailable for this run.
    """
    if csv_path is None or not csv_path.exists():
        return None
    counts: Dict[str, int] = {}
    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                op = (raw.get("OP CODE") or "").strip()
                op_type = (raw.get("OP TYPE") or "").strip().lower()
                if not op or op_type == "signpost":
                    continue
                counts[op] = counts.get(op, 0) + 1
    except OSError:
        return None
    if not counts:
        return None

    operations: Dict[str, object] = {}
    for op_name, cnt in counts.items():
        operations[op_name] = {
            "configurations": [
                {
                    "arguments": {},
                    "executions": [{"counter": int(cnt)}],
                    "sweep_source_hash": None,
                    "config_id": "synthesized_from_tracy",
                }
            ]
        }
    payload = {"operations": operations}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2))
    return dst


def collect_run(
    *,
    model_id: str,
    box_override: Optional[str] = None,
    mesh_override: Optional[Tuple[int, int]] = None,
    dtype_override: Optional[str] = None,
    baseline_run_id: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    run_dir_root: Optional[Path] = None,
    trace: bool = False,
    max_generated_tokens: int = 64,
    max_seq_len: int = 128,
) -> RunArtifacts:
    """Run the brought-up demo with Tracy + model_tracer enabled and
    package the artifacts into perf-data/<run_id>/.

    `trace=False` (the default) runs the demo with `--disable_trace`. Trace
    replay + multi-device + long generation can desync tracy's host-side op
    log from the device-side perf CSV, surfacing as
    `AssertionError: Device data missing: Op N not present ...`. The
    `trace_capturer` optimizer block proposes re-enabling trace and
    measuring the speedup; that's the right place for it.

    `max_generated_tokens` defaults to 64 (vs. the bring-up demo's 200)
    because the device-profiler ring buffer can drop records on very long
    multi-device runs even without trace replay.

    Raises RuntimeError if `prepare_bringup` produced no runnable command.
    """
    plan = prepare_bringup(
        model_id=model_id,
        box_override=box_override,
        mesh_override=mesh_override,
        dtype_override=dtype_override,
        trace=trace,
        max_generated_tokens=max_generated_tokens,
        max_seq_len=max_seq_len,
    )
    if plan.invocation is None:
        # Surface the blockers inline so the user does not have to chase
        # them in a second `prepare` invocation. Model-agnostic: we read
        # whatever the planner found.
        lines: List[str] = []
        lines.append(
            f"perf collect: no runnable command for {model_id} on "
            f"{plan.box_name} mesh [{plan.mesh_shape[0]},{plan.mesh_shape[1]}]."
        )
        lines.append(f"  Planner verdict: {plan.fit_verdict}; compat: {plan.compat_overall}")
        if plan.notes:
            lines.append("  Notes:")
            for n in plan.notes:
                lines.append(f"    - {n}")
        if plan.kernel_blockers:
            lines.append("  Kernel blockers:")
            for k in plan.kernel_blockers:
                lines.append(f"    - {k}")
        if plan.compat_blocking:
            lines.append("  Missing building blocks:")
            for b in plan.compat_blocking:
                lines.append(f"    - {b}")
        # Suggest the next concrete action the user can take right now,
        # without rerunning a separate sub-command.
        lines.append("")
        lines.append("  How to unblock:")
        if plan.kernel_blockers:
            lines.append("    - Try a different --mesh (e.g. one whose TP divides num_key_value_heads / num_heads).")
            lines.append(
                "    - Run `python -m scripts.tt_hw_planner plan "
                f"{model_id} --box {plan.box_name} --all-meshes` to see "
                "every canonical mesh's feasibility side-by-side."
            )
        if plan.compat_blocking:
            lines.append(
                f"    - `tt_hw_planner compat {model_id}` lists the building "
                "blocks still missing; those need a port before this model can run."
            )
        raise CollectError("\n".join(lines))

    run_root = run_dir_root if run_dir_root else PERF_DATA_ROOT
    run_id = _new_run_id()
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tracer_dir = run_dir / TRACER_OUTPUT_SUBDIR
    tracer_dir.mkdir(parents=True, exist_ok=True)

    argv = _build_tracy_argv(plan, tracer_dir)

    env = {**os.environ, **plan.invocation.env}
    env["TTNN_OPERATION_TRACE_DIR"] = str(tracer_dir)
    # Belt-and-braces with --dump-device-data-mid-run in the argv: this env
    # var is what tracy actually exports, so setting it directly removes any
    # ordering issue between tracy's option parser and the demo's child
    # process. See `_build_tracy_argv` for the rationale.
    env["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
    if extra_env:
        env.update(extra_env)

    log_path = run_dir / "collect.log"
    meta_path = run_dir / "run_meta.json"

    # Probe HF config for architectural metadata used by block inference.
    # Failure is non-fatal — we just lose auto-block-segmentation in the
    # report (it still renders, just without per-layer overlays).
    hf_n_layers: Optional[int] = None
    hf_n_heads: Optional[int] = None
    hf_n_kv_heads: Optional[int] = None
    try:
        from ..probe import probe_model

        probe = probe_model(model_id)
        if probe and probe.raw_config:
            cfg = probe.raw_config
            hf_n_layers = cfg.get("num_hidden_layers")
            hf_n_heads = cfg.get("num_attention_heads")
            hf_n_kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
    except Exception:
        pass

    meta = RunMeta(
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model_id=model_id,
        box=plan.box_name,
        arch=plan.arch,
        mesh_shape=plan.mesh_shape,
        mesh_device=plan.mesh_device,
        dtype=plan.dtype,
        git_sha=_git_sha(),
        baseline_run_id=baseline_run_id,
        pytest_argv=argv,
        env={k: v for k, v in plan.invocation.env.items()},
        test_path=plan.invocation.test_path,
        num_hidden_layers=hf_n_layers,
        num_attention_heads=hf_n_heads,
        num_key_value_heads=hf_n_kv_heads,
        notes=list(plan.notes),
    )
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2))

    if dry_run:
        return RunArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            tracy_csv=None,
            tracer_master=None,
            run_meta=meta_path,
            log=log_path,
        )

    start = time.time()
    with log_path.open("w") as lf:
        lf.write(f"# argv: {argv}\n")
        lf.write(f"# cwd:  {REPO_ROOT}\n")
        lf.flush()
        proc = subprocess.run(argv, cwd=REPO_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    search_roots = [
        REPO_ROOT / "generated" / "profiler",
        tracer_dir,
        run_dir,
    ]
    # Reject any artifact older than the start of this run. Otherwise a crashed
    # pytest can falsely "succeed" by picking up a stale CSV/JSON from a
    # previous, unrelated run. Generic; not model-specific.
    csv_src, tracer_src, hierarchy_src = _locate_outputs(search_roots, not_older_than=start)

    csv_dst = _copy_into_run_dir(csv_src, run_dir / "ops_perf_results.csv")
    tracer_dst = _copy_into_run_dir(tracer_src, run_dir / "ttnn_operations_master.json")
    _copy_into_run_dir(hierarchy_src, run_dir / "ttnn_module_hierarchy.json")
    tracer_synthesized = False
    if tracer_dst is None:
        tracer_dst = _synthesize_tracer_master_from_csv(csv_dst, run_dir / "ttnn_operations_master.json")
        tracer_synthesized = tracer_dst is not None
        if tracer_synthesized:
            meta_dict = json.loads(meta_path.read_text())
            notes = list(meta_dict.get("notes") or [])
            notes.append(
                "model_tracer master JSON was missing; synthesized minimal "
                "`ttnn_operations_master.json` from Tracy op counts (arguments unavailable)."
            )
            meta_dict["notes"] = notes
            meta_dict["tracer_master_synthesized"] = True
            meta_path.write_text(json.dumps(meta_dict, indent=2, default=str))

    device_rows = _count_device_rows(csv_dst)
    failure_reason: Optional[str] = None
    if proc.returncode != 0:
        failure_reason = (
            f"pytest exited with code {proc.returncode}; "
            f"the demo did not complete. See {log_path.name} for the full traceback."
        )
    elif csv_dst is None:
        failure_reason = "no Tracy CSV produced; profiling did not start (check collect.log)."
    elif device_rows < MIN_DEVICE_ROWS_FOR_HEALTHY_RUN:
        failure_reason = (
            f"only {device_rows} device-kernel rows in Tracy CSV "
            f"(< {MIN_DEVICE_ROWS_FOR_HEALTHY_RUN}); the model almost certainly "
            f"crashed before iterating. See {log_path.name}."
        )

    completion_payload = {
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 2),
        "tracy_csv_found": csv_dst is not None,
        "tracer_master_found": tracer_dst is not None,
        "tracer_master_synthesized": tracer_synthesized,
        "device_kernel_rows": device_rows,
        "failed": failure_reason is not None,
        "failure_reason": failure_reason,
    }
    (run_dir / "completion.json").write_text(json.dumps(completion_payload, indent=2))

    if failure_reason is not None:
        from ._postmortem import analyze_log

        pm = analyze_log(log_path, current_mesh=plan.mesh_shape)
        postmortem_text = pm.render()
        (run_dir / "postmortem.txt").write_text(postmortem_text)
        meta_dict = json.loads(meta_path.read_text())
        meta_dict["failed"] = True
        meta_dict["failure_reason"] = failure_reason
        meta_dict["exit_code"] = proc.returncode
        meta_dict["postmortem"] = {
            "matched_pattern": pm.matched_pattern,
            "assertion_text": pm.assertion_text,
            "file_line": list(pm.file_line) if pm.file_line else None,
            "quantified": pm.quantified,
            "next_retry_args": pm.next_retry_args,
            "retry_explanation": pm.retry_explanation,
        }
        meta_path.write_text(json.dumps(meta_dict, indent=2, default=str))
        raise CollectError(
            f"perf collect failed for run {run_id}: {failure_reason}\n"
            f"  log: {log_path}\n"
            f"\n{postmortem_text}\n"
            f"  Fix the underlying error, then re-run perf collect.",
            run_dir=run_dir,
            run_id=run_id,
            post_mortem=pm,
        )

    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        tracy_csv=csv_dst,
        tracer_master=tracer_dst,
        run_meta=meta_path,
        log=log_path,
        exit_code=proc.returncode,
        failed=False,
        failure_reason=None,
    )


MIN_DEVICE_ROWS_FOR_HEALTHY_RUN = 50


def _count_device_rows(csv_path: Optional[Path]) -> int:
    """Cheap sanity check: how many non-signpost rows are in the Tracy CSV.

    A real run produces thousands; a crashed pytest produces 0-2 (just the
    enter/exit signposts) or none at all.
    """
    if csv_path is None or not csv_path.exists():
        return 0
    rows = 0
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if not header:
                return 0
            for line in f:
                if not line.strip():
                    continue
                if line.startswith("signpost,signpost,"):
                    continue
                rows += 1
    except OSError:
        return 0
    return rows


def list_runs(run_dir_root: Optional[Path] = None) -> List[Path]:
    """Return all run directories under `perf-data/`, newest first."""
    root = run_dir_root if run_dir_root else PERF_DATA_ROOT
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def find_run(run_id: str, run_dir_root: Optional[Path] = None) -> Path:
    root = run_dir_root if run_dir_root else PERF_DATA_ROOT
    candidate = root / run_id
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"run not found: {candidate}")


def load_run_meta(run_dir: Path) -> Dict[str, object]:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_meta.json not found in {run_dir}")
    return json.loads(meta_path.read_text())


def require_healthy_run(run_dir: Path) -> Dict[str, object]:
    """Load `run_meta.json` and refuse to proceed if the underlying perf
    collect crashed.

    Downstream tools (`perf report`, `perf status`, `perf compare`,
    `perf apply`, `perf finalize`) all call this so a misleading "GREEN"
    status can never be reported on a run where pytest never produced
    real device-kernel rows.
    """
    meta = load_run_meta(run_dir)
    if meta.get("failed"):
        reason = meta.get("failure_reason", "unknown failure")
        raise CollectError(
            f"run {run_dir.name} is marked failed: {reason}\n"
            f"  Re-run `tt_hw_planner perf collect ...` after fixing the "
            f"underlying error. See {run_dir / 'collect.log'} for the "
            f"original traceback."
        )
    return meta

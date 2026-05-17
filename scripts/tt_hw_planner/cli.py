# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Command-line interface for tt_hw_planner.

Subcommands:
    plan          (default for HF model IDs)   — memory-budget recommendation
    compat        — list which TT building blocks the model needs and which exist
    scaffold      — generate a first-draft port (table entries + per-model JSONs)
    prepare       — emit env + pytest invocation to run the model on the recommended box
    calibrate     — open a mesh on real hw, measure usable HBM, write to YAML
    smoke-test    — open a mesh, run hot ops at the model's shapes
    validate      — run the full robustness + smoke battery for a (model, box, mesh)
    list-meshes   — print the canonical mesh topology table
    show-overhead — print current overhead constants (calibrated or analytical)

Usage:
    python -m scripts.tt_hw_planner Qwen/Qwen3-32B                    # plan (implicit)
    python -m scripts.tt_hw_planner plan Qwen/Qwen3-32B               # plan (explicit)
    python -m scripts.tt_hw_planner compat Qwen/Qwen3-32B             # bring-up checklist
    python -m scripts.tt_hw_planner prepare Qwen/Qwen3-32B            # runnable command
    python -m scripts.tt_hw_planner calibrate --box QB2 --mesh 1,4
    python -m scripts.tt_hw_planner smoke-test --model Qwen/Qwen3-32B --box QB2 --mesh 1,4
    python -m scripts.tt_hw_planner list-meshes
    python -m scripts.tt_hw_planner show-overhead

Backward-compat: `python scripts/tt_hw_recommender.py <hf_id>` still works
and routes to `plan`.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

from .architecture import DTYPE_BYTES
from .bringup import (
    BringupError,
    REPO_ROOT,
    prepare_bringup,
    render_json as render_bringup_json,
    render_script as render_bringup_script,
    render_text as render_bringup_text,
)
from .compatibility import check_compatibility
from .hardware import HARDWARE, find_box, reload_calibration, _CALIBRATED_OVERHEAD
from .kernel_constraints import evaluate_kernels
from .probe import probe_model
from .report import (
    render_compat_json,
    render_compat_table,
    render_json,
    render_markdown,
    render_table,
)
from .verdict import evaluate_all


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _dtypes_for(category: str, user: List[str], source_dtype: str = "") -> List[str]:
    if user:
        return user
    if category in ("LLM", "VLM"):
        base = ["bf16", "bfp8_b"]
        if source_dtype in ("fp8", "f8_e8m0"):
            return ["fp8", "bfp8_b", "bf16"]
        return base
    return ["bf16"]


def _parse_mesh(s: str) -> Tuple[int, int]:
    parts = s.replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"mesh must be 'rows,cols' or 'rowsxcols' (got '{s}')")
    return int(parts[0]), int(parts[1])


# ---------------------------------------------------------------------------
# Subcommand: plan
# ---------------------------------------------------------------------------


def cmd_plan(args) -> int:
    probe = probe_model(args.model_id)

    boxes = HARDWARE if not args.box else [find_box(n) for n in args.box]
    dtypes = _dtypes_for(probe.category, args.dtype, probe.saved_dtype)

    if probe.memory_model is None:
        return _render_weights_only(probe, boxes, dtypes, args)

    kv_bpe = 2.0 if args.kv_dtype == "bf16" else (4.0 if args.kv_dtype == "fp32" else 2.0)
    verdict = evaluate_all(
        model=probe.memory_model,
        boxes=boxes,
        dtypes=dtypes,
        batch=args.batch,
        seq=args.seq,
        kv_dtype_bytes=kv_bpe,
        all_meshes=args.all_meshes,
        explore_pp=args.explore_pp,
    )

    if args.format == "json":
        print(render_json(probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes))
    elif args.format == "markdown":
        print(render_markdown(probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes))
    else:
        print(
            render_table(
                probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes, show_overhead=not args.no_overhead_detail
            )
        )
    return 0


def _render_weights_only(probe, boxes, dtypes, args) -> int:
    from .verdict import FitRow, FitVerdict, Tightness, pick_best
    from .parallelism import ParallelConfig, ShardedMemory

    rows: List[FitRow] = []
    for box in boxes:
        if args.all_meshes:
            meshes = list(box.mesh_shapes)
        else:
            max_tp = max(r * c for r, c in box.mesh_shapes)
            meshes = [next(s for s in box.mesh_shapes if s[0] * s[1] == max_tp)]
        for dtype in dtypes:
            per_param = DTYPE_BYTES.get(dtype, 2.0)
            on_disk_per_param = probe.bytes_per_param_on_disk or 2.0
            scale = per_param / on_disk_per_param if on_disk_per_param else 1.0
            scaled_weights = int(probe.weight_bytes_total * scale)
            for shape in meshes:
                tp = shape[0] * shape[1]
                pcfg = ParallelConfig(tp=tp)
                per_chip_w = scaled_weights // tp
                per_chip_total = per_chip_w + 1_000_000_000  # +1 GB activation slack
                usable_gb = box.usable_per_chip_gb(tp)
                per_chip_gb = per_chip_total / 1e9
                headroom_gb = usable_gb - per_chip_gb
                rows.append(
                    FitRow(
                        box=box,
                        dtype=dtype,
                        mesh_shape=shape,
                        parallel=pcfg,
                        sharded=ShardedMemory(
                            weights_bytes=per_chip_w, kv_cache_bytes=0, activation_bytes=1_000_000_000
                        ),
                        usable_per_chip_gb=usable_gb,
                        per_chip_gb=per_chip_gb,
                        headroom_gb=headroom_gb,
                        tightness=Tightness.classify(headroom_gb, usable_gb),
                    )
                )

    verdict = FitVerdict(rows=rows, best=pick_best(rows), notes=["weights-only estimate; no transformer memory model"])

    if args.format == "json":
        print(render_json(probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes))
    elif args.format == "markdown":
        print(render_markdown(probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes))
    else:
        print(
            render_table(
                probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes, show_overhead=not args.no_overhead_detail
            )
        )
    return 0


def cmd_compat(args) -> int:
    probe = probe_model(args.model_id)
    if not probe.raw_config:
        print(
            f"ERROR: could not load config.json for {args.model_id}. "
            "Compatibility analysis needs the HuggingFace config; check that "
            "the repo is public (or HF_TOKEN is set) and that model_type is "
            "exposed in config.json.",
            file=sys.stderr,
        )
        return 1

    report = check_compatibility(args.model_id, probe.raw_config)

    kernel_report = None
    if not args.skip_kernel_check:
        tp_grid = args.tp_grid if args.tp_grid else None
        kernel_report = evaluate_kernels(probe.raw_config, tp_grid=tp_grid)

    if args.format == "json":
        print(render_compat_json(report, kernel_report))
    else:
        print(render_compat_table(report, kernel_report, verbose=args.verbose))

    if report.overall == "BLOCKED":
        return 2
    if kernel_report is not None and kernel_report.has_blockers(tp=1):
        return 2
    return 0


def cmd_scaffold(args) -> int:
    from .scaffold import (
        ScaffoldError,
        apply_scaffold,
        plan_scaffold,
        render_apply,
        render_json as render_scaffold_json,
        render_patch,
        render_text,
    )

    try:
        plan = plan_scaffold(args.model_id)
    except ScaffoldError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.format == "json":
        applied = apply_scaffold(plan) if args.apply else None
        print(render_scaffold_json(plan, applied))
        return 0
    if args.format == "patch":
        print(render_patch(plan))
        if args.apply:
            applied = apply_scaffold(plan)
            print(render_apply(plan, applied), file=sys.stderr)
        return 0

    print(render_text(plan, show_diff=not args.no_diff))

    if args.apply:
        applied = apply_scaffold(plan)
        print()
        print(render_apply(plan, applied))

    return 0


def cmd_prepare(args) -> int:
    mesh_override: Optional[Tuple[int, int]] = None
    if args.mesh:
        try:
            mesh_override = _parse_mesh(args.mesh)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    try:
        plan = prepare_bringup(
            model_id=args.model_id,
            box_override=args.box,
            mesh_override=mesh_override,
            dtype_override=args.dtype,
            batch=args.batch,
            max_seq_len=args.max_seq_len,
            max_generated_tokens=args.max_generated_tokens,
            accuracy=args.accuracy,
            trace=not args.no_trace,
            paged_attention=not args.no_paged_attention,
            instruct=not args.no_instruct,
        )
    except BringupError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(render_bringup_json(plan))
    elif args.format == "script":
        print(render_bringup_script(plan))
    else:
        print(render_bringup_text(plan))

    if args.write_script:
        from pathlib import Path

        # Resolve the user-supplied path and refuse anything that would land
        # under a system directory. Mode is owner-only (0o700) — the script
        # is private to the user invoking the planner.
        path = Path(args.write_script).expanduser().resolve()
        _SYSTEM_PREFIXES = ("/etc", "/usr", "/bin", "/sbin", "/boot", "/sys", "/proc", "/dev")
        s = str(path)
        if any(s == d or s.startswith(d + "/") for d in _SYSTEM_PREFIXES):
            print(f"\nERROR: refusing to write bring-up script to system path: {path}", file=sys.stderr)
            return 1
        path.write_text(render_bringup_script(plan))
        path.chmod(0o700)
        print(f"\nWrote bring-up script: {path}", file=sys.stderr)

    if args.execute:
        if plan.invocation is None:
            print("\nERROR: no executable command (see blockers above).", file=sys.stderr)
            return 2
        if args.strict and plan.compat_overall not in {"ALREADY SUPPORTED", "READY"}:
            print(
                f"\nERROR: --strict refused — compat verdict is '{plan.compat_overall}'. "
                "Remove --strict to execute despite PARTIAL blocks.",
                file=sys.stderr,
            )
            return 2
        import subprocess

        full_env = {**os.environ, **plan.invocation.env}
        print(f"\nExecuting in {REPO_ROOT} …", file=sys.stderr)
        return subprocess.run(plan.invocation.argv(), cwd=REPO_ROOT, env=full_env).returncode

    if plan.invocation is None:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Subcommand: calibrate
# ---------------------------------------------------------------------------


def cmd_calibrate(args) -> int:
    from .calibration import calibrate_box

    try:
        mesh = _parse_mesh(args.mesh)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Calibrating {args.box} @ mesh {mesh}")
    print(f"  This opens the mesh on hardware and binary-searches the")
    print(f"  largest single DRAM allocation that succeeds.")
    print()
    try:
        run = calibrate_box(args.box, mesh, source_label=args.label or "calibrate CLI")
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print()
    print(f"  Measured per-chip usable:   {run.measured_per_chip_gb:.2f} GB")
    print(f"  Predicted per-chip usable:  {run.predicted_per_chip_gb:.2f} GB")
    print(f"  Implied per-chip overhead:  {run.implied_overhead_gb:.2f} GB")
    box = find_box(args.box)
    print(f"  (Box HBM/chip:              {box.hbm_per_chip_gb:.2f} GB)")
    print()
    delta = run.predicted_per_chip_gb - run.measured_per_chip_gb
    if abs(delta) < 0.5:
        print("  Analytical estimate matches measurement within 0.5 GB. ")
    elif delta > 0:
        print(
            f"  Analytical estimate is OPTIMISTIC by {delta:.2f} GB/chip — "
            "the planner currently over-promises memory."
        )
    else:
        print(
            f"  Analytical estimate is PESSIMISTIC by {-delta:.2f} GB/chip — "
            "the planner has room to be more aggressive."
        )
    print()
    reload_calibration()
    print(f"  Wrote calibration entry. Now active for box {args.box}.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: smoke-test
# ---------------------------------------------------------------------------


def cmd_smoke_test(args) -> int:
    from .smoke import run_smoke_suite, render_smoke_results

    probe = probe_model(args.model)
    if probe.arch_spec is None:
        print(
            f"ERROR: smoke-test currently supports transformer-family models only " f"(got category={probe.category}).",
            file=sys.stderr,
        )
        return 1

    try:
        mesh = _parse_mesh(args.mesh)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Smoke-testing {args.model} on mesh {mesh}")
    print(
        f"  hidden={probe.arch_spec.hidden_size}, "
        f"n_heads={probe.arch_spec.num_attention_heads}, "
        f"head_dim={probe.arch_spec.head_dim}"
    )
    print()
    try:
        results = run_smoke_suite(probe.arch_spec, mesh, batch=args.batch, seq=args.seq)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(render_smoke_results(results))
    n_fail = sum(1 for r in results if not r["ok"])
    return 0 if n_fail == 0 else 2


# ---------------------------------------------------------------------------
# Subcommand: list-meshes
# ---------------------------------------------------------------------------


def cmd_list_meshes(args) -> int:
    print("Canonical mesh topology (probe with ttnn.open_mesh_device):")
    print()
    print(f"  {'BOX':<8} {'ARCH':<10} {'CHIPS':>5}  CANONICAL MESHES")
    print("  " + "-" * 76)
    for b in HARDWARE:
        shapes = ", ".join(f"[{r},{c}]" for r, c in b.mesh_shapes[:8])
        if len(b.mesh_shapes) > 8:
            shapes += ", ..."
        print(f"  {b.name:<8} {b.arch:<10} {b.chips:>5}  {shapes}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Subcommand: validate
# ---------------------------------------------------------------------------
#
# Runs the full test battery for a (model, box, mesh) triple in one shot:
# paper-side robustness sweep, calibration audit, hardware smoke-test of the
# target's hot ops, and a positive-control smoke-test on a known-good model.


def _capture_verdict(
    model_id: str,
    box_filter: str,
    seq: int = 8192,
    batch: int = 1,
    dtype: Optional[str] = None,
    explore_pp: bool = False,
) -> str:
    """Run plan logic internally and return a one-line verdict string."""
    from .verdict import evaluate_all

    try:
        probe = probe_model(model_id)
    except SystemExit as e:
        return f"ERROR: {e}"
    if probe.memory_model is None:
        return "ERROR: no transformer memory model (non-transformer category)"

    box = find_box(box_filter)
    dtypes = [dtype] if dtype else _dtypes_for(probe.category, [], probe.saved_dtype)
    verdict = evaluate_all(
        model=probe.memory_model,
        boxes=[box],
        dtypes=dtypes,
        batch=batch,
        seq=seq,
        kv_dtype_bytes=2.0,
        all_meshes=False,
        explore_pp=explore_pp,
    )
    if verdict.best is None:
        return "NO FIT"
    r = verdict.best
    return (
        f"{r.tightness.value} on {r.box.name} "
        f"[{r.mesh_shape[0]},{r.mesh_shape[1]}] ({r.parallel.label}) "
        f"{r.dtype}, headroom={r.headroom_gb:.1f}G"
    )


def _stage1_robustness(model_id: str, box: str) -> Tuple[bool, List[Tuple[str, str]]]:
    """Run 4 plan variants and report whether they all agree on the fit class."""
    variants = [
        ("baseline (seq=8192)", dict(seq=8192)),
        ("with --explore-pp", dict(seq=8192, explore_pp=True)),
        ("--seq 1024 (minimal KV)", dict(seq=1024)),
        ("--dtype bfp4_b", dict(seq=8192, dtype="bfp4_b")),
    ]
    results = []
    for label, kwargs in variants:
        v = _capture_verdict(model_id, box, **kwargs)
        results.append((label, v))
    fit_classes = {("FIT" if r.startswith("FITS") else "NO" if r.startswith("NO FIT") else "ERR") for _, r in results}
    return (len(fit_classes) == 1), results


def _stage2_calibration_audit(box_name: str) -> Tuple[bool, List[str]]:
    from .calibration import load, DEFAULT_CALIBRATION_PATH

    db = load(DEFAULT_CALIBRATION_PATH)
    matching = [r for r in db.runs if r.box == box_name]
    notes = []
    if not matching:
        return False, [f"{box_name}: no calibration data — run `calibrate --box {box_name} --mesh <r,c>`"]
    for r in matching:
        notes.append(
            f"{box_name} mesh {r.mesh}: measured {r.measured_per_chip_gb:.2f} GB/chip "
            f"(overhead {r.implied_overhead_gb:.2f} GB) [{r.source}]"
        )
    return True, notes


def _stage3_smoke_test(model_id: str, mesh: Tuple[int, int], seq: int) -> Tuple[bool, List[dict], Optional[str]]:
    from .smoke import run_smoke_suite

    try:
        probe = probe_model(model_id)
    except SystemExit as e:
        return False, [], str(e)
    if probe.arch_spec is None:
        return False, [], "no transformer config for this model"
    try:
        results = run_smoke_suite(probe.arch_spec, mesh, batch=1, seq=seq)
    except RuntimeError as e:
        return False, [], str(e)
    all_ok = all(r["ok"] for r in results)
    return all_ok, results, None


def _render_smoke_brief(results: List[dict]) -> List[str]:
    out = []
    for r in results:
        if r.get("skipped"):
            status = "SKIP"
        elif r["ok"]:
            status = "OK  "
        else:
            status = "FAIL"
        op = r["op"][:50]
        out.append(f"    {status}  {op:<50}  {r['elapsed_s']:>5.2f}s")
        if r.get("skipped") and r.get("skip_reason"):
            out.append(f"            -> {r['skip_reason']}")
        elif not r["ok"] and r.get("error"):
            out.append(f"            -> {r['error']}")
    return out


def _run_smoke_stage(
    label: str,
    model_id: str,
    mesh: Tuple[int, int],
    smoke_seq: int,
    skip_hardware: bool,
    skip_msg: Optional[str] = None,
) -> Optional[bool]:
    """Render one smoke-test stage. Returns True/False/None (None = skipped)."""
    print(label)
    if skip_hardware:
        print("    SKIPPED (--skip-hardware)")
        return None
    if skip_msg:
        print(f"    SKIPPED ({skip_msg})")
        return None
    ok, results, err = _stage3_smoke_test(model_id, mesh, smoke_seq)
    if err:
        print(f"    -> SKIPPED: {err}")
        return None
    for line in _render_smoke_brief(results):
        print(line)
    n_run = sum(1 for r in results if not r.get("skipped"))
    n_ok = sum(1 for r in results if r["ok"] and not r.get("skipped"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    print(f"    -> {n_ok}/{n_run} ops succeeded" + (f", {n_skip} skipped" if n_skip else ""))
    return ok


def cmd_validate(args) -> int:
    try:
        mesh = _parse_mesh(args.mesh)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    width = 78
    print("=" * width)
    print(f"  VALIDATING: {args.model} on {args.box} mesh {mesh}")
    print("=" * width)
    print()

    overall_ok = True

    print("[STAGE 1] Paper verdict + robustness sweep")
    robust, rows = _stage1_robustness(args.model, args.box)
    for label, verdict in rows:
        print(f"    {label:<30}  {verdict}")
    if robust:
        print("    -> Verdict is ROBUST (all four variants agree).")
    else:
        print("    -> Verdict is NOT robust; different knobs give different fit-classes.")
        overall_ok = False
    print()

    print("[STAGE 2] Calibration audit")
    has_cal, notes = _stage2_calibration_audit(args.box)
    for n in notes:
        print(f"    {n}")
    if has_cal:
        from .calibration import load, DEFAULT_CALIBRATION_PATH

        db = load(DEFAULT_CALIBRATION_PATH)
        cal_meshes = {tuple(r.mesh) for r in db.runs if r.box == args.box}
        if mesh in cal_meshes:
            print(f"    -> Target mesh {mesh} IS calibrated.")
        else:
            print(f"    -> Target mesh {mesh} is NOT calibrated. Run:")
            print(f"       python -m scripts.tt_hw_planner calibrate --box {args.box} --mesh {mesh[0]},{mesh[1]}")
    else:
        overall_ok = False
    print()

    target_ok = _run_smoke_stage(
        f"[STAGE 3] Smoke-test {args.model} hot ops on {args.box} mesh {mesh}",
        args.model,
        mesh,
        args.smoke_seq,
        args.skip_hardware,
    )
    if target_ok is False:
        overall_ok = False
    print()

    control_skip = "control would be the same model as the target" if args.control == args.model else None
    control_ok = _run_smoke_stage(
        f"[STAGE 4] Positive control: {args.control} on {args.box} mesh {mesh}",
        args.control,
        mesh,
        args.smoke_seq,
        args.skip_hardware,
        skip_msg=control_skip,
    )
    if control_ok is False:
        overall_ok = False
    print()

    print("=" * width)
    print("  FINAL VERDICT")
    print("=" * width)
    paper = rows[0][1] if rows else "?"
    print(f"  Paper verdict (baseline):  {paper}")
    if target_ok is True:
        print(f"  Op-shape compatibility:    PASS (all hot ops run at the model's shapes)")
    elif target_ok is False:
        print(f"  Op-shape compatibility:    FAIL (some ops do not run at these shapes)")
        print(f"                              -> blocker even on bigger hardware")
    else:
        print(f"  Op-shape compatibility:    not tested")
    if control_ok is True:
        print(f"  Smoke harness sanity:      PASS (control model's ops all run)")
    elif control_ok is False:
        print(f"  Smoke harness sanity:      FAIL — investigate before trusting other results")
    else:
        print(f"  Smoke harness sanity:      not tested")

    print()
    print("  Action:")
    if paper.startswith("FITS"):
        if target_ok:
            print(f"    Plan looks executable. Proceed to a port using mesh {mesh}.")
        elif target_ok is False:
            print(f"    Memory budget fits but some ops fail; fix TTNN before porting.")
        else:
            print(f"    Plan looks fits-on-paper. Re-run with hardware to confirm.")
    elif paper.startswith("NO FIT"):
        print(f"    Model does not fit on {args.box}.  Re-plan with --explore-pp,")
        print(f"    a larger box, or more aggressive quantization to find a viable")
        print(f"    target.  Op-shape result above tells you whether to file a")
        print(f"    TTNN bug for the unsupported shapes.")
    print()

    return 0 if overall_ok else 2


# ---------------------------------------------------------------------------
# Subcommand: show-overhead
# ---------------------------------------------------------------------------


def cmd_show_overhead(args) -> int:
    print("Current per-chip overhead model:")
    print()
    print(f"  {'BOX':<8} {'HBM/CHIP':>9}  {'USABLE':>9}  SOURCE")
    print("  " + "-" * 76)
    for b in HARDWARE:
        usable = b.usable_per_chip_gb(b.chips)
        cal = _CALIBRATED_OVERHEAD.get(b.name)
        source = (
            f"measured ({cal:.2f} GB overhead, from calibration.yaml)"
            if cal is not None
            else f"analytical: {b.overhead.source}"
        )
        print(f"  {b.name:<8} {b.hbm_per_chip_gb:>7.1f}G  {usable:>7.2f}G  {source}")
    print()
    if not _CALIBRATED_OVERHEAD:
        print("  No calibration data yet.  Run:")
        print("    python -m scripts.tt_hw_planner calibrate --box <name> --mesh <r,c>")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Backward-compat: if the first arg looks like an HF model ID (has a
    # slash and isn't a known subcommand), inject `plan`.
    SUBCOMMANDS = {
        "plan",
        "compat",
        "scaffold",
        "prepare",
        "calibrate",
        "smoke-test",
        "validate",
        "list-meshes",
        "show-overhead",
        "-h",
        "--help",
    }
    if argv and argv[0] not in SUBCOMMANDS and ("/" in argv[0] or argv[0].startswith("-")):
        argv = ["plan"] + argv

    parser = argparse.ArgumentParser(
        prog="tt_hw_planner",
        description="Pre-flight memory planner for Tenstorrent hardware.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- plan ----------------------------------------------------------------
    pp = sub.add_parser("plan", help="memory-budget recommendation (default)")
    pp.add_argument("model_id", help="HuggingFace model id, e.g. Qwen/Qwen3-32B")
    pp.add_argument("--batch", type=int, default=1)
    pp.add_argument("--seq", type=int, default=8192)
    pp.add_argument("--kv-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    pp.add_argument("--dtype", action="append", default=[], choices=list(DTYPE_BYTES.keys()))
    pp.add_argument("--box", action="append", default=[], choices=[b.name for b in HARDWARE])
    pp.add_argument(
        "--all-meshes", action="store_true", help="Show every canonical mesh per box, not just the largest TP."
    )
    pp.add_argument("--explore-pp", action="store_true", help="Also enumerate TP×PP combinations (e.g. T3K TP=4,PP=2).")
    pp.add_argument("--format", choices=["table", "json", "markdown"], default="table")
    pp.add_argument("--no-overhead-detail", action="store_true")
    pp.set_defaults(func=cmd_plan)

    # --- compat --------------------------------------------------------------
    pcompat = sub.add_parser(
        "compat",
        help="list which TT building blocks + kernel constraints the model needs",
    )
    pcompat.add_argument("model_id", help="HuggingFace model id, e.g. Qwen/Qwen3-32B")
    pcompat.add_argument("--format", choices=["table", "json"], default="table")
    pcompat.add_argument("--verbose", action="store_true", help="show notes for every block + every kernel finding")
    pcompat.add_argument(
        "--skip-kernel-check",
        action="store_true",
        help="only check building-block availability, not kernel constraints",
    )
    pcompat.add_argument(
        "--tp-grid", type=int, nargs="+", default=None, help="TP values to check for divisibility (default: 1 2 4 8 32)"
    )
    pcompat.set_defaults(func=cmd_compat)

    # --- scaffold ------------------------------------------------------------
    pscaf = sub.add_parser(
        "scaffold",
        help="generate first-draft port (table entries + per-model JSONs) for a READY model",
    )
    pscaf.add_argument("model_id", help="HuggingFace model id of the NEW model to port")
    pscaf.add_argument(
        "--apply",
        action="store_true",
        help="actually write the changes to the working tree (default: dry-run)",
    )
    pscaf.add_argument(
        "--format",
        choices=["text", "patch", "json"],
        default="text",
        help="text: human-readable plan + diff; patch: emit `git apply`-compatible diff; json: structured",
    )
    pscaf.add_argument("--no-diff", action="store_true", help="omit the inline diff in text format")
    pscaf.set_defaults(func=cmd_scaffold)

    # --- prepare -------------------------------------------------------------
    pprep = sub.add_parser(
        "prepare",
        help="emit ready-to-run env + pytest invocation for the recommended box",
    )
    pprep.add_argument("model_id", help="HuggingFace model id, e.g. Qwen/Qwen3-32B")
    pprep.add_argument(
        "--box",
        default=None,
        choices=[b.name for b in HARDWARE],
        help="override the planner's box pick",
    )
    pprep.add_argument("--mesh", default=None, help="override the planner's mesh (requires --box, e.g. 1,4)")
    pprep.add_argument(
        "--dtype",
        default=None,
        choices=list(DTYPE_BYTES.keys()),
        help="override the planner's dtype pick",
    )
    pprep.add_argument("--batch", type=int, default=1, help="pytest --batch_size (default 1)")
    pprep.add_argument("--max-seq-len", type=int, default=1024, help="pytest --max_seq_len (default 1024)")
    pprep.add_argument(
        "--max-generated-tokens", type=int, default=200, help="pytest --max_generated_tokens (default 200)"
    )
    pprep.add_argument(
        "--accuracy",
        action="store_true",
        help="use the accuracy parametrization instead of performance",
    )
    pprep.add_argument("--no-trace", action="store_true", help="disable --enable_trace (slower; needed for accuracy)")
    pprep.add_argument("--no-paged-attention", action="store_true", help="disable paged attention")
    pprep.add_argument("--no-instruct", action="store_true", help="use raw completion path instead of chat template")
    pprep.add_argument("--format", choices=["text", "script", "json"], default="text")
    pprep.add_argument(
        "--write-script",
        default=None,
        metavar="PATH",
        help="also write a self-contained bash script to PATH",
    )
    pprep.add_argument(
        "--execute",
        action="store_true",
        help="run the emitted pytest command in-process (requires a runnable plan)",
    )
    pprep.add_argument(
        "--strict",
        action="store_true",
        help="refuse --execute unless compat is ALREADY SUPPORTED or READY (CI-friendly)",
    )
    pprep.add_argument(
        "--allow-port",
        action="store_true",
        help="DEPRECATED: kept for backward compatibility; permissive execution is now the default",
    )
    pprep.set_defaults(func=cmd_prepare)

    # --- calibrate -----------------------------------------------------------
    pc = sub.add_parser("calibrate", help="measure usable per-chip HBM on hardware")
    pc.add_argument("--box", required=True, choices=[b.name for b in HARDWARE])
    pc.add_argument("--mesh", required=True, help="mesh shape, e.g. 1,4 or 1x4")
    pc.add_argument("--label", default="", help="source label written into calibration.yaml")
    pc.set_defaults(func=cmd_calibrate)

    # --- smoke-test ----------------------------------------------------------
    ps = sub.add_parser("smoke-test", help="run hot ops at the model's shapes on hardware")
    ps.add_argument("--model", required=True, help="HuggingFace model id")
    ps.add_argument("--box", required=True, choices=[b.name for b in HARDWARE])
    ps.add_argument("--mesh", required=True, help="mesh shape, e.g. 1,4")
    ps.add_argument("--batch", type=int, default=1)
    ps.add_argument(
        "--seq", type=int, default=2048, help="probe seq length (default 2048 — kept low for the smoke test)"
    )
    ps.set_defaults(func=cmd_smoke_test)

    # --- validate ------------------------------------------------------------
    pv = sub.add_parser("validate", help="run the full test battery for a (model, box, mesh) triple")
    pv.add_argument("--model", required=True, help="HuggingFace model id to validate")
    pv.add_argument("--box", required=True, choices=[b.name for b in HARDWARE])
    pv.add_argument("--mesh", required=True, help="mesh shape, e.g. 1,4")
    pv.add_argument(
        "--control",
        default="Qwen/Qwen3-1.7B",
        help="known-good model for the smoke-harness sanity check " "(default: Qwen/Qwen3-1.7B)",
    )
    pv.add_argument("--smoke-seq", type=int, default=2048, help="seq length for the smoke probes (default 2048)")
    pv.add_argument("--skip-hardware", action="store_true", help="paper-only validation; skip hardware-touching stages")
    pv.set_defaults(func=cmd_validate)

    # --- list-meshes ---------------------------------------------------------
    pl = sub.add_parser("list-meshes", help="print canonical mesh topology")
    pl.set_defaults(func=cmd_list_meshes)

    # --- show-overhead -------------------------------------------------------
    po = sub.add_parser("show-overhead", help="print active per-chip overhead constants")
    po.set_defaults(func=cmd_show_overhead)

    args = parser.parse_args(argv)
    return args.func(args)

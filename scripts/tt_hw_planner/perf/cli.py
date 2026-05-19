# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt_hw_planner perf` subcommand router.

Entry points:

  collect   <model>            Profile + write artifacts.
  join      <run>              Re-join Tracy + tracer + classify regions.
  report    <run>              Write report.html.
  dashboard <run>              Launch the Dash app.
  status    <run>              Show the block status board.
  blocks    list / show NAME   Inspect the optimizer-block catalog.
  apply     NAME --run R [--cluster CID] [--dry-run]
  revert    NAME --run R [--cluster CID]
  compare   RUN_A RUN_B
  finalize  <run>
  validate  <run>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ..hardware import HARDWARE


def _add_collect(sub) -> None:
    p = sub.add_parser("collect", help="profile a model and stash artifacts under perf-data/<run_id>/")
    p.add_argument("model_id")
    p.add_argument("--box", default=None, choices=[b.name for b in HARDWARE])
    p.add_argument("--mesh", default=None, help="override mesh, e.g. 1,4")
    p.add_argument("--dtype", default=None)
    p.add_argument("--baseline", default=None, help="run id to use as baseline for comparison")
    p.add_argument("--dry-run", action="store_true", help="prepare the command but don't execute it")
    # Auto-retry: when a run fails, the post-mortem may compute concrete retry
    # args (e.g. halve --max-generated-tokens on buffer overflow). The CLI
    # loop honors those by default; opt out for one-shot debugging.
    p.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="maximum auto-retries when the post-mortem proposes concrete retry args (default 2)",
    )
    p.add_argument(
        "--no-auto-retry",
        action="store_true",
        help="disable the auto-retry loop; fail on the first run",
    )
    # Tracy + multi-device + trace-replay + long generation can desync the host
    # op log from the device perf CSV (see _postmortem `trace-replay device
    # data desync`). For a profiling BASELINE we therefore default to:
    #   - --disable_trace at the demo level (every op fires once, easy to correlate)
    #   - a shorter --max_generated_tokens (less buffer pressure)
    # The `trace_capturer` optimizer block then proposes turning trace back on
    # and re-measuring. This matches the standard baseline-then-optimize flow.
    trace_grp = p.add_mutually_exclusive_group()
    trace_grp.add_argument(
        "--no-trace",
        dest="trace",
        action="store_false",
        help="run the demo with --disable_trace (default for the baseline collection)",
    )
    trace_grp.add_argument(
        "--trace",
        dest="trace",
        action="store_true",
        help="run the demo with --enable_trace (use only after the baseline collection passed)",
    )
    p.set_defaults(trace=False)
    p.add_argument(
        "--max-generated-tokens",
        type=int,
        default=8,
        help="generated tokens for the profiled run (default 8). The Metal device-side "
        "profiler keeps markers in a per-(device,core,RISC) DRAM ring of 12000 entries; "
        "we ALSO enable mid-run dump to flush the ring, but a small token count is the "
        "safety belt. Raise it once you've validated the run completes.",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="prefill length for the profiled run (default 128, down from the demo's "
        "1024). On large models (>=7B at full mesh) the on-chip 12000-marker ring fills "
        "rapidly during prefill; a shorter prefill is the single biggest knob for fitting "
        "the whole run in the buffer. Raise it once you've validated the run completes.",
    )
    p.set_defaults(func=_cmd_collect)


def _add_join(sub) -> None:
    p = sub.add_parser("join", help="re-join the Tracy CSV + tracer JSON for a run")
    p.add_argument("run_id")
    p.set_defaults(func=_cmd_join)


def _add_report(sub) -> None:
    p = sub.add_parser("report", help="render report.html for a run")
    p.add_argument("run_id")
    p.add_argument("--baseline", default=None, help="baseline run id for diff overlays")
    p.add_argument("--op-scatter-filter", default="ttnn.matmul")
    p.set_defaults(func=_cmd_report)


def _add_dashboard(sub) -> None:
    p = sub.add_parser("dashboard", help="launch the interactive Dash app for a run")
    p.add_argument("--run", required=True, help="run id")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--debug", action="store_true")
    p.set_defaults(func=_cmd_dashboard)


def _add_status(sub) -> None:
    p = sub.add_parser("status", help="show the optimizer status board")
    p.add_argument("run_id")
    p.set_defaults(func=_cmd_status)


def _add_blocks(sub) -> None:
    p = sub.add_parser("blocks", help="inspect the optimizer-block catalog")
    psub = p.add_subparsers(dest="blocks_cmd", required=True)
    psub.add_parser("list", help="list all blocks")
    pshow = psub.add_parser("show", help="show one block's metadata")
    pshow.add_argument("name")
    p.set_defaults(func=_cmd_blocks)


def _add_apply(sub) -> None:
    p = sub.add_parser("apply", help="apply an optimizer block to a cluster")
    p.add_argument("block_name")
    p.add_argument("--run", required=True)
    p.add_argument("--cluster", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_apply)


def _add_revert(sub) -> None:
    p = sub.add_parser("revert", help="revert a previously applied block (single-file deletion)")
    p.add_argument("block_name")
    p.add_argument("--run", required=True)
    p.add_argument("--cluster", default=None)
    p.set_defaults(func=_cmd_revert)


def _add_compare(sub) -> None:
    p = sub.add_parser("compare", help="diff two runs side-by-side")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.set_defaults(func=_cmd_compare)


def _add_finalize(sub) -> None:
    p = sub.add_parser("finalize", help="convergence gate -> optimized_config.yaml + runner.sh")
    p.add_argument("run_id")
    p.add_argument("--baseline", default=None)
    p.add_argument("--target-tps", type=float, default=None)
    p.add_argument("--target-ratio", type=float, default=0.85)
    p.add_argument("--suppress", default=[], action="append", help="block to mark suppressed; repeatable")
    p.set_defaults(func=_cmd_finalize)


def _add_suggest(sub) -> None:
    p = sub.add_parser(
        "suggest",
        help="propose per-module optimizer blocks by comparing this run against the reference DB",
    )
    p.add_argument("run_id")
    p.add_argument(
        "--arch-family",
        default=None,
        help="override the arch_family used for reference matching "
        "(autodetected from run_meta.json's model_id when omitted)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of the table",
    )
    p.set_defaults(func=_cmd_suggest)


def _add_validate(sub) -> None:
    p = sub.add_parser(
        "validate",
        help="validate that a run has all required artifacts and usable module attribution",
    )
    p.add_argument("run_id")
    p.add_argument(
        "--min-attribution-ratio",
        type=float,
        default=0.25,
        help="minimum fraction of device rows that must have module_path (default 0.25)",
    )
    p.set_defaults(func=_cmd_validate)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _parse_mesh(s: Optional[str]):
    if s is None:
        return None
    parts = s.replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"mesh must be 'rows,cols' (got '{s}')")
    return int(parts[0]), int(parts[1])


def _cmd_collect(args) -> int:
    from .collect import collect_run, CollectError
    from ._postmortem import analyze_log

    mesh = _parse_mesh(args.mesh) if args.mesh else None
    base_kwargs: dict = dict(
        model_id=args.model_id,
        box_override=args.box,
        mesh_override=mesh,
        dtype_override=args.dtype,
        baseline_run_id=args.baseline,
        dry_run=args.dry_run,
        trace=args.trace,
        max_generated_tokens=args.max_generated_tokens,
        max_seq_len=args.max_seq_len,
    )

    max_retries = 0 if args.no_auto_retry else max(0, args.max_retries)
    attempt = 0
    artifacts = None
    last_err: Optional[CollectError] = None
    prev_evidence: Optional[dict] = None
    while True:
        try:
            artifacts = collect_run(**base_kwargs)
            break
        except CollectError as e:
            last_err = e
            # On retries, re-analyze the new log against the PREVIOUS attempt's
            # evidence. This lets the post-mortem detect "no change" and refuse
            # to recommend the same useless retry again. Also pass the mesh
            # that was actually used so retry recommendations are mesh-aware
            # (e.g. "we're already at [1,1], don't bisect further").
            current_mesh_used = base_kwargs.get("mesh_override")
            if attempt > 0 and e.run_dir is not None:
                pm = analyze_log(
                    e.run_dir / "collect.log",
                    prev_evidence=prev_evidence,
                    current_mesh=current_mesh_used,
                )
                e.post_mortem = pm  # overwrite with the comparative analysis
                # Rewrite postmortem.txt with the better diagnosis.
                (e.run_dir / "postmortem.txt").write_text(pm.render())
            else:
                pm = getattr(e, "post_mortem", None)

            retry_args = getattr(pm, "next_retry_args", None) if pm is not None else None
            prev_evidence = getattr(pm, "quantified", None) if pm is not None else None

            if attempt >= max_retries or not retry_args:
                # Out of retries, or framework decided this is not auto-recoverable.
                if pm is not None and retry_args is None and getattr(pm, "retry_explanation", None):
                    print(
                        f"\nAUTO-RETRY HALTED after {attempt} attempt(s): " f"{pm.retry_explanation}\n",
                        file=sys.stderr,
                    )
                break

            # Apply the post-mortem's recommended kwargs and try again.
            print(
                f"\nAUTO-RETRY {attempt + 1}/{max_retries}: post-mortem recommends "
                f"{retry_args}; reason: {getattr(pm, 'retry_explanation', '(none)')}\n",
                file=sys.stderr,
            )
            base_kwargs.update(retry_args)
            attempt += 1
            continue
        except Exception as e:
            print(f"ERROR: perf collect failed: {e}", file=sys.stderr)
            return 2

    if artifacts is None:
        assert last_err is not None
        print(f"ERROR: {last_err}", file=sys.stderr)
        return 2

    if attempt > 0:
        print(f"(succeeded after {attempt} auto-retry attempt(s))")
        # If the last failed attempt was a fabric-timeout bisect, surface the
        # bisect result loudly. The user submitted the run wanting multi-chip
        # data; they got single-chip instead because the link was the problem.
        if last_err is not None and last_err.post_mortem is not None:
            prev_pm = last_err.post_mortem
            if prev_pm.matched_pattern and "Fabric router sync timeout" in prev_pm.matched_pattern:
                final_mesh = base_kwargs.get("mesh_override")
                if final_mesh == (1, 1):
                    stuck = prev_pm.quantified.get("stuck_device") if prev_pm.quantified else "?"
                    print()
                    print("=" * 70)
                    print("BISECT RESULT: model runs on mesh=[1,1]; multi-chip fabric is the problem.")
                    print("=" * 70)
                    print(
                        f"  The model itself is fine on this hardware. Device {stuck} failed "
                        f"the multi-chip ethernet handshake (fabric router sync timeout) "
                        f"before any model op ran."
                    )
                    print()
                    print("  This perf data is for the single-chip mesh. To get multi-chip data:")
                    print("    1. Reset:           tt-smi -r")
                    print("    2. Check live link: tt-smi -s --snapshot_no_tty | rg ETH_LIVE_STATUS")
                    print("                        (any chip with ETH_LIVE_STATUS=0x0 has a dead link;")
                    print("                         no flag in this repo can fix that.)")
                    print("    3. Re-run:          tt_hw_planner perf collect ... --mesh <multi-chip>")
                    print("=" * 70)
    print(f"run_id:  {artifacts.run_id}")
    print(f"run_dir: {artifacts.run_dir}")
    if artifacts.tracy_csv:
        print(f"tracy:   {artifacts.tracy_csv}")
    if artifacts.tracer_master:
        print(f"tracer:  {artifacts.tracer_master}")
    print(f"log:     {artifacts.log}")
    print()
    print("Next:")
    print(f"  tt_hw_planner perf report {artifacts.run_id}")
    return 0


def _cmd_join(args) -> int:
    from .collect import find_run, require_healthy_run
    from .ceilings import load_box_spec
    from .cluster import cluster_rows
    from .join import join_run, write_joined_json
    from .regions import classify_all

    run_dir = find_run(args.run_id)
    meta = require_healthy_run(run_dir)
    box = load_box_spec(meta["box"], tuple(meta["mesh_shape"]))  # type: ignore[arg-type]
    rows = join_run(
        run_id=args.run_id,
        tracy_csv=run_dir / "ops_perf_results.csv" if (run_dir / "ops_perf_results.csv").exists() else None,
        tracer_master=run_dir / "ttnn_operations_master.json"
        if (run_dir / "ttnn_operations_master.json").exists()
        else None,
        num_hidden_layers=meta.get("num_hidden_layers"),
        module_hierarchy=run_dir / "ttnn_module_hierarchy.json"
        if (run_dir / "ttnn_module_hierarchy.json").exists()
        else None,
    )
    cluster_rows(rows)
    classify_all(rows, box)
    out = write_joined_json(rows, run_dir / "joined.json")
    print(f"joined: {out}  ({len(rows)} rows)")
    return 0


def _cmd_report(args) -> int:
    from .report import build_report

    try:
        path = build_report(
            args.run_id,
            baseline_run_id=args.baseline,
            op_scatter_filter=args.op_scatter_filter,
        )
    except Exception as e:
        print(f"ERROR: perf report failed: {e}", file=sys.stderr)
        return 2
    print(f"report: {path}")
    return 0


def _cmd_dashboard(args) -> int:
    try:
        from .dashboard.app import run_dashboard
    except ImportError as e:
        print(f"ERROR: dashboard import failed: {e}", file=sys.stderr)
        return 2
    run_dashboard(run_id=args.run, host=args.host, port=args.port, debug=args.debug)
    return 0


def _cmd_status(args) -> int:
    from .collect import CollectError
    from .status_board import render_status_board

    try:
        print(render_status_board(args.run_id))
    except CollectError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


def _cmd_blocks(args) -> int:
    from .optimizers.catalog import render_blocks_list_text, render_block_show_text

    if args.blocks_cmd == "list":
        print(render_blocks_list_text())
    else:
        print(render_block_show_text(args.name))
    return 0


def _cmd_apply(args) -> int:
    from .runner import apply_block

    try:
        res = apply_block(
            args.block_name,
            run_id=args.run,
            cluster_id=args.cluster,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"ERROR: apply failed: {e}", file=sys.stderr)
        return 2
    if res.patch_path:
        print(f"applied: {res.block} -> {res.patch_path}")
    elif res.dry_run:
        print(f"dry-run: {res.block} would produce {res.findings} finding(s) — {res.rationale}")
    else:
        print(f"no-op:   {res.block} found {res.findings} finding(s) but produced no patches.")
    return 0


def _cmd_revert(args) -> int:
    from .collect import CollectError
    from .runner import revert_block

    try:
        removed = revert_block(args.block_name, run_id=args.run, cluster_id=args.cluster)
    except CollectError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if not removed:
        print(f"no patches to revert for {args.block_name} (cluster={args.cluster})")
    for p in removed:
        print(f"reverted: {p}")
    return 0


def _cmd_compare(args) -> int:
    from .collect import CollectError
    from .report import compare_runs

    try:
        print(compare_runs(args.run_a, args.run_b))
    except CollectError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


def _infer_arch_family(model_id: str) -> str:
    """Cheap heuristic: derive `arch_family` from the HF model id.

    "Qwen/Qwen2.5-7B-Instruct" -> "qwen2"
    "meta-llama/Llama-3.1-8B-Instruct" -> "llama"
    "mistralai/Mistral-7B-Instruct-v0.2" -> "mistral"
    "state-spaces/mamba-2.8b" -> "mamba"

    Mostly the loader part of the id name; biased toward lower-case and
    the first alpha cluster. Falls back to "" so the matcher still tries
    the alias table.
    """
    if not model_id:
        return ""
    tail = model_id.split("/")[-1].lower()
    # Strip versions/numbers from the prefix.
    prefix = ""
    for ch in tail:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return prefix


def _cmd_suggest(args) -> int:
    from .collect import find_run, load_run_meta, require_healthy_run
    from .join import join_run
    from .module_graph import build_module_graph, parse_hierarchy_sidecar
    from .suggestion_engine import propose_optimizations, render_suggestions_text
    import json as _json

    run_dir = find_run(args.run_id)
    try:
        meta = require_healthy_run(run_dir)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    arch_family = args.arch_family or _infer_arch_family(meta.get("model_id", ""))
    hierarchy_path = run_dir / "ttnn_module_hierarchy.json"
    if not hierarchy_path.exists():
        print(
            "ERROR: this run has no ttnn_module_hierarchy.json sidecar.\n"
            "  Per-module suggestions require the sidecar; re-run `perf collect`\n"
            "  on a tt-metal version that includes the module_hierarchy patch.",
            file=sys.stderr,
        )
        return 2

    rows = join_run(
        run_id=args.run_id,
        tracy_csv=run_dir / "ops_perf_results.csv" if (run_dir / "ops_perf_results.csv").exists() else None,
        tracer_master=run_dir / "ttnn_operations_master.json"
        if (run_dir / "ttnn_operations_master.json").exists()
        else None,
        num_hidden_layers=meta.get("num_hidden_layers"),
        module_hierarchy=hierarchy_path,
    )
    hierarchy = parse_hierarchy_sidecar(hierarchy_path)
    graph = build_module_graph(rows, hierarchy)
    if not graph.nodes:
        print(
            "ERROR: module graph is empty. The sidecar exists but no op-counter\n"
            "  attributions were recorded. This usually means the hooks failed\n"
            "  to install (check collect.log for 'module_hierarchy' warnings).",
            file=sys.stderr,
        )
        return 2

    suggestions = propose_optimizations(
        graph=graph,
        arch_family=arch_family,
        mesh_shape=tuple(meta.get("mesh_shape", (0, 0))),  # type: ignore[arg-type]
        dtype=meta.get("dtype"),
        box=meta.get("box"),
    )

    if args.json:
        print(_json.dumps([s.to_dict() for s in suggestions], indent=2, default=str))
    else:
        print(f"perf suggest — run {args.run_id} (arch_family={arch_family})")
        print(f"  model:  {meta.get('model_id')}")
        print(f"  box:    {meta.get('box')}   mesh: {meta.get('mesh_shape')}   dtype: {meta.get('dtype')}")
        print()
        print(render_suggestions_text(suggestions))
    return 0


def _cmd_validate(args) -> int:
    from .collect import find_run, require_healthy_run
    from .join import join_run
    from .module_graph import build_module_graph, parse_hierarchy_sidecar

    run_dir = find_run(args.run_id)
    try:
        meta = require_healthy_run(run_dir)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    required = {
        "ops_perf_results.csv": (run_dir / "ops_perf_results.csv"),
        "ttnn_operations_master.json": (run_dir / "ttnn_operations_master.json"),
        "ttnn_module_hierarchy.json": (run_dir / "ttnn_module_hierarchy.json"),
    }
    missing = [name for name, p in required.items() if not p.exists()]

    rows = join_run(
        run_id=args.run_id,
        tracy_csv=required["ops_perf_results.csv"] if required["ops_perf_results.csv"].exists() else None,
        tracer_master=required["ttnn_operations_master.json"]
        if required["ttnn_operations_master.json"].exists()
        else None,
        num_hidden_layers=meta.get("num_hidden_layers"),
        module_hierarchy=required["ttnn_module_hierarchy.json"]
        if required["ttnn_module_hierarchy.json"].exists()
        else None,
    )

    device_rows = len(rows)
    attributed_rows = sum(1 for r in rows if r.module_path)
    attr_ratio = (attributed_rows / device_rows) if device_rows else 0.0

    graph_nodes = 0
    if required["ttnn_module_hierarchy.json"].exists():
        graph_nodes = len(
            build_module_graph(rows, parse_hierarchy_sidecar(required["ttnn_module_hierarchy.json"])).nodes
        )

    print(f"run:        {args.run_id}")
    print(f"model:      {meta.get('model_id')}")
    print(f"box/mesh:   {meta.get('box')} / {meta.get('mesh_shape')}")
    print("artifacts:")
    for name, p in required.items():
        print(f"  - {name}: {'OK' if p.exists() else 'MISSING'}")
    print(f"joined rows:                 {device_rows}")
    print(f"rows with module attribution:{attributed_rows}")
    print(f"attribution ratio:           {attr_ratio:.3f}")
    print(f"module graph nodes:          {graph_nodes}")

    if missing:
        print("\nFAIL: required artifacts missing:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(
            "Hint: rerun `tt_hw_planner perf collect ...` and check collect.log for model_tracer/plugin errors.",
            file=sys.stderr,
        )
        return 2
    if attr_ratio < args.min_attribution_ratio:
        print(
            f"\nFAIL: attribution ratio {attr_ratio:.3f} < threshold {args.min_attribution_ratio:.3f}",
            file=sys.stderr,
        )
        print(
            "Hint: sidecar exists but attribution is sparse/empty; check collect.log for "
            "`module_hierarchy` install/export warnings and tracer output.",
            file=sys.stderr,
        )
        return 2
    if graph_nodes == 0:
        print(
            "\nFAIL: no module graph nodes could be built from this run.",
            file=sys.stderr,
        )
        print(
            "Hint: verify sidecar `op_module_log` is populated and join rows have module_path values.",
            file=sys.stderr,
        )
        return 2

    print("\nPASS: run artifacts and module attribution look healthy.")
    return 0


def _cmd_finalize(args) -> int:
    from .finalize import finalize_run

    try:
        result = finalize_run(
            args.run_id,
            baseline_run_id=args.baseline,
            target_tps=args.target_tps,
            target_ratio=args.target_ratio,
            suppressed=args.suppress,
        )
    except Exception as e:
        print(f"ERROR: finalize failed: {e}", file=sys.stderr)
        return 2
    print(result.summary)
    return 0 if result.passed else 2


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tt_hw_planner perf", description="perf subcommand router")
    sub = p.add_subparsers(dest="perf_cmd", required=True)
    _add_collect(sub)
    _add_join(sub)
    _add_report(sub)
    _add_dashboard(sub)
    _add_status(sub)
    _add_blocks(sub)
    _add_apply(sub)
    _add_revert(sub)
    _add_compare(sub)
    _add_finalize(sub)
    _add_suggest(sub)
    _add_validate(sub)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

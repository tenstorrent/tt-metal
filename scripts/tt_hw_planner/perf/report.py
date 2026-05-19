# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""High-level `perf report` entry point.

Ties together: load_run + join + cluster + classify regions + build all
eight chart figures + write a self-contained report.html.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .ceilings import BoxSpec, load_box_spec
from .charts.block_stack import make_block_stack
from .charts.block_util_runtime import make_block_util_runtime
from .charts.cache_heatmap import make_cache_heatmap
from .charts.config_scatter import make_config_scatter
from .charts.hash_diff import make_hash_diff
from .charts.risc_stack import make_risc_stack
from .charts.roofline import make_roofline
from .charts.sol_bars import make_sol_bars
from .charts.waterfall import make_waterfall
from .cluster import Cluster, cluster_rows
from .collect import find_run, load_run_meta, require_healthy_run
from .compare import biggest_movers, render_compare_summary
from .join import JoinedRow, join_run, write_joined_json
from .regions import classify_all
from .report_html import ReportContext, write_report
from .dashboard.timeline import make_nsight_timeline


def _load(run_dir: Path, run_id: str) -> Tuple[List[JoinedRow], List[Cluster], BoxSpec, dict]:
    meta = require_healthy_run(run_dir)
    box = load_box_spec(meta["box"], tuple(meta["mesh_shape"]))  # type: ignore[arg-type]
    rows = join_run(
        run_id=run_id,
        tracy_csv=run_dir / "ops_perf_results.csv" if (run_dir / "ops_perf_results.csv").exists() else None,
        tracer_master=run_dir / "ttnn_operations_master.json"
        if (run_dir / "ttnn_operations_master.json").exists()
        else None,
        num_hidden_layers=meta.get("num_hidden_layers"),
        module_hierarchy=run_dir / "ttnn_module_hierarchy.json"
        if (run_dir / "ttnn_module_hierarchy.json").exists()
        else None,
    )
    clusters = cluster_rows(rows)
    classify_all(rows, box)
    return rows, clusters, box, meta


def build_report(
    run_id: str,
    *,
    baseline_run_id: Optional[str] = None,
    catalog_entries: Optional[List[Tuple[str, int, str]]] = None,
    op_scatter_filter: str = "ttnn.matmul",
    run_dir_root: Optional[Path] = None,
) -> Path:
    """Build the full HTML report for one run (with optional baseline)."""
    run_dir = find_run(run_id, run_dir_root)
    rows, clusters, box, meta = _load(run_dir, run_id)
    write_joined_json(rows, run_dir / "joined.json")

    baseline_rows: List[JoinedRow] = []
    baseline_clusters: List[Cluster] = []
    if baseline_run_id:
        try:
            bdir = find_run(baseline_run_id, run_dir_root)
            baseline_rows, baseline_clusters, _, _ = _load(bdir, baseline_run_id)
        except FileNotFoundError:
            pass

    if catalog_entries is None:
        # Import lazily so the chart layer doesn't pay the catalog import cost.
        from .optimizers.catalog import catalog_for_sidebar

        catalog_entries = catalog_for_sidebar()

    ctx = ReportContext(
        run_id=run_id,
        run_dir=run_dir,
        model_id=meta["model_id"],
        box=box,
        rows=rows,
        clusters=clusters,
        baseline_run_id=baseline_run_id,
        baseline_rows=baseline_rows or None,
        baseline_clusters=baseline_clusters or None,
        extra_meta=meta,
    )

    # Put the per-block utilization-vs-runtime chart FIRST: that's the
    # deliverable Mohamed asked for. The rest are supporting detail.
    charts = {
        "nsight_timeline": make_nsight_timeline(rows, baseline_rows=baseline_rows or None),
        "per_block_util_vs_runtime": make_block_util_runtime(
            rows, baseline_rows=baseline_rows or None, util_axis="FPU"
        ),
        "per_block_dram": make_block_util_runtime(rows, baseline_rows=baseline_rows or None, util_axis="DRAM"),
        "roofline": make_roofline(rows, clusters, box),
        "speed_of_light": make_sol_bars(clusters),
        "per_risc": make_risc_stack(rows, clusters),
        "per_block": make_block_stack(rows, clusters),
        "config_scatter": make_config_scatter(clusters, op_filter=op_scatter_filter),
        "waterfall": make_waterfall(rows),
        "cache_heatmap": make_cache_heatmap(rows),
        "hash_diff": make_hash_diff(rows, clusters, baseline_rows or None),
    }

    path = write_report(ctx, charts, catalog_entries)
    return path


def compare_runs(run_a: str, run_b: str, run_dir_root: Optional[Path] = None) -> str:
    """Plain-text comparison between two run ids (for the `perf compare` CLI)."""
    da = find_run(run_a, run_dir_root)
    db = find_run(run_b, run_dir_root)
    rows_a, clusters_a, _, _ = _load(da, run_a)
    rows_b, clusters_b, _, _ = _load(db, run_b)
    return render_compare_summary(clusters_a, clusters_b, run_a, run_b)

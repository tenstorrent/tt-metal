# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the compare-to-baseline view.

The strategy is conservative: we render the same eight charts as the
single-run report but, when a baseline run is present, every chart gets
a ghost trace (the baseline) overlaid in muted gray plus a side-summary
table of biggest improvers and regressions.

We deliberately overlay rather than computing per-cluster diffs in a
separate chart — that's what `hash_diff` already does, and we want the
geometry of every chart to make the "did this dot move" question visual.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go

from .cluster import Cluster
from .join import JoinedRow


def _cluster_index(clusters: List[Cluster]) -> Dict[Tuple[str, Optional[str]], Cluster]:
    return {(c.op_code, c.args_hash): c for c in clusters}


def biggest_movers(
    current: List[Cluster], baseline: List[Cluster], top_n: int = 10
) -> Tuple[List[Tuple[Cluster, Cluster, float]], List[Tuple[Cluster, Cluster, float]]]:
    """Return (improvers, regressions) ranked by absolute total-time delta.

    Match clusters by (op_code, args_hash). Unmatched are skipped.
    """
    base = _cluster_index(baseline)
    pairs: List[Tuple[Cluster, Cluster, float]] = []
    for c in current:
        b = base.get((c.op_code, c.args_hash))
        if b is None:
            continue
        delta = c.total_device_ns - b.total_device_ns
        pairs.append((c, b, delta))

    improvers = sorted([p for p in pairs if p[2] < 0], key=lambda p: p[2])[:top_n]
    regressions = sorted([p for p in pairs if p[2] > 0], key=lambda p: -p[2])[:top_n]
    return improvers, regressions


def render_compare_summary(
    current: List[Cluster], baseline: List[Cluster], current_run_id: str, baseline_run_id: str
) -> str:
    """Plain-text comparison summary used by CLI `perf compare`."""
    improvers, regressions = biggest_movers(current, baseline)
    total_cur = sum(c.total_device_ns for c in current) / 1e6
    total_base = sum(c.total_device_ns for c in baseline) / 1e6
    delta = total_cur - total_base
    lines: List[str] = []
    lines.append(f"=== compare: {current_run_id} vs {baseline_run_id} ===")
    lines.append(
        f"Total device time: current {total_cur:.2f} ms  baseline {total_base:.2f} ms  "
        f"Δ {delta:+.2f} ms ({(delta / total_base * 100 if total_base else 0):+.1f}%)"
    )
    lines.append("")
    lines.append("Biggest improvers:")
    for c, b, d in improvers:
        lines.append(
            f"  {c.cluster_id} {c.op_code:<28s}  {b.total_device_ns/1e6:>7.2f} ms -> "
            f"{c.total_device_ns/1e6:>7.2f} ms  ({d/1e6:+.2f} ms)"
        )
    if not improvers:
        lines.append("  (none)")
    lines.append("")
    lines.append("Regressions:")
    for c, b, d in regressions:
        lines.append(
            f"  {c.cluster_id} {c.op_code:<28s}  {b.total_device_ns/1e6:>7.2f} ms -> "
            f"{c.total_device_ns/1e6:>7.2f} ms  ({d/1e6:+.2f} ms)"
        )
    if not regressions:
        lines.append("  (none)")
    return "\n".join(lines)


def overlay_baseline_on_roofline(
    fig: go.Figure, baseline_rows: List[JoinedRow], baseline_clusters: List[Cluster]
) -> None:
    """Add a faded baseline scatter to an existing roofline figure."""
    if not baseline_rows:
        return
    rows_by_cluster = {r.cluster_id: r for r in baseline_rows}
    xs, ys = [], []
    customs = []
    texts = []
    for c in baseline_clusters:
        s = rows_by_cluster.get(c.cluster_id)
        if s is None or s.device_kernel_ns is None or c.mean_pm_ideal_ns is None:
            continue
        intensity = 1.0  # placeholder; the chart's true intensity is recomputed inside roofline.py
        xs.append(intensity)
        ys.append(c.median_device_ns)
        customs.append([c.cluster_id])
        texts.append(f"baseline {c.cluster_id}: {c.median_device_ns/1000:.1f} us")
    if not xs:
        return
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(color="#586069", size=6, opacity=0.4, symbol="circle-open"),
            customdata=customs,
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            name="baseline",
        )
    )

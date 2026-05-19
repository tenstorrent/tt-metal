# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline chart (chart 1).

Each cluster is one point: X = arithmetic intensity (FLOPs / off-chip byte),
Y = achieved FLOPS. Ceiling lines from `ceilings.py` are overlaid. Points
are colored by region label (from regions.py) and sized by total time
contribution (`device_kernel_ns * n_calls`).

We approximate arithmetic intensity from Tracy's PM_REQ_I_BW / PM_REQ_O_BW
when available, otherwise from PM_IDEAL + PM_COMPUTE. We compute FLOPS by
inferring an effective FLOP count from PM_COMPUTE * peak FLOPS at the
op's fidelity (when present) — this gives a stable cross-cluster scaling
even though we don't recompute per-op FLOPs from shape directly.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import plotly.graph_objects as go

from ..ceilings import BoxSpec, CeilingLine, ceilings_for
from ..cluster import Cluster
from ..join import JoinedRow
from ..regions import REGIONS
from .theme import REGION_COLORS, color_for_op


def _cluster_intensity_and_flops(c: Cluster, box: BoxSpec) -> Optional[Tuple[float, float]]:
    """Return (intensity, achieved_flops) or None if we can't compute.

    Intensity is a relative number: PM_COMPUTE / PM_BANDWIDTH is the
    natural Tracy roofline coordinate (ns of compute per ns of bandwidth)
    multiplied by the BW ceiling to put it on the same axis as bytes.
    achieved_flops = peak_flops_for_fidelity * (PM_COMPUTE / device_ns).
    """
    if c.mean_pm_ideal_ns is None or c.median_device_ns <= 0:
        return None
    peak = box.fpu_peak_flops_per_chip(c.math_fidelity or "HiFi2") * box.total_chips
    if peak <= 0:
        return None
    achieved = peak * (c.mean_pm_ideal_ns / c.median_device_ns)
    intensity = peak / max(box.dram_bw_bytes_per_s(), 1.0)
    if intensity <= 0:
        return None
    return intensity, achieved


def _intensity_axis(box: BoxSpec) -> Tuple[float, float]:
    low = 0.01
    high = max(
        2.0 * box.fpu_peak_flops_per_chip("LoFi") * box.total_chips / max(box.dram_bw_bytes_per_s(), 1.0), 1024.0
    )
    return low, high


def _ceiling_trace(line: CeilingLine, x_lo: float, x_hi: float, color: str) -> go.Scatter:
    if line.flops_per_s is not None:
        xs = [x_lo, x_hi]
        ys = [line.flops_per_s, line.flops_per_s]
    else:
        xs = [x_lo, x_hi]
        ys = [line.y_at(x_lo), line.y_at(x_hi)]
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=color, width=1.5, dash="dot"),
        name=line.label,
        hoverinfo="name",
        showlegend=True,
    )


def _bottleneck(row: JoinedRow) -> str:
    candidates = [
        ("FPU", row.pm_fpu_util_pct),
        ("DRAM", row.dram_bw_util_pct),
        ("NoC", row.noc_util_pct),
        ("ETH", row.eth_bw_util_pct),
    ]
    best = max(candidates, key=lambda kv: kv[1] or 0)
    if best[1] is None or best[1] <= 0:
        return "dispatch"
    return best[0]


def make_roofline(rows: List[JoinedRow], clusters: List[Cluster], box: BoxSpec) -> go.Figure:
    fig = go.Figure()
    x_lo, x_hi = _intensity_axis(box)

    # Diagonal BW ceilings first (drawn under FPU horizontals).
    bw_colors = {"dram": "#f0883e", "noc": "#a371f7", "eth": "#d29922"}
    fpu_colors = ["#56d364", "#7ee787", "#3fb950", "#2ea043"]
    fpu_idx = 0
    for line in ceilings_for(box):
        if line.kind == "fpu":
            color = fpu_colors[fpu_idx % len(fpu_colors)]
            fpu_idx += 1
        else:
            color = bw_colors.get(line.kind, "#8b949e")
        fig.add_trace(_ceiling_trace(line, x_lo, x_hi, color))

    # Per-region scatter traces (so legend has one entry per region).
    rows_by_cluster = {r.cluster_id: r for r in rows}
    by_region: dict = {label: dict(x=[], y=[], size=[], text=[], custom=[], color=[]) for label in REGIONS}
    for c in clusters:
        sample = rows_by_cluster.get(c.cluster_id)
        if sample is None:
            continue
        coord = _cluster_intensity_and_flops(c, box)
        if coord is None:
            continue
        intensity, flops = coord
        region = sample.region or "?"
        bucket = by_region[region]
        bucket["x"].append(intensity)
        bucket["y"].append(flops)
        bucket["size"].append(math.sqrt(max(c.total_device_ns, 1.0)))
        bucket["custom"].append([c.cluster_id])
        tooltip = (
            f"<b>{c.op_code}</b><br>cluster {c.cluster_id}<br>"
            f"shape: {c.shape_signature[:80]}<br>"
            f"median {c.median_device_ns/1000:.1f} us · {c.n_calls} calls<br>"
            f"region <b>{region}</b> — {sample.region_reason or ''}<br>"
            f"bottleneck: {_bottleneck(sample)}"
        )
        bucket["text"].append(tooltip)
        bucket["color"].append(color_for_op(c.op_code))

    # Plot each region as a separate trace so the legend tells the story.
    for region, data in by_region.items():
        if not data["x"]:
            continue
        sizes = data["size"]
        max_s = max(sizes) if sizes else 1.0
        scaled = [6 + 24 * (s / max_s) for s in sizes]
        info = REGIONS[region]
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="markers",
                marker=dict(
                    color=REGION_COLORS.get(region, "#8b949e"),
                    size=scaled,
                    line=dict(color="#0d1117", width=0.5),
                    opacity=0.85,
                ),
                customdata=data["custom"],
                hovertemplate="%{text}<extra></extra>",
                text=data["text"],
                name=f"{region} — {info.name}",
            )
        )

    fig.update_xaxes(
        type="log",
        title="Arithmetic intensity (FLOPs / off-chip byte)",
        range=[math.log10(x_lo), math.log10(x_hi)],
    )
    fig.update_yaxes(
        type="log",
        title="Achieved FLOPS",
    )
    fig.update_layout(
        title="Roofline — click any point to inspect; ceilings are dotted",
        legend=dict(orientation="v", x=1.02, y=1),
        height=600,
    )
    return fig

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Speed-of-Light bars (chart 2).

For each top-N cluster, four horizontal bars: FPU / DRAM / NoC / ETH
utilization as % of peak. The tallest is highlighted in red so users
immediately see the active bottleneck. Mirrors Nsight Compute's SoL chart.
"""

from __future__ import annotations

from typing import List

import plotly.graph_objects as go

from ..cluster import Cluster


def make_sol_bars(clusters: List[Cluster], top_n: int = 12) -> go.Figure:
    fig = go.Figure()
    if not clusters:
        fig.add_annotation(
            text="No clusters available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8b949e"),
        )
        return fig

    top = clusters[:top_n]
    kinds = ["FPU", "DRAM", "NoC", "ETH"]
    kind_color = {"FPU": "#56d364", "DRAM": "#f0883e", "NoC": "#a371f7", "ETH": "#d29922"}

    for kind in kinds:
        attr = {
            "FPU": "mean_fpu_util_pct",
            "DRAM": "mean_dram_bw_util_pct",
            "NoC": "mean_noc_util_pct",
            "ETH": "mean_eth_bw_util_pct",
        }[kind]
        xs = []
        ys = []
        colors = []
        customs = []
        for c in top:
            v = getattr(c, attr) or 0
            xs.append(v)
            ys.append(f"{c.cluster_id}  {c.op_code}")
            customs.append([c.cluster_id])
            # color: kind's color, but if this kind is the bottleneck for
            # this cluster, paint it red.
            kind_vals = [
                ("FPU", c.mean_fpu_util_pct or 0),
                ("DRAM", c.mean_dram_bw_util_pct or 0),
                ("NoC", c.mean_noc_util_pct or 0),
                ("ETH", c.mean_eth_bw_util_pct or 0),
            ]
            tallest = max(kind_vals, key=lambda kv: kv[1])[0]
            colors.append("#F85149" if tallest == kind and v > 5 else kind_color[kind])

        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys,
                orientation="h",
                name=kind + " util",
                marker=dict(color=colors),
                customdata=customs,
                hovertemplate="<b>%{y}</b><br>" + kind + " util: %{x:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="group",
        title="Speed-of-Light bars — tallest red bar = active bottleneck",
        height=max(360, 24 * len(top) + 120),
        xaxis=dict(title="% of peak", range=[0, 100]),
        legend=dict(orientation="v", x=1.02, y=1),
    )
    return fig

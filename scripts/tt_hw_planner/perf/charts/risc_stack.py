# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-RISC stacked bars (chart 3).

For each top-N cluster, decompose the device kernel time across the six
RISCs (BRISC/NCRISC/TRISC0-2/ERISC). The longest segment is the long
pole; users use this to know whether data movement (BRISC/NCRISC) or
compute (TRISC0-2) dominates inside a slow kernel.
"""

from __future__ import annotations

from typing import List

import plotly.graph_objects as go

from ..cluster import Cluster
from ..join import JoinedRow


RISC_KINDS = [
    ("BRISC", "brisc_ns", "#58a6ff"),
    ("NCRISC", "ncrisc_ns", "#1f6feb"),
    ("TRISC0", "trisc0_ns", "#56d364"),
    ("TRISC1", "trisc1_ns", "#3fb950"),
    ("TRISC2", "trisc2_ns", "#2ea043"),
    ("ERISC", "erisc_ns", "#d29922"),
]


def make_risc_stack(rows: List[JoinedRow], clusters: List[Cluster], top_n: int = 12) -> go.Figure:
    fig = go.Figure()
    rows_by_cluster = {r.cluster_id: r for r in rows}

    top = clusters[:top_n]
    if not top:
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

    labels = [f"{c.cluster_id}  {c.op_code}" for c in top]
    samples = [rows_by_cluster.get(c.cluster_id) for c in top]
    customs = [[c.cluster_id] for c in top]

    for name, attr, color in RISC_KINDS:
        xs = [(getattr(s, attr) or 0) / 1e3 if s else 0 for s in samples]
        fig.add_trace(
            go.Bar(
                x=xs,
                y=labels,
                orientation="h",
                name=name,
                marker=dict(color=color),
                customdata=customs,
                hovertemplate=f"<b>{name}</b><br>%{{x:.2f}} us<br>%{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Per-RISC kernel time breakdown (us)",
        height=max(360, 24 * len(top) + 120),
        xaxis=dict(title="us"),
        legend=dict(orientation="v", x=1.02, y=1),
    )
    return fig

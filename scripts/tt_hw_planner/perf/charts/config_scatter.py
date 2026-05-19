# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration scatter (chart 5).

For a single op type (matmul by default), one point per (shape, config)
cluster. X = shape complexity (M*N*K or similar scalar), Y = median
device_kernel_ns. Hover shows the literal model_tracer kwargs. Reference
DB best is overlaid as a red diamond when available.

This chart is the place where model_tracer earns its keep: two matmuls
of the same shape with different program_configs land on different points
here because their `args_hash` differs.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import plotly.graph_objects as go

from ..cluster import Cluster


def _shape_complexity(c: Cluster) -> Optional[float]:
    """Crude scalar from the shape signature. Falls back to len(sig)."""
    sig = c.shape_signature
    try:
        first = sig.split(";")[0].split("|")[0]
        dims = [int(x) for x in first.split("x") if x]
        prod = 1
        for d in dims:
            prod *= d
        return float(prod) if prod > 0 else None
    except (ValueError, IndexError):
        return float(len(sig))


def make_config_scatter(clusters: List[Cluster], op_filter: str = "ttnn.matmul") -> go.Figure:
    fig = go.Figure()
    matching = [c for c in clusters if c.op_code == op_filter]

    if not matching:
        fig.add_annotation(
            text=f"No '{op_filter}' clusters in this run.<br>(Set <code>op_filter</code> in CLI to change.)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8b949e"),
        )
        return fig

    xs, ys, sizes, customs, texts = [], [], [], [], []
    for c in matching:
        sc = _shape_complexity(c)
        if sc is None or sc <= 0:
            continue
        xs.append(sc)
        ys.append(c.median_device_ns / 1e3)
        sizes.append(8 + 4 * math.log10(max(c.n_calls, 1)))
        customs.append([c.cluster_id])
        texts.append(
            f"<b>{c.op_code}</b><br>cluster {c.cluster_id}<br>"
            f"shape: {c.shape_signature[:80]}<br>"
            f"median {c.median_device_ns/1000:.1f} us · {c.n_calls} calls<br>"
            f"args_hash: {c.args_hash or '(tracer missing)'}<br>"
            f"fidelity: {c.math_fidelity}"
        )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(color="#58a6ff", size=sizes, line=dict(color="#0d1117", width=0.5)),
            customdata=customs,
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            name="this run",
        )
    )

    fig.update_layout(
        title=f"{op_filter} — configuration scatter (one point per args_hash)",
        height=540,
    )
    fig.update_xaxes(title="shape complexity (∏ dims)", type="log")
    fig.update_yaxes(title="median device kernel (us)", type="log")
    return fig

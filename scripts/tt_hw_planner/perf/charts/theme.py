# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Plotly templates + shared color palette for the perf report.

We define two themes (dark for the report.html default, light for slides
or screenshots). The light theme is intentionally minimal so cluster
points stay readable.
"""

from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go
import plotly.io as pio


# Op-code -> color, used everywhere consistent op coloring is desired.
OP_PALETTE: Dict[str, str] = {
    "ttnn.matmul": "#58a6ff",
    "ttnn.linear": "#58a6ff",
    "ttnn.add": "#56d364",
    "ttnn.mul": "#56d364",
    "ttnn.layer_norm": "#bc8cff",
    "ttnn.rms_norm": "#bc8cff",
    "ttnn.softmax": "#f0883e",
    "ttnn.attention": "#f85149",
    "ttnn.scaled_dot_product_attention": "#f85149",
    "ttnn.all_gather": "#d29922",
    "ttnn.reduce_scatter": "#d29922",
    "ttnn.reshard": "#a371f7",
    "ttnn.tilize": "#7d8590",
    "ttnn.untilize": "#7d8590",
    "ttnn.embedding": "#3fb950",
}
DEFAULT_OP_COLOR = "#8b949e"


def color_for_op(op_code: str) -> str:
    return OP_PALETTE.get(op_code, DEFAULT_OP_COLOR)


# Region overlay colors (mirrored from regions.py for chart reuse).
REGION_COLORS: Dict[str, str] = {
    "A": "#2EA043",
    "B": "#FFD93D",
    "C": "#FFA630",
    "D": "#1F6FEB",
    "E": "#F85149",
    "F": "#8B949E",
    "?": "#586069",
}


THEME_DARK = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", size=12, color="#c9d1d9"),
        title=dict(font=dict(size=14, color="#f0f6fc")),
        xaxis=dict(
            gridcolor="#30363d",
            zerolinecolor="#30363d",
            linecolor="#30363d",
            tickcolor="#30363d",
            title=dict(font=dict(size=12, color="#8b949e")),
        ),
        yaxis=dict(
            gridcolor="#30363d",
            zerolinecolor="#30363d",
            linecolor="#30363d",
            tickcolor="#30363d",
            title=dict(font=dict(size=12, color="#8b949e")),
        ),
        legend=dict(bgcolor="rgba(13,17,23,0.85)", bordercolor="#30363d", borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(
            bgcolor="#161b22", bordercolor="#30363d", font=dict(family="JetBrains Mono, monospace", size=11)
        ),
        margin=dict(l=60, r=20, t=40, b=60),
    )
)


THEME_LIGHT = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f6f8fa",
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", size=12, color="#1f2328"),
        title=dict(font=dict(size=14, color="#1f2328")),
        xaxis=dict(gridcolor="#d0d7de", zerolinecolor="#d0d7de", linecolor="#d0d7de", tickcolor="#d0d7de"),
        yaxis=dict(gridcolor="#d0d7de", zerolinecolor="#d0d7de", linecolor="#d0d7de", tickcolor="#d0d7de"),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#d0d7de", borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor="#d0d7de", font=dict(family="JetBrains Mono, monospace", size=11)
        ),
        margin=dict(l=60, r=20, t=40, b=60),
    )
)


pio.templates["tt_perf_dark"] = THEME_DARK
pio.templates["tt_perf_light"] = THEME_LIGHT


def apply_theme(fig: go.Figure, dark: bool = True) -> go.Figure:
    """Apply our theme to an existing figure (in-place + return)."""
    fig.update_layout(template="tt_perf_dark" if dark else "tt_perf_light")
    return fig

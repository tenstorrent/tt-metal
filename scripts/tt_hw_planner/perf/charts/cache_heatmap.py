# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Program-cache heatmap (chart 7).

Heatmap of (block_path × op_code) -> fraction of calls that hit the
program cache. Cold cells (low hit rate) point the user at the
`cache_warmer` and `trace_capturer` blocks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import plotly.graph_objects as go

from ..join import JoinedRow


def make_cache_heatmap(rows: List[JoinedRow]) -> go.Figure:
    cells: Dict[Tuple[str, str], Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in rows:
        if r.program_cache_hit is None:
            continue
        k = (r.block_path, r.op_code or "?")
        hits, total = cells[k]
        cells[k] = (hits + (1 if r.program_cache_hit else 0), total + 1)

    fig = go.Figure()
    if not cells:
        fig.add_annotation(
            text="No PROGRAM CACHE HIT data in this run.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8b949e"),
        )
        return fig

    blocks = sorted({k[0] for k in cells.keys()})
    ops = sorted({k[1] for k in cells.keys()})

    z = []
    text = []
    for b in blocks:
        row = []
        trow = []
        for o in ops:
            hits, total = cells.get((b, o), (0, 0))
            rate = (hits / total * 100.0) if total else None
            row.append(rate)
            trow.append(f"{hits}/{total}" if total else "")
        z.append(row)
        text.append(trow)

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=ops,
            y=blocks,
            colorscale=[[0, "#F85149"], [0.5, "#FFA630"], [0.9, "#FFD93D"], [1, "#2EA043"]],
            zmin=0,
            zmax=100,
            text=text,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b> · %{x}<br>hit rate: %{z:.0f}%<extra></extra>",
            colorbar=dict(title="hit %"),
        )
    )
    fig.update_layout(
        title="Program cache hit rate per (block × op)",
        height=max(360, 22 * len(blocks) + 120),
    )
    return fig

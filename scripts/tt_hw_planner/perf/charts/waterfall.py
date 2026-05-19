# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Op-to-op waterfall (chart 6).

Timeline view: each op is a horizontal bar from t_start to t_start +
kernel_ns. Gaps between bars are op-to-op latency (host/dispatch). Long
gaps relative to kernel time imply dispatch-bound (region F).
"""

from __future__ import annotations

from typing import List

import plotly.graph_objects as go

from ..join import JoinedRow
from .theme import color_for_op


def make_waterfall(rows: List[JoinedRow], top_n: int = 200) -> go.Figure:
    fig = go.Figure()
    if not rows:
        return fig

    # Reconstruct an approximate t_start from cumulative (op_to_op + kernel) ns.
    work_rows = [r for r in rows if r.device_kernel_ns is not None][:top_n]
    t = 0.0
    intervals = []
    for r in work_rows:
        gap = r.op_to_op_latency_ns or 0
        t += gap
        intervals.append((t, t + (r.device_kernel_ns or 0), gap, r))
        t += r.device_kernel_ns or 0

    if not intervals:
        return fig

    # Plot kernels as bars; gaps as red ticks above the timeline.
    starts = [iv[0] / 1e3 for iv in intervals]
    durations = [(iv[1] - iv[0]) / 1e3 for iv in intervals]
    labels = [iv[3].op_code for iv in intervals]
    colors = [color_for_op(iv[3].op_code) for iv in intervals]
    customs = [[iv[3].cluster_id] for iv in intervals]
    texts = [
        f"<b>{iv[3].op_code}</b><br>kernel {iv[3].device_kernel_ns/1000:.1f} us<br>"
        f"gap before: {iv[2]/1000:.1f} us<br>block: {iv[3].block_path}"
        for iv in intervals
    ]

    fig.add_trace(
        go.Bar(
            x=durations,
            y=labels,
            base=starts,
            orientation="h",
            marker=dict(color=colors),
            customdata=customs,
            hovertemplate="%{text}<extra></extra>",
            text=texts,
            name="kernel",
        )
    )

    # Mark large gaps.
    big_gaps = [iv for iv in intervals if (iv[2] or 0) > 5_000]  # >5us
    if big_gaps:
        fig.add_trace(
            go.Scatter(
                x=[iv[0] / 1e3 for iv in big_gaps],
                y=[iv[3].op_code for iv in big_gaps],
                mode="markers",
                marker=dict(color="#F85149", size=10, symbol="triangle-down"),
                name="big gap (>5us)",
                hovertemplate="gap %{x:.1f} us<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Op-to-op waterfall (first {len(intervals)} ops; red triangles flag gaps > 5us)",
        height=max(420, 16 * len(intervals)),
        xaxis=dict(title="time (us)"),
        showlegend=True,
    )
    return fig

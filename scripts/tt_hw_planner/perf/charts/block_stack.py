# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-transformer-block utilization vs runtime (chart 4).

The deliverable Mohamed asked for. Two side-by-side stacks per block:
left bar = total runtime (ms), stacked by op type. Right bar = mean
utilization (FPU/DRAM/NoC/ETH), stacked.

Blocks are derived from `tracy.block_scope` signposts via `_walk_signposts`
in join.py. Rows that fall outside any block end up in `root`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..cluster import Cluster
from ..join import JoinedRow
from .theme import color_for_op


def _aggregate_by_block_op(rows: List[JoinedRow]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Returns (runtime_ms_per_block_per_op, util_means_per_block)."""
    runtime: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    util_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    util_count: Dict[str, int] = defaultdict(int)
    for r in rows:
        if r.device_kernel_ns is None:
            continue
        runtime[r.block_path][r.op_code or "?"] += r.device_kernel_ns / 1e6
        util_sum[r.block_path]["FPU"] += r.pm_fpu_util_pct or 0
        util_sum[r.block_path]["DRAM"] += r.dram_bw_util_pct or 0
        util_sum[r.block_path]["NoC"] += r.noc_util_pct or 0
        util_sum[r.block_path]["ETH"] += r.eth_bw_util_pct or 0
        util_count[r.block_path] += 1

    util_means: Dict[str, Dict[str, float]] = {}
    for block, sums in util_sum.items():
        n = max(util_count[block], 1)
        util_means[block] = {k: v / n for k, v in sums.items()}
    return runtime, util_means


def make_block_stack(rows: List[JoinedRow], clusters: List[Cluster]) -> go.Figure:
    runtime, utils = _aggregate_by_block_op(rows)

    blocks = sorted(runtime.keys(), key=lambda b: -sum(runtime[b].values()))
    if not blocks:
        fig = go.Figure()
        fig.add_annotation(
            text="No signposted blocks found.<br>Wrap decoder loops in tracy.block_scope(...) to populate this chart.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8b949e"),
        )
        return fig

    op_codes = sorted(
        {op for blk in runtime.values() for op in blk.keys()}, key=lambda o: -sum(runtime[b].get(o, 0) for b in blocks)
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=("Runtime per block (ms, stacked by op)", "Mean utilization (%)"),
        horizontal_spacing=0.08,
    )

    for op in op_codes:
        values = [runtime[b].get(op, 0.0) for b in blocks]
        fig.add_trace(
            go.Bar(
                x=values,
                y=blocks,
                orientation="h",
                name=op,
                marker=dict(color=color_for_op(op)),
                hovertemplate=f"<b>{op}</b><br>%{{x:.2f}} ms<br>%{{y}}<extra></extra>",
                legendgroup="ops",
            ),
            row=1,
            col=1,
        )

    util_kinds = [("FPU", "#56d364"), ("DRAM", "#f0883e"), ("NoC", "#a371f7"), ("ETH", "#d29922")]
    for kind, color in util_kinds:
        values = [utils.get(b, {}).get(kind, 0.0) for b in blocks]
        fig.add_trace(
            go.Bar(
                x=values,
                y=blocks,
                orientation="h",
                name=kind + " util",
                marker=dict(color=color, opacity=0.85),
                hovertemplate=f"<b>{kind} util</b><br>%{{x:.1f}}%<br>%{{y}}<extra></extra>",
                legendgroup="util",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        barmode="stack",
        title="Per-block runtime + utilization (signposted blocks)",
        height=max(360, 24 * len(blocks) + 120),
        legend=dict(orientation="v", x=1.02, y=1),
    )
    fig.update_xaxes(title="ms", row=1, col=1)
    fig.update_xaxes(title="%", row=1, col=2, range=[0, 100])
    return fig

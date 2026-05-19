# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-transformer-block utilization vs runtime (Mohamed's headline chart).

For each transformer block i in [0, num_hidden_layers):

  - X axis: runtime in ms (sum of device kernel time for ops in that block)
  - Y axis: utilization (% of FPU / DRAM / NoC peak, configurable)
  - one marker per block, line connecting consecutive layers

This is the "TTNN-visualizer style, but per block" plot. Cross-run overlay
is supported: pass a `baseline_rows` argument to draw the baseline run's
per-block curve in muted color underneath the current run, so the user can
see at a glance whether their last optimization moved each block up-and-
left (faster + better utilized) or backwards.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go

from ..join import JoinedRow


_UTIL_FIELDS: Dict[str, str] = {
    "FPU": "pm_fpu_util_pct",
    "DRAM": "dram_bw_util_pct",
    "NoC": "noc_util_pct",
    "ETH": "eth_bw_util_pct",
}


def _per_block_summary(
    rows: List[JoinedRow],
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Aggregate rows into per-block (runtime_ms, util_pct_mean) tuples.

    Returns (ordered_block_labels, summary_dict).

    Block labels look like `decoder.layers.00`, `decoder.layers.01`, ... and
    sort lexicographically into the right order because they are
    zero-padded by `block_inference._layer_label`.
    """
    runtime_ns: Dict[str, float] = defaultdict(float)
    util_sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    util_counts: Dict[str, int] = defaultdict(int)
    op_count: Dict[str, int] = defaultdict(int)

    for r in rows:
        if r.block_path == "root" or r.device_kernel_ns is None:
            continue
        runtime_ns[r.block_path] += r.device_kernel_ns or 0.0
        op_count[r.block_path] += 1
        for util_name, field in _UTIL_FIELDS.items():
            v = getattr(r, field, None)
            if v is None:
                continue
            util_sums[r.block_path][util_name] += float(v)
            util_counts[r.block_path] += 1

    blocks = sorted(runtime_ns.keys())
    summary: Dict[str, Dict[str, float]] = {}
    for b in blocks:
        denom = max(util_counts[b] // max(1, len(_UTIL_FIELDS)), 1)
        summary[b] = {
            "runtime_ms": runtime_ns[b] / 1e6,
            "op_count": op_count[b],
        }
        for util_name in _UTIL_FIELDS:
            summary[b][util_name] = util_sums[b].get(util_name, 0.0) / denom
    return blocks, summary


def make_block_util_runtime(
    rows: List[JoinedRow],
    *,
    baseline_rows: Optional[List[JoinedRow]] = None,
    util_axis: str = "FPU",
) -> go.Figure:
    """Build the scatter+line per-block chart.

    Args:
        rows: joined rows of the current run.
        baseline_rows: optional, for overlay. If provided, draws a muted
            companion curve so optimization deltas are visually obvious.
        util_axis: which utilization to plot on Y. One of FPU / DRAM / NoC /
            ETH; defaults to FPU which is usually the bottleneck for LLM
            decode.
    """
    blocks, summary = _per_block_summary(rows)

    if not blocks:
        fig = go.Figure()
        fig.add_annotation(
            text=(
                "No transformer blocks could be inferred from this run.<br>"
                "Either the model's HF config (num_hidden_layers) is unknown,<br>"
                "or no anchor op was found.<br><br>"
                "Re-run `perf collect` so run_meta.json records num_hidden_layers,<br>"
                "or wrap decoder loops in `tracy.block_scope(...)` for explicit signposts."
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8b949e", size=13),
            align="center",
        )
        fig.update_layout(
            title="Per-transformer-block utilization vs runtime",
            height=380,
        )
        return fig

    util_axis = util_axis.upper() if util_axis else "FPU"
    if util_axis not in _UTIL_FIELDS:
        util_axis = "FPU"

    xs = [summary[b]["runtime_ms"] for b in blocks]
    ys = [summary[b][util_axis] for b in blocks]
    layer_idxs = [int(b.split(".")[-1]) for b in blocks]
    op_counts = [summary[b]["op_count"] for b in blocks]
    hover = [
        f"<b>{b}</b><br>" f"runtime: {x:.3f} ms<br>" f"{util_axis} util: {y:.1f}%<br>" f"ops: {n}"
        for b, x, y, n in zip(blocks, xs, ys, op_counts)
    ]

    fig = go.Figure()

    if baseline_rows:
        _b, _s = _per_block_summary(baseline_rows)
        if _b:
            base_xs = [_s[b]["runtime_ms"] for b in _b]
            base_ys = [_s[b][util_axis] for b in _b]
            fig.add_trace(
                go.Scatter(
                    x=base_xs,
                    y=base_ys,
                    mode="lines+markers",
                    name="baseline",
                    line=dict(color="#6e7681", width=2, dash="dot"),
                    marker=dict(size=8, color="#6e7681", opacity=0.6),
                    hoverinfo="skip",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers+text",
            name="current",
            line=dict(color="#56d364", width=2),
            marker=dict(
                size=12,
                color=layer_idxs,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="layer", tickformat="d"),
                line=dict(color="#0d1117", width=1),
            ),
            text=[str(i) for i in layer_idxs],
            textposition="top center",
            textfont=dict(size=9, color="#c9d1d9"),
            hovertext=hover,
            hoverinfo="text",
        )
    )

    median_x = sorted(xs)[len(xs) // 2]
    fig.add_vline(
        x=median_x,
        line=dict(color="#30363d", dash="dash", width=1),
        annotation_text=f"median {median_x:.2f} ms",
        annotation_position="top",
    )

    fig.update_layout(
        title=(f"Per-transformer-block {util_axis} utilization vs runtime " f"({len(blocks)} block(s) inferred)"),
        xaxis=dict(title="block runtime (ms, sum of device-kernel time)"),
        yaxis=dict(title=f"{util_axis} utilization (%)", range=[0, 100]),
        height=440,
        showlegend=baseline_rows is not None,
    )
    return fig

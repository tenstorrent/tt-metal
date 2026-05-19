# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Nsight-style per-device/per-RISC timeline for perf dashboard.

This is intentionally not a replacement for the existing aggregate charts.
It gives the "systems view": who executed when, on which device lane, and
for how long. The figure is built from JoinedRow records without requiring
additional tracing formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import plotly.graph_objects as go

from ..join import JoinedRow


_RISCS: Tuple[Tuple[str, str], ...] = (
    ("brisc_ns", "BRISC"),
    ("ncrisc_ns", "NCRISC"),
    ("trisc0_ns", "TRISC0"),
    ("trisc1_ns", "TRISC1"),
    ("trisc2_ns", "TRISC2"),
    ("erisc_ns", "ERISC"),
)


@dataclass
class _Interval:
    run_label: str
    lane: str
    start_ms: float
    dur_ms: float
    op_code: str
    module_path: Optional[str]
    block_path: str
    row_index: int
    device_id: Optional[int]
    global_call_count: int


def _reconstruct_intervals(rows: List[JoinedRow], run_label: str) -> List[_Interval]:
    """Reconstruct timeline intervals from cumulative op stream.

    Tracy CSV in this pipeline does not always provide an explicit absolute
    timestamp per row in the joined layer, so we rebuild per-device time from:

      t += op_to_op_latency_ns
      op_start = t
      t += device_kernel_ns

    and then place each RISC interval on its lane with the same op_start and
    RISC-specific duration.
    """
    per_dev_t_ns: Dict[int, float] = {}
    out: List[_Interval] = []
    compress_gaps = True
    for r in rows:
        dev = int(r.device_id) if r.device_id is not None else -1
        t = per_dev_t_ns.get(dev, 0.0)
        if not compress_gaps:
            t += float(r.op_to_op_latency_ns or 0.0)
        op_start_ns = t
        kernel_ns = float(r.device_kernel_ns or 0.0)
        if kernel_ns <= 0.0:
            # Fallback to max RISC duration if kernel duration is absent.
            kernel_ns = max(float(getattr(r, k) or 0.0) for k, _ in _RISCS)
        if kernel_ns <= 0.0:
            # Nothing to plot for this row.
            continue

        for attr, risc_name in _RISCS:
            dur_ns = float(getattr(r, attr) or 0.0)
            if dur_ns <= 0.0:
                continue
            lane = f"dev{dev}:{risc_name}"
            out.append(
                _Interval(
                    run_label=run_label,
                    lane=lane,
                    start_ms=op_start_ns / 1e6,
                    dur_ms=dur_ns / 1e6,
                    op_code=r.op_code,
                    module_path=r.module_path,
                    block_path=r.block_path,
                    row_index=r.row_index,
                    device_id=r.device_id,
                    global_call_count=r.global_call_count,
                )
            )
        # Active-time axis by default: keep bars contiguous so short kernels
        # remain visible even when dispatch gaps are very large.
        per_dev_t_ns[dev] = op_start_ns + kernel_ns
    return out


def _lane_order(intervals: Iterable[_Interval]) -> List[str]:
    lanes = sorted({i.lane for i in intervals})

    # Stable order: dev0 BRISC..ERISC, dev1 BRISC..ERISC, ...
    def key(lane: str) -> Tuple[int, int]:
        # lane format: dev{n}:{risc}
        left, risc = lane.split(":")
        dev = int(left.replace("dev", ""))
        risc_idx = next((idx for idx, (_, name) in enumerate(_RISCS) if name == risc), 99)
        return dev, risc_idx

    return sorted(lanes, key=key)


def make_nsight_timeline(
    rows: List[JoinedRow],
    *,
    baseline_rows: Optional[List[JoinedRow]] = None,
    highlight_module_path: Optional[str] = None,
) -> go.Figure:
    """Render a lane timeline with optional baseline overlay.

    - Primary run bars: bright blue.
    - Baseline bars: translucent amber overlay on same lanes.
    - Highlighted module bars (if selected): red.
    """
    intervals = _reconstruct_intervals(rows, "run")
    if baseline_rows:
        intervals.extend(_reconstruct_intervals(baseline_rows, "baseline"))

    fig = go.Figure()
    if not intervals:
        fig.update_layout(
            template="plotly_dark",
            title="Nsight Timeline (no intervals available)",
            xaxis_title="Time (ms)",
            yaxis_title="Lane",
            height=650,
        )
        return fig

    lanes = _lane_order(intervals)

    for run_label, base_color, opacity in (
        ("run", "#58a6ff", 0.85),
        ("baseline", "#d29922", 0.45),
    ):
        sub = [i for i in intervals if i.run_label == run_label]
        if not sub:
            continue

        # Split highlighted/non-highlighted for better visual contrast.
        normal = [i for i in sub if not highlight_module_path or i.module_path != highlight_module_path]
        highlighted = [i for i in sub if highlight_module_path and i.module_path == highlight_module_path]

        def add_trace(data: List[_Interval], color: str, name_suffix: str = "") -> None:
            if not data:
                return
            fig.add_trace(
                go.Bar(
                    orientation="h",
                    y=[i.lane for i in data],
                    x=[i.dur_ms for i in data],
                    base=[i.start_ms for i in data],
                    marker={"color": color},
                    opacity=opacity,
                    name=f"{run_label}{name_suffix}",
                    customdata=[
                        [
                            i.module_path or "",
                            i.block_path,
                            i.op_code,
                            i.row_index,
                            i.device_id if i.device_id is not None else -1,
                            i.global_call_count,
                            i.run_label,
                        ]
                        for i in data
                    ],
                    hovertemplate=(
                        "lane=%{y}<br>"
                        "op=%{customdata[2]}<br>"
                        "module=%{customdata[0]}<br>"
                        "block=%{customdata[1]}<br>"
                        "row=%{customdata[3]}<br>"
                        "device=%{customdata[4]}<br>"
                        "call=%{customdata[5]}<br>"
                        "run=%{customdata[6]}<br>"
                        "start=%{base:.3f}ms<br>"
                        "dur=%{x:.3f}ms<extra></extra>"
                    ),
                )
            )

        add_trace(normal, base_color)
        # Highlight in red for selected module.
        add_trace(highlighted, "#f85149", " (highlight)")

    fig.update_layout(
        template="plotly_dark",
        title="Nsight Timeline (active-time): device × RISC lanes",
        barmode="overlay",
        xaxis_title="Active time (ms)",
        yaxis_title="Lane",
        yaxis={"categoryorder": "array", "categoryarray": lanes},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        height=700,
        margin={"l": 120, "r": 20, "t": 60, "b": 60},
        hovermode="closest",
    )
    # Drag to pan (default with rangeslider off); users can zoom with box.
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    return fig

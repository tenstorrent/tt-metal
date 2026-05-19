# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kernel-hash diff vs baseline (chart 8).

Detects clusters whose `COMPUTE KERNEL HASH` or `DATA MOVEMENT KERNEL HASH`
changed vs a baseline run, with the time delta. Useful for catching PRs
that silently rewrote a kernel and made it slower (or faster).

Only renders meaningfully when both `rows` and `baseline_rows` are given.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go

from ..cluster import Cluster
from ..join import JoinedRow


def _by_signature(rows: List[JoinedRow]) -> Dict[Tuple[str, str], JoinedRow]:
    """Map (op_code, args_hash or kernel_hash+shape) -> first sample row."""
    out: Dict[Tuple[str, str], JoinedRow] = {}
    for r in rows:
        key_id = r.args_hash or (r.compute_kernel_hash + r.outputs_summary[:32])
        key = (r.op_code or "?", key_id)
        if key not in out:
            out[key] = r
    return out


def make_hash_diff(
    rows: List[JoinedRow],
    clusters: List[Cluster],
    baseline_rows: Optional[List[JoinedRow]],
) -> go.Figure:
    fig = go.Figure()
    if not baseline_rows:
        fig.add_annotation(
            text="No baseline supplied.<br>Run <code>perf collect --baseline run_NNN</code> to populate.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8b949e"),
        )
        return fig

    cur_by_sig = _by_signature(rows)
    base_by_sig = _by_signature(baseline_rows)

    diffs = []
    for sig, cur in cur_by_sig.items():
        base = base_by_sig.get(sig)
        if base is None:
            continue
        ch_compute = cur.compute_kernel_hash != base.compute_kernel_hash
        ch_dm = cur.dm_kernel_hash != base.dm_kernel_hash
        if not (ch_compute or ch_dm):
            continue
        dt = (cur.device_kernel_ns or 0) - (base.device_kernel_ns or 0)
        diffs.append((cur, base, dt, ch_compute, ch_dm))

    if not diffs:
        fig.add_annotation(
            text="No kernel hash changes vs baseline.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8b949e"),
        )
        return fig

    diffs.sort(key=lambda d: -abs(d[2]))
    diffs = diffs[:30]

    labels = [f"{d[0].op_code} ({d[0].cluster_id or '?'})" for d in diffs]
    dts = [d[2] / 1e3 for d in diffs]
    colors = ["#F85149" if dt > 0 else "#2EA043" for dt in dts]
    texts = [
        (
            f"<b>{d[0].op_code}</b><br>"
            f"current: kernel {(d[0].device_kernel_ns or 0)/1000:.1f} us<br>"
            f"baseline: kernel {(d[1].device_kernel_ns or 0)/1000:.1f} us<br>"
            f"Δ = {dts[i]:.2f} us<br>"
            f"compute hash changed: {d[3]}<br>"
            f"DM hash changed: {d[4]}"
        )
        for i, d in enumerate(diffs)
    ]

    fig.add_trace(
        go.Bar(
            x=dts,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Kernel hash diff vs baseline (red=slower, green=faster)",
        height=max(360, 24 * len(diffs) + 120),
        xaxis=dict(title="Δ device kernel (us); positive = regression"),
    )
    return fig

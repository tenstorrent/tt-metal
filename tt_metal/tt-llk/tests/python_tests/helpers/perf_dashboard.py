# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parquet -> HTML performance dashboard (closes the loop).

Reads a shared-schema Parquet batch and writes one plotly scatter plot per test:
the timing stats (``mean(<run_type>)``) against the sweep configuration. This
lets the dashboard consume Parquet directly, so a run's output can be visualized
straight after CSV -> Parquet conversion, with no database in the loop.

Mirrors the in-run scatter (perf.dump_scatter) but reads a Parquet file instead
of a live PerfReport. Needs plotly + pyarrow; no device libraries.
"""

from pathlib import Path

import plotly.graph_objects as go
import pyarrow.parquet as pq

from .perf_parquet import safe_stem
from .perf_wide_schema import DB_SCHEMA

_SWEEP_CATEGORIES = {"formats", "flags", "key", "configuration"}
_SWEEP_COLUMNS = {c.name for c in DB_SCHEMA if c.category in _SWEEP_CATEGORIES}


def dashboard_from_parquet(parquet_path, out_dir):
    """Write one ``<test_name>.html`` scatter plot per test in the batch.

    x = sweep index (hover shows the sweep config), y = each ``mean(<run_type>)``
    column. A test with no populated mean column is skipped. Returns
    {test_name: html_path}.
    """
    df = pq.read_table(parquet_path).to_pandas()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for test_name, group in df.groupby("test_name"):
        mean_columns = [
            c for c in group.columns if c.startswith("mean(") and group[c].notna().any()
        ]
        if not mean_columns:
            continue

        sweep_columns = [
            c for c in group.columns if c in _SWEEP_COLUMNS and group[c].notna().any()
        ]
        hover = [
            ", ".join(f"{c}={row[c]}" for c in sweep_columns)
            for _, row in group.iterrows()
        ]

        fig = go.Figure()
        x = list(range(len(group)))
        for col in mean_columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=list(group[col]),
                    mode="markers+lines",
                    name=col,
                    text=hover,
                    hoverinfo="text+y",
                )
            )
        fig.update_layout(
            title=f"Performance: {test_name}",
            xaxis_title="Sweep index (see hover for config)",
            yaxis_title="Cycles / Tile",
            legend_title="Run type / stat",
        )

        path = out_dir / f"{safe_stem(test_name)}.html"
        fig.write_html(str(path))
        written[test_name] = path
    return written

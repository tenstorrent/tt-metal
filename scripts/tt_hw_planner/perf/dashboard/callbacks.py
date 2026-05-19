# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dash callbacks — wire the UI buttons to the runner code paths.

Apply / Revert / Dry-run buttons go through `runner.apply_block` and
`runner.revert_block` (same code paths as the CLI). The Re-collect
button shells out to `collect_run` for the same model id and pops the
new run id into the URL.

All callbacks are written to never block the main thread for long: long
work (collect) is launched in a daemon thread and the user gets a
spinning status pill while it runs.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from typing import Any, Dict, List

from dash import ALL, Dash, Input, Output, State, ctx, html, no_update

from .cytoscape_layout import CYTOSCAPE_STYLESHEET


_COLLECT_STATE: Dict[str, str] = {"status": "idle", "last_run": ""}


def register_callbacks(app: Dash) -> None:
    CYTO_LIGHT_OVERRIDES: List[Dict[str, Any]] = [
        {
            "selector": 'node[id = "__root__"]',
            "style": {"background-color": "#fde68a", "border-color": "#f59e0b", "color": "#111827"},
        },
        {
            "selector": 'node[id = "model"]',
            "style": {"background-color": "#bfdbfe", "border-color": "#3b82f6", "color": "#111827"},
        },
        {
            "selector": 'node[id = "decoder"]',
            "style": {"background-color": "#bbf7d0", "border-color": "#22c55e", "color": "#111827"},
        },
        {
            "selector": 'node[id = "decoder.layers"]',
            "style": {"background-color": "#fecdd3", "border-color": "#f43f5e", "color": "#111827"},
        },
        {
            "selector": 'node[id = "layers"]',
            "style": {"background-color": "#fecdd3", "border-color": "#f43f5e", "color": "#111827"},
        },
        {
            "selector": "node[kind = 'root']",
            "style": {
                "background-color": "#eef2ff",
                "border-color": "#94a3b8",
                "color": "#0f172a",
            },
        },
        {
            "selector": "node[kind = 'parent']",
            "style": {
                "background-color": "#e0ecff",
                "border-color": "#93c5fd",
                "color": "#0f172a",
            },
        },
        {"selector": "node[kind = 'parent'][layer_band = 'b0']", "style": {"background-color": "#dbeafe"}},
        {"selector": "node[kind = 'parent'][layer_band = 'b1']", "style": {"background-color": "#dcfce7"}},
        {"selector": "node[kind = 'parent'][layer_band = 'b2']", "style": {"background-color": "#fef3c7"}},
        {"selector": "node[kind = 'parent'][layer_band = 'b3']", "style": {"background-color": "#fce7f3"}},
        {"selector": "node[kind = 'parent'][layer_band = 'b4']", "style": {"background-color": "#fae8ff"}},
        {"selector": "node[kind = 'parent'][layer_band = 'b5']", "style": {"background-color": "#fee2e2"}},
        {"selector": "node[kind = 'parent'][layer_band = 'b6']", "style": {"background-color": "#e0f2fe"}},
        {"selector": "node[kind = 'parent'][layer_band = 'b7']", "style": {"background-color": "#ecfccb"}},
        {
            "selector": "node[kind = 'leaf']",
            "style": {
                "background-color": "#cbd5e1",
                "border-color": "#94a3b8",
                "color": "#111827",
            },
        },
        {"selector": "node[kind = 'leaf'][op_role = 'matmul']", "style": {"background-color": "#60a5fa"}},
        {"selector": "node[kind = 'leaf'][op_role = 'sdpa']", "style": {"background-color": "#34d399"}},
        {"selector": "node[kind = 'leaf'][op_role = 'norm']", "style": {"background-color": "#fbbf24"}},
        {"selector": "node[kind = 'leaf'][op_role = 'layout_move']", "style": {"background-color": "#a78bfa"}},
        {"selector": "node[kind = 'leaf'][op_role = 'all_gather']", "style": {"background-color": "#22d3ee"}},
        {"selector": "node[kind = 'leaf'][op_role = 'reduce_scatter']", "style": {"background-color": "#2dd4bf"}},
        {"selector": "node[kind = 'leaf'][op_role = 'binary']", "style": {"background-color": "#f472b6"}},
        {"selector": "node[kind = 'leaf'][op_role = 'typecast']", "style": {"background-color": "#93c5fd"}},
        {"selector": "node[kind = 'leaf'][op_role = 'qkv_heads']", "style": {"background-color": "#f59e0b"}},
        {"selector": "node[kind = 'leaf'][op_role = 'concat_heads']", "style": {"background-color": "#fb7185"}},
        {"selector": "node[kind = 'leaf'][op_role = 'rotary']", "style": {"background-color": "#4ade80"}},
        {"selector": "node[kind = 'leaf'][op_role = 'other']", "style": {"background-color": "#a3a3a3"}},
        {
            "selector": "edge",
            "style": {
                "line-color": "#64748b",
                "target-arrow-color": "#64748b",
                "opacity": 0.5,
            },
        },
        {
            "selector": "node:selected",
            "style": {"border-color": "#ef4444", "border-width": 5},
        },
    ]
    THEME_STYLES = {
        "dark": {
            "root": {
                "background": "#0d1117",
                "color": "#c9d1d9",
                "minHeight": "100vh",
                "fontFamily": "Inter, -apple-system, system-ui, sans-serif",
                "margin": 0,
                "--tt-bg": "#0d1117",
                "--tt-pane-bg": "#161b22",
                "--tt-border": "#30363d",
                "--tt-fg": "#c9d1d9",
                "--tt-fg-dim": "#8b949e",
                "--tt-heading": "#f0f6fc",
                "--tt-accent": "#58a6ff",
                "--tt-accent-bg-soft": "#1f6feb33",
                "--tt-hot-bg": "#3d1518",
                "--tt-good-bg": "#11321a",
                "--tt-bar-track": "#21262d",
            },
            "topbar": {
                "padding": "10px 20px",
                "background": "#161b22",
                "borderBottom": "1px solid #30363d",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
            },
            "left_rail": {
                "background": "#161b22",
                "borderRight": "1px solid #30363d",
                "padding": "12px",
                "overflow": "auto",
            },
            "inspector": {
                "background": "#161b22",
                "borderLeft": "1px solid #30363d",
                "padding": "16px",
                "overflow": "auto",
                "fontSize": "12px",
            },
            "toolbar": {
                "display": "flex",
                "alignItems": "center",
                "gap": "16px",
                "padding": "8px 12px",
                "borderBottom": "1px solid #30363d",
            },
            "graph": {"width": "100%", "height": "100%", "background": "#0d1117"},
            "tabs": {"background": "#0d1117", "primary": "#1f6feb", "border": "#30363d"},
            "template": "plotly_dark",
        },
        "light": {
            "root": {
                "background": "#ffffff",
                "color": "#111827",
                "minHeight": "100vh",
                "fontFamily": "Inter, -apple-system, system-ui, sans-serif",
                "margin": 0,
                "--tt-bg": "#ffffff",
                "--tt-pane-bg": "#ffffff",
                "--tt-border": "#d1d5db",
                "--tt-fg": "#111827",
                "--tt-fg-dim": "#4b5563",
                "--tt-heading": "#0f172a",
                "--tt-accent": "#1d4ed8",
                "--tt-accent-bg-soft": "#dbeafe",
                "--tt-hot-bg": "#fee2e2",
                "--tt-good-bg": "#dcfce7",
                "--tt-bar-track": "#e5e7eb",
            },
            "topbar": {
                "padding": "10px 20px",
                "background": "#ffffff",
                "borderBottom": "1px solid #d1d5db",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
            },
            "left_rail": {
                "background": "#ffffff",
                "borderRight": "1px solid #d1d5db",
                "padding": "12px",
                "overflow": "auto",
            },
            "inspector": {
                "background": "#ffffff",
                "borderLeft": "1px solid #d1d5db",
                "padding": "16px",
                "overflow": "auto",
                "fontSize": "12px",
            },
            "toolbar": {
                "display": "flex",
                "alignItems": "center",
                "gap": "16px",
                "background": "#ffffff",
                "padding": "8px 12px",
                "borderBottom": "1px solid #d1d5db",
            },
            "graph": {"width": "100%", "height": "100%", "background": "#ffffff"},
            "tabs": {"background": "#ffffff", "primary": "#2563eb", "border": "#d1d5db"},
            "template": "plotly_white",
        },
    }

    def _role_key(path: str) -> str:
        parts = (path or "").split(".")
        out = []
        for p in parts:
            out.append("*" if p.isdigit() else p)
        return ".".join(out)

    def _layer_from_node_id(node_id: str) -> str:
        m = re.search(r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)", node_id or "")
        return m.group(1) if m else ""

    @app.callback(
        Output("module-graph", "elements"),
        Input("layer-focus-selector", "value"),
        Input("btn-refresh-graph", "n_clicks"),
        State("module-elements-data", "data"),
        prevent_initial_call=True,
    )
    def filter_graph_elements_by_layer(layer_value, _refresh_clicks, elements_blob):
        if layer_value in (None, "", "all"):
            try:
                return json.loads(elements_blob) if isinstance(elements_blob, str) else (elements_blob or [])
            except (TypeError, json.JSONDecodeError):
                return []
        try:
            elements = json.loads(elements_blob) if isinstance(elements_blob, str) else (elements_blob or [])
        except (TypeError, json.JSONDecodeError):
            return []

        try:
            selected_idx = int(str(layer_value))
        except ValueError:
            selected_idx = None

        selected = str(layer_value).zfill(2)
        node_ids = set()
        edge_elements = []
        node_elements = []
        for e in elements:
            data = e.get("data") or {}
            if "source" in data and "target" in data:
                edge_elements.append(e)
            else:
                node_elements.append(e)

        for e in node_elements:
            data = e.get("data") or {}
            nid = str(data.get("id") or "")
            if not nid:
                continue
            layer = _layer_from_node_id(nid)
            layer_idx_data = data.get("layer_index")
            layer_match = False
            if selected_idx is not None and layer_idx_data is not None:
                try:
                    layer_match = int(layer_idx_data) == selected_idx
                except (TypeError, ValueError):
                    layer_match = False
            if not layer_match:
                layer_match = layer == selected or layer == str(selected_idx or "")
            if not layer_match:
                continue
            cur = nid
            node_ids.add(cur)
            while "." in cur:
                cur = cur.rsplit(".", 1)[0]
                node_ids.add(cur)
            node_ids.add("__root__")

        for keep in ("model", "decoder", "decoder.layers", "layers", "root", "__root__"):
            node_ids.add(keep)

        filtered = []
        for e in node_elements:
            nid = str((e.get("data") or {}).get("id") or "")
            if nid in node_ids:
                filtered.append(e)
        for e in edge_elements:
            d = e.get("data") or {}
            s = str(d.get("source") or "")
            t = str(d.get("target") or "")
            if s in node_ids and t in node_ids:
                filtered.append(e)
        return filtered

    @app.callback(
        Output("selected-cluster", "data"),
        Input({"type": "chart", "name": "roofline"}, "clickData"),
        Input({"type": "chart", "name": "speed_of_light"}, "clickData"),
        Input({"type": "chart", "name": "per_risc"}, "clickData"),
        Input({"type": "chart", "name": "per_block"}, "clickData"),
        Input({"type": "chart", "name": "config_scatter"}, "clickData"),
        Input({"type": "chart", "name": "waterfall"}, "clickData"),
        Input({"type": "chart", "name": "cache_heatmap"}, "clickData"),
        Input({"type": "chart", "name": "hash_diff"}, "clickData"),
        prevent_initial_call=True,
    )
    def update_selected_cluster(*click_data):
        for d in click_data:
            if not d:
                continue
            pts = d.get("points") or []
            for p in pts:
                cd = p.get("customdata")
                if cd and isinstance(cd, list) and cd:
                    return cd[0]
        return no_update

    @app.callback(
        Output("inspector-body", "children", allow_duplicate=True),
        Input("selected-cluster", "data"),
        State("clusters-data", "data"),
        prevent_initial_call=True,
    )
    def render_inspector(cid: str, clusters_blob: str):
        if not cid:
            return [html.Em("Click a point on any chart to populate.", style={"color": "#8b949e"})]
        try:
            payload = json.loads(clusters_blob) if isinstance(clusters_blob, str) else (clusters_blob or [])
        except json.JSONDecodeError:
            payload = []
        c = next((x for x in payload if x.get("cluster_id") == cid), None)
        if c is None:
            return [html.Em(f"No data for cluster {cid}", style={"color": "#8b949e"})]

        def fmt_ns(v):
            if v is None:
                return ""
            return f"{v/1e6:.2f} ms" if v >= 1e6 else f"{v/1e3:.1f} us"

        def fmt_pct(v):
            return "" if v is None else f"{v:.1f}%"

        region = c.get("region") or "?"
        region_color = {
            "A": "#2EA043",
            "B": "#FFD93D",
            "C": "#FFA630",
            "D": "#1F6FEB",
            "E": "#F85149",
            "F": "#8B949E",
            "?": "#586069",
        }.get(region, "#586069")
        suggestions = {
            "A": [],
            "B": ["math_fidelity_downcast"],
            "C": ["math_fidelity_downcast", "program_config_tuner"],
            "D": ["fusion_rewriter"],
            "E": ["dram_l1_promoter", "fusion_rewriter", "layout_unifier"],
            "F": ["trace_capturer", "cache_warmer"],
            "?": [],
        }.get(region, [])

        kv_items = [
            ("region", f"{region} — {c.get('region_reason', '')}"),
            ("op", c.get("op_code")),
            ("cluster_id", cid),
            ("n_calls", c.get("n_calls")),
            ("median", fmt_ns(c.get("median_device_ns"))),
            ("total", fmt_ns(c.get("total_device_ns"))),
            ("% of peak", "" if c.get("pct_of_peak") is None else f"{c['pct_of_peak']:.1f}%"),
            ("FPU util", fmt_pct(c.get("mean_fpu_util_pct"))),
            ("DRAM util", fmt_pct(c.get("mean_dram_bw_util_pct"))),
            ("NoC util", fmt_pct(c.get("mean_noc_util_pct"))),
            ("ETH util", fmt_pct(c.get("mean_eth_bw_util_pct"))),
            ("args_hash", c.get("args_hash") or "(tracer missing)"),
            ("kernel_hash", c.get("compute_kernel_hash") or "(none)"),
            ("compute src", c.get("compute_kernel_source") or "(none)"),
            ("DM src", c.get("dm_kernel_source") or "(none)"),
            ("blocks", ", ".join(c.get("blocks") or [])),
        ]
        rows = []
        for k, v in kv_items:
            rows.append(html.Div(k, style={"color": "#8b949e", "fontSize": "11px"}))
            rows.append(
                html.Div(
                    str(v) if v is not None else "",
                    style={
                        "color": "#c9d1d9",
                        "fontFamily": "JetBrains Mono, monospace",
                        "fontSize": "11px",
                        "wordBreak": "break-all",
                    },
                )
            )

        sug_items = []
        for s in suggestions:
            sug_items.append(
                html.Div(
                    style={
                        "border": "1px solid #30363d",
                        "borderRadius": "4px",
                        "padding": "8px",
                        "marginBottom": "8px",
                        "background": "#0d1117",
                    },
                    children=[
                        html.Div(s, style={"color": "#58a6ff", "fontWeight": "600", "fontSize": "12px"}),
                        html.Div(
                            f"tt_hw_planner perf apply {s} --cluster {cid}",
                            style={
                                "background": "#0d1117",
                                "border": "1px solid #30363d",
                                "borderRadius": "4px",
                                "padding": "6px 8px",
                                "fontFamily": "JetBrains Mono, monospace",
                                "fontSize": "11px",
                                "wordBreak": "break-all",
                                "color": "#c9d1d9",
                            },
                        ),
                    ],
                )
            )
        if not sug_items:
            sug_items.append(html.Em(f"No applicable block for region {region}.", style={"color": "#8b949e"}))

        return [
            html.H3(f"Cluster {cid}", style={"color": "#f0f6fc", "fontSize": "14px"}),
            html.Span(
                region,
                style={
                    "background": region_color,
                    "color": "#0d1117",
                    "padding": "1px 6px",
                    "borderRadius": "8px",
                    "fontSize": "10px",
                    "fontWeight": "600",
                },
            ),
            html.Div(
                rows,
                style={"display": "grid", "gridTemplateColumns": "130px 1fr", "gap": "4px 12px", "margin": "10px 0"},
            ),
            html.H4("Suggested optimizer blocks", style={"color": "#f0f6fc", "fontSize": "12px"}),
            *sug_items,
        ]

    @app.callback(
        Output("selected-block", "data"),
        Input({"type": "catalog-block", "name": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def remember_block(_all_clicks):
        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and "name" in triggered:
            return triggered["name"]
        return no_update

    @app.callback(
        Output("selected-block", "data", allow_duplicate=True),
        Input("selected-module-path", "data"),
        State("module-suggestion-index-data", "data"),
        prevent_initial_call=True,
    )
    def auto_select_block_from_node(module_path, suggestion_index_blob):
        if not module_path:
            return no_update
        try:
            idx = (
                json.loads(suggestion_index_blob)
                if isinstance(suggestion_index_blob, str)
                else (suggestion_index_blob or {})
            )
        except (TypeError, json.JSONDecodeError):
            idx = {}
        by_path = idx.get("by_path") or {}
        by_role = idx.get("by_role") or {}
        s = by_path.get(module_path)
        if s is None:
            target_role = _role_key(module_path)
            s = by_role.get(target_role)
        if s is None:
            return no_update
        blocks = list(s.get("proposed_blocks") or [])
        return blocks[0] if blocks else no_update

    @app.callback(
        Output("selected-block-label", "children"),
        Input("selected-block", "data"),
    )
    def show_selected_block(block_name):
        if not block_name:
            return "Selected block: (none)"
        return f"Selected block: {block_name}"

    @app.callback(
        Output("selected-node-label", "children"),
        Input("selected-module-path", "data"),
        Input("selected-cluster", "data"),
    )
    def show_selected_node(module_path, cluster_id):
        if module_path:
            return f"Selected node: {module_path}"
        if cluster_id:
            return f"Selected node: cluster {cluster_id}"
        return "Selected node: (none)"

    @app.callback(
        Output("global-action-panel", "style"),
        Input("selected-module-path", "data"),
        Input("selected-cluster", "data"),
    )
    def toggle_global_action_panel(module_path, _cluster_id):
        return {"display": "block"}

    @app.callback(
        Output("action-result-banner", "children"),
        Input("btn-apply", "n_clicks"),
        Input("btn-revert", "n_clicks"),
        Input("btn-dryrun", "n_clicks"),
        State("selected-block", "data"),
        State("selected-cluster", "data"),
        State("selected-module-path", "data"),
        State("run-id", "data"),
        prevent_initial_call=True,
    )
    def do_action(
        _apply_clicks,
        _revert_clicks,
        _dry_clicks,
        block_name,
        cluster_id,
        module_path,
        run_id,
    ):
        if not block_name:
            return "Select a block from the catalog first."
        if not run_id:
            return "Run id missing."
        from ..runner import apply_block, revert_block

        trig = ctx.triggered_id
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        module_hint = f" (module={module_path})" if module_path else ""
        action_cluster_id = None if module_path else cluster_id
        scope_hint = f"scope={'module' if module_path else 'cluster'}"
        try:
            if trig == "btn-apply":
                res = apply_block(block_name, run_id=run_id, cluster_id=action_cluster_id)
                return f"[{stamp}] apply {res.block} {scope_hint}{module_hint} -> {res.patch_path or 'no patch'}  ({res.findings} findings)"
            if trig == "btn-revert":
                removed = revert_block(block_name, run_id=run_id, cluster_id=cluster_id)
                return f"[{stamp}] revert {block_name} -> removed {len(removed)} patch file(s)"
            if trig == "btn-dryrun":
                res = apply_block(block_name, run_id=run_id, cluster_id=action_cluster_id, dry_run=True)
                return f"[{stamp}] dry-run {res.block} {scope_hint}{module_hint}: {res.findings} finding(s) — {res.rationale}"
        except Exception as e:
            return f"[{stamp}] ERROR: {e}"
        return f"[{stamp}] No action fired (trigger={trig})."

    @app.callback(
        Output("action-result-banner", "children", allow_duplicate=True),
        Input("btn-apply", "n_clicks"),
        Input("btn-revert", "n_clicks"),
        Input("btn-dryrun", "n_clicks"),
        prevent_initial_call=True,
    )
    def show_action_started(_apply_clicks, _revert_clicks, _dry_clicks):
        trig = ctx.triggered_id
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if trig == "btn-apply":
            return f"[{stamp}] Running apply..."
        if trig == "btn-revert":
            return f"[{stamp}] Running revert..."
        if trig == "btn-dryrun":
            return f"[{stamp}] Running dry-run..."
        return f"[{stamp}] Running action..."

    @app.callback(
        Output("theme-mode", "data"),
        Output("dashboard-root", "className"),
        Output("dashboard-root", "style"),
        Output("dashboard-topbar", "style"),
        Output("left-rail", "style"),
        Output("inspector-pane", "style"),
        Output("module-graph-toolbar", "style"),
        Output("module-graph", "style"),
        Output("tabs", "colors"),
        Input("theme-selector", "value"),
        prevent_initial_call=False,
    )
    def apply_theme(theme_value):
        mode = "light" if theme_value == "light" else "dark"
        style = THEME_STYLES[mode]

        return (
            mode,
            f"theme-{mode}",
            style["root"],
            style["topbar"],
            style["left_rail"],
            style["inspector"],
            style["toolbar"],
            style["graph"],
            style["tabs"],
        )

    @app.callback(
        Output("status-pill", "children"),
        Input("status-refresh", "n_intervals"),
        Input("btn-refresh", "n_clicks"),
        State("run-id", "data"),
    )
    def refresh_status(_n, _clicks, run_id):
        if not run_id:
            return ""
        try:
            from ..status_board import compute_status

            statuses = compute_status(run_id)
            counts: Dict[str, int] = {}
            for s in statuses:
                counts[s.status] = counts.get(s.status, 0) + 1
            parts = [f"{k}: {v}" for k, v in sorted(counts.items())]
            return "  ".join(parts) or "no status"
        except Exception as e:  # pragma: no cover
            return f"ERROR: {e}"

    @app.callback(
        Output({"type": "chart", "name": "hash_diff"}, "figure"),
        Output("action-result-banner", "children", allow_duplicate=True),
        Input("baseline-run-selector", "value"),
        State("run-id", "data"),
        prevent_initial_call=True,
    )
    def refresh_hash_diff_from_dashboard(baseline_run_id, run_id):
        from ..charts.hash_diff import make_hash_diff
        from ..ceilings import load_box_spec
        from ..cluster import cluster_rows
        from ..collect import find_run, require_healthy_run
        from ..join import join_run
        from ..regions import classify_all

        if not run_id:
            return no_update, "Run id missing."

        try:
            run_dir = find_run(run_id)
            meta = require_healthy_run(run_dir)
            box = load_box_spec(meta["box"], tuple(meta["mesh_shape"]))  # type: ignore[arg-type]
            rows = join_run(
                run_id=run_id,
                tracy_csv=run_dir / "ops_perf_results.csv" if (run_dir / "ops_perf_results.csv").exists() else None,
                tracer_master=run_dir / "ttnn_operations_master.json"
                if (run_dir / "ttnn_operations_master.json").exists()
                else None,
                num_hidden_layers=meta.get("num_hidden_layers"),
                module_hierarchy=run_dir / "ttnn_module_hierarchy.json"
                if (run_dir / "ttnn_module_hierarchy.json").exists()
                else None,
            )
            clusters = cluster_rows(rows)
            classify_all(rows, box)

            baseline_rows = None
            baseline_msg = "Hash Diff baseline cleared."
            if baseline_run_id:
                bdir = find_run(baseline_run_id)
                bmeta = require_healthy_run(bdir)
                baseline_rows = join_run(
                    run_id=baseline_run_id,
                    tracy_csv=bdir / "ops_perf_results.csv" if (bdir / "ops_perf_results.csv").exists() else None,
                    tracer_master=bdir / "ttnn_operations_master.json"
                    if (bdir / "ttnn_operations_master.json").exists()
                    else None,
                    num_hidden_layers=bmeta.get("num_hidden_layers"),
                    module_hierarchy=bdir / "ttnn_module_hierarchy.json"
                    if (bdir / "ttnn_module_hierarchy.json").exists()
                    else None,
                )
                cluster_rows(baseline_rows)
                classify_all(baseline_rows, box)
                baseline_msg = f"Hash Diff baseline set to {baseline_run_id}."

            return make_hash_diff(rows, clusters, baseline_rows), baseline_msg
        except Exception as exc:
            return no_update, f"ERROR: failed to refresh Hash Diff: {exc}"

    @app.callback(
        Output("action-log", "children"),
        Input("btn-recollect", "n_clicks"),
        State("run-id", "data"),
        prevent_initial_call=True,
    )
    def recollect(_n, run_id):
        if not run_id:
            return "no run id"
        from ..collect import collect_run, load_run_meta, find_run

        try:
            meta = load_run_meta(find_run(run_id))
        except FileNotFoundError as e:
            return f"ERROR: {e}"

        def worker():
            _COLLECT_STATE["status"] = "running"
            try:
                artifacts = collect_run(
                    model_id=meta["model_id"],
                    box_override=meta["box"],
                    mesh_override=tuple(meta["mesh_shape"]),
                    baseline_run_id=run_id,
                )
                _COLLECT_STATE["last_run"] = artifacts.run_id
                _COLLECT_STATE["status"] = "done"
            except Exception as e:  # pragma: no cover
                _COLLECT_STATE["status"] = f"error: {e}"

        threading.Thread(target=worker, daemon=True).start()
        return "re-collect launched (refresh status pill to see new run id)"

    @app.callback(
        Output("selected-module-path", "data"),
        Input("module-graph", "tapNodeData"),
        Input({"type": "chart", "name": "nsight_timeline"}, "clickData"),
        prevent_initial_call=True,
    )
    def select_module_from_graph_or_timeline(graph_tap, timeline_click):
        trig = ctx.triggered_id
        if trig == "module-graph":
            if not graph_tap:
                return no_update
            data = graph_tap
            node_id = (data or {}).get("id")
            if not node_id or node_id == "__root__":
                return no_update
            return node_id
        if isinstance(trig, dict) and trig.get("type") == "chart" and trig.get("name") == "nsight_timeline":
            if not timeline_click:
                return no_update
            pts = timeline_click.get("points") or []
            if not pts:
                return no_update
            cd = pts[0].get("customdata") or []
            module_path = cd[0] if isinstance(cd, list) and len(cd) >= 1 else ""
            return module_path or no_update
        return no_update

    @app.callback(
        Output("inspector-body", "children", allow_duplicate=True),
        Input("selected-module-path", "data"),
        State("module-node-index-data", "data"),
        State("module-suggestion-index-data", "data"),
        prevent_initial_call=True,
    )
    def on_module_node_selected(selected_module_path, node_index_blob, suggestion_index_blob):
        from .inspector import build_inspector_for_node, build_empty_inspector
        from ..module_graph import ModuleNode
        from ..suggestion_engine import Suggestion
        from ..reference_db import ModuleReference

        if not selected_module_path:
            return no_update

        try:
            node_idx = json.loads(node_index_blob) if isinstance(node_index_blob, str) else (node_index_blob or {})
            sug_idx = (
                json.loads(suggestion_index_blob)
                if isinstance(suggestion_index_blob, str)
                else (suggestion_index_blob or {})
            )
        except (TypeError, json.JSONDecodeError):
            node_idx, sug_idx = {}, {}

        by_path = node_idx or {}
        sug_by_path = (sug_idx or {}).get("by_path") or {}
        sug_by_role = (sug_idx or {}).get("by_role") or {}

        node_dict = by_path.get(selected_module_path)
        if node_dict is None:
            return build_empty_inspector(f"No data for {selected_module_path}")

        node = ModuleNode(
            attribute_path=node_dict.get("attribute_path", ""),
            module_class=node_dict.get("module_class", ""),
            op_count=int(node_dict.get("op_count", 0) or 0),
            total_device_ns=float(node_dict.get("total_device_ns", 0.0) or 0.0),
            median_device_ns=float(node_dict.get("median_device_ns", 0.0) or 0.0),
            mean_fpu_util_pct=float(node_dict.get("mean_fpu_util_pct", 0.0) or 0.0),
            mean_dram_bw_util_pct=float(node_dict.get("mean_dram_bw_util_pct", 0.0) or 0.0),
            mean_noc_util_pct=float(node_dict.get("mean_noc_util_pct", 0.0) or 0.0),
            mean_eth_bw_util_pct=float(node_dict.get("mean_eth_bw_util_pct", 0.0) or 0.0),
            program_cache_hit_rate=float(node_dict.get("program_cache_hit_rate", 0.0) or 0.0),
            row_indices=list(node_dict.get("row_indices", []) or []),
            op_code_counts=dict(node_dict.get("op_code_counts", {}) or {}),
            layer_index=node_dict.get("layer_index"),
        )

        suggestion = None
        sug_dict = sug_by_path.get(selected_module_path)
        if sug_dict is None:
            target_role = _role_key(selected_module_path)
            sug_dict = sug_by_role.get(target_role)
        if sug_dict is not None:
            ref = ModuleReference(
                reference_id=sug_dict.get("reference_id", ""),
                model_id=sug_dict.get("reference_model_id", ""),
                arch_family="",
                box="",
                mesh_shape=(0, 0),
                dtype="",
                source_run_id=sug_dict.get("reference_source_run_id"),
                curator="synthetic" if sug_dict.get("reference_quality") == "synthetic" else "curated",
                runtime_ms_p50=float(sug_dict.get("reference_runtime_ms_p50", 0.0)),
                fpu_util_pct=float(sug_dict.get("reference_fpu_util_pct", 0.0)),
            )
            suggestion = Suggestion(
                attribute_path=sug_dict.get("attribute_path", ""),
                module_class=sug_dict.get("module_class", ""),
                layer_index=sug_dict.get("layer_index"),
                user_runtime_ms_p50=float(sug_dict.get("user_runtime_ms_p50", 0.0)),
                user_fpu_util_pct=float(sug_dict.get("user_fpu_util_pct", 0.0)),
                user_op_count=int(sug_dict.get("user_op_count", 0)),
                reference=ref,
                delta_runtime_ms=float(sug_dict.get("delta_runtime_ms", 0.0)),
                delta_fpu_util_pct=float(sug_dict.get("delta_fpu_util_pct", 0.0)),
                is_significant=bool(sug_dict.get("is_significant", False)),
                proposed_blocks=list(sug_dict.get("proposed_blocks", []) or []),
                rationale=sug_dict.get("rationale", ""),
            )

        return build_inspector_for_node(node, suggestion)

    @app.callback(
        Output({"type": "chart", "name": "nsight_timeline"}, "figure"),
        Input("selected-module-path", "data"),
        State({"type": "chart", "name": "nsight_timeline"}, "figure"),
        prevent_initial_call=True,
    )
    def highlight_timeline_for_module(selected_module_path, fig):
        if not fig or "data" not in fig:
            return no_update
        if not selected_module_path:
            return no_update

        for trace in fig.get("data", []):
            cds = trace.get("customdata") or []
            if not cds:
                continue
            base_color = "#d29922" if str(trace.get("name", "")).startswith("baseline") else "#58a6ff"
            colors = []
            for cd in cds:
                module_path = cd[0] if isinstance(cd, list) and len(cd) >= 1 else ""
                colors.append("#f85149" if module_path == selected_module_path else base_color)
            trace.setdefault("marker", {})
            trace["marker"]["color"] = colors
        return fig

    @app.callback(
        Output("module-graph", "stylesheet"),
        Input("theme-mode", "data"),
        prevent_initial_call=True,
    )
    def highlight_graph_for_module(theme_mode):
        stylesheet = list(CYTOSCAPE_STYLESHEET)
        if theme_mode == "light":
            stylesheet += CYTO_LIGHT_OVERRIDES
        return stylesheet

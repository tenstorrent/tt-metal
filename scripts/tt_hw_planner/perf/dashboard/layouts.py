# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dash layout for the perf dashboard.

Mirrors the static report.html layout: topbar / left rail / main pane /
right inspector. The chart figures come from the same `charts/*.py`
functions the static HTML uses, so the two views stay in lockstep.

The headline view is the **module graph** (Cytoscape) — a node-link
representation of the model where each node is an nn.Module, colored
by utilization and sized by runtime. Clicking a node populates the
inspector with the per-module performance comparison and the
suggestion engine's recommendation. The classic plotly chart tabs
live to the right of the graph tab so the existing analysis workflows
keep working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import dash_cytoscape as cyto
from dash import dcc, html

from .cytoscape_layout import CYTOSCAPE_STYLESHEET, DEFAULT_CYTO_LAYOUT

cyto.load_extra_layouts()


DARK_BG = "var(--tt-bg)"
PANE_BG = "var(--tt-pane-bg)"
BORDER = "var(--tt-border)"
FG = "var(--tt-fg)"
FG_DIM = "var(--tt-fg-dim)"


def _topbar(
    run_id: str,
    model_id: str,
    box_name: str,
    mesh_shape,
    baseline_run_id: Optional[str] = None,
    baseline_run_options: Optional[List[Dict[str, str]]] = None,
) -> html.Div:
    baseline_text = ""
    if baseline_run_id:
        baseline_text = f" vs baseline {baseline_run_id}"
    return html.Div(
        id="dashboard-topbar",
        style={
            "padding": "10px 20px",
            "background": PANE_BG,
            "borderBottom": f"1px solid {BORDER}",
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
        },
        children=[
            html.Div(
                children=[
                    html.Span(
                        f"tt_hw_planner perf — {model_id}",
                        style={"fontSize": "16px", "fontWeight": "600", "color": "var(--tt-heading)"},
                    ),
                    html.Span(
                        f"  run {run_id}  ·  box {box_name}  ·  mesh {mesh_shape[0]}x{mesh_shape[1]}{baseline_text}",
                        style={"color": FG_DIM, "fontSize": "12px"},
                    ),
                ]
            ),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "12px"},
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Span("Baseline", style={"fontSize": "11px", "color": FG_DIM}),
                            dcc.Dropdown(
                                id="baseline-run-selector",
                                options=[{"label": "(none)", "value": ""}] + (baseline_run_options or []),
                                value=baseline_run_id or "",
                                clearable=False,
                                style={"width": "220px", "fontSize": "11px"},
                            ),
                            html.Span("Theme", style={"fontSize": "11px", "color": FG_DIM}),
                            dcc.Dropdown(
                                id="theme-selector",
                                options=[
                                    {"label": "Dark", "value": "dark"},
                                    {"label": "Light", "value": "light"},
                                ],
                                value="dark",
                                clearable=False,
                                style={"width": "120px", "fontSize": "11px"},
                            ),
                        ],
                    ),
                    html.Div(id="status-pill", children=[], style={"fontSize": "12px", "color": FG_DIM}),
                ],
            ),
        ],
    )


def _left_rail(catalog_entries: List[Tuple[str, int, str]]) -> html.Div:
    catalog_items = [
        html.Div(
            id={"type": "catalog-block", "name": name},
            n_clicks=0,
            children=[
                html.Span(name),
                html.Span(f"L{level}", style={"float": "right", "color": FG_DIM, "fontSize": "10px"}),
            ],
            style={
                "padding": "6px 8px",
                "borderRadius": "4px",
                "cursor": "pointer",
                "fontSize": "12px",
                "color": FG,
            },
        )
        for name, level, _desc in catalog_entries
    ]
    return html.Div(
        id="left-rail",
        style={"background": PANE_BG, "borderRight": f"1px solid {BORDER}", "padding": "12px", "overflow": "auto"},
        children=[
            html.H4(
                "Optimizer catalog",
                style={"color": "var(--tt-heading)", "fontSize": "12px", "textTransform": "uppercase"},
            ),
            *catalog_items,
            html.Div(
                id="selected-block-label",
                children="Selected block: (none)",
                style={
                    "marginTop": "8px",
                    "padding": "6px 8px",
                    "border": f"1px solid {BORDER}",
                    "borderRadius": "4px",
                    "fontSize": "11px",
                    "fontFamily": "ui-monospace, Menlo, monospace",
                    "color": FG_DIM,
                },
            ),
            html.Hr(style={"borderColor": BORDER}),
            html.H4("Actions", style={"color": "var(--tt-heading)", "fontSize": "12px", "textTransform": "uppercase"}),
            html.Button(
                "Re-collect",
                id="btn-recollect",
                n_clicks=0,
                style={
                    "width": "100%",
                    "marginTop": "6px",
                    "background": "var(--tt-accent-bg-soft)",
                    "color": "var(--tt-accent)",
                    "border": f"1px solid {BORDER}",
                    "padding": "6px",
                    "borderRadius": "4px",
                    "cursor": "pointer",
                },
            ),
            html.Button(
                "Refresh status",
                id="btn-refresh",
                n_clicks=0,
                style={
                    "width": "100%",
                    "marginTop": "6px",
                    "background": "transparent",
                    "color": FG,
                    "border": f"1px solid {BORDER}",
                    "padding": "6px",
                    "borderRadius": "4px",
                    "cursor": "pointer",
                },
            ),
            html.Div(id="action-log", style={"marginTop": "12px", "fontSize": "11px", "color": FG_DIM}),
        ],
    )


def _cytoscape_tab(
    cyto_elements: List[Dict[str, Any]],
    reference_options: List[Dict[str, str]],
    layer_options: List[Dict[str, str]],
    default_layer_value: str,
) -> html.Div:
    """The headline tab: interactive module graph + reference selector.

    The reference dropdown lets the user override which reference DB
    entry is used for the comparison (when multiple are present); the
    default is "auto" which lets the suggestion engine pick.
    """
    return html.Div(
        id="module-graph-tab",
        style={"display": "grid", "gridTemplateRows": "auto 1fr", "height": "100%"},
        children=[
            html.Div(
                id="module-graph-toolbar",
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "16px",
                    "padding": "8px 12px",
                    "borderBottom": f"1px solid {BORDER}",
                },
                children=[
                    html.Span("Reference:", style={"color": FG_DIM, "fontSize": "11px"}),
                    dcc.Dropdown(
                        id="reference-selector",
                        options=[{"label": "auto (best match)", "value": "auto"}] + reference_options,
                        value="auto",
                        clearable=False,
                        style={"width": "320px", "fontSize": "11px"},
                    ),
                    html.Span("Layer:", style={"color": FG_DIM, "fontSize": "11px"}),
                    dcc.Dropdown(
                        id="layer-focus-selector",
                        options=[{"label": "all layers", "value": "all"}] + layer_options,
                        value=default_layer_value,
                        clearable=False,
                        style={"width": "180px", "fontSize": "11px"},
                    ),
                    html.Button(
                        "Refresh Graph",
                        id="btn-refresh-graph",
                        n_clicks=0,
                        style={
                            "padding": "6px 10px",
                            "background": "transparent",
                            "color": FG,
                            "border": f"1px solid {BORDER}",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "fontSize": "11px",
                        },
                    ),
                    html.Div(style={"flex": 1}),
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "12px",
                            "fontSize": "10px",
                            "color": FG_DIM,
                            "fontFamily": "ui-monospace, Menlo, monospace",
                        },
                        children=[
                            html.Span("util:"),
                            *[
                                html.Span(
                                    [
                                        html.Span(
                                            "",
                                            style={
                                                "display": "inline-block",
                                                "width": "10px",
                                                "height": "10px",
                                                "background": color,
                                                "marginRight": "4px",
                                                "verticalAlign": "middle",
                                                "borderRadius": "2px",
                                            },
                                        ),
                                        label,
                                    ]
                                )
                                for label, color in [
                                    ("<25%", "#388bfd"),
                                    ("25-50%", "#7ee787"),
                                    ("50-75%", "#d29922"),
                                    ("75%+", "#f85149"),
                                ]
                            ],
                            html.Span("  ·  size: total runtime", style={"marginLeft": "8px"}),
                        ],
                    ),
                ],
            ),
            cyto.Cytoscape(
                id="module-graph",
                elements=cyto_elements,
                style={"width": "100%", "height": "100%", "background": DARK_BG},
                layout=DEFAULT_CYTO_LAYOUT,
                stylesheet=CYTOSCAPE_STYLESHEET,
                zoom=0.6,
                pan={"x": 40, "y": 30},
                minZoom=0.2,
                maxZoom=3.0,
                wheelSensitivity=0.2,
            ),
        ],
    )


def _tab_panel(
    charts: Dict[str, go.Figure],
    cyto_elements: List[Dict[str, Any]],
    reference_options: List[Dict[str, str]],
    layer_options: List[Dict[str, str]],
    default_layer_value: str,
) -> html.Div:
    chart_tabs = [
        dcc.Tab(
            label=k.replace("_", " ").title(),
            value=k,
            children=dcc.Graph(
                id={"type": "chart", "name": k},
                figure=fig if fig is not None else go.Figure(),
                style={"height": "640px"},
            ),
        )
        for k, fig in charts.items()
    ]

    tabs_children: List[Any] = [
        dcc.Tab(
            label="Module Graph",
            value="module_graph",
            children=html.Div(
                _cytoscape_tab(cyto_elements, reference_options, layer_options, default_layer_value),
                style={"height": "640px"},
            ),
        )
    ]
    default_value = "roofline"
    if cyto_elements:
        default_value = "module_graph"
    elif charts:
        default_value = next(iter(charts.keys()))

    tabs_children.extend(chart_tabs)

    tabs = dcc.Tabs(
        id="tabs",
        value=default_value,
        children=tabs_children,
        colors={"background": DARK_BG, "primary": "#1f6feb", "border": BORDER},
    )
    return html.Div(tabs, style={"overflow": "auto"})


def _inspector_pane() -> html.Div:
    return html.Div(
        id="inspector-pane",
        style={
            "background": PANE_BG,
            "borderLeft": f"1px solid {BORDER}",
            "padding": "16px",
            "overflow": "auto",
            "fontSize": "12px",
        },
        children=[
            html.H3("Inspector", style={"margin": "0 0 12px 0", "color": "var(--tt-heading)", "fontSize": "14px"}),
            html.Div(
                id="selected-node-label",
                children="Selected node: (none)",
                style={
                    "padding": "6px 8px",
                    "marginBottom": "8px",
                    "border": f"1px solid {BORDER}",
                    "borderRadius": "4px",
                    "fontSize": "11px",
                    "fontFamily": "ui-monospace, Menlo, monospace",
                    "color": FG_DIM,
                },
            ),
            html.Div(
                id="action-result-banner",
                children="",
                style={
                    "padding": "6px 8px",
                    "marginBottom": "8px",
                    "border": f"1px solid {BORDER}",
                    "borderRadius": "4px",
                    "fontSize": "11px",
                    "fontFamily": "ui-monospace, Menlo, monospace",
                    "color": FG_DIM,
                    "minHeight": "28px",
                },
            ),
            html.Div(
                id="inspector-body",
                children=[html.Em("Click a point on any chart to populate.", style={"color": FG_DIM})],
            ),
            html.Div(
                id="global-action-panel",
                style={"display": "block"},
                children=[
                    html.Hr(style={"borderColor": BORDER}),
                    html.H4("Apply / revert", style={"color": "var(--tt-heading)", "fontSize": "12px"}),
                    html.Div(
                        style={"display": "flex", "gap": "6px", "marginTop": "6px"},
                        children=[
                            html.Button(
                                "Apply",
                                id="btn-apply",
                                n_clicks=0,
                                style={
                                    "flex": 1,
                                    "background": "var(--tt-accent-bg-soft)",
                                    "color": "var(--tt-accent)",
                                    "border": f"1px solid {BORDER}",
                                    "padding": "6px",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Button(
                                "Revert",
                                id="btn-revert",
                                n_clicks=0,
                                style={
                                    "flex": 1,
                                    "background": "transparent",
                                    "color": FG,
                                    "border": f"1px solid {BORDER}",
                                    "padding": "6px",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Button(
                                "Dry-run",
                                id="btn-dryrun",
                                n_clicks=0,
                                style={
                                    "flex": 1,
                                    "background": "transparent",
                                    "color": FG,
                                    "border": f"1px solid {BORDER}",
                                    "padding": "6px",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Store(id="selected-cluster"),
            dcc.Store(id="selected-block"),
        ],
    )


def build_layout(
    run_id: str,
    model_id: str,
    box_name: str,
    mesh_shape,
    charts: Dict[str, go.Figure],
    catalog_entries: List[Tuple[str, int, str]],
    clusters_json: str,
    baseline_run_id: Optional[str] = None,
    cyto_elements: Optional[List[Dict[str, Any]]] = None,
    reference_options: Optional[List[Dict[str, str]]] = None,
    nodes_json: str = "[]",
    suggestions_json: str = "[]",
    node_index_json: Optional[Dict[str, Dict[str, Any]]] = None,
    suggestion_index_json: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    elements_json: str = "[]",
    layer_options: Optional[List[Dict[str, str]]] = None,
    default_layer_value: str = "all",
    baseline_run_options: Optional[List[Dict[str, str]]] = None,
) -> html.Div:
    return html.Div(
        id="dashboard-root",
        className="theme-dark",
        style={
            "background": DARK_BG,
            "color": FG,
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
        children=[
            _topbar(
                run_id,
                model_id,
                box_name,
                mesh_shape,
                baseline_run_id,
                baseline_run_options=baseline_run_options,
            ),
            html.Div(
                id="dashboard-main-grid",
                style={"display": "grid", "gridTemplateColumns": "240px 1fr 380px", "height": "calc(100vh - 50px)"},
                children=[
                    _left_rail(catalog_entries),
                    _tab_panel(
                        charts,
                        cyto_elements or [],
                        reference_options or [],
                        layer_options or [],
                        default_layer_value,
                    ),
                    _inspector_pane(),
                ],
            ),
            dcc.Store(id="clusters-data", data=clusters_json),
            dcc.Store(id="run-id", data=run_id),
            dcc.Store(id="module-nodes-data", data=nodes_json),
            dcc.Store(id="module-suggestions-data", data=suggestions_json),
            dcc.Store(id="module-node-index-data", data=node_index_json or {}),
            dcc.Store(id="module-suggestion-index-data", data=suggestion_index_json or {"by_path": {}, "by_role": {}}),
            dcc.Store(id="module-elements-data", data=elements_json),
            dcc.Store(id="selected-module-path"),
            dcc.Store(id="layer-options-data", data=layer_options or []),
            dcc.Store(id="theme-mode", data="dark"),
            dcc.Interval(id="status-refresh", interval=30_000, n_intervals=0),
        ],
    )

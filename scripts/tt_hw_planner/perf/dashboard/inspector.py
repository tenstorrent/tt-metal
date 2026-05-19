# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspector panel: the right-side detail view that appears when a user
clicks a node in the cytoscape graph.

Three sections, top-to-bottom:

  1. **Identity** — attribute path, module class, layer index, op count.
  2. **Performance** — your metrics vs reference metrics, side-by-side,
     with delta bars. Cleanly degraded if there's no matching reference
     (just shows "your" metrics + a hint to curate a reference).
  3. **Recommendation** — the proposed optimizer block(s), expected
     speedup, an Apply button.

Pure-function factories so this module can be imported without bringing
in the Dash app construction code; the callbacks layer calls these to
build content on demand.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from dash import html

from ..module_graph import ModuleGraph, ModuleNode
from ..suggestion_engine import Suggestion


# ---------------------------------------------------------------------------
# Color palette (matches cytoscape stylesheet)
# ---------------------------------------------------------------------------


_FG = "var(--tt-fg)"
_FG_DIM = "var(--tt-fg-dim)"
_BG = "var(--tt-pane-bg)"
_BORDER = "var(--tt-border)"
_BG_HOT = "var(--tt-hot-bg)"  # subtle red tint for "user worse"
_BG_GOOD = "var(--tt-good-bg)"  # subtle green tint for "user better"
_ACCENT = "var(--tt-accent)"
_BAR_TRACK = "var(--tt-bar-track, #21262d)"


# ---------------------------------------------------------------------------
# Section: identity
# ---------------------------------------------------------------------------


def _row(label: str, value: str, *, mono: bool = False, dim_value: bool = False) -> html.Div:
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between", "padding": "2px 0", "fontSize": "12px"},
        children=[
            html.Span(label, style={"color": _FG_DIM}),
            html.Span(
                value,
                style={
                    "color": _FG_DIM if dim_value else _FG,
                    "fontFamily": "ui-monospace, Menlo, Consolas, monospace" if mono else "inherit",
                    "fontWeight": "500",
                    "textAlign": "right",
                },
            ),
        ],
    )


def _section_identity(node: ModuleNode) -> html.Div:
    layer = "—" if node.layer_index is None else str(node.layer_index)
    op_codes_str = ", ".join(f"{c}×{op}" for op, c in sorted(node.op_code_counts.items(), key=lambda kv: -kv[1])[:4])
    return html.Div(
        style={"padding": "10px 0"},
        children=[
            html.Div(
                "Identity",
                style={
                    "color": "var(--tt-heading)",
                    "fontSize": "11px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "6px",
                },
            ),
            _row("Path", node.attribute_path, mono=True),
            _row("Class", node.module_class, mono=True),
            _row("Layer", layer),
            _row("Op count", str(node.op_count)),
            _row("Top ops", op_codes_str or "—", mono=True, dim_value=True),
        ],
    )


# ---------------------------------------------------------------------------
# Section: performance (your vs reference)
# ---------------------------------------------------------------------------


def _bar(value: float, max_value: float, color: str) -> html.Div:
    """A simple horizontal bar inside a track (no css framework needed)."""
    pct = max(0.0, min(1.0, (value / max_value) if max_value > 0 else 0.0))
    return html.Div(
        style={
            "flex": "1",
            "height": "6px",
            "background": _BAR_TRACK,
            "borderRadius": "3px",
            "overflow": "hidden",
            "margin": "0 6px",
        },
        children=html.Div(
            style={"width": f"{pct * 100:.0f}%", "height": "100%", "background": color, "borderRadius": "3px"}
        ),
    )


def _compare_row(
    label: str, you: float, ref: Optional[float], unit: str = "", max_value: Optional[float] = None
) -> html.Div:
    """Two-column metric row: you (left bar) vs reference (right bar)."""
    you_str = f"{you:.2f}{unit}"
    ref_str = f"{ref:.2f}{unit}" if ref is not None else "—"
    if max_value is None:
        max_value = max(you, ref or 0.0, 1.0)
    your_bar = _bar(you, max_value, _ACCENT)
    ref_bar = _bar(ref or 0.0, max_value, "#7ee787")
    return html.Div(
        style={"padding": "4px 0"},
        children=[
            html.Div(label, style={"fontSize": "10px", "color": _FG_DIM, "marginBottom": "2px"}),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "0"},
                children=[
                    html.Span(
                        you_str,
                        style={
                            "color": _ACCENT,
                            "fontSize": "11px",
                            "fontFamily": "ui-monospace, monospace",
                            "width": "60px",
                            "textAlign": "right",
                        },
                    ),
                    your_bar,
                    html.Span("vs", style={"color": _FG_DIM, "fontSize": "10px"}),
                    ref_bar,
                    html.Span(
                        ref_str,
                        style={
                            "color": "#7ee787",
                            "fontSize": "11px",
                            "fontFamily": "ui-monospace, monospace",
                            "width": "60px",
                            "textAlign": "left",
                        },
                    ),
                ],
            ),
        ],
    )


def _section_performance(node: ModuleNode, suggestion: Optional[Suggestion]) -> html.Div:
    user_runtime_ms = node.median_device_ns / 1e6
    user_fpu = node.mean_fpu_util_pct
    user_dram = node.mean_dram_bw_util_pct

    if suggestion is not None:
        ref = suggestion.reference
        ref_runtime = ref.runtime_ms_p50
        ref_fpu = ref.fpu_util_pct
        ref_dram = ref.dram_bw_util_pct
        ref_label_text = (
            f"Reference: {ref.model_id}  ·  {ref.box}  ·  mesh {ref.mesh_shape}  ·  {ref.dtype}  "
            f"· quality={ref.quality}"
        )
        if ref.source_run_id:
            ref_label_text += f"  · source_run={ref.source_run_id}"
        delta_color = _BG_HOT if suggestion.delta_runtime_ms > 0 else _BG_GOOD
    else:
        ref_runtime = None
        ref_fpu = None
        ref_dram = None
        ref_label_text = "Reference: (no match in DB — see references/README.md to curate one)"
        delta_color = _BG

    return html.Div(
        style={
            "padding": "10px 0",
            "background": delta_color,
            "borderRadius": "4px",
            "marginTop": "8px",
            "padding": "8px",
        },
        children=[
            html.Div(
                "Performance",
                style={
                    "color": "var(--tt-heading)",
                    "fontSize": "11px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "6px",
                },
            ),
            html.Div(
                ref_label_text,
                style={
                    "color": _FG_DIM,
                    "fontSize": "10px",
                    "fontFamily": "ui-monospace, monospace",
                    "marginBottom": "4px",
                },
            ),
            _compare_row("Median runtime", user_runtime_ms, ref_runtime, " ms"),
            _compare_row("FPU util", user_fpu, ref_fpu, "%", max_value=100.0),
            _compare_row("DRAM BW util", user_dram, ref_dram, "%", max_value=100.0),
        ],
    )


# ---------------------------------------------------------------------------
# Section: recommendation
# ---------------------------------------------------------------------------


def _section_recommendation(suggestion: Optional[Suggestion]) -> html.Div:
    status_text = "Unknown"
    status_color = "#7d8590"
    if suggestion is not None:
        if suggestion.delta_runtime_ms > 0 and suggestion.is_significant:
            status_text = "Needs Optimization"
            status_color = "#f85149"
        elif suggestion.delta_runtime_ms <= 0:
            status_text = "Optimized / At Reference"
            status_color = "#2ea043"
        else:
            status_text = "Near Reference"
            status_color = "#d29922"

    if suggestion is None:
        return html.Div(
            style={"padding": "10px 0"},
            children=[
                html.Div(
                    "Recommendation",
                    style={
                        "color": "var(--tt-heading)",
                        "fontSize": "11px",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.05em",
                        "marginBottom": "6px",
                    },
                ),
                html.Div(
                    [
                        html.Span("Hardware optimization status: ", style={"color": _FG_DIM}),
                        html.Span(status_text, style={"color": status_color, "fontWeight": "700"}),
                    ],
                    style={"fontSize": "11px", "marginBottom": "8px"},
                ),
                html.Em(
                    "No reference matched this module. The dashboard can still show "
                    "this submodule's standalone metrics, but it cannot recommend a "
                    "specific optimizer block until a reference is curated.",
                    style={"color": _FG_DIM, "fontSize": "11px"},
                ),
            ],
        )

    proposed_chips = [
        html.Span(
            block,
            style={
                "display": "inline-block",
                "padding": "3px 8px",
                "margin": "2px 4px 2px 0",
                "background": "var(--tt-accent-bg-soft)",
                "color": _ACCENT,
                "border": f"1px solid {_BORDER}",
                "borderRadius": "10px",
                "fontFamily": "ui-monospace, monospace",
                "fontSize": "11px",
            },
        )
        for block in suggestion.proposed_blocks
    ] or [html.Em("(no candidate blocks listed)", style={"color": _FG_DIM, "fontSize": "11px"})]

    return html.Div(
        style={"padding": "10px 0"},
        children=[
            html.Div(
                "Recommendation",
                style={
                    "color": "var(--tt-heading)",
                    "fontSize": "11px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "6px",
                },
            ),
            html.Div(
                [
                    html.Span("Hardware optimization status: ", style={"color": _FG_DIM}),
                    html.Span(status_text, style={"color": status_color, "fontWeight": "700"}),
                ],
                style={"fontSize": "11px", "marginBottom": "8px"},
            ),
            html.Div(
                suggestion.rationale,
                style={"color": _FG, "fontSize": "11px", "lineHeight": "1.5", "marginBottom": "8px"},
            ),
            html.Div("Apply one of:", style={"color": _FG_DIM, "fontSize": "10px", "marginBottom": "4px"}),
            html.Div(proposed_chips),
        ],
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_inspector_for_node(
    node: Optional[ModuleNode],
    suggestion: Optional[Suggestion],
) -> List[Any]:
    """Top-level: produce the children list for the cytoscape inspector
    pane given a clicked-on ``ModuleNode`` (or None if nothing is
    selected)."""
    if node is None:
        return [html.Em("Click a node to populate.", style={"color": _FG_DIM, "fontSize": "11px"})]

    return [
        _section_identity(node),
        _section_performance(node, suggestion),
        _section_recommendation(suggestion),
    ]


def build_empty_inspector(reason: str = "") -> List[Any]:
    """When the dashboard loaded a run without module-hierarchy data."""
    msg = reason or "Click a node to populate."
    return [html.Em(msg, style={"color": _FG_DIM, "fontSize": "11px"})]


__all__ = ["build_inspector_for_node", "build_empty_inspector"]

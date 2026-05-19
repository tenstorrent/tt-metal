# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert a ModuleGraph into Cytoscape-flavored node/edge JSON + a
stylesheet that gives the dashboard's headline tab a polished
Nsight-like look.

Cytoscape's data model:

  elements = [
    {"data": {"id": "...", "label": "...", "parent": "...", ...}},
    {"data": {"id": "edge-1", "source": "...", "target": "..."}},
    ...
  ]

We use the ``parent`` field to express the module hierarchy — Cytoscape
renders each parent as a compound node (a box) containing its children.
This gives the user a collapsible hierarchical view for free.

Color/size mappings live in the stylesheet (``CYTOSCAPE_STYLESHEET``)
and reference data attributes set on each node:

  ``util_pct``        - 0..100, drives a green→yellow→red gradient.
  ``runtime_norm``    - 0..1, drives node size.
  ``layer_index``     - integer or null; the layout uses it for grouping.
  ``op_count``        - int; shown in tooltip.

All values come from ``ModuleNode`` (direct, not subtree) for LEAF
nodes, and from ``ModuleGraph.subtree_metrics`` for INTERIOR / parent
nodes — so a wrapper module's color summarizes its descendants. This
matches the way Nsight-style tools surface "hot regions".
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from ..module_graph import ModuleGraph, ModuleNode


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _short_label(attribute_path: str) -> str:
    """Shorten the path for display on the node itself.

    "model.decoder.layers.0.self_attn.q_proj" -> "q_proj"
    "model.decoder.layers.0.self_attn"        -> "self_attn"
    "layers.0"                                 -> "layer 0"
    ""                                          -> "root"
    """
    if not attribute_path:
        return "root"
    parts = attribute_path.split(".")
    # Special-case "layers.<N>" to "layer N" — nicer than "0", "1", etc.
    if len(parts) >= 2 and parts[-2] in ("layers", "h", "blocks") and parts[-1].isdigit():
        return f"layer {parts[-1]}"
    return parts[-1]


def _util_color_data(util_pct: float) -> str:
    """Map util to a category the stylesheet colors via a single mapping.

    Cytoscape's stylesheet doesn't do continuous interpolation as easily
    as plotly, so we bin into discrete classes the stylesheet styles
    explicitly.
    """
    if util_pct >= 75:
        return "hot"
    if util_pct >= 50:
        return "warm"
    if util_pct >= 25:
        return "cool"
    return "cold"


def _node_data_for_leaf(node: ModuleNode, runtime_norm_max: float) -> Dict[str, Any]:
    util = node.mean_fpu_util_pct
    runtime_norm = (node.total_device_ns / runtime_norm_max) if runtime_norm_max > 0 else 0.0
    return {
        "id": node.attribute_path,
        "label": _short_label(node.attribute_path),
        "module_class": node.module_class,
        "op_role": node.attribute_path.rsplit(".", 1)[-1] if "." in node.attribute_path else node.attribute_path,
        "util_pct": round(util, 1),
        "util_bin": _util_color_data(util),
        "runtime_ms": round(node.total_device_ns / 1e6, 3),
        "runtime_norm": round(runtime_norm, 3),
        "op_count": node.op_count,
        "layer_index": node.layer_index,
        "layer_band": f"b{(node.layer_index or 0) % 8}" if node.layer_index is not None else "b0",
        "median_ns": round(node.median_device_ns, 0),
        "dram_bw_util_pct": round(node.mean_dram_bw_util_pct, 1),
        "noc_util_pct": round(node.mean_noc_util_pct, 1),
        "kind": "leaf",
    }


def _node_data_for_parent(node: ModuleNode, graph: ModuleGraph, runtime_norm_max: float) -> Dict[str, Any]:
    subtree = graph.subtree_metrics(node.attribute_path)
    util = subtree["subtree_fpu_util_pct"]
    runtime_norm = (subtree["subtree_total_device_ns"] / runtime_norm_max) if runtime_norm_max > 0 else 0.0
    return {
        "id": node.attribute_path,
        "label": _short_label(node.attribute_path),
        "module_class": node.module_class,
        "op_role": node.attribute_path.rsplit(".", 1)[-1] if "." in node.attribute_path else node.attribute_path,
        "util_pct": round(util, 1),
        "util_bin": _util_color_data(util),
        "runtime_ms": round(subtree["subtree_total_device_ns"] / 1e6, 3),
        "runtime_norm": round(runtime_norm, 3),
        "op_count": int(subtree["subtree_op_count"]),
        "layer_index": node.layer_index,
        "layer_band": f"b{(node.layer_index or 0) % 8}" if node.layer_index is not None else "b0",
        "median_ns": round(node.median_device_ns, 0),
        "dram_bw_util_pct": round(subtree["subtree_dram_bw_util_pct"], 1),
        "noc_util_pct": round(subtree["subtree_noc_util_pct"], 1),
        "kind": "parent",
    }


def graph_to_cytoscape_elements(
    graph: ModuleGraph,
    *,
    include_root_synthetic: bool = True,
) -> List[Dict[str, Any]]:
    """Convert a ``ModuleGraph`` into the list Cytoscape consumes.

    Cytoscape's compound-node model expects each child node to carry a
    ``parent`` field pointing at the parent node's id. We set that
    field on every node whose ``parent_path`` exists in the graph.

    ``include_root_synthetic`` adds a single ROOT node above every
    top-level node so the layout always has one entry point — helpful
    for the "fit to viewport" behaviour.
    """
    if not graph.nodes:
        return []

    # Compute the runtime normalizer once — total wall-time of any one
    # node (subtree) capped at the largest. This gives node sizes a
    # consistent visual budget.
    runtime_max = 0.0
    for path in graph.nodes:
        st = graph.subtree_metrics(path)
        if st["subtree_total_device_ns"] > runtime_max:
            runtime_max = st["subtree_total_device_ns"]
    runtime_max = max(runtime_max, 1.0)

    elements: List[Dict[str, Any]] = []

    # Precompute a deterministic role index for leaf nodes so we can place
    # siblings consistently in "preset" layout mode.
    role_names = sorted({p.rsplit(".", 1)[-1] for p, n in graph.nodes.items() if n.op_count > 0 and "." in p})
    role_to_idx = {r: i for i, r in enumerate(role_names)}

    def _position_for(path: str, node: ModuleNode) -> Dict[str, float]:
        # Columnar tree:
        #   x grows by layer index (each layer is its own vertical column).
        #   y grows by role index so each layer reads top->bottom.
        parts = path.split(".")
        depth = max(0, len(parts) - 1)
        layer = node.layer_index if node.layer_index is not None else -1

        # Default root/container placement.
        x = 120.0 + depth * 280.0
        y = 80.0 + depth * 110.0

        # Leaf nodes: column per layer, row per role.
        if node.op_count > 0:
            role = parts[-1] if parts else ""
            ridx = role_to_idx.get(role, 0)
            x = 260.0 + max(layer, 0) * 150.0
            y = 220.0 + ridx * 74.0
        elif path.endswith("layers"):
            # Decoder/layers container spans top area.
            x = 220.0
            y = 90.0
        elif layer >= 0 and path.count(".") == 2:
            # decoder.layers.<N> container anchors each layer column.
            x = 260.0 + layer * 150.0
            y = 150.0
        return {"x": x, "y": y}

    if include_root_synthetic:
        elements.append(
            {
                "data": {
                    "id": "__root__",
                    "label": "model",
                    "module_class": "Model",
                    "util_pct": 0.0,
                    "util_bin": "cold",
                    "runtime_ms": round(runtime_max / 1e6, 3),
                    "runtime_norm": 1.0,
                    "op_count": sum(n.op_count for n in graph.nodes.values()),
                    "layer_index": None,
                    "kind": "root",
                },
                "position": {"x": 900.0, "y": 20.0},
            }
        )

    children_set = {e.child_path for e in graph.edges}
    for path, node in graph.nodes.items():
        is_parent = bool(graph.children_of.get(path))
        data = _node_data_for_parent(node, graph, runtime_max) if is_parent else _node_data_for_leaf(node, runtime_max)
        if path in graph.parent_of:
            data["parent"] = graph.parent_of[path]
        elif include_root_synthetic and path not in children_set:
            # Top-level nodes attach to the synthetic root for a single
            # tree, which the user can collapse to declutter.
            data["parent"] = "__root__"
        elements.append({"data": data, "position": _position_for(path, node)})

    # Emit hierarchy edges so layout engines (especially breadthfirst)
    # have explicit structure. Without these, Cytoscape can flatten all
    # nodes into one rank because compound nesting alone is not enough
    # for stable top-down placement on larger graphs.
    for e in graph.edges:
        elements.append(
            {
                "data": {
                    "id": f"edge:{e.parent_path}->{e.child_path}",
                    "source": e.parent_path,
                    "target": e.child_path,
                    "kind": "hierarchy_edge",
                }
            }
        )

    if include_root_synthetic:
        for root in graph.root_paths():
            elements.append(
                {
                    "data": {
                        "id": f"edge:__root__->{root}",
                        "source": "__root__",
                        "target": root,
                        "kind": "hierarchy_edge",
                    }
                }
            )

    return elements


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------


# Colors chosen for AAA dark-theme contrast. Heat scale matches the
# rest of the dashboard (red = hot, green = idle).
_HEAT = {
    "hot": "#ff6b57",  # bottleneck
    "warm": "#f0b429",  # medium pressure
    "cool": "#4ecdc4",  # healthy
    "cold": "#7f8ea3",  # idle-ish neutral slate
}


CYTOSCAPE_STYLESHEET: List[Dict[str, Any]] = [
    # Root container
    {
        "selector": "node[kind = 'root']",
        "style": {
            "background-color": "#121722",
            "border-color": "#2f3a4a",
            "border-width": 1,
            "label": "data(label)",
            "color": "#d9e2ef",
            "font-size": 11,
            "font-family": "ui-monospace, Menlo, Consolas, monospace",
            "text-valign": "top",
            "text-margin-y": -8,
            "padding": "18px",
            "shape": "round-rectangle",
            "background-opacity": 0.85,
            "min-zoomed-font-size": 8,
        },
    },
    # Parent / compound nodes (carry children)
    {
        "selector": "node[kind = 'parent']",
        "style": {
            "background-color": "#171d2a",
            "border-color": "#2f3a4a",
            "border-width": 1,
            "shape": "round-rectangle",
            "padding": "12px",
            "label": "data(label)",
            "color": "#c9d1d9",
            "font-size": 10,
            "font-family": "ui-monospace, Menlo, Consolas, monospace",
            "text-valign": "top",
            "text-margin-y": -4,
            "min-zoomed-font-size": 8,
        },
    },
    # Layer container tinting (subtle) so different layers are easier to scan.
    {"selector": "node[kind = 'parent'][layer_band = 'b0']", "style": {"background-color": "#1a2231"}},
    {"selector": "node[kind = 'parent'][layer_band = 'b1']", "style": {"background-color": "#1c2432"}},
    {"selector": "node[kind = 'parent'][layer_band = 'b2']", "style": {"background-color": "#1d2634"}},
    {"selector": "node[kind = 'parent'][layer_band = 'b3']", "style": {"background-color": "#1f2735"}},
    {"selector": "node[kind = 'parent'][layer_band = 'b4']", "style": {"background-color": "#202a38"}},
    {"selector": "node[kind = 'parent'][layer_band = 'b5']", "style": {"background-color": "#222b3a"}},
    {"selector": "node[kind = 'parent'][layer_band = 'b6']", "style": {"background-color": "#232d3b"}},
    {"selector": "node[kind = 'parent'][layer_band = 'b7']", "style": {"background-color": "#25303e"}},
    # Leaf nodes (have direct ops). Color by util_bin, size by runtime.
    {
        "selector": "node[kind = 'leaf']",
        "style": {
            "background-color": "#232b39",
            "border-color": "#516179",
            "border-width": 2,
            "label": "data(label)",
            "color": "#dce6f5",
            "font-size": 9,
            "font-family": "ui-monospace, Menlo, Consolas, monospace",
            "text-valign": "center",
            "text-halign": "center",
            "text-wrap": "wrap",
            "text-max-width": 80,
            # Size proportional to runtime norm in [0,1].
            "width": "mapData(runtime_norm, 0, 1, 40, 110)",
            "height": "mapData(runtime_norm, 0, 1, 28, 62)",
            "min-zoomed-font-size": 7,
        },
    },
    # Operation-role colors (leaf fill) for quick op-type distinction.
    {"selector": "node[kind = 'leaf'][op_role = 'matmul']", "style": {"background-color": "#1d4e89"}},
    {"selector": "node[kind = 'leaf'][op_role = 'sdpa']", "style": {"background-color": "#365d2b"}},
    {"selector": "node[kind = 'leaf'][op_role = 'norm']", "style": {"background-color": "#6a3d1a"}},
    {"selector": "node[kind = 'leaf'][op_role = 'layout_move']", "style": {"background-color": "#4d3f7a"}},
    {"selector": "node[kind = 'leaf'][op_role = 'all_gather']", "style": {"background-color": "#0f5a6d"}},
    {"selector": "node[kind = 'leaf'][op_role = 'reduce_scatter']", "style": {"background-color": "#245f55"}},
    {"selector": "node[kind = 'leaf'][op_role = 'binary']", "style": {"background-color": "#5a3042"}},
    {"selector": "node[kind = 'leaf'][op_role = 'typecast']", "style": {"background-color": "#3d556b"}},
    {"selector": "node[kind = 'leaf'][op_role = 'qkv_heads']", "style": {"background-color": "#5d4a1f"}},
    {"selector": "node[kind = 'leaf'][op_role = 'concat_heads']", "style": {"background-color": "#6a3c2c"}},
    {"selector": "node[kind = 'leaf'][op_role = 'rotary']", "style": {"background-color": "#2f5a34"}},
    {"selector": "node[kind = 'leaf'][op_role = 'other']", "style": {"background-color": "#2f3648"}},
    # Heat-binning: override background color by util_bin
    {"selector": "node[kind = 'leaf'][util_bin = 'hot']", "style": {"border-color": _HEAT["hot"], "border-width": 4}},
    {"selector": "node[kind = 'leaf'][util_bin = 'warm']", "style": {"border-color": _HEAT["warm"], "border-width": 4}},
    {"selector": "node[kind = 'leaf'][util_bin = 'cool']", "style": {"border-color": _HEAT["cool"], "border-width": 4}},
    {"selector": "node[kind = 'leaf'][util_bin = 'cold']", "style": {"border-color": _HEAT["cold"], "border-width": 3}},
    # Optimization status (from reference comparison) takes precedence
    # over util-bin colors so the graph answers "optimized for HW?" at
    # a glance.
    {
        "selector": "node[kind = 'leaf'][opt_status = 'optimized']",
        "style": {"border-color": "#2ea043", "border-width": 5},
    },
    {
        "selector": "node[kind = 'leaf'][opt_status = 'near_ref']",
        "style": {"border-color": "#d29922", "border-width": 5},
    },
    {
        "selector": "node[kind = 'leaf'][opt_status = 'needs_opt']",
        "style": {"border-color": "#f85149", "border-width": 5},
    },
    {
        "selector": "node[kind = 'leaf'][opt_status = 'unknown']",
        "style": {"border-color": "#7d8590", "border-width": 4},
    },
    # Hover
    {
        "selector": "node:active",
        "style": {
            "overlay-color": "#1f6feb",
            "overlay-opacity": 0.2,
            "overlay-padding": 3,
        },
    },
    # Selected state — bright outline so the user knows what the
    # inspector pane is showing.
    {
        "selector": "node:selected",
        "style": {
            "border-color": "#ff9f1c",
            "border-width": 5,
        },
    },
    # Edges (none currently emitted, but ready if we add them later)
    {
        "selector": "edge",
        "style": {
            "width": 1.5,
            "line-color": "#3a4455",
            "target-arrow-color": "#3a4455",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "opacity": 0.25,
        },
    },
]


# Cytoscape layout.
# Use `dagre` (loaded via `dash_cytoscape.load_extra_layouts()` in
# dashboard/layouts.py) for deterministic top-to-bottom (TB) hierarchy.
# This avoids the "single horizontal strip" effect from breadthfirst on
# dense, wide module trees.
DEFAULT_CYTO_LAYOUT: Dict[str, Any] = {
    "name": "preset",
    # Keep caller in control of viewport; fit=True tends to compress
    # large graphs into a thin strip.
    "fit": False,
    "padding": 16,
    "animate": False,
}


__all__ = [
    "CYTOSCAPE_STYLESHEET",
    "DEFAULT_CYTO_LAYOUT",
    "graph_to_cytoscape_elements",
]

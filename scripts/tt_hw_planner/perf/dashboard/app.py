# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dash app entry point.

The app reuses the same eight figure functions as the static report and
the same `OptimizerBlock` catalog as the CLI. Buttons in the inspector
trigger callbacks in `dashboard/callbacks.py` that funnel through the
`runner.apply_block` / `runner.revert_block` code paths — the user sees
the same effect whether they click here or run `tt_hw_planner perf apply`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dash import Dash

from ..ceilings import load_box_spec
from ..charts.block_stack import make_block_stack
from ..charts.block_util_runtime import make_block_util_runtime
from ..charts.cache_heatmap import make_cache_heatmap
from ..charts.config_scatter import make_config_scatter
from ..charts.hash_diff import make_hash_diff
from ..charts.risc_stack import make_risc_stack
from ..charts.roofline import make_roofline
from ..charts.sol_bars import make_sol_bars
from ..charts.waterfall import make_waterfall
from ..cluster import cluster_rows
from ..collect import find_run, load_run_meta, require_healthy_run
from ..join import join_run
from ..module_graph import build_module_graph, parse_hierarchy_sidecar
from ..optimizers.catalog import catalog_for_sidebar
from ..reference_db import list_module_references
from ..regions import classify_all
from ..suggestion_engine import propose_optimizations
from .cytoscape_layout import graph_to_cytoscape_elements
from .layouts import build_layout
from .timeline import make_nsight_timeline


def _filter_elements_by_layer(elements: List[Dict[str, Any]], layer_value: Optional[str]) -> List[Dict[str, Any]]:
    if not elements:
        return []
    if not layer_value or layer_value == "all":
        return elements
    target = str(layer_value).zfill(2)

    def _layer_from_node_id(node_id: str) -> str:
        import re

        m = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", node_id or "")
        return m.group(1).zfill(2) if m else ""

    node_ids: set[str] = set()
    node_elements: List[Dict[str, Any]] = []
    edge_elements: List[Dict[str, Any]] = []
    for e in elements:
        d = e.get("data") or {}
        if "source" in d and "target" in d:
            edge_elements.append(e)
        else:
            node_elements.append(e)

    for e in node_elements:
        d = e.get("data") or {}
        nid = str(d.get("id") or "")
        if not nid:
            continue
        if _layer_from_node_id(nid) != target:
            continue
        cur = nid
        node_ids.add(cur)
        while "." in cur:
            cur = cur.rsplit(".", 1)[0]
            node_ids.add(cur)
        node_ids.add("__root__")
    for keep in ("model", "decoder", "decoder.layers", "layers", "root", "__root__"):
        node_ids.add(keep)

    filtered: List[Dict[str, Any]] = []
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


def _clusters_payload(rows, clusters) -> str:
    rows_by_cluster = {r.cluster_id: r for r in rows}
    payload: List[dict] = []
    for c in clusters:
        s = rows_by_cluster.get(c.cluster_id)
        payload.append(
            {
                "cluster_id": c.cluster_id,
                "op_code": c.op_code,
                "args_hash": c.args_hash,
                "compute_kernel_hash": c.compute_kernel_hash,
                "shape_signature": c.shape_signature,
                "math_fidelity": c.math_fidelity,
                "n_calls": c.n_calls,
                "median_device_ns": c.median_device_ns,
                "total_device_ns": c.total_device_ns,
                "pct_of_peak": c.percent_of_peak,
                "mean_fpu_util_pct": c.mean_fpu_util_pct,
                "mean_dram_bw_util_pct": c.mean_dram_bw_util_pct,
                "mean_noc_util_pct": c.mean_noc_util_pct,
                "mean_eth_bw_util_pct": c.mean_eth_bw_util_pct,
                "program_cache_hit_rate": c.program_cache_hit_rate,
                "blocks": c.blocks,
                "arguments_example": c.arguments_example,
                "region": s.region if s else None,
                "region_reason": s.region_reason if s else None,
                "compute_kernel_source": s.compute_kernel_source if s else "",
                "dm_kernel_source": s.dm_kernel_source if s else "",
            }
        )
    return json.dumps(payload)


def _baseline_run_options(current_run_id: str, run_dir: Path) -> List[Dict[str, str]]:
    run_root = run_dir.parent
    options: List[Dict[str, str]] = []
    for p in sorted(run_root.glob("run_*"), key=lambda x: x.name, reverse=True):
        rid = p.name
        if rid == current_run_id:
            continue
        options.append({"label": rid, "value": rid})
    return options


def _role_key(path: str) -> str:
    return ".".join("*" if p.isdigit() else p for p in (path or "").split("."))


def build_app(run_id: str, baseline_run_id: Optional[str] = None, run_dir_root: Optional[Path] = None) -> Dash:
    run_dir = find_run(run_id, run_dir_root)
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
    if baseline_run_id:
        try:
            bdir = find_run(baseline_run_id, run_dir_root)
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
        except FileNotFoundError:
            baseline_rows = None

    charts = {
        "nsight_timeline": make_nsight_timeline(rows, baseline_rows=baseline_rows),
        "per_block_util_vs_runtime": make_block_util_runtime(rows, baseline_rows=baseline_rows, util_axis="FPU"),
        "per_block_dram": make_block_util_runtime(rows, baseline_rows=baseline_rows, util_axis="DRAM"),
        "roofline": make_roofline(rows, clusters, box),
        "speed_of_light": make_sol_bars(clusters),
        "per_risc": make_risc_stack(rows, clusters),
        "per_block": make_block_stack(rows, clusters),
        "config_scatter": make_config_scatter(clusters),
        "waterfall": make_waterfall(rows),
        "cache_heatmap": make_cache_heatmap(rows),
        "hash_diff": make_hash_diff(rows, clusters, baseline_rows),
    }

    catalog_entries = catalog_for_sidebar()
    clusters_json = _clusters_payload(rows, clusters)

    hierarchy_path = run_dir / "ttnn_module_hierarchy.json"
    module_graph = (
        build_module_graph(rows, parse_hierarchy_sidecar(hierarchy_path)) if hierarchy_path.exists() else None
    )
    cyto_elements: List[Dict[str, Any]] = []
    suggestions_payload: List[Dict[str, Any]] = []
    nodes_payload: List[Dict[str, Any]] = []
    node_index_payload: Dict[str, Dict[str, Any]] = {}
    suggestion_index_payload: Dict[str, Dict[str, Any]] = {"by_path": {}, "by_role": {}}
    layer_options: List[Dict[str, str]] = []
    default_layer_value = "all"
    if module_graph is not None and module_graph.nodes:
        all_cyto_elements = graph_to_cytoscape_elements(module_graph)
        from ..cli import _infer_arch_family

        arch_family = _infer_arch_family(meta.get("model_id", ""))
        try:
            suggestions = propose_optimizations(
                graph=module_graph,
                arch_family=arch_family,
                mesh_shape=tuple(meta["mesh_shape"]),  # type: ignore[arg-type]
                dtype=meta.get("dtype"),
                box=meta.get("box"),
            )
            suggestions_payload = [s.to_dict() for s in suggestions]
        except Exception as exc:
            print(f"dashboard: suggestion engine failed: {exc}")
            suggestions_payload = []
        nodes_payload = [asdict(n) for n in module_graph.nodes.values()]
        node_index_payload = {str(n.get("attribute_path") or ""): n for n in nodes_payload if n.get("attribute_path")}
        suggestion_index_payload["by_path"] = {
            str(s.get("attribute_path") or ""): s for s in suggestions_payload if s.get("attribute_path")
        }
        for s in suggestions_payload:
            path = str(s.get("attribute_path") or "")
            if not path:
                continue
            rk = _role_key(path)
            suggestion_index_payload["by_role"].setdefault(rk, s)
        layers = sorted({n.layer_index for n in module_graph.nodes.values() if n.layer_index is not None})
        layer_options = [{"label": f"layer {int(i):02d}", "value": f"{int(i):02d}"} for i in layers]
        if layer_options:
            default_layer_value = layer_options[0]["value"]

        sug_by_path = {s["attribute_path"]: s for s in suggestions_payload}
        for e in all_cyto_elements:
            d = e.get("data") or {}
            if d.get("kind") != "leaf":
                continue
            nid = str(d.get("id") or "")
            s = sug_by_path.get(nid)
            if s is None:
                d["opt_status"] = "unknown"
                continue
            delta = float(s.get("delta_runtime_ms", 0.0))
            significant = bool(s.get("is_significant", False))
            if delta > 0.0 and significant:
                d["opt_status"] = "needs_opt"
            elif delta <= 0.0:
                d["opt_status"] = "optimized"
            else:
                d["opt_status"] = "near_ref"

        cyto_elements = _filter_elements_by_layer(all_cyto_elements, default_layer_value)
    else:
        all_cyto_elements = []

    reference_options = [
        {"label": f"{rid}  ({mid}, {fam}, mesh={list(mesh)}, {dtype})", "value": rid}
        for rid, mid, fam, mesh, dtype in list_module_references()
    ]
    baseline_run_options = _baseline_run_options(run_id, run_dir)

    app = Dash(
        __name__,
        title=f"tt_perf — {run_id}",
        update_title=None,
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(
        run_id=run_id,
        model_id=meta["model_id"],
        box_name=meta["box"],
        mesh_shape=tuple(meta["mesh_shape"]),
        charts=charts,
        catalog_entries=catalog_entries,
        clusters_json=clusters_json,
        baseline_run_id=baseline_run_id,
        cyto_elements=cyto_elements,
        reference_options=reference_options,
        nodes_json=json.dumps(nodes_payload, default=str),
        suggestions_json=json.dumps(suggestions_payload, default=str),
        node_index_json=node_index_payload,
        suggestion_index_json=suggestion_index_payload,
        elements_json=json.dumps(all_cyto_elements, default=str),
        layer_options=layer_options,
        default_layer_value=default_layer_value,
        baseline_run_options=baseline_run_options,
    )

    from . import callbacks

    callbacks.register_callbacks(app)
    return app


def run_dashboard(*, run_id: str, host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    app = build_app(run_id)
    app.run_server(host=host, port=port, debug=debug)

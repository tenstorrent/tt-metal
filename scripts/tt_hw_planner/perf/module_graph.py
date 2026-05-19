# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Build a node-graph view of the model from joined-rows + hierarchy sidecar.

The cytoscape headline view, the inspector pane, the reference-comparison
logic, and the suggestion engine all consume ``ModuleGraph``. Building it
here means none of those downstream consumers need to know about the
sidecar's JSON shape.

A ``ModuleGraph`` has:

- Nodes: one per unique attribute path (e.g. ``model.layers.0.self_attn``).
  Each node aggregates the perf metrics of every op that ran inside that
  submodule (and *only* inside that submodule, not its descendants). This
  makes the per-layer comparison clean — clicking ``layers.0.self_attn``
  shows only attention work, not the whole layer.

- Edges: parent → child relationships derived from the dot-separated
  attribute paths. (Tensor-dataflow edges would require ttnn-side tensor
  IDs we don't currently emit; the parent/child tree is the practical
  approximation that still gives a useful graph.)

The module-class field is also captured; that's what the reference DB
matches on when computing comparisons.

Generic across any HF model. Works on runs that DO have the hierarchy
sidecar; for runs that don't, ``build_module_graph`` returns an empty
graph and downstream code falls back to the flat per-op view.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .join import JoinedRow


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ModuleNode:
    """One node in the module graph. Self-aggregates (not subtree)."""

    attribute_path: str
    module_class: str

    # Direct-only aggregates over all ops attributed to this module.
    # Subtree aggregates are computed lazily by the chart/inspector when
    # rendering a parent.
    op_count: int = 0
    total_device_ns: float = 0.0
    median_device_ns: float = 0.0
    mean_fpu_util_pct: float = 0.0
    mean_dram_bw_util_pct: float = 0.0
    mean_noc_util_pct: float = 0.0
    mean_eth_bw_util_pct: float = 0.0
    program_cache_hit_rate: float = 0.0

    # Indices into the parent run's JoinedRow list. Useful for the
    # inspector when the user wants to drill into the actual ops.
    row_indices: List[int] = field(default_factory=list)

    # Op-code distribution: {op_code: count}. Lets the inspector show
    # "this submodule fired 32 matmuls + 32 add + 1 softmax" without
    # needing to re-walk row_indices.
    op_code_counts: Dict[str, int] = field(default_factory=dict)

    # Layer index parsed from the attribute path (e.g. "layers.4.mlp" -> 4).
    # None for non-layered submodules. Used by the cytoscape layout to
    # group nodes by layer.
    layer_index: Optional[int] = None

    @property
    def parent_path(self) -> str:
        """The path of this module's parent, or "" for top-level roots.

        Splits on the last dot, but treats numeric indices ("0", "1", ...)
        as part of the parent (so "layers.0.self_attn" -> parent is
        "layers.0", not "layers"). This makes per-layer aggregation
        intuitive.
        """
        if "." not in self.attribute_path:
            return ""
        return self.attribute_path.rsplit(".", 1)[0]


@dataclass
class ModuleEdge:
    parent_path: str
    child_path: str


@dataclass
class ModuleGraph:
    """The graph + per-node aggregates + helpers."""

    nodes: Dict[str, ModuleNode] = field(default_factory=dict)
    edges: List[ModuleEdge] = field(default_factory=list)

    # Convenience indexes
    children_of: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    parent_of: Dict[str, str] = field(default_factory=dict)
    nodes_by_class: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def root_paths(self) -> List[str]:
        """Top-level nodes (no parent in the graph)."""
        all_paths = set(self.nodes.keys())
        children = {e.child_path for e in self.edges}
        roots = sorted(all_paths - children)
        return roots

    def descendants(self, path: str) -> List[str]:
        out: List[str] = []
        stack = list(self.children_of.get(path, []))
        while stack:
            p = stack.pop()
            out.append(p)
            stack.extend(self.children_of.get(p, []))
        return out

    def subtree_metrics(self, path: str) -> Dict[str, float]:
        """Aggregate metrics over a node + all of its descendants.

        The cytoscape layout uses this to color a parent node by the
        sum of its children's runtime even when the parent has no ops
        directly attributed (e.g. wrapper modules that just contain
        children).
        """
        paths = [path] + self.descendants(path)
        total = 0.0
        count = 0
        sum_fpu = 0.0
        sum_dram = 0.0
        sum_noc = 0.0
        weight = 0.0
        for p in paths:
            n = self.nodes.get(p)
            if n is None:
                continue
            total += n.total_device_ns
            count += n.op_count
            # Time-weighted utilization average (so a long-running op
            # dominates a short one).
            if n.op_count and n.total_device_ns > 0:
                sum_fpu += n.mean_fpu_util_pct * n.total_device_ns
                sum_dram += n.mean_dram_bw_util_pct * n.total_device_ns
                sum_noc += n.mean_noc_util_pct * n.total_device_ns
                weight += n.total_device_ns
        if weight > 0:
            return {
                "subtree_total_device_ns": total,
                "subtree_op_count": count,
                "subtree_fpu_util_pct": sum_fpu / weight,
                "subtree_dram_bw_util_pct": sum_dram / weight,
                "subtree_noc_util_pct": sum_noc / weight,
            }
        return {
            "subtree_total_device_ns": total,
            "subtree_op_count": count,
            "subtree_fpu_util_pct": 0.0,
            "subtree_dram_bw_util_pct": 0.0,
            "subtree_noc_util_pct": 0.0,
        }


# ---------------------------------------------------------------------------
# Hierarchy parsing
# ---------------------------------------------------------------------------


@dataclass
class HierarchyIndex:
    """Parsed contents of ``ttnn_module_hierarchy.json`` with a fast
    op-counter → attribute-path lookup.
    """

    module_paths: Dict[int, str] = field(default_factory=dict)
    module_classes: Dict[int, str] = field(default_factory=dict)
    # op_counter -> (innermost_path, innermost_class)
    op_attribution: Dict[int, Tuple[str, str]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.op_attribution


def parse_hierarchy_sidecar(path: Optional[Path]) -> HierarchyIndex:
    """Load the sidecar and produce a fast lookup table.

    Returns an empty index if the path is None or missing — callers
    that don't have module data fall back gracefully.
    """
    idx = HierarchyIndex()
    if path is None or not path.exists():
        return idx
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return idx

    paths = data.get("module_paths") or {}
    classes = data.get("module_classes") or {}
    idx.module_paths = {int(k): str(v) for k, v in paths.items()}
    idx.module_classes = {int(k): str(v) for k, v in classes.items()}

    for entry in data.get("op_module_log") or []:
        counter = int(entry.get("op_counter", -1))
        stack_ids = entry.get("stack_mt_ids") or []
        if counter < 0 or not stack_ids:
            continue
        # The INNERMOST module is the one whose forward() was most-deeply
        # nested when the op fired — that's the most useful attribution.
        # ("layers.0.self_attn.q_proj" not "layers" not "Qwen2Model".)
        innermost = stack_ids[-1]
        path_str = idx.module_paths.get(innermost, f"<unknown:{innermost}>")
        class_str = idx.module_classes.get(innermost, "Unknown")
        idx.op_attribution[counter] = (path_str, class_str)
    return idx


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


_LAYER_INDEX_RE_PARTS = ("layers", "h", "blocks", "encoder.layer")


def _extract_layer_index(attribute_path: str) -> Optional[int]:
    """Parse a layer index from common patterns.

    Examples:
      "model.layers.7.self_attn" -> 7
      "transformer.h.3.mlp" -> 3
      "blocks.12.attention" -> 12
      "encoder.layer.4.intermediate" -> 4

    Returns None for paths that don't contain one of those segment names
    immediately followed by an integer.
    """
    parts = attribute_path.split(".")
    for i, p in enumerate(parts):
        token = p
        # Handle two-segment markers like "encoder.layer".
        if i + 1 < len(parts):
            two = f"{p}.{parts[i + 1]}"
            if two in _LAYER_INDEX_RE_PARTS and i + 2 < len(parts):
                try:
                    return int(parts[i + 2])
                except ValueError:
                    continue
        if token in _LAYER_INDEX_RE_PARTS and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return None


def _aggregate(rows: List[JoinedRow], indices: List[int]) -> Dict[str, float]:
    if not indices:
        return {
            "op_count": 0,
            "total_device_ns": 0.0,
            "median_device_ns": 0.0,
            "mean_fpu_util_pct": 0.0,
            "mean_dram_bw_util_pct": 0.0,
            "mean_noc_util_pct": 0.0,
            "mean_eth_bw_util_pct": 0.0,
            "program_cache_hit_rate": 0.0,
        }
    devs: List[float] = []
    fpu: List[float] = []
    dram: List[float] = []
    noc: List[float] = []
    eth: List[float] = []
    hits = 0
    seen_hit_data = 0
    for i in indices:
        r = rows[i]
        if r.device_kernel_ns is not None:
            devs.append(float(r.device_kernel_ns))
        if r.pm_fpu_util_pct is not None:
            fpu.append(float(r.pm_fpu_util_pct))
        if r.dram_bw_util_pct is not None:
            dram.append(float(r.dram_bw_util_pct))
        if r.noc_util_pct is not None:
            noc.append(float(r.noc_util_pct))
        if r.eth_bw_util_pct is not None:
            eth.append(float(r.eth_bw_util_pct))
        if r.program_cache_hit is not None:
            seen_hit_data += 1
            if r.program_cache_hit:
                hits += 1
    devs_sorted = sorted(devs)
    median = devs_sorted[len(devs_sorted) // 2] if devs_sorted else 0.0
    return {
        "op_count": len(indices),
        "total_device_ns": sum(devs),
        "median_device_ns": median,
        "mean_fpu_util_pct": sum(fpu) / len(fpu) if fpu else 0.0,
        "mean_dram_bw_util_pct": sum(dram) / len(dram) if dram else 0.0,
        "mean_noc_util_pct": sum(noc) / len(noc) if noc else 0.0,
        "mean_eth_bw_util_pct": sum(eth) / len(eth) if eth else 0.0,
        "program_cache_hit_rate": (hits / seen_hit_data) if seen_hit_data else 0.0,
    }


def _enumerate_ancestors(attribute_path: str) -> List[str]:
    """For "model.layers.0.self_attn.q_proj" returns
    ["model", "model.layers", "model.layers.0", "model.layers.0.self_attn",
     "model.layers.0.self_attn.q_proj"]. Includes the input as the last
    entry. Used to wire up parent->child edges.
    """
    parts = attribute_path.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def build_module_graph(
    rows: List[JoinedRow],
    hierarchy: HierarchyIndex,
) -> ModuleGraph:
    """Construct a ModuleGraph from joined rows + a hierarchy index.

    Each row's ``global_call_count`` (Tracy's monotonic op counter) is
    used to look up its attribute path in the hierarchy. Rows whose
    counter isn't in the hierarchy are dropped from the graph (they
    still appear elsewhere — this only affects the cytoscape view).
    """
    g = ModuleGraph()

    # 1. Group row indices by attribute path.
    by_path: Dict[str, List[int]] = defaultdict(list)
    cls_for_path: Dict[str, str] = {}
    op_counts_per_path: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i, r in enumerate(rows):
        if hierarchy.is_empty:
            path = r.module_path or ""
            klass = r.module_class or "Unknown"
        else:
            attr = hierarchy.op_attribution.get(r.global_call_count)
            if attr is None:
                # Fall back to module_path populated by join's block-scope
                # fallback, if present.
                path = r.module_path or ""
                klass = r.module_class or "Unknown"
            else:
                path, klass = attr
        if not path:
            continue
        by_path[path].append(i)
        cls_for_path[path] = klass
        op_counts_per_path[path][r.op_code] += 1
    if not by_path:
        return g

    # 2. Materialize nodes for every path AND its ancestors (so the
    #    parent wrapper modules show up as graph nodes even if they
    #    don't have any directly-attributed ops).
    all_paths: Set[str] = set()
    for p in by_path.keys():
        for a in _enumerate_ancestors(p):
            all_paths.add(a)

    for p in all_paths:
        agg = _aggregate(rows, by_path.get(p, []))
        node = ModuleNode(
            attribute_path=p,
            module_class=cls_for_path.get(p, "Container"),
            op_count=int(agg["op_count"]),
            total_device_ns=float(agg["total_device_ns"]),
            median_device_ns=float(agg["median_device_ns"]),
            mean_fpu_util_pct=float(agg["mean_fpu_util_pct"]),
            mean_dram_bw_util_pct=float(agg["mean_dram_bw_util_pct"]),
            mean_noc_util_pct=float(agg["mean_noc_util_pct"]),
            mean_eth_bw_util_pct=float(agg["mean_eth_bw_util_pct"]),
            program_cache_hit_rate=float(agg["program_cache_hit_rate"]),
            row_indices=list(by_path.get(p, [])),
            op_code_counts=dict(op_counts_per_path.get(p, {})),
            layer_index=_extract_layer_index(p),
        )
        g.nodes[p] = node
        g.nodes_by_class[node.module_class].append(p)

    # 3. Wire up parent/child edges by walking attribute paths.
    for p in g.nodes:
        if "." not in p:
            continue
        parent = p.rsplit(".", 1)[0]
        if parent not in g.nodes:
            continue
        g.edges.append(ModuleEdge(parent_path=parent, child_path=p))
        g.children_of[parent].append(p)
        g.parent_of[p] = parent

    return g


# ---------------------------------------------------------------------------
# Path-pattern matcher (used by reference_db + suggestion_engine)
# ---------------------------------------------------------------------------


def matches_role_path(attribute_path: str, role_pattern: str) -> bool:
    """Does ``attribute_path`` match the reference DB's ``role_path`` pattern?

    The pattern uses ``*`` as a single-segment wildcard, similar to a
    very narrow glob:

      "decoder.layers.*.self_attn.q_proj"      matches all of
      "model.decoder.layers.0.self_attn.q_proj"   (extra prefix in user)
      "decoder.layers.7.self_attn.q_proj"         (exact)
      "layers.7.self_attn.q_proj"                 (extra prefix in pattern)

    Both ends use **suffix-anchored matching**: we align the two strings
    from the rightmost segment leftward and walk back. The match
    succeeds if every segment in the overlap matches (treating ``*`` as
    a wildcard). Whichever side is longer is allowed to have extra
    leading segments — this handles both `model.` wrappers in the user
    path AND `decoder.` wrappers in the reference path. Conceptually,
    role_path describes a submodule's LEAF role, and the prefix is
    incidental architecture wrapping.

    Returns False on empty inputs (so callers can use it as a filter).
    """
    if not attribute_path or not role_pattern:
        return False
    a_parts = attribute_path.split(".")
    p_parts = role_pattern.split(".")
    # Walk from the right; require all overlapping segments to match.
    overlap = min(len(a_parts), len(p_parts))
    if overlap == 0:
        return False
    # The LAST segment is the role's "head" and must match exactly
    # (no wildcards allowed there — guards against e.g. "self_attn"
    # accidentally matching "self_attn.q_proj").
    if p_parts[-1] != "*" and p_parts[-1] != a_parts[-1]:
        return False
    if a_parts[-1] != p_parts[-1] and p_parts[-1] != "*":
        return False
    for i in range(1, overlap + 1):
        pp = p_parts[-i]
        ap = a_parts[-i]
        if pp == "*" or ap == "*":
            continue
        if pp != ap:
            return False
    return True


__all__ = [
    "ModuleNode",
    "ModuleEdge",
    "ModuleGraph",
    "HierarchyIndex",
    "parse_hierarchy_sidecar",
    "build_module_graph",
    "matches_role_path",
]

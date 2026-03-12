# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Chaitin-Briggs graph coloring for CB slot allocation.

Implements the standard Chaitin-Briggs register allocation algorithm
adapted for circular buffer slot assignment:

1. Build interference graph: CBs with overlapping lifetimes interfere
2. Coalesce: SAME_CORE chained CBs share the same slot
3. Simplify: iteratively remove nodes with degree < K (K=32)
4. Spill: if a node has degree >= K, mark for spilling
5. Select: pop stack and assign lowest available color

The hardware has 32 CB slots (indices 0-31). Unlike CPU registers,
spilling a CB means splitting the lifetime and using L1 double-buffering,
which is extremely rare for fusion graphs (usually < 10 CBs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from models.demos.deepseek_v3_b1.auto_fusion.types import DataEdge, OpNode, TransferType


@dataclass
class ColoringResult:
    """Result of Chaitin-Briggs graph coloring."""

    assignments: Dict[FrozenSet[Tuple[str, str]], int]  # group -> CB slot
    port_assignments: Dict[Tuple[str, str], int]  # (op_id, port) -> CB slot
    num_colors_used: int
    spilled: List[FrozenSet[Tuple[str, str]]]  # groups that couldn't be colored
    interference_edges: int  # number of interference edges


class ChaitinBriggsAllocator:
    """
    Chaitin-Briggs graph coloring allocator for CB slots.

    Adapts the classic register allocation algorithm:
    - Nodes = CB lifetime groups (coalesced via SAME_CORE edges)
    - Colors = CB slot indices (0-31)
    - Interference = overlapping lifetimes on compatible core ranges
    - K = 32 (number of hardware CB slots)
    """

    K = 32  # Number of CB slots

    def __init__(
        self,
        nodes: List[OpNode],
        edges: List[DataEdge],
        schedule: List[str],
        liveness: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None,
        chain_groups: Optional[Dict[Tuple[str, str], Set[Tuple[str, str]]]] = None,
    ):
        """
        Args:
            nodes: List of OpNodes
            edges: List of DataEdges
            schedule: Execution order
            liveness: Pre-computed liveness intervals (from CBAllocator)
            chain_groups: Pre-computed chain groups (from CBAllocator)
        """
        self._nodes_map = {n.id: n for n in nodes}
        self._edges = edges
        self._schedule = schedule
        self._step = {nid: i for i, nid in enumerate(schedule)}

        # Use provided or compute fresh
        self._liveness = liveness
        self._chain_groups = chain_groups

    def build_interference_graph(self) -> Dict[FrozenSet[Tuple[str, str]], Set[FrozenSet[Tuple[str, str]]]]:
        """
        Build the interference graph.

        Two CB groups interfere if their lifetimes overlap.

        Returns:
            Adjacency dict: group -> set of interfering groups.
        """
        groups = self._get_coalesced_groups()
        group_intervals = self._compute_group_intervals(groups)

        adj: Dict[FrozenSet[Tuple[str, str]], Set[FrozenSet[Tuple[str, str]]]] = {g: set() for g in groups}

        group_list = list(groups)
        for i in range(len(group_list)):
            for j in range(i + 1, len(group_list)):
                gi, gj = group_list[i], group_list[j]
                si, ei = group_intervals[gi]
                sj, ej = group_intervals[gj]
                # Check overlap
                if not (ei < sj or ej < si):
                    adj[gi].add(gj)
                    adj[gj].add(gi)

        return adj

    def simplify(self) -> List[FrozenSet[Tuple[str, str]]]:
        """
        Simplify phase: iteratively remove nodes with degree < K.

        Returns a stack of removed nodes (last removed = first to color).
        """
        adj = self.build_interference_graph()
        remaining = set(adj.keys())
        stack = []

        while remaining:
            # Find a node with degree < K
            removed_any = False
            for node in list(remaining):
                degree = len(adj[node] & remaining)
                if degree < self.K:
                    stack.append(node)
                    remaining.remove(node)
                    removed_any = True
                    break

            if not removed_any:
                # All nodes have degree >= K — potential spill
                # Heuristic: pick node with highest degree
                spill = max(remaining, key=lambda n: len(adj[n] & remaining))
                stack.append(spill)
                remaining.remove(spill)

        return stack

    def select_colors(self, stack: List[FrozenSet[Tuple[str, str]]]) -> ColoringResult:
        """
        Select phase: assign colors by popping the stack.

        For each node, assign the lowest color not used by any neighbor
        that has already been colored.
        """
        adj = self.build_interference_graph()
        assignments: Dict[FrozenSet[Tuple[str, str]], int] = {}
        spilled: List[FrozenSet[Tuple[str, str]]] = []

        for group in reversed(stack):
            # Find colors used by colored neighbors
            used = set()
            for neighbor in adj.get(group, set()):
                if neighbor in assignments:
                    used.add(assignments[neighbor])

            # Assign lowest available color
            color = 0
            while color in used:
                color += 1

            if color >= self.K:
                spilled.append(group)
            else:
                assignments[group] = color

        # Build per-port assignments
        port_assignments = {}
        for group, color in assignments.items():
            for port_key in group:
                port_assignments[port_key] = color

        return ColoringResult(
            assignments=assignments,
            port_assignments=port_assignments,
            num_colors_used=len(set(assignments.values())) if assignments else 0,
            spilled=spilled,
            interference_edges=sum(len(v) for v in adj.values()) // 2,
        )

    def handle_spills(self, spilled: List[FrozenSet[Tuple[str, str]]]) -> List[str]:
        """
        Handle spilled nodes by suggesting strategies.

        For CB allocation, spilling means:
        - Split the CB lifetime into shorter intervals
        - Use double-buffering to reduce concurrent live CBs
        - In extreme cases, serialize ops to reduce register pressure

        Returns:
            List of human-readable spill strategies.
        """
        strategies = []
        for group in spilled:
            ports = ", ".join(f"{op}.{port}" for op, port in group)
            strategies.append(
                f"Spill group [{ports}]: consider splitting lifetime or "
                f"adding explicit barrier to reduce concurrent CBs"
            )
        return strategies

    def allocate(self) -> ColoringResult:
        """
        Run the full Chaitin-Briggs allocation pipeline.

        Returns:
            ColoringResult with CB slot assignments.
        """
        stack = self.simplify()
        result = self.select_colors(stack)

        if result.spilled:
            strategies = self.handle_spills(result.spilled)
            # For now, just report — actual spill handling is future work
            for s in strategies:
                print(f"[graph_coloring] WARNING: {s}")

        return result

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_coalesced_groups(self) -> List[FrozenSet[Tuple[str, str]]]:
        """Get CB groups, coalescing SAME_CORE chained ports."""
        if self._chain_groups:
            return [frozenset(members) for members in self._chain_groups.values()]

        # Build from scratch using Union-Find
        parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for edge in self._edges:
            if edge.transfer == TransferType.SAME_CORE:
                union((edge.src_node, edge.src_port), (edge.dst_node, edge.dst_port))

        groups_dict: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
        for nid in self._schedule:
            node = self._nodes_map[nid]
            for port_name in node.spec.cb_ports:
                key = (nid, port_name)
                root = find(key)
                groups_dict.setdefault(root, set()).add(key)

        return [frozenset(members) for members in groups_dict.values()]

    def _get_liveness(self) -> Dict[Tuple[str, str], Tuple[int, int]]:
        """Get or compute liveness intervals."""
        if self._liveness:
            return self._liveness

        # Simple computation (mirrors CBAllocator._compute_liveness)
        from models.demos.deepseek_v3_b1.auto_fusion.types import CBDirection

        num_steps = len(self._schedule)
        intervals: Dict[Tuple[str, str], List[int]] = {}

        for nid in self._schedule:
            node = self._nodes_map[nid]
            s = self._step[nid]
            for port_name, port_spec in node.spec.cb_ports.items():
                key = (nid, port_name)
                if port_spec.is_sharded:
                    intervals[key] = [0, s]
                elif port_spec.direction == CBDirection.OUTPUT:
                    intervals[key] = [s, num_steps]
                else:
                    intervals[key] = [s, s]

        for edge in self._edges:
            src_key = (edge.src_node, edge.src_port)
            dst_key = (edge.dst_node, edge.dst_port)
            dst_step = self._step[edge.dst_node]
            if edge.transfer == TransferType.SAME_CORE:
                if src_key in intervals and dst_key in intervals:
                    merged_s = min(intervals[src_key][0], intervals[dst_key][0])
                    merged_e = max(intervals[src_key][1], intervals[dst_key][1])
                    intervals[src_key] = [merged_s, merged_e]
                    intervals[dst_key] = [merged_s, merged_e]
            else:
                if src_key in intervals:
                    intervals[src_key][1] = max(intervals[src_key][1], dst_step)

        return {k: (v[0], v[1]) for k, v in intervals.items()}

    def _compute_group_intervals(
        self, groups: List[FrozenSet[Tuple[str, str]]]
    ) -> Dict[FrozenSet[Tuple[str, str]], Tuple[int, int]]:
        """Compute merged liveness interval for each group."""
        liveness = self._get_liveness()
        result = {}
        for group in groups:
            starts = [liveness[m][0] for m in group if m in liveness]
            ends = [liveness[m][1] for m in group if m in liveness]
            if starts:
                result[group] = (min(starts), max(ends))
            else:
                result[group] = (0, 0)
        return result

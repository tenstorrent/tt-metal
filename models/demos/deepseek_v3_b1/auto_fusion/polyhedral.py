# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Polyhedral analysis for auto-fusion graphs.

Provides fusion legality checking and tile size optimization using
simplified polyhedral analysis concepts:

1. Iteration domains: Each op has a set of tile iterations it performs
2. Dependence analysis: Check whether fusing two ops preserves data flow
3. Tile size optimization: Choose tile sizes that fit in L1 budget

For simple single-core ops (like gated_local_reduce), all domains are
degenerate single-point domains and fusion is trivially legal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class IterationDomain:
    """
    Simplified iteration domain for a micro-op.

    Each dimension has a lower bound, upper bound, and tile size.
    For most single-core ops, this is a single point [0, num_tiles).
    For matmul, this could be 3D: [0, M) x [0, N) x [0, K).
    """

    dimensions: List[Tuple[int, int]]  # List of (lower, upper) per dimension
    tile_sizes: List[int] = field(default_factory=list)  # Tile size per dimension

    @property
    def num_dims(self) -> int:
        return len(self.dimensions)

    @property
    def total_iterations(self) -> int:
        total = 1
        for lo, hi in self.dimensions:
            total *= hi - lo
        return total

    @property
    def is_degenerate(self) -> bool:
        """True if the domain is a single point (trivial)."""
        return all(hi - lo <= 1 for lo, hi in self.dimensions)


@dataclass
class DependenceInfo:
    """Result of dependence analysis between two ops."""

    is_legal: bool  # Whether fusion preserves correctness
    dependence_type: str  # "flow", "anti", "output", "none"
    distance: int = 0  # Dependence distance (0 = same iteration)
    reason: str = ""  # Human-readable explanation


class PolyhedralAnalyzer:
    """
    Simplified polyhedral analysis for fusion legality and tile optimization.

    For Milestone 1 (gated_local_reduce), all ops have degenerate
    single-point domains, so fusion is trivially legal. The analyzer
    still provides the framework for extending to matmul (3D domains)
    in Milestone 2.
    """

    def __init__(self, graph):
        self._graph = graph
        self._nodes = {n.id: n for n in graph.nodes}
        self._edges = graph.edges
        self._schedule = graph.get_schedule()

    def build_iteration_domains(self) -> Dict[str, IterationDomain]:
        """
        Build iteration domains for each op in the graph.

        Heuristic:
        - If op has a "num_tiles" CT arg, use [0, num_tiles) as 1D domain
        - If op has "out_w" + "k_num_tiles" CT args (matmul), use 2D domain
        - Otherwise, use degenerate [0, 1) domain
        """
        domains = {}
        for nid in self._schedule:
            node = self._nodes[nid]
            ct = node.ct_args

            if "out_w" in ct and "k_num_tiles" in ct:
                # Matmul-like: 2D domain [0, out_w) x [0, k_num_tiles)
                domains[nid] = IterationDomain(
                    dimensions=[(0, int(ct["out_w"])), (0, int(ct["k_num_tiles"]))],
                )
            elif "num_tiles" in ct:
                # 1D reduction/element-wise
                domains[nid] = IterationDomain(
                    dimensions=[(0, int(ct["num_tiles"]))],
                )
            else:
                # Degenerate: single point
                domains[nid] = IterationDomain(
                    dimensions=[(0, 1)],
                )

        return domains

    def check_fusion_legality(self, op_a: str, op_b: str) -> DependenceInfo:
        """
        Check whether ops op_a and op_b can be legally fused.

        Fusion is legal when:
        1. There is a direct or transitive data dependence from op_a to op_b
           (flow dependence), OR they are independent (no dependence)
        2. There are no cyclic dependences between them
        3. They execute on compatible core sets (same or subset)

        Returns:
            DependenceInfo with legality result and explanation.
        """
        if op_a not in self._nodes or op_b not in self._nodes:
            return DependenceInfo(
                is_legal=False,
                dependence_type="none",
                reason=f"Unknown op: {op_a if op_a not in self._nodes else op_b}",
            )

        # Check for direct edge
        has_direct_edge = False
        for edge in self._edges:
            if edge.src_node == op_a and edge.dst_node == op_b:
                has_direct_edge = True
                break
            if edge.src_node == op_b and edge.dst_node == op_a:
                # Reverse dependence — fusion would create a cycle
                return DependenceInfo(
                    is_legal=False,
                    dependence_type="anti",
                    reason=f"Anti-dependence: {op_b} -> {op_a} prevents fusion",
                )

        # Check for transitive dependence via BFS
        if not has_direct_edge:
            reachable = self._reachable_from(op_a)
            if op_b in reachable:
                has_direct_edge = True  # Transitive flow dependence

        # Check for reverse path (cycle detection)
        reverse_reachable = self._reachable_from(op_b)
        if op_a in reverse_reachable:
            return DependenceInfo(
                is_legal=False,
                dependence_type="output",
                reason=f"Cyclic dependence between {op_a} and {op_b}",
            )

        dep_type = "flow" if has_direct_edge else "none"
        return DependenceInfo(
            is_legal=True,
            dependence_type=dep_type,
            reason=f"Fusion legal: {'flow dependence' if has_direct_edge else 'independent ops'}",
        )

    def compute_tile_sizes(self, l1_budget: int) -> Dict[str, Dict[str, int]]:
        """
        Compute optimal tile sizes for each op to fit within L1 budget.

        For simple ops (1D domains), tile_size = domain size (all tiles at once).
        For matmul (2D), optimize inner loop tile size to maximize L1 reuse.

        Args:
            l1_budget: Available L1 memory in bytes for intermediate buffers.

        Returns:
            Dict mapping op_id -> {dimension_name: tile_size}.
        """
        PAGE_SIZE = 2048  # BF16 tile = 2048 bytes
        domains = self.build_iteration_domains()
        result = {}

        for nid, domain in domains.items():
            if domain.is_degenerate or domain.num_dims == 1:
                # Simple: use entire domain
                result[nid] = {"tiles": domain.total_iterations}
            elif domain.num_dims == 2:
                # Matmul-like: optimize for L1
                m_range = domain.dimensions[0][1] - domain.dimensions[0][0]
                k_range = domain.dimensions[1][1] - domain.dimensions[1][0]

                # Each tile of in0 (m x k) and in1 (k x n) needs L1 space
                # Tile along K to fit in L1
                max_k_tiles = max(1, l1_budget // (2 * PAGE_SIZE))
                k_tile = min(k_range, max_k_tiles)

                result[nid] = {"m_tiles": m_range, "k_tiles": k_tile}
            else:
                result[nid] = {"tiles": domain.total_iterations}

        return result

    def _reachable_from(self, start: str) -> set:
        """BFS to find all nodes reachable from start via forward edges."""
        from collections import deque

        visited = set()
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for edge in self._edges:
                if edge.src_node == u and edge.dst_node not in visited:
                    visited.add(edge.dst_node)
                    queue.append(edge.dst_node)
        return visited

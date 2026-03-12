# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Synchronous Dataflow (SDF) analysis for auto-fusion graphs.

Computes repetition vectors, buffer bounds, and double-buffering suggestions
from the SDF rates declared on each micro-op's CB ports.

Key concepts:
- Each micro-op is an SDF actor that fires once per invocation
- Each CB port has a production/consumption rate (tiles per firing)
- The topology matrix Gamma has one row per edge and one column per actor
- The repetition vector q satisfies Gamma * q = 0 (balance equation)
- Buffer bounds = min pages needed on each edge to avoid deadlock
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from models.demos.deepseek_v3_b1.auto_fusion.types import OpNode, TransferType


class SDFAnalyzer:
    """
    Analyzes a fusion graph using Synchronous Dataflow theory.

    Given a graph of micro-ops with SDF rates on CB ports, computes:
    1. The topology matrix (edges x actors)
    2. The repetition vector (how many times each actor fires)
    3. Minimum buffer bounds per edge
    4. Double-buffering suggestions for throughput
    """

    def __init__(self, graph, ct_args_resolver: Optional[Dict[str, Dict[str, int]]] = None):
        """
        Args:
            graph: FusionGraph instance
            ct_args_resolver: Optional mapping of op_id -> {param_name: value}
                for resolving parametric SDF rates. If None, uses node.ct_args.
        """
        self._graph = graph
        self._nodes = {n.id: n for n in graph.nodes}
        self._edges = graph.edges
        self._schedule = graph.get_schedule()
        self._ct_resolver = ct_args_resolver or {}

    def _resolve_rate(self, node: OpNode, port_name: str) -> int:
        """Resolve the SDF rate for a port, handling parametric expressions."""
        port_spec = node.spec.cb_ports.get(port_name)
        if port_spec is None or port_spec.sdf_rate is None:
            return 1  # Default: 1 tile per firing

        rate = port_spec.sdf_rate
        if not rate.is_parametric:
            return rate.tokens

        # Resolve from ct_args_resolver or node.ct_args
        resolver = self._ct_resolver.get(node.id, node.ct_args)
        param = rate.param_expr
        if param in resolver:
            return int(resolver[param])

        return rate.tokens if rate.tokens > 0 else 1

    def build_topology_matrix(self) -> np.ndarray:
        """
        Build the SDF topology matrix Gamma.

        Gamma is |E| x |V| where:
        - Row i corresponds to edge i
        - Column j corresponds to actor j
        - Gamma[i,j] = production rate if j is source of edge i
        - Gamma[i,j] = -consumption rate if j is sink of edge i
        - Gamma[i,j] = 0 otherwise

        Returns:
            numpy array of shape (num_edges, num_actors)
        """
        actor_ids = self._schedule
        actor_idx = {nid: i for i, nid in enumerate(actor_ids)}
        num_actors = len(actor_ids)
        num_edges = len(self._edges)

        if num_edges == 0:
            return np.zeros((0, num_actors), dtype=int)

        gamma = np.zeros((num_edges, num_actors), dtype=int)

        for i, edge in enumerate(self._edges):
            src_node = self._nodes[edge.src_node]
            dst_node = self._nodes[edge.dst_node]

            prod_rate = self._resolve_rate(src_node, edge.src_port)
            cons_rate = self._resolve_rate(dst_node, edge.dst_port)

            gamma[i, actor_idx[edge.src_node]] = prod_rate
            gamma[i, actor_idx[edge.dst_node]] = -cons_rate

        return gamma

    def compute_repetition_vector(self) -> Dict[str, int]:
        """
        Compute the repetition vector q such that Gamma * q = 0.

        For a consistent SDF graph, the repetition vector exists and is unique
        up to a scalar multiple. We normalize to the smallest positive integers.

        For single-rate graphs (all rates equal), q = [1, 1, ..., 1].

        Returns:
            Dict mapping actor_id -> number of firings per schedule period.
        """
        actor_ids = self._schedule
        gamma = self.build_topology_matrix()

        if gamma.shape[0] == 0:
            # No edges: each actor fires once
            return {nid: 1 for nid in actor_ids}

        # For connected SDF graphs, use BFS on the rate ratios
        # Start from first actor with q[0] = 1, propagate via edges
        num_actors = len(actor_ids)
        actor_idx = {nid: i for i, nid in enumerate(actor_ids)}
        q = [0] * num_actors

        # BFS from first actor
        from collections import deque

        q[0] = 1
        visited = {0}
        queue = deque([0])

        # Build adjacency: for each edge, src and dst are connected
        adj: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(num_actors)}
        for i, edge in enumerate(self._edges):
            si = actor_idx[edge.src_node]
            di = actor_idx[edge.dst_node]
            src_node = self._nodes[edge.src_node]
            dst_node = self._nodes[edge.dst_node]
            prod = self._resolve_rate(src_node, edge.src_port)
            cons = self._resolve_rate(dst_node, edge.dst_port)
            adj[si].append((di, prod, cons))
            adj[di].append((si, cons, prod))

        while queue:
            u = queue.popleft()
            for v, rate_u, rate_v in adj[u]:
                if v not in visited:
                    # q[u] * rate_u = q[v] * rate_v
                    # q[v] = q[u] * rate_u / rate_v
                    q[v] = q[u] * rate_u // rate_v if rate_v > 0 else q[u]
                    visited.add(v)
                    queue.append(v)

        # Handle disconnected components (each fires once)
        for i in range(num_actors):
            if q[i] == 0:
                q[i] = 1

        # Normalize to smallest positive integers (divide by GCD)
        from functools import reduce
        from math import gcd

        g = reduce(gcd, q)
        q = [x // g for x in q]

        return {actor_ids[i]: q[i] for i in range(num_actors)}

    def compute_buffer_bounds(self) -> Dict[Tuple[str, str], int]:
        """
        Compute minimum buffer sizes (in tiles/pages) per edge to prevent deadlock.

        For each edge (src, dst):
          min_buffer = max(production_rate, consumption_rate)

        This ensures the consumer can always find enough data and the producer
        can always find enough space.

        Returns:
            Dict mapping (src_node_id, dst_node_id) -> minimum pages.
        """
        bounds = {}
        for edge in self._edges:
            src_node = self._nodes[edge.src_node]
            dst_node = self._nodes[edge.dst_node]
            prod = self._resolve_rate(src_node, edge.src_port)
            cons = self._resolve_rate(dst_node, edge.dst_port)
            bounds[(edge.src_node, edge.dst_node)] = max(prod, cons)
        return bounds

    def suggest_double_buffering(self) -> Dict[Tuple[str, str], int]:
        """
        Suggest double-buffered sizes for edges where it would improve throughput.

        Double buffering is beneficial when producer and consumer can overlap:
        - Different RISCs (e.g., reader producing while compute consumes)
        - Cross-core transfers (producer can push next batch while consumer works)

        Returns:
            Dict mapping (src_node_id, dst_node_id) -> suggested initial tokens.
            Only includes edges where double buffering is beneficial.
        """
        suggestions = {}
        bounds = self.compute_buffer_bounds()

        for edge in self._edges:
            key = (edge.src_node, edge.dst_node)
            min_buf = bounds[key]

            if edge.transfer != TransferType.SAME_CORE:
                # Cross-core: double buffer for overlap
                suggestions[key] = min_buf
            else:
                # Same-core: check if different RISCs can overlap
                src_node = self._nodes[edge.src_node]
                dst_node = self._nodes[edge.dst_node]
                src_trisc_noop = src_node.spec.trisc.is_noop
                dst_trisc_noop = dst_node.spec.trisc.is_noop
                if src_trisc_noop != dst_trisc_noop:
                    # Different RISCs: double buffer
                    suggestions[key] = min_buf

        return suggestions

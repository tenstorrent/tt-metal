# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Lifetime-aware circular buffer allocator for auto-fused kernels.

Three-phase allocation:

1. **Liveness analysis**: Compute [first_needed, last_needed] intervals for each
   (op, port) pair based on schedule order and data edges.

2. **CB index assignment**: Interval graph coloring assigns physical CB indices
   (0-31) to ports. Ports with non-overlapping lifetimes on the same cores can
   share indices. SAME_CORE edges chain output→input into a single group.

3. **L1 memory packing**: For intermediate CBs (not backed by user tensors),
   pack into minimal L1 using first-fit-decreasing on the timeline. Two CBs
   sharing L1 memory must have non-overlapping lifetimes. This is the key
   optimization that lets auto-fused kernels match hand-optimized L1 usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from models.demos.deepseek_v3_b1.auto_fusion.types import CBDirection, DataEdge, OpNode, TransferType


@dataclass
class CBAllocation:
    """Result of allocating a CB for a specific (op, port) pair."""

    index: int  # Physical CB index (0-31)
    op_id: str  # Which op this belongs to
    port_name: str  # Which port on the op
    live_start: int = 0  # Schedule step when first needed
    live_end: int = 0  # Schedule step when last needed
    is_external: bool = True  # Backed by user-provided tensor?
    # L1 packing fields (populated by pack_l1)
    total_size: int = 0  # Total size in bytes
    pool_offset: int = -1  # Byte offset in L1 pool (-1 = not packed / external)


class CBAllocator:
    """
    Allocates physical CB indices and L1 memory for all ops in a fusion graph.

    Usage:
        allocator = CBAllocator(nodes, edges, schedule)
        allocs = allocator.allocate(external_ports={("rmsnorm", "input"), ...})
        pool_size = allocator.pack_l1(cb_sizes={("matmul", "out"): 4096, ...})
    """

    MAX_CB_INDEX = 31  # Hardware limit

    def __init__(self, nodes: List[OpNode], edges: List[DataEdge], schedule: List[str]):
        self._nodes = {n.id: n for n in nodes}
        self._edges = edges
        self._schedule = schedule
        self._step = {nid: i for i, nid in enumerate(schedule)}
        self._allocations: Dict[Tuple[str, str], CBAllocation] = {}

    def allocate(
        self,
        external_ports: Optional[Set[Tuple[str, str]]] = None,
        allow_index_reuse: bool = True,
    ) -> Dict[Tuple[str, str], CBAllocation]:
        """
        Allocate CB indices with liveness-aware reuse.

        Args:
            external_ports: Set of (op_id, port_name) backed by user tensors.
                If None, all sharded ports are assumed external.
            allow_index_reuse: If False, every chain group gets a unique index.
                Use False for single-phase kernels where CB descriptors persist
                for the entire execution (no inter-phase barriers to update them).

        Returns:
            Dict mapping (op_id, port_name) -> CBAllocation.
        """
        # Phase 1: Liveness analysis
        intervals = self._compute_liveness()

        # When index reuse is disabled, extend all lifetimes to full schedule
        # so that the greedy coloring assigns unique indices to each group.
        if not allow_index_reuse:
            num_steps = len(self._schedule)
            for key in intervals:
                intervals[key] = (0, num_steps)

        # Phase 2: Build chain groups (SAME_CORE edges)
        groups = self._build_chain_groups()

        # Phase 3: Interval graph coloring
        self._allocations = self._assign_indices(intervals, groups, external_ports)

        # Write back to nodes
        for (op_id, port_name), alloc in self._allocations.items():
            self._nodes[op_id].cb_bindings[port_name] = alloc.index

        return self._allocations

    def pack_l1(
        self,
        cb_sizes: Dict[Tuple[str, str], int],
    ) -> int:
        """
        Pack internal (non-external) CBs into L1 memory pools.

        Uses first-fit-decreasing bin packing on the lifetime timeline:
        two CBs can share the same L1 region only if their lifetimes
        don't overlap.

        Args:
            cb_sizes: Dict mapping (op_id, port_name) -> total_size in bytes
                      for each intermediate CB that needs pool allocation.

        Returns:
            Peak L1 pool size in bytes needed to hold all internal CBs.
        """
        # Set sizes on allocations
        for key, size in cb_sizes.items():
            if key in self._allocations:
                self._allocations[key].total_size = size

        # Get internal CBs that need packing
        internal = [(k, a) for k, a in self._allocations.items() if not a.is_external and a.total_size > 0]

        if not internal:
            return 0

        # Sort by size descending (first-fit decreasing heuristic)
        internal.sort(key=lambda x: -x[1].total_size)

        # Pack using timeline-aware first-fit
        placed: List[Tuple[int, int, int, int]] = []  # (offset, size, start, end)

        for _key, alloc in internal:
            offset = self._find_l1_offset(placed, alloc)
            alloc.pool_offset = offset
            placed.append((offset, alloc.total_size, alloc.live_start, alloc.live_end))

        # Return peak pool size
        return max(o + s for o, s, _, _ in placed) if placed else 0

    def get_internal_allocs(self) -> List[CBAllocation]:
        """Get all internal (non-external) allocations that have been packed."""
        return [a for a in self._allocations.values() if not a.is_external and a.pool_offset >= 0]

    def get_external_allocs(self) -> List[CBAllocation]:
        """Get all external (tensor-backed) allocations."""
        return [a for a in self._allocations.values() if a.is_external]

    # =========================================================================
    # Phase 1: Liveness analysis
    # =========================================================================

    def _compute_liveness(self) -> Dict[Tuple[str, str], Tuple[int, int]]:
        """
        Compute [start, end] interval for each (op, port).

        Rules:
        - Sharded input/weight: alive from step 0 to the owning op's step
        - Non-sharded input: alive only at the owning op's step
        - Output: alive from producing op's step to end of schedule
        - Scratch: alive only during the owning op's step
        - Edges refine intervals: outputs extend to last consumer, SAME_CORE
          edges merge producer and consumer intervals
        """
        num_steps = len(self._schedule)
        intervals: Dict[Tuple[str, str], List[int]] = {}

        for nid in self._schedule:
            node = self._nodes[nid]
            s = self._step[nid]

            for port_name, port_spec in node.spec.cb_ports.items():
                key = (nid, port_name)

                if port_spec.is_sharded:
                    # Pre-loaded in L1 before kernel starts
                    intervals[key] = [0, s]
                elif port_spec.direction == CBDirection.OUTPUT:
                    # Produced at step s, may persist to end
                    intervals[key] = [s, num_steps]
                elif port_spec.direction == CBDirection.INPUT:
                    # Consumed at step s
                    intervals[key] = [s, s]
                elif port_spec.direction == CBDirection.SCRATCH:
                    # Only during step s
                    intervals[key] = [s, s]
                else:
                    intervals[key] = [s, s]

        # Refine with edge information
        for edge in self._edges:
            src_key = (edge.src_node, edge.src_port)
            dst_key = (edge.dst_node, edge.dst_port)
            dst_step = self._step[edge.dst_node]
            src_step = self._step[edge.src_node]

            if edge.transfer == TransferType.SAME_CORE:
                # Merge lifetimes: the shared CB spans both ops
                if src_key in intervals and dst_key in intervals:
                    merged_start = min(intervals[src_key][0], intervals[dst_key][0])
                    merged_end = max(intervals[src_key][1], intervals[dst_key][1])
                    intervals[src_key] = [merged_start, merged_end]
                    intervals[dst_key] = [merged_start, merged_end]
            else:
                # Cross-core: source must stay alive until consumer reads
                if src_key in intervals:
                    intervals[src_key][1] = max(intervals[src_key][1], dst_step)
                # Destination needs to be alive when data arrives
                if dst_key in intervals:
                    intervals[dst_key][0] = min(intervals[dst_key][0], src_step)

        return {k: (v[0], v[1]) for k, v in intervals.items()}

    # =========================================================================
    # Phase 2: Chain group construction (Union-Find)
    # =========================================================================

    def _build_chain_groups(self) -> Dict[Tuple[str, str], Set[Tuple[str, str]]]:
        """
        Group chained ports using Union-Find with sibling coalescing.

        Two passes:
        1. Chain SAME_CORE edges: output→input pairs share a CB.
        2. Coalesce siblings: when multiple SAME_CORE edges feed different
           input ports of the same downstream op, merge their source output
           groups into one CB. This matches hand-fused patterns where e.g.
           two reduce outputs share one intermed CB with 2 pages.

        Returns mapping from group representative -> set of member (op_id, port) keys.
        """
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

        # Pass 1: Chain SAME_CORE output→input pairs
        for edge in self._edges:
            if edge.transfer == TransferType.SAME_CORE:
                union((edge.src_node, edge.src_port), (edge.dst_node, edge.dst_port))

        # Pass 2: Coalesce siblings — merge source outputs that feed
        # different input ports of the same downstream op via SAME_CORE edges.
        # Group edges by destination op.
        from collections import defaultdict

        dst_groups: Dict[str, List[DataEdge]] = defaultdict(list)
        for edge in self._edges:
            if edge.transfer == TransferType.SAME_CORE:
                dst_groups[edge.dst_node].append(edge)

        for dst_node, edges in dst_groups.items():
            if len(edges) < 2:
                continue
            # Only coalesce siblings whose source ops have the exact same core set.
            # This prevents merging CBs that live on different core subsets
            # (e.g., matmul.out on 112 cores vs mcast2.dst on 130 cores).
            from models.demos.deepseek_v3_b1.auto_fusion.graph import _cores_to_set

            first_src = (edges[0].src_node, edges[0].src_port)
            first_cores = _cores_to_set(self._nodes[edges[0].src_node].placement.core_range_set)
            for edge in edges[1:]:
                src_cores = _cores_to_set(self._nodes[edge.src_node].placement.core_range_set)
                if src_cores == first_cores:
                    union(first_src, (edge.src_node, edge.src_port))

        # Build groups from all ports
        groups: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
        for nid in self._schedule:
            node = self._nodes[nid]
            for port_name in node.spec.cb_ports:
                key = (nid, port_name)
                root = find(key)
                groups.setdefault(root, set()).add(key)

        return groups

    # =========================================================================
    # Phase 3: Interval graph coloring
    # =========================================================================

    def _assign_indices(
        self,
        intervals: Dict[Tuple[str, str], Tuple[int, int]],
        groups: Dict[Tuple[str, str], Set[Tuple[str, str]]],
        external_ports: Optional[Set[Tuple[str, str]]],
    ) -> Dict[Tuple[str, str], CBAllocation]:
        """
        Assign CB indices using greedy interval graph coloring.

        Groups sorted by start time; each group gets the lowest available
        index whose previous user's lifetime has ended.
        """
        # Compute per-group merged intervals
        group_intervals: Dict[Tuple[str, str], Tuple[int, int]] = {}
        for root, members in groups.items():
            starts = [intervals[m][0] for m in members if m in intervals]
            ends = [intervals[m][1] for m in members if m in intervals]
            if starts:
                group_intervals[root] = (min(starts), max(ends))

        # Sort groups by start time (stable sort preserves insertion order for ties)
        sorted_roots = sorted(
            group_intervals.keys(),
            key=lambda r: (group_intervals[r][0], group_intervals[r][1]),
        )

        # Greedy coloring: assign lowest available index
        index_end: Dict[int, int] = {}  # index -> last end time using that index
        group_index: Dict[Tuple[str, str], int] = {}

        for root in sorted_roots:
            start, end = group_intervals[root]

            # Find lowest index whose previous user has ended (strictly before start)
            best_idx = None
            for idx in sorted(index_end.keys()):
                if index_end[idx] < start:
                    best_idx = idx
                    break

            if best_idx is None:
                best_idx = len(index_end)

            if best_idx > self.MAX_CB_INDEX:
                raise RuntimeError(
                    f"CB index overflow: need index {best_idx} but max is {self.MAX_CB_INDEX}. "
                    f"Too many simultaneously live CBs."
                )

            group_index[root] = best_idx
            index_end[best_idx] = end

        # Build allocations
        allocations: Dict[Tuple[str, str], CBAllocation] = {}
        for root, members in groups.items():
            idx = group_index.get(root, 0)
            for op_id, port_name in members:
                key = (op_id, port_name)
                live = intervals.get(key, (0, len(self._schedule)))

                # Determine if external
                is_ext = False
                if external_ports and key in external_ports:
                    is_ext = True
                else:
                    node = self._nodes[op_id]
                    port_spec = node.spec.cb_ports.get(port_name)
                    if port_spec and port_spec.is_sharded:
                        is_ext = True

                allocations[key] = CBAllocation(
                    index=idx,
                    op_id=op_id,
                    port_name=port_name,
                    live_start=live[0],
                    live_end=live[1],
                    is_external=is_ext,
                )

        return allocations

    # =========================================================================
    # L1 memory packing
    # =========================================================================

    def _find_l1_offset(
        self,
        placed: List[Tuple[int, int, int, int]],
        alloc: CBAllocation,
    ) -> int:
        """
        Find the lowest L1 byte offset where alloc fits without overlapping
        any concurrently-live placement.

        Args:
            placed: List of (offset, size, live_start, live_end) already placed.
            alloc: The allocation to place.

        Returns:
            Byte offset within the L1 pool.
        """
        # Filter to only concurrent placements (overlapping lifetimes)
        concurrent = [
            (offset, size) for (offset, size, ls, le) in placed if not (le < alloc.live_start or ls > alloc.live_end)
        ]

        # Sort by offset
        concurrent.sort()

        # Find first gap that fits
        candidate = 0
        for co, cs in concurrent:
            if candidate + alloc.total_size <= co:
                return candidate
            candidate = max(candidate, co + cs)

        return candidate

    # =========================================================================
    # Backward-compatible accessors
    # =========================================================================

    def get_cb_index(self, op_id: str, port_name: str) -> int:
        """Get the allocated CB index for an (op, port) pair."""
        key = (op_id, port_name)
        if key not in self._allocations:
            raise KeyError(f"No CB allocation for ({op_id}, {port_name})")
        return self._allocations[key].index

    def get_all_indices(self) -> Set[int]:
        """Get all CB indices that were allocated."""
        return {a.index for a in self._allocations.values()}

    def get_liveness_summary(self) -> str:
        """Return a human-readable summary of liveness intervals and L1 layout."""
        lines = []
        for key, alloc in sorted(self._allocations.items()):
            ext = "EXT" if alloc.is_external else "INT"
            pool = f"@{alloc.pool_offset}" if alloc.pool_offset >= 0 else ""
            size = f"{alloc.total_size}B" if alloc.total_size > 0 else ""
            lines.append(
                f"  CB{alloc.index:2d} [{alloc.live_start}-{alloc.live_end}] "
                f"{alloc.op_id}.{alloc.port_name} {ext} {size}{pool}"
            )
        return "\n".join(lines)

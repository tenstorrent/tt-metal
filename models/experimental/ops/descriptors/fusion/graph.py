# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
OpGraph: Tree-based fusion topology.

Builds fused descriptors from a tree of OpNode objects. Each node holds one
operation. Parent->child edges encode sequential ordering; sibling nodes run
in parallel on disjoint core subsets.

Usage::

    root = OpNode(ln_op, children=[OpNode(rms_op_A), OpNode(rms_op_B)])
    fused = OpGraphBuilder(root).build()
    # Returns a single FusedOp, suitable for composite.launch()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.fusion.common import (
    BarrierConfig,
    BarrierSegment,
    MultiBarrierSpec,
    _BuildResult,
    _core_range_set_to_coords,
    _core_ranges_key,
    _coords_to_core_range_set,
    _get_node_core_range,
)
from models.experimental.ops.descriptors.fusion.cb_allocator import (
    _save_cb_state,
    _restore_cb_state,
    _verify_cb_restore,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class OpNode:
    """A node in the fusion tree.

    Each node holds one operation.  Parent->child edges encode sequential
    ordering (parent runs before child).  Sibling nodes run in parallel on
    disjoint core subsets.  A leaf node has no children.

    The node's core range is derived from its op's ProgramDescriptor
    kernels via ``_get_node_core_range()`` -- there is no separate
    core_range field.
    """

    op: OpDescriptor
    children: List["OpNode"] = field(default_factory=list)


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_device_from_tree(node: OpNode):
    """Walk an OpNode tree, return device from first tensor found."""
    for t in node.op.input_tensors:
        return t.device()
    for t in node.op.output_tensors:
        return t.device()
    for child in node.children:
        dev = _extract_device_from_tree(child)
        if dev is not None:
            return dev
    return None


def _merge_build_results(results: List[_BuildResult]) -> _BuildResult:
    """Merge multiple _BuildResults into one.

    Combines ProgramDescriptors, deduplicates input tensors (by identity),
    concatenates output tensors, and unions semaphore refs.
    """
    if len(results) == 1:
        return results[0]

    merged_desc = ttnn.merge_program_descriptors([r.descriptor for r in results])

    # Deduplicate input tensors (shared inputs appear in multiple ops)
    all_inputs: List = []
    seen_ids: Set[int] = set()
    for r in results:
        for t in r.input_tensors:
            tid = id(t)
            if tid not in seen_ids:
                all_inputs.append(t)
                seen_ids.add(tid)

    # Output tensors: one per result, in order
    all_outputs = [t for r in results for t in r.output_tensors]

    # Union semaphore refs
    all_semaphores = tuple(ref for r in results for ref in r.semaphores)

    return _BuildResult(
        descriptor=merged_desc,
        input_tensors=all_inputs,
        output_tensors=all_outputs,
        semaphores=all_semaphores,
    )


# =============================================================================
# OpGraph Builder
# =============================================================================


class OpGraphBuilder:
    """Builds fused descriptors from a tree of OpNode objects.

    The fusion tree is a standard tree where each node holds one operation.
    Parent->child edges encode sequential ordering; sibling nodes run in
    parallel on disjoint core subsets.  For each root-to-leaf path, a
    separate fused kernel binary is generated.  Nodes sharing a path
    segment synchronize via shared GlobalSemaphore addresses.

    Usage::

        root = OpNode(ln_op, children=[OpNode(rms_op_A), OpNode(rms_op_B)])
        fused = OpGraphBuilder(root).build()
        # Returns a single FusedOp, suitable for composite.launch()
    """

    def __init__(self, root: OpNode):
        self._root = root
        self._built = False

    def build(self, device: Any = None):
        """Build a fused descriptor from the tree.

        Returns a single self-contained FusedOp.  For branching trees,
        path ProgramDescriptors are merged internally so the result can be
        dispatched as one unit via ``composite.launch([result])``.

        Output tensors are ordered by leaf in left-to-right DFS order.

        Args:
            device: Optional device for GlobalSemaphore allocation.
                If *None*, auto-extracted from the first tensor found
                in the tree's OpDescriptors.
        """
        from models.experimental.ops.descriptors.fusion.fusion import FusedOp

        r = self._build_internal(device)
        return FusedOp(
            op=OpDescriptor(r.descriptor, r.input_tensors, r.output_tensors),
            semaphores=r.semaphores,
        )

    def _build_internal(self, device: Any = None) -> _BuildResult:
        """Internal build returning intermediate _BuildResult."""
        from models.experimental.ops.descriptors.fusion.codegen import (
            _create_barrier_segment_config,
        )

        if self._built:
            raise ValueError("Already built")
        self._built = True

        # Single node with no children = nothing to fuse
        if not self._root.children:
            op = self._root.op
            return _BuildResult(
                descriptor=op.descriptor,
                input_tensors=op.input_tensors,
                output_tensors=op.output_tensors,
            )

        # Validate tree topology before doing any device allocation
        self._validate_topology()

        # Auto-extract device from tensors if not provided
        if device is None:
            device = _extract_device_from_tree(self._root)
            if device is None:
                raise ValueError("Cannot auto-extract device: no tensors found in tree. " "Pass device explicitly.")

        # Trace all root-to-leaf paths
        paths = self._trace_paths()

        # Compute union of all leaf core ranges
        union_range = self._compute_union_ranges()

        # Allocate shared per-core monotonic semaphores on union range.
        sem_compute_done = ttnn.create_global_semaphore(device, union_range, 0)
        sem_writer_done = ttnn.create_global_semaphore(device, union_range, 0)
        compute_done_addr = ttnn.get_global_semaphore_address(sem_compute_done)
        writer_done_addr = ttnn.get_global_semaphore_address(sem_writer_done)
        all_sem_refs = [sem_compute_done, sem_writer_done]

        # Pre-allocate barrier configs for each unique core range across all
        # path segments.  Paths that share a segment MUST use the same
        # arrive/release GlobalSemaphore L1 addresses so that cores running
        # different kernel binaries synchronize at the same barrier.
        segment_cache: Dict[frozenset, BarrierConfig] = {}
        for path in paths:
            for core_range, _ in path:
                key = _core_ranges_key(core_range)
                if key not in segment_cache:
                    segment_cache[key] = _create_barrier_segment_config(device, core_range)
                    all_sem_refs.extend(segment_cache[key]._sem_refs)

        results = []
        for path in paths:
            # Save CB descriptor state before building each path.
            path_prog_descs = [op.descriptor for _, phases in path for op in phases]
            saved_cb_state = _save_cb_state(path_prog_descs)

            result = self._build_path(
                device,
                path,
                compute_done_addr,
                writer_done_addr,
                all_sem_refs,
                segment_cache,
            )
            results.append(result)

            # Restore original state so the next path sees uncorrupted indices
            _restore_cb_state(saved_cb_state)
            _verify_cb_restore(saved_cb_state)

        return _merge_build_results(results)

    def _trace_paths(self) -> List[List[Tuple[Any, List[OpDescriptor]]]]:
        """Trace all root-to-leaf paths through the tree.

        Returns a list of paths.  Each path is a list of
        ``(core_range, [phases])`` segments.  Consecutive nodes with the
        same core range are grouped into one segment.
        """
        raw_paths: List[List[Tuple[Any, OpDescriptor]]] = []

        def _collect(node: OpNode, prefix: List[Tuple[Any, OpDescriptor]]):
            if not node.children:
                # Leaf: use the node's own core range
                core_range = _get_node_core_range(node)
                raw_paths.append(prefix + [(core_range, node.op)])
            else:
                # Internal node: use effective range (union of descendant leaves)
                eff_coords = self._effective_leaf_range(node)
                eff_range = _coords_to_core_range_set(eff_coords)
                current = prefix + [(eff_range, node.op)]
                for child in node.children:
                    _collect(child, current)

        _collect(self._root, [])
        return [self._group_into_segments(raw) for raw in raw_paths]

    @staticmethod
    def _group_into_segments(
        raw_path: List[Tuple[Any, OpDescriptor]],
    ) -> List[Tuple[Any, List[OpDescriptor]]]:
        """Group consecutive same-range nodes into segments."""
        segments: List[Tuple[Any, List[OpDescriptor]]] = []
        for core_range, op in raw_path:
            if segments and _core_ranges_key(segments[-1][0]) == _core_ranges_key(core_range):
                segments[-1][1].append(op)
            else:
                segments.append((core_range, [op]))
        return segments

    def _compute_union_ranges(self) -> Any:
        """Compute the union CoreRangeSet of all leaf core ranges."""
        all_coords: Set[Tuple[int, int]] = set()

        def _collect_leaves(node: OpNode):
            if not node.children:
                all_coords.update(_core_range_set_to_coords(_get_node_core_range(node)))
            else:
                for child in node.children:
                    _collect_leaves(child)

        _collect_leaves(self._root)
        return _coords_to_core_range_set(all_coords)

    def _validate_topology(self) -> None:
        """Validate the tree topology.

        Checks:
        - Each child's core_range is a subset of its parent's range.
        - Sibling nodes have disjoint core ranges.

        Raises:
            ValueError: On any topology violation.
        """

        def _validate_children(node: OpNode, parent_coords: Set[Tuple[int, int]], depth: int):
            if not node.children:
                return

            # Check sibling disjointness
            seen_coords: Set[Tuple[int, int]] = set()
            for child in node.children:
                child_coords = _core_range_set_to_coords(_get_node_core_range(child))
                overlap = seen_coords & child_coords
                if overlap:
                    raise ValueError(
                        f"OpGraph topology error: sibling nodes at depth {depth + 1} "
                        f"have overlapping cores {sorted(overlap)}"
                    )
                seen_coords |= child_coords

            # Check each child is subset of parent, then recurse
            for child in node.children:
                child_coords = _core_range_set_to_coords(_get_node_core_range(child))
                if not child_coords.issubset(parent_coords):
                    extra = sorted(child_coords - parent_coords)
                    raise ValueError(
                        f"OpGraph topology error: node at depth {depth + 1} "
                        f"(cores {sorted(child_coords)}) has cores {extra} "
                        f"outside parent range {sorted(parent_coords)}"
                    )
                _validate_children(child, child_coords, depth + 1)

        root_coords = _core_range_set_to_coords(_get_node_core_range(self._root))
        _validate_children(self._root, root_coords, depth=0)

    @staticmethod
    def _effective_leaf_range(node: OpNode) -> Set[Tuple[int, int]]:
        """Compute the union of all descendant leaf core coordinates."""
        if not node.children:
            return _core_range_set_to_coords(_get_node_core_range(node))
        coords: Set[Tuple[int, int]] = set()
        for child in node.children:
            coords |= OpGraphBuilder._effective_leaf_range(child)
        return coords

    def _build_path(
        self,
        device: Any,
        path: List[Tuple[Any, List[OpDescriptor]]],
        compute_done_addr: int,
        writer_done_addr: int,
        shared_sem_refs: List[Any],
        segment_cache: Dict[frozenset, BarrierConfig],
    ) -> _BuildResult:
        """Build a fused _BuildResult for one root-to-leaf path."""
        from models.experimental.ops.descriptors.fusion.codegen import (
            _build_fused_descriptor,
            _create_phase_info,
        )

        # Flatten phases and determine leaf core range
        all_phases: List[OpDescriptor] = []
        for _, phases in path:
            all_phases.extend(phases)

        # Leaf core range = last segment's core range.
        leaf_core_range = path[-1][0] if len(path) > 1 else None

        # Build barrier segments and transition map
        segments, transition_map = self._build_barrier_segments(
            path,
            all_phases,
            segment_cache,
        )

        # Build MultiBarrierSpec
        multi_barrier = MultiBarrierSpec(
            segments=segments,
            compute_done_addr=compute_done_addr,
            writer_done_addr=writer_done_addr,
            transition_map=transition_map,
            _sem_refs=list(shared_sem_refs),
        )

        # Build PhaseInfo list and call _build_fused_descriptor
        phase_infos = [_create_phase_info(op, i) for i, op in enumerate(all_phases)]

        return _build_fused_descriptor(
            phase_infos,
            device,
            core_range_override=leaf_core_range,
            multi_barrier=multi_barrier,
        )

    def _build_barrier_segments(
        self,
        path: List[Tuple[Any, List[OpDescriptor]]],
        all_phases: List[OpDescriptor],
        segment_cache: Dict[frozenset, BarrierConfig],
    ) -> Tuple[List[BarrierSegment], Dict[int, Tuple[int, int]]]:
        """Build barrier segments and transition map for a path.

        Uses ``segment_cache`` (keyed by core_ranges_key) so that paths
        sharing a segment (e.g. the stem) reuse the same GlobalSemaphore
        arrive/release addresses for cross-kernel synchronization.

        Returns (segments, transition_map).
        """
        segments: List[BarrierSegment] = []

        transition_map: Dict[int, Tuple[int, int]] = {}
        global_phase_offset = 0

        for seg_path_idx, (core_range, phases) in enumerate(path):
            num_phases_in_seg = len(phases)
            if num_phases_in_seg == 0:
                global_phase_offset += num_phases_in_seg
                continue

            is_last_segment = seg_path_idx == len(path) - 1
            num_transitions_within = num_phases_in_seg - 1
            num_transitions = num_transitions_within
            if not is_last_segment:
                num_transitions += 1  # transition to next segment

            if num_transitions > 0:
                # Look up pre-allocated barrier config from cache
                key = _core_ranges_key(core_range)
                barrier_cfg = segment_cache[key]
                seg = BarrierSegment(
                    config=barrier_cfg,
                    arrive_addr=barrier_cfg.global_arrive_addr,
                    release_addr=barrier_cfg.global_release_addr,
                )
                segments.append(seg)

                seg_idx = len(segments) - 1
                call_idx = 0

                # Map transitions within this segment
                for t in range(num_transitions_within):
                    global_transition = global_phase_offset + t
                    transition_map[global_transition] = (seg_idx, call_idx)
                    call_idx += 1

                # Map transition to next segment (if any)
                if not is_last_segment:
                    global_transition = global_phase_offset + num_phases_in_seg - 1
                    transition_map[global_transition] = (seg_idx, call_idx)

            global_phase_offset += num_phases_in_seg

        return segments, transition_map


# =============================================================================
# Convenience Functions
# =============================================================================


def build_op_graph(
    root_phases: List[OpDescriptor],
    children: List[OpNode],
    device: Any = None,
):
    """Build a fused descriptor for a tree topology.

    Convenience wrapper around :class:`OpGraphBuilder`.  Converts
    ``root_phases`` into a chain of nodes, attaches ``children`` to
    the last root-phase node, then builds.

    Args:
        root_phases: Phases that run before the tree splits.  Converted
            into a chain of OpNode objects.
        children: Subtrees to attach to the last root-phase node.
        device: Optional device for GlobalSemaphore allocation.
            If *None*, auto-extracted from the first tensor found
            in the tree's OpDescriptors.

    Returns:
        A single self-contained FusedOp suitable for
        ``composite.launch([result])``.
    """
    if not root_phases:
        raise ValueError("root_phases cannot be empty")
    # Last root phase gets children attached
    last = OpNode(root_phases[-1], children=list(children))
    node = last
    for desc in reversed(root_phases[:-1]):
        node = OpNode(desc, children=[node])
    return OpGraphBuilder(node).build(device)


__all__ = [
    "OpNode",
    "OpGraphBuilder",
    "build_op_graph",
    "_extract_device_from_tree",
    "_merge_build_results",
]

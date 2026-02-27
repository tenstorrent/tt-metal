# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
OpGraph: Tree-based fusion topology.

Builds fused descriptors from a tree of OpNode objects. Each node holds one
operation. Parent->child edges encode sequential ordering; sibling nodes run
in parallel on disjoint core subsets.

The builder uses a per-core group approach: it walks the tree to determine
each core's phase sequence, groups cores with identical sequences, and builds
one fused kernel binary per group.

Usage::

    root = OpNode(ln_op, children=[OpNode(rms_op_A), OpNode(rms_op_B)])
    fused = OpGraphBuilder(root).build()
    # Returns a single FusedOp, suitable for composite.launch()
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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


@dataclass
class CoreGroup:
    """A group of cores that execute the same phase sequence.

    Each group becomes one fused kernel binary.  The group's core_range
    is derived from the kernel descriptors of the leaf node in each
    core's root-to-leaf path through the tree.

    Attributes:
        core_range: CoreRangeSet of cores in this group.
        phases: Ordered list of OpDescriptors these cores execute.
        barrier_scopes: CoreRangeSet per phase transition
            (len = len(phases) - 1).  Each entry is the effective leaf
            range of the tree node at that position, determining which
            cores must synchronize at that transition.
    """

    core_range: Any
    phases: List[OpDescriptor]
    barrier_scopes: List[Any]


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


def _compute_shared_cb_remaps(
    groups: List[CoreGroup],
) -> List[Optional[Dict[int, Dict[int, int]]]]:
    """Compute forced CB remaps for phases shared across multiple groups.

    When an op appears in multiple groups (e.g. block-sharded LN with sender
    on group 0 and receiver on group 1), its CB slot assignments must be
    identical across groups.  Otherwise, multicast writes hit wrong L1
    addresses on receiver cores.

    Strategy: build a single reference pool from the union of ALL shared ops
    (in the order they first appear across groups), then force each group to
    use that pool's slot assignments for its shared phases.

    Returns a list (one per group) of forced_phase_remaps dicts, or None for
    groups that don't need forcing.
    """
    from models.experimental.ops.descriptors.fusion.cb_allocator import (
        CBPoolAllocator,
        _get_phantom_cb_indices,
        _extract_remote_cb_indices,
    )
    from models.experimental.ops.descriptors.fusion.codegen import _create_phase_info

    # Count how many groups each op (by identity) appears in.
    op_group_count: Dict[int, int] = defaultdict(int)
    for group in groups:
        seen_ops: Set[int] = set()
        for op in group.phases:
            op_id = id(op)
            if op_id not in seen_ops:
                op_group_count[op_id] += 1
                seen_ops.add(op_id)

    shared_op_ids = {op_id for op_id, count in op_group_count.items() if count > 1}
    if not shared_op_ids:
        return [None] * len(groups)

    # Build an ordered list of unique shared ops (first occurrence order).
    shared_ops_ordered: List[OpDescriptor] = []
    shared_ops_seen: Set[int] = set()
    for group in groups:
        for op in group.phases:
            op_id = id(op)
            if op_id in shared_op_ids and op_id not in shared_ops_seen:
                shared_ops_ordered.append(op)
                shared_ops_seen.add(op_id)

    # Build a reference pool from the shared ops.
    ref_pool = CBPoolAllocator(max_slots=32)
    ref_phase_infos = [_create_phase_info(op, i) for i, op in enumerate(shared_ops_ordered)]
    for pi in ref_phase_infos:
        for remote_idx in _extract_remote_cb_indices(pi.op_descriptor.descriptor):
            ref_pool.reserve_index(remote_idx)
    for phase_idx, pi in enumerate(ref_phase_infos):
        phantom_indices = _get_phantom_cb_indices(pi)
        ref_pool.allocate_phase(phase_idx, pi.cb_info, phantom_indices)

    # Build op_id -> reference remap lookup.
    ref_remaps: Dict[int, Dict[int, int]] = {}
    for ref_idx, op in enumerate(shared_ops_ordered):
        ref_remaps[id(op)] = ref_pool.get_remap(ref_idx)

    # For each group, build forced_phase_remaps for its shared phases.
    result: List[Optional[Dict[int, Dict[int, int]]]] = []
    for group in groups:
        forced: Dict[int, Dict[int, int]] = {}
        for phase_idx, op in enumerate(group.phases):
            if id(op) in shared_op_ids:
                forced[phase_idx] = ref_remaps[id(op)]
        result.append(forced if forced else None)

    return result


def _equalize_cb_sizes(results: List[_BuildResult]) -> None:
    """Equalize CB total_sizes across multi-group build results.

    When multiple groups share an op whose kernels communicate across groups
    via multicast (e.g. block-sharded LN sender/receiver), the CB L1
    addresses must be identical on all cores.  CB addresses are allocated
    sequentially by buffer_index, so any per-slot size difference shifts
    all subsequent addresses.

    Fix: for each buffer_index, compute the max total_size across all
    groups and pad smaller allocations to match.
    """
    # Collect max total_size per buffer_index across all groups.
    max_sizes: Dict[int, int] = {}
    for result in results:
        for cb in result.descriptor.cbs:
            for fmt in cb.format_descriptors:
                idx = fmt.buffer_index
                if idx not in max_sizes or cb.total_size > max_sizes[idx]:
                    max_sizes[idx] = cb.total_size

    # Pad any CB whose total_size is below the cross-group max.
    for result in results:
        for cb in result.descriptor.cbs:
            for fmt in cb.format_descriptors:
                idx = fmt.buffer_index
                if cb.total_size < max_sizes[idx]:
                    cb.total_size = max_sizes[idx]


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

    # Concatenate kernel labels
    all_labels = tuple(label for r in results for label in r.kernel_labels)

    # Concatenate kernel_phase_map (order matches merge_program_descriptors)
    all_kpm = tuple(entry for r in results for entry in r.kernel_phase_map)

    return _BuildResult(
        descriptor=merged_desc,
        input_tensors=all_inputs,
        output_tensors=all_outputs,
        semaphores=all_semaphores,
        kernel_labels=all_labels,
        kernel_phase_map=all_kpm,
    )


# =============================================================================
# OpGraph Builder
# =============================================================================


class OpGraphBuilder:
    """Builds fused descriptors from a tree of OpNode objects.

    The fusion tree is a standard tree where each node holds one operation.
    Parent->child edges encode sequential ordering; sibling nodes run in
    parallel on disjoint core subsets.

    The builder uses a per-core group approach:

    1. Walk the tree to determine each core's phase sequence.
    2. Group cores with identical phase sequences into :class:`CoreGroup`
       objects.
    3. Build one fused kernel binary per group.
    4. Merge all group binaries into a single ProgramDescriptor.

    Groups sharing a barrier scope (e.g. the stem) reuse the same
    GlobalSemaphore addresses for cross-kernel synchronization.

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
        group ProgramDescriptors are merged internally so the result can
        be dispatched as one unit via ``composite.launch([result])``.

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

        # Compute per-core groups
        groups = self._compute_core_groups()

        # Compute union of all leaf core ranges
        union_range = self._compute_union_ranges()

        # Allocate shared per-core monotonic semaphores on union range.
        sem_compute_done = ttnn.create_global_semaphore(device, union_range, 0)
        sem_writer_done = ttnn.create_global_semaphore(device, union_range, 0)
        compute_done_addr = ttnn.get_global_semaphore_address(sem_compute_done)
        writer_done_addr = ttnn.get_global_semaphore_address(sem_writer_done)
        all_sem_refs = [sem_compute_done, sem_writer_done]

        # Pre-allocate barrier configs for each unique barrier scope across
        # all groups.  Groups sharing a scope MUST use the same arrive/release
        # GlobalSemaphore L1 addresses so that cores running different kernel
        # binaries synchronize at the same barrier.
        segment_cache: Dict[frozenset, BarrierConfig] = {}
        for group in groups:
            for scope in group.barrier_scopes:
                key = _core_ranges_key(scope)
                if key not in segment_cache:
                    segment_cache[key] = _create_barrier_segment_config(device, scope)
                    all_sem_refs.extend(segment_cache[key]._sem_refs)

        # When there are multiple groups (branching tree), phases may have
        # different native core ranges than the group's range (e.g. stem
        # covers 16 cores, group only covers 8).  In this case, we need to
        # pass target_core_range to restrict the fused kernel to the group.
        multi_group = len(groups) > 1

        # For multi-group builds, compute forced CB remaps for shared phases.
        # Ops whose kernels span multiple groups (e.g. block-sharded LN with
        # mcast sender on row 0 and receiver on row 1) MUST get identical CB
        # slot assignments in every group, otherwise multicast writes hit wrong
        # L1 addresses.  We build a reference pool from the shared phases and
        # force each group to use those slot assignments.
        per_group_forced: List[Optional[Dict[int, Dict[int, int]]]] = [None] * len(groups)
        if multi_group:
            per_group_forced = _compute_shared_cb_remaps(groups)

        results = []
        for g_idx, group in enumerate(groups):
            # Save CB descriptor state before building each group.
            group_prog_descs = [op.descriptor for op in group.phases]
            saved_cb_state = _save_cb_state(group_prog_descs)

            result = self._build_group(
                device,
                group,
                compute_done_addr,
                writer_done_addr,
                all_sem_refs,
                segment_cache,
                needs_target_core_range=multi_group,
                forced_phase_remaps=per_group_forced[g_idx],
            )
            results.append(result)

            # Restore original state so the next group sees uncorrupted indices
            _restore_cb_state(saved_cb_state)
            _verify_cb_restore(saved_cb_state)

        # Equalize CB total_sizes across groups so that all fused kernel
        # binaries produce the same L1 layout.  This is critical even with
        # forced remaps: non-shared phases can still change slot sizes which
        # shifts L1 addresses for subsequent slots.
        if multi_group:
            _equalize_cb_sizes(results)

        return _merge_build_results(results)

    @staticmethod
    def _kernel_variant_key(op: OpDescriptor, coord: Tuple[int, int]) -> tuple:
        """Return tuple of kernel indices whose core ranges contain *coord*.

        For interleaved ops (all kernels on the same range), every core
        produces the same key.  For block-sharded ops with per-core-subset
        kernels (e.g. mcast sender on row 0, receiver on row 1), different
        cores produce different keys, causing them to land in separate groups.
        """
        indices = []
        for k_idx, kernel in enumerate(op.descriptor.kernels):
            if coord in _core_range_set_to_coords(kernel.core_ranges):
                indices.append(k_idx)
        return tuple(indices)

    def _compute_core_groups(self) -> List[CoreGroup]:
        """Compute per-core phase sequences and group by identity.

        Walks the tree pre-order.  For each node, records its op and
        barrier scope (effective leaf range) for every core in the node's
        range.  Cores with identical phase sequences are grouped together.

        The grouping key includes both op identity and a kernel variant key
        (which kernel indices cover each core).  This ensures that cores
        seeing different kernel subsets for the same op (e.g. mcast sender
        vs receiver in block-sharded LayerNorm) end up in separate groups.

        Returns list of CoreGroups, one per unique phase sequence.
        """
        # Per-core entries: coord -> [(OpDescriptor, barrier_scope_CoreRangeSet)]
        per_core: Dict[Tuple[int, int], List[Tuple[OpDescriptor, Any]]] = defaultdict(list)

        def _walk(node: OpNode):
            eff_coords = self._effective_leaf_range(node)
            eff_range = _coords_to_core_range_set(eff_coords)
            node_coords = _core_range_set_to_coords(_get_node_core_range(node))
            for coord in node_coords:
                per_core[coord].append((node.op, eff_range))
            for child in node.children:
                _walk(child)

        _walk(self._root)

        # Group cores by identical phase sequence (by op identity + kernel variant)
        groups_by_key: Dict[tuple, Set[Tuple[int, int]]] = defaultdict(set)
        for coord, entries in per_core.items():
            key = tuple((id(op), self._kernel_variant_key(op, coord)) for op, _ in entries)
            groups_by_key[key].add(coord)

        # Build CoreGroups
        result = []
        for key, coords in groups_by_key.items():
            representative = per_core[next(iter(coords))]
            core_range = _coords_to_core_range_set(coords)
            phases = [op for op, _ in representative]
            barrier_scopes = [eff_range for _, eff_range in representative[:-1]]
            result.append(
                CoreGroup(
                    core_range=core_range,
                    phases=phases,
                    barrier_scopes=barrier_scopes,
                )
            )

        return result

    def _build_group(
        self,
        device: Any,
        group: CoreGroup,
        compute_done_addr: int,
        writer_done_addr: int,
        shared_sem_refs: List[Any],
        segment_cache: Dict[frozenset, BarrierConfig],
        needs_target_core_range: bool = False,
        forced_phase_remaps: Optional[Dict[int, Dict[int, int]]] = None,
    ) -> _BuildResult:
        """Build a fused _BuildResult for one core group.

        Args:
            needs_target_core_range: If True, pass the group's core_range
                as target_core_range to _build_fused_descriptor.  This is
                needed for branching trees where the stem's native core
                range differs from the group's (leaf) core range.  For
                single-group linear chains, this should be False.
            forced_phase_remaps: If provided, maps phase_idx -> {orig_cb -> hw_slot}
                for phases whose CB remaps must match across groups.
        """
        from models.experimental.ops.descriptors.fusion.codegen import (
            _build_fused_descriptor,
            _create_phase_info,
        )

        # Build barrier segments and transition map from barrier scopes
        segments, transition_map = self._build_group_barriers(
            group.barrier_scopes,
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
        phase_infos = [_create_phase_info(op, i) for i, op in enumerate(group.phases)]

        # Only set target_core_range for branching trees where phases may
        # have different native core ranges than the group's core range.
        target_cr = group.core_range if needs_target_core_range else None

        return _build_fused_descriptor(
            phase_infos,
            device,
            target_core_range=target_cr,
            multi_barrier=multi_barrier,
            forced_phase_remaps=forced_phase_remaps,
        )

    @staticmethod
    def _build_group_barriers(
        barrier_scopes: List[Any],
        segment_cache: Dict[frozenset, BarrierConfig],
    ) -> Tuple[List[BarrierSegment], Dict[int, Tuple[int, int]]]:
        """Build barrier segments and transition map from barrier scopes.

        Consecutive transitions with the same barrier scope share a segment.
        The ``segment_cache`` ensures groups sharing a scope (e.g. the stem)
        use the same GlobalSemaphore addresses for synchronization.

        Returns (segments, transition_map).
        """
        segments: List[BarrierSegment] = []
        transition_map: Dict[int, Tuple[int, int]] = {}

        prev_scope_key = None
        seg_idx = -1
        call_idx = 0

        for t_idx, scope in enumerate(barrier_scopes):
            scope_key = _core_ranges_key(scope)
            if scope_key != prev_scope_key:
                # New barrier segment
                barrier_cfg = segment_cache[scope_key]
                seg = BarrierSegment(
                    config=barrier_cfg,
                    arrive_addr=barrier_cfg.global_arrive_addr,
                    release_addr=barrier_cfg.global_release_addr,
                )
                segments.append(seg)
                seg_idx += 1
                call_idx = 0
                prev_scope_key = scope_key
            transition_map[t_idx] = (seg_idx, call_idx)
            call_idx += 1

        return segments, transition_map

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
    "CoreGroup",
    "OpGraphBuilder",
    "build_op_graph",
    "_extract_device_from_tree",
    "_merge_build_results",
]

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
    _get_node_allowed_coords,
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
    allowed_core_range: Optional[Any] = field(default=None, repr=False)


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
        phase_op_indices: Index into the ``unique_ops`` list for each
            phase.  Used by the global CB pool to map group-local phases
            to global allocation indices.
    """

    core_range: Any
    phases: List[OpDescriptor]
    barrier_scopes: List[Any]
    phase_op_indices: List[int] = field(default_factory=list)


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


def _build_global_cb_pool(
    unique_ops: List[OpDescriptor],
) -> "CBPoolAllocator":
    """Build a single global CB pool from all unique ops in the tree.

    Allocates slots for every unique op in declaration order.  The resulting
    pool has consistent slot assignments that can be projected per-group
    via ``project_to_group()``.

    Args:
        unique_ops: Ordered list of unique OpDescriptors from the tree walk.

    Returns:
        A ``CBPoolAllocator`` with one phase_remap per unique op.
    """
    from models.experimental.ops.descriptors.fusion.cb_allocator import (
        CBPoolAllocator,
        _get_phantom_cb_indices,
        _extract_remote_cb_indices,
    )
    from models.experimental.ops.descriptors.fusion.codegen import _create_phase_info

    pool = CBPoolAllocator()

    # Create PhaseInfo for each unique op
    phase_infos = [_create_phase_info(op, i) for i, op in enumerate(unique_ops)]

    # Reserve remote CB indices from all ops first
    for pi in phase_infos:
        for remote_idx in _extract_remote_cb_indices(pi.op_descriptor.descriptor):
            pool.reserve_index(remote_idx)

    # Allocate each unique op as a phase
    for phase_idx, pi in enumerate(phase_infos):
        phantom_indices = _get_phantom_cb_indices(pi)
        pool.allocate_phase(phase_idx, pi.cb_info, phantom_indices)

    return pool


def _project_pools_for_groups(
    groups: List[CoreGroup],
    global_pool: "CBPoolAllocator",
) -> List["CBPoolAllocator"]:
    """Project the global pool to per-group pools with L1-alignment padding.

    Identifies shared slots (referenced by >=2 groups), computes the max
    shared slot index M, and pads every group's pool to include all global
    slots with index <= M.  This ensures identical L1 layout for CB indices
    0..M across all groups, which is required for multicast correctness.

    Args:
        groups: CoreGroups with ``phase_op_indices`` populated.
        global_pool: The global pool from ``_build_global_cb_pool()``.

    Returns:
        List of projected ``CBPoolAllocator`` pools, one per group.
    """
    # Compute which slots each group references
    per_group_slots: List[Set[int]] = []
    for group in groups:
        slots: Set[int] = set()
        for global_idx in group.phase_op_indices:
            remap = global_pool.phase_remaps[global_idx]
            slots.update(remap.values())
        per_group_slots.append(slots)

    # Identify shared slots (in >=2 groups)
    slot_group_count: Dict[int, int] = defaultdict(int)
    for slots in per_group_slots:
        for s in slots:
            slot_group_count[s] += 1
    shared_slots = {s for s, count in slot_group_count.items() if count > 1}

    # Compute padding: all global pool slots with index <= max(shared slots)
    padding_slots: Set[int] = set()
    if shared_slots:
        max_shared = max(shared_slots)
        padding_slots = {s for s in global_pool.get_all_slot_indices() if s <= max_shared}

    # Project each group
    projected: List["CBPoolAllocator"] = []
    for group in groups:
        proj = global_pool.project_to_group(group.phase_op_indices, padding_slots)
        projected.append(proj)

    return projected


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

        # Compute per-core groups and unique op list
        groups, unique_ops = self._compute_core_groups()

        # Compute union of all leaf core ranges
        union_range = self._compute_union_ranges()

        # Allocate shared per-core monotonic semaphores on union range.
        sem_compute_done = ttnn.create_global_semaphore(device, union_range, 0)
        sem_writer_done = ttnn.create_global_semaphore(device, union_range, 0)
        sem_reset_done = ttnn.create_global_semaphore(device, union_range, 0)
        compute_done_addr = ttnn.get_global_semaphore_address(sem_compute_done)
        writer_done_addr = ttnn.get_global_semaphore_address(sem_writer_done)
        reset_done_addr = ttnn.get_global_semaphore_address(sem_reset_done)
        all_sem_refs = [sem_compute_done, sem_writer_done, sem_reset_done]

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

        # Build global CB pool and project per-group pools.
        # The global pool allocates slots for all unique ops in one pass,
        # producing consistent slot assignments by construction.  Per-group
        # pools are projections with L1-alignment padding, replacing the
        # old shared-op detection + forced-remap + equalize approach.
        per_group_pools: List[Optional["CBPoolAllocator"]] = [None] * len(groups)
        if multi_group:
            global_pool = _build_global_cb_pool(unique_ops)
            per_group_pools = _project_pools_for_groups(groups, global_pool)

        # Save CB descriptor state for ALL unique ops (not just per-group).
        # Projected pools include padding slots whose source_fmt references
        # may point to ops outside the current group.  build_merged_cb_descriptors
        # mutates source_fmt.buffer_index on all slots including padding.  If we
        # only saved per-group ops, mutations on padding slots' source_fmt would
        # corrupt later groups' descriptors.
        all_prog_descs = [op.descriptor for op in unique_ops] if multi_group else []
        saved_all_cb_state = _save_cb_state(all_prog_descs) if multi_group else []

        results = []
        for g_idx, group in enumerate(groups):
            if multi_group:
                # Restore ALL unique ops before each group build so that
                # each group starts from pristine descriptor state.
                if g_idx > 0:
                    _restore_cb_state(saved_all_cb_state)
            else:
                # Single-group: save/restore only the group's own phases
                # (no padding slots from other ops to worry about).
                saved_all_cb_state = _save_cb_state([op.descriptor for op in group.phases])

            result = self._build_group(
                device,
                group,
                compute_done_addr,
                writer_done_addr,
                reset_done_addr,
                all_sem_refs,
                segment_cache,
                needs_target_core_range=multi_group,
                cb_pool=per_group_pools[g_idx],
            )
            results.append(result)

        # Restore original state after all groups are built
        _restore_cb_state(saved_all_cb_state)
        _verify_cb_restore(saved_all_cb_state)

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

    def _compute_core_groups(self) -> Tuple[List[CoreGroup], List[OpDescriptor]]:
        """Compute per-core phase sequences and group by identity.

        Walks the tree pre-order.  For each node, records its op and
        barrier scope (effective leaf range) for every core in the node's
        range.  Cores with identical phase sequences are grouped together.

        The grouping key includes both op identity and a kernel variant key
        (which kernel indices cover each core).  This ensures that cores
        seeing different kernel subsets for the same op (e.g. mcast sender
        vs receiver in block-sharded LayerNorm) end up in separate groups.

        Returns (groups, unique_ops):
            groups: List of CoreGroups, one per unique phase sequence.
            unique_ops: Ordered list of unique OpDescriptors encountered
                during the tree walk (deduped by ``id()``).
        """
        # Per-core entries: coord -> [(OpDescriptor, barrier_scope_CoreRangeSet)]
        per_core: Dict[Tuple[int, int], List[Tuple[OpDescriptor, Any]]] = defaultdict(list)

        # Track unique ops in first-encounter order
        unique_ops: List[OpDescriptor] = []
        op_id_to_index: Dict[int, int] = {}

        def _walk(node: OpNode):
            eff_coords = self._effective_leaf_range(node)
            eff_range = _coords_to_core_range_set(eff_coords)
            node_coords = _core_range_set_to_coords(_get_node_core_range(node))
            for coord in node_coords:
                per_core[coord].append((node.op, eff_range))
            # Register unique op
            op_id = id(node.op)
            if op_id not in op_id_to_index:
                op_id_to_index[op_id] = len(unique_ops)
                unique_ops.append(node.op)
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
            phase_op_indices = [op_id_to_index[id(op)] for op in phases]
            result.append(
                CoreGroup(
                    core_range=core_range,
                    phases=phases,
                    barrier_scopes=barrier_scopes,
                    phase_op_indices=phase_op_indices,
                )
            )

        return result, unique_ops

    def _build_group(
        self,
        device: Any,
        group: CoreGroup,
        compute_done_addr: int,
        writer_done_addr: int,
        reset_done_addr: int,
        shared_sem_refs: List[Any],
        segment_cache: Dict[frozenset, BarrierConfig],
        needs_target_core_range: bool = False,
        cb_pool: Optional["CBPoolAllocator"] = None,
    ) -> _BuildResult:
        """Build a fused _BuildResult for one core group.

        Args:
            needs_target_core_range: If True, pass the group's core_range
                as target_core_range to _build_fused_descriptor.  This is
                needed for branching trees where the stem's native core
                range differs from the group's (leaf) core range.  For
                single-group linear chains, this should be False.
            cb_pool: If provided, a pre-projected CB pool for this group
                (from the global pool).  When None, the builder self-allocates.
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
            reset_done_addr=reset_done_addr,
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
            cb_pool=cb_pool,
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

    @staticmethod
    def _propagate_allowed_ranges(root: OpNode) -> None:
        """Propagate allowed_core_range bottom-up through the tree.

        Post-order traversal:
        1. If ``node.op.allowed_core_range`` is set, use it.
        2. Else if leaf, default to actual core range.
        3. Else, union of own actual range + children's allowed ranges.

        After this, every node has ``allowed_core_range`` set.
        """

        def _propagate(node: OpNode):
            # Recurse children first (post-order)
            for child in node.children:
                _propagate(child)

            if node.op.allowed_core_range is not None:
                node.allowed_core_range = node.op.allowed_core_range
            elif not node.children:
                # Leaf: default to actual core range
                node.allowed_core_range = _get_node_core_range(node)
            else:
                # Internal node: union of own actual range + children's allowed ranges.
                # Including the actual range ensures partial coverage is valid
                # (parent may use more cores than children collectively cover).
                all_coords: Set[Tuple[int, int]] = _core_range_set_to_coords(_get_node_core_range(node))
                for child in node.children:
                    all_coords |= _core_range_set_to_coords(child.allowed_core_range)
                node.allowed_core_range = _coords_to_core_range_set(all_coords)

        _propagate(root)

    def _validate_topology(self) -> None:
        """Validate the tree topology using allowed core ranges.

        First propagates ``allowed_core_range`` on all nodes, then checks:

        1. **Actual subset of allowed**: Each node's actual core range is a
           subset of its allowed range.
        2. **Child allowed subset of parent allowed**: Each child's allowed
           range is a subset of its parent's allowed range.
        3. **Sibling disjointness**: Siblings' allowed ranges are pairwise
           disjoint.

        Raises:
            ValueError: On any topology violation.
        """
        # Propagate allowed ranges before validation
        self._propagate_allowed_ranges(self._root)

        def _validate_node(node: OpNode, parent_allowed_coords: Set[Tuple[int, int]], depth: int):
            # Check actual ⊆ allowed for this node
            actual_coords = _core_range_set_to_coords(_get_node_core_range(node))
            allowed_coords = _get_node_allowed_coords(node)
            if not actual_coords.issubset(allowed_coords):
                extra = sorted(actual_coords - allowed_coords)
                raise ValueError(
                    f"OpGraph topology error: node at depth {depth} "
                    f"has actual cores {extra} outside its allowed range "
                    f"{sorted(allowed_coords)}"
                )

            if not node.children:
                return

            # Check sibling disjointness (using allowed ranges)
            seen_coords: Set[Tuple[int, int]] = set()
            for child in node.children:
                child_allowed = _get_node_allowed_coords(child)
                overlap = seen_coords & child_allowed
                if overlap:
                    raise ValueError(
                        f"OpGraph topology error: sibling nodes at depth {depth + 1} "
                        f"have overlapping cores {sorted(overlap)}"
                    )
                seen_coords |= child_allowed

            # Check each child's allowed ⊆ parent's allowed, then recurse
            for child in node.children:
                child_allowed = _get_node_allowed_coords(child)
                if not child_allowed.issubset(allowed_coords):
                    extra = sorted(child_allowed - allowed_coords)
                    raise ValueError(
                        f"OpGraph topology error: node at depth {depth + 1} "
                        f"(cores {sorted(child_allowed)}) has cores {extra} "
                        f"outside parent range {sorted(allowed_coords)}"
                    )
                _validate_node(child, allowed_coords, depth + 1)

        root_allowed = _get_node_allowed_coords(self._root)
        _validate_node(self._root, root_allowed, depth=0)

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

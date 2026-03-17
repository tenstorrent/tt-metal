# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
    # Returns a single FusedOp, suitable for .launch()
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
    _NOOP_OP,
    _core_range_set_to_coords,
    _core_ranges_key,
    _coords_to_core_range_set,
    _get_node_core_range,
)
from models.experimental.ops.descriptors.fusion.cb_allocator import (
    PhaseInfo,
    _save_cb_state,
    _restore_cb_state,
    _verify_cb_restore,
)


# No-op phases have no entry in the global CB pool
NOOP_PHASE_INDEX = None

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
            (len = len(phases) - 1).  Each entry is the union of all
            descendant core ranges at that position, determining which
            cores must synchronize (release) at that transition.
        barrier_arrive_scopes: CoreRangeSet per phase transition
            (len = len(phases) - 1).  Subset of barrier_scopes — only
            cores that did real work in the completed phase.  Used for
            the arrive threshold in asymmetric barriers.
        phase_op_indices: Index into the ``unique_ops`` list for each
            phase.  Used by the global CB pool to map group-local phases
            to global allocation indices.
    """

    core_range: Any
    phases: List[OpDescriptor]
    barrier_scopes: List[Any]
    barrier_arrive_scopes: List[Any] = field(default_factory=list)
    phase_op_indices: List[int] = field(default_factory=list)
    has_trailing_barrier: bool = False


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
    device: Any = None,
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
        num_cbs_for_device,
        _get_phantom_cb_indices,
        _extract_remote_cb_indices,
    )
    from models.experimental.ops.descriptors.fusion.codegen import _create_phase_info

    pool = CBPoolAllocator(max_slots=num_cbs_for_device(device))

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
            if global_idx is NOOP_PHASE_INDEX:
                continue  # No-op phases have no CB allocation
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

    # Project each group — filter out no-op phases for CB pool projection,
    # then re-insert empty remaps at no-op positions so phase_remaps aligns
    # with group.phases (builder indexes by phase position).
    projected: List["CBPoolAllocator"] = []
    for group in groups:
        real_indices = [idx for idx in group.phase_op_indices if idx is not NOOP_PHASE_INDEX]
        proj = global_pool.project_to_group(real_indices, padding_slots)

        # Re-insert empty remaps for no-op phases at the correct positions
        noop_positions = [i for i, idx in enumerate(group.phase_op_indices) if idx is NOOP_PHASE_INDEX]
        if noop_positions:
            for pos in noop_positions:
                proj.phase_remaps.insert(pos, {})

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
        # Returns a single FusedOp, suitable for .launch()
    """

    def __init__(self, root: OpNode):
        self._root = root
        self._built = False

    def build(self, device: Any = None):
        """Build a fused descriptor from the tree.

        Returns a single self-contained FusedOp.  For branching trees,
        group ProgramDescriptors are merged internally so the result can
        be dispatched as one unit via ``result.launch()``.

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

        # Pre-allocate barrier configs for each unique (release, arrive) scope
        # pair across all groups.  Groups sharing a release scope MUST use the
        # same arrive/release GlobalSemaphore L1 addresses so that cores
        # running different kernel binaries synchronize at the same barrier.
        # Different arrive scopes need different arrive thresholds, so the
        # cache key includes both release AND arrive scopes.
        segment_cache: Dict[Tuple[frozenset, frozenset], BarrierConfig] = {}
        for group in groups:
            for release_scope, arrive_scope in zip(group.barrier_scopes, group.barrier_arrive_scopes):
                release_key = _core_ranges_key(release_scope)
                arrive_key = _core_ranges_key(arrive_scope) if arrive_scope is not None else frozenset()
                cache_key = (release_key, arrive_key)
                if cache_key not in segment_cache:
                    segment_cache[cache_key] = _create_barrier_segment_config(
                        device, release_scope, arrive_ranges=arrive_scope
                    )
                    all_sem_refs.extend(segment_cache[cache_key]._sem_refs)

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
            global_pool = _build_global_cb_pool(unique_ops, device=device)
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
        # Per-core entries: coord -> [(OpDescriptor, release_scope, arrive_scope)]
        # release_scope = node_coords ∪ desc_coords (all cores in the barrier)
        # arrive_scope = node_coords (cores that did real work in this phase)
        per_core: Dict[Tuple[int, int], List[Tuple[OpDescriptor, Any, Any]]] = defaultdict(list)

        # Cores that need a trailing barrier after their last real phase.
        # These cores finish real work but have no subsequent phase — the
        # trailing barrier lets them arrive ("I'm done") so other cores
        # waiting on the barrier can proceed.
        trailing_barrier_coords: Dict[Tuple[int, int], Tuple[Any, Any]] = {}

        # Track unique ops in first-encounter order
        unique_ops: List[OpDescriptor] = []
        op_id_to_index: Dict[int, int] = {}

        def _walk(node: OpNode):
            desc_coords = self._all_descendant_coords(node)
            node_coords = _core_range_set_to_coords(_get_node_core_range(node))

            # Barrier scope = everyone who participates: node cores + descendant cores.
            # Arrive scope = cores that did real work (node cores).
            # No-op cores (in descendants but not node) wait only.
            barrier_coords = desc_coords | node_coords
            release_scope = _coords_to_core_range_set(barrier_coords) if barrier_coords else None
            arrive_scope = _coords_to_core_range_set(node_coords) if node_coords else None

            # Real phase entries for cores in this node
            for coord in node_coords:
                per_core[coord].append((node.op, release_scope, arrive_scope))

            # No-op entries for cores in descendants but not this node
            for coord in desc_coords - node_coords:
                per_core[coord].append((_NOOP_OP, release_scope, arrive_scope))

            # Register unique op (real ops only — noop is not a unique op)
            op_id = id(node.op)
            if op_id not in op_id_to_index:
                op_id_to_index[op_id] = len(unique_ops)
                unique_ops.append(node.op)

            # Early-exit cores (in node but not in any descendant) need a
            # trailing barrier after their last phase so they can arrive
            # at the cross-core barrier before exiting.
            exit_coords = node_coords - desc_coords
            if exit_coords and desc_coords:
                for coord in exit_coords:
                    trailing_barrier_coords[coord] = (release_scope, arrive_scope)

            for child in node.children:
                _walk(child)

        _walk(self._root)

        # Group cores by identical phase sequence (by op identity + kernel variant).
        # The grouping key also distinguishes trailing-barrier vs non-trailing
        # cores so they land in separate groups.
        groups_by_key: Dict[tuple, Set[Tuple[int, int]]] = defaultdict(set)
        for coord, entries in per_core.items():
            has_trailing = coord in trailing_barrier_coords
            key = (tuple((id(op), self._kernel_variant_key(op, coord)) for op, _, _ in entries), has_trailing)
            groups_by_key[key].add(coord)

        # Build CoreGroups
        result = []
        for key, coords in groups_by_key.items():
            rep_coord = next(iter(coords))
            representative = per_core[rep_coord]
            core_range = _coords_to_core_range_set(coords)
            phases = [op for op, _, _ in representative]
            barrier_scopes = [release for _, release, _ in representative[:-1]]
            barrier_arrive_scopes = [arrive for _, _, arrive in representative[:-1]]

            # Early-exit cores: append trailing barrier scope so they can
            # arrive after their last real phase and exit cleanly.
            has_trailing = rep_coord in trailing_barrier_coords
            if has_trailing:
                trailing_release, trailing_arrive = trailing_barrier_coords[rep_coord]
                barrier_scopes.append(trailing_release)
                barrier_arrive_scopes.append(trailing_arrive)

            phase_op_indices = [op_id_to_index[id(op)] if op is not _NOOP_OP else NOOP_PHASE_INDEX for op in phases]
            result.append(
                CoreGroup(
                    core_range=core_range,
                    phases=phases,
                    barrier_scopes=barrier_scopes,
                    barrier_arrive_scopes=barrier_arrive_scopes,
                    phase_op_indices=phase_op_indices,
                    has_trailing_barrier=has_trailing,
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
        segment_cache: Dict[Tuple[frozenset, frozenset], BarrierConfig],
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
            barrier_arrive_scopes=group.barrier_arrive_scopes,
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

        # Build PhaseInfo list — no-op phases get empty cb_info
        phase_infos = []
        for i, op in enumerate(group.phases):
            if op is _NOOP_OP:
                phase_infos.append(PhaseInfo(phase_idx=i, op_descriptor=op, cb_info={}))
            else:
                phase_infos.append(_create_phase_info(op, i))

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
        segment_cache: Dict[Tuple[frozenset, frozenset], BarrierConfig],
        barrier_arrive_scopes: Optional[List[Any]] = None,
    ) -> Tuple[List[BarrierSegment], Dict[int, Tuple[int, int]]]:
        """Build barrier segments and transition map from barrier scopes.

        Consecutive transitions with the same barrier scope share a segment.
        The ``segment_cache`` ensures groups sharing a scope (e.g. the stem)
        use the same GlobalSemaphore addresses for synchronization.

        The cache key is ``(release_scope_key, arrive_scope_key)`` — different
        arrive counts at the same release scope produce different segments
        (they need different arrive thresholds).

        Returns (segments, transition_map).
        """
        segments: List[BarrierSegment] = []
        transition_map: Dict[int, Tuple[int, int]] = {}

        prev_cache_key = None
        seg_idx = -1
        call_idx = 0

        for t_idx, scope in enumerate(barrier_scopes):
            release_key = _core_ranges_key(scope)
            arrive_scope = barrier_arrive_scopes[t_idx] if barrier_arrive_scopes else None
            arrive_key = _core_ranges_key(arrive_scope) if arrive_scope is not None else frozenset()
            cache_key = (release_key, arrive_key)
            if cache_key != prev_cache_key:
                # New barrier segment
                barrier_cfg = segment_cache[cache_key]
                seg = BarrierSegment(
                    config=barrier_cfg,
                    arrive_addr=barrier_cfg.global_arrive_addr,
                    release_addr=barrier_cfg.global_release_addr,
                )
                segments.append(seg)
                seg_idx += 1
                call_idx = 0
                prev_cache_key = cache_key
            transition_map[t_idx] = (seg_idx, call_idx)
            call_idx += 1

        return segments, transition_map

    def _compute_union_ranges(self) -> Any:
        """Compute the union CoreRangeSet of ALL node core ranges.

        Includes every node in the tree — not just leaves — so that
        early-exit parent cores (in disjoint topologies) also get
        GlobalSemaphore allocations for compute_done/writer_done/reset_done.
        """
        all_coords: Set[Tuple[int, int]] = set()

        def _collect_all(node: OpNode):
            all_coords.update(_core_range_set_to_coords(_get_node_core_range(node)))
            for child in node.children:
                _collect_all(child)

        _collect_all(self._root)
        return _coords_to_core_range_set(all_coords)

    def _validate_topology(self) -> None:
        """Validate the tree topology.

        Checks sibling disjointness: siblings' actual core ranges must be
        pairwise disjoint.  Child ranges may be wider or narrower than their
        parent, and may be fully disjoint from the parent (narrow→wide and
        disjoint topologies are supported via no-op phase entries and
        trailing barriers).

        Raises:
            ValueError: On any topology violation.
        """

        def _validate_node(node: OpNode, depth: int):
            if not node.children:
                return

            # Check sibling disjointness (using actual core ranges)
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

            for child in node.children:
                _validate_node(child, depth + 1)

        _validate_node(self._root, depth=0)

    @staticmethod
    def _effective_leaf_range(node: OpNode) -> Set[Tuple[int, int]]:
        """Compute the union of all descendant leaf core coordinates."""
        if not node.children:
            return _core_range_set_to_coords(_get_node_core_range(node))
        coords: Set[Tuple[int, int]] = set()
        for child in node.children:
            coords |= OpGraphBuilder._effective_leaf_range(child)
        return coords

    @staticmethod
    def _all_descendant_coords(node: OpNode) -> Set[Tuple[int, int]]:
        """Union of ALL descendant core coords (not just leaves).

        Used for barrier scope computation: the barrier after a node must
        cover every core that continues past that node — i.e., every core
        in any descendant's kernel range, not just leaves.  For shrinking
        core ranges (e.g. matmul(16) → slice(8) → rms(4)), the leaf-only
        union would be 4 cores, but 10 cores actually continue (8 + 2
        noop-padded), so the barrier scope must be the full descendant
        union (8 + 4 = 12 unique coords in this example).
        """
        coords: Set[Tuple[int, int]] = set()
        for child in node.children:
            coords |= _core_range_set_to_coords(_get_node_core_range(child))
            coords |= OpGraphBuilder._all_descendant_coords(child)
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
        ``result.launch()``.
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

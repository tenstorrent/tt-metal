# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
High-Level Fusion API: Sequential and Parallel.

Fused execution uses ``fusion_dispatch_op`` so the device program cache can patch
only tensor-address slots on repeat launches.

**Fusion build cache** (``_BUILD_CACHE``): LRU-bounded ``OrderedDict`` with
collision-free tuple key ``(container kind, tree shape, branch
program_cache_key / descriptor hash)``.  Max entries controlled by
``TT_METAL_FUSION_BUILD_CACHE_MAX_ENTRIES`` (default 256).  Cache lookup
never accesses :attr:`OpDescriptor.descriptor`.

The cache stores only the fused ``ProgramDescriptor``, semaphore allocation specs,
kernel labels, and an output-source map — **no IO tensors or L1 buffers**. On a cache hit, a fresh
:class:`FusedOp` is constructed from the cached descriptor and the caller's
current branch ops' tensors. This avoids pinning device buffers in the cache.

The cache key includes **mesh identity** (:meth:`MeshDevice.id` when available,
else Python ``id`` of the device object) from ``build(device=...)`` or inferred
from branch tensors, so entries never cross different open meshes — required
because the device program cache behind ``fusion_dispatch_op`` is tied to a
specific mesh.

**Generation counter** (``_BUILD_CACHE_GEN``): monotonically incremented by
:func:`clear_build_cache`.  Each persistent container stores
``_cached_entry_gen`` alongside ``_cached_entry``; the hot path in
:func:`_container_run` checks the generation before using the cached entry,
falling through to the cache lookup / cold path when stale.

**Steady state:** each ``build()`` call creates new branch descriptors (cheap:
params + hash, no factory), gets a cache hit (reuses the fused
``ProgramDescriptor``), and ``launch()`` dispatches via ``fusion_dispatch_op``
which patches only changed tensor-address slots.

**Launch path:** :meth:`FusedOp.launch` copies merged IO from the branch ops
captured at ``build()`` before dispatch (so in-place branch tensor updates are
visible). Pass ``launch(*ops)`` to refresh from a different op tuple instead.

**Steady state API:** :meth:`Sequential.run` and :meth:`Parallel.run` dispatch
via :func:`_container_run`.  Persistent containers cache a ``_cached_entry``
pointer to the shared ``_CacheEntry``; on call 2+ the C++
``FusionDispatchState`` allocates ephemeral outputs, patches, and dispatches
in a single call — zero L1 pinning between forward passes.  Inline
containers (new each call) always use the lightweight warm path (3-arg
``fusion_dispatch_op``, reused output tensors).
"""

import os

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import (
    OpDescriptor,
    _DeferredOutput,
    _clear_all_program_key_caches,
    is_op_descriptor,
)
from models.experimental.ops.descriptors.fusion.common import (
    _SemaphoreSpec,
    _get_risc_type,
)


# =============================================================================
# Fusion Build Cache — LRU-bounded
# =============================================================================

_BUILD_CACHE: OrderedDict[tuple, "_CacheEntry"] = OrderedDict()
_BUILD_CACHE_MAX = int(os.environ.get("TT_METAL_FUSION_BUILD_CACHE_MAX_ENTRIES", "256"))


def _build_cache_get(key):
    entry = _BUILD_CACHE.get(key)
    if entry is not None:
        _BUILD_CACHE.move_to_end(key)
    return entry


def _build_cache_put(key, entry):
    _BUILD_CACHE[key] = entry
    _BUILD_CACHE.move_to_end(key)
    while len(_BUILD_CACHE) > _BUILD_CACHE_MAX:
        _BUILD_CACHE.popitem(last=False)


@dataclass
class _CacheEntry:
    """Stored in ``_BUILD_CACHE``.  Fully initialized at construction, immutable thereafter.

    Contains everything needed to dispatch a fused program on cache hit
    without re-running codegen/merge.  No tensor references or live L1
    buffers are held — barrier semaphores are stored as allocation specs
    (``sem_specs``) and re-allocated ephemerally at each dispatch.

    The cached ``ProgramDescriptor`` may have stale buffer/semaphore
    addresses (CB buffer pointers and runtime arg values) after the
    original tensors/semaphores are freed.  ``address_slots`` (opaque C++
    ``AddressSlots``) records every descriptor position that references an
    IO tensor or semaphore address so the dispatch path can refresh them
    from live objects before dispatch.
    """

    cached_descriptor: Any  # ProgramDescriptor — dispatched via fusion_dispatch_op on hit
    sem_specs: Tuple[_SemaphoreSpec, ...]  # Allocation blueprints for ephemeral barrier semaphores
    kernel_labels: tuple  # For _apply_kernel_dir file naming
    # (op_idx, tensor_idx) per merged output; None when the fused op has no outputs.
    output_sources: Optional[Tuple[Tuple[int, int], ...]]
    # len(merged input_tensors) after identity dedupe; prealloc on cache hit (no tensor refs).
    merged_input_len: Optional[int] = None
    # Opaque C++ AddressSlots — maps every stale descriptor position (CB buffer
    # pointers, runtime args, common args) to an IO tensor index.  Computed once
    # at build time via compute_address_slots, passed to fusion_dispatch_op.
    address_slots: Any = None
    # result_reorder[j] = index into the outputs list (output_sources order)
    # for default_results[j].  Used on the hot path to reorder the ephemeral
    # outputs from output_sources order to the return order callers expect.
    # Stored as a list (not tuple) for zero-conversion C++ consumption.
    result_reorder: List[int] = field(default_factory=list)
    # Cached output TensorSpecs (output_sources order) and shared_output_map
    # for the allocating fusion_dispatch_op overload.  Computed eagerly at build time.
    output_specs: Optional[List] = None
    shared_output_map: List = field(default_factory=list)
    # C++ FusionDispatchState — created eagerly at build time when output_specs
    # and device are available.  Used by both persistent and inline dispatch paths.
    dispatch_state: Any = None


def _flatten_ops(items) -> List[OpDescriptor]:
    """Recursively flatten Sequential/Parallel items into an ordered list of OpDescriptors."""
    result = []
    for item in items:
        if is_op_descriptor(item):
            result.append(item)
        elif isinstance(item, Sequential):
            result.extend(_flatten_ops(item._items))
        elif isinstance(item, Parallel):
            result.extend(_flatten_ops(item._items))
    return result


def _topology_fingerprint(items) -> str:
    """Encode nested tree shape of *items* (``O``, ``S(...)``, ``P(...)``)."""
    parts = []
    for item in items:
        if is_op_descriptor(item):
            parts.append("O")
        elif isinstance(item, Sequential):
            parts.append(f"S({_topology_fingerprint(item._items)})")
        elif isinstance(item, Parallel):
            parts.append(f"P({_topology_fingerprint(item._items)})")
    return ",".join(parts)


def _build_cache_surface_key(items, container_prefix: str) -> str:
    """Human-readable surface key: ``P(O,O)`` vs ``S(O,O)`` for top-level Parallel vs Sequential."""
    return f"{container_prefix}({_topology_fingerprint(items)})"


def _fusion_mesh_runtime_id(device) -> int:
    """Process-local mesh identity for fusion cache keys.

    Prefer :meth:`MeshDevice.id` (monotonic mesh id from Metal) over ``id(device)``.
    """
    if device is None:
        return 0
    mid = getattr(device, "id", None)
    if callable(mid):
        try:
            return int(mid())
        except (TypeError, ValueError):
            pass
    return id(device)


def _make_rebind_output_sources(ops: List, output_source_map) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Convert merge ``output_source_map`` (op object, tensor_idx) to (op_index, tensor_idx).

    Returns ``None`` when there is no merge map (single-leaf fuse); :meth:`FusedOp.refresh_merged_io`
    then accepts exactly one source op.
    """
    if not output_source_map:
        return None
    op_id_to_idx = {id(op): idx for idx, op in enumerate(ops)}
    return tuple((op_id_to_idx[id(op)], t_idx) for op, t_idx in output_source_map)


def _infer_output_sources_from_merged(fused_op: "FusedOp", ops: List) -> Tuple[Tuple[int, int], ...]:
    """Recover (op_idx, tensor_idx) pairs when ``output_source_map`` was empty at cache write.

    Must match tensors in :attr:`FusedOp.output_tensors` to branch ``output_tensors`` by identity.
    """
    op_id_to_idx = {id(op): i for i, op in enumerate(ops)}
    pairs: List[Tuple[int, int]] = []
    for t in fused_op.output_tensors:
        tid = id(t)
        matched = False
        for op in ops:
            for t_idx, ot in enumerate(op.output_tensors):
                if id(ot) == tid:
                    pairs.append((op_id_to_idx[id(op)], t_idx))
                    matched = True
                    break
            if matched:
                break
        if not matched:
            raise ValueError("fusion cache: merged output tensor not found on branch ops; cannot infer output_sources")
    return tuple(pairs)


def _coerce_mutable_io_opdescriptor(op: OpDescriptor) -> OpDescriptor:
    """Ensure ``input_tensors`` / ``output_tensors`` are lists so in-place rebind works."""
    ins = op.input_tensors if isinstance(op.input_tensors, list) else list(op.input_tensors)
    outs = op.output_tensors if isinstance(op.output_tensors, list) else list(op.output_tensors)
    if ins is op.input_tensors and outs is op.output_tensors:
        return op
    return OpDescriptor(op.descriptor, ins, outs)


def _compute_address_slots(desc, io_tensors, sem_addrs=()):
    """Compute the full address-slot mapping (opaque C++ ``AddressSlots``).

    Called once at build time when buffer pointers and runtime arg addresses
    are still valid.  Uses the same address-matching logic as
    ``discover_address_slots`` in the program factory.

    If ``sem_addrs`` is provided, runtime arg positions matching those
    semaphore addresses are also recorded so they can be patched with
    fresh addresses on each dispatch (ephemeral semaphores).
    """
    return ttnn._ttnn.operations.experimental.compute_address_slots(desc, io_tensors, list(sem_addrs))


def _cache_build_result(
    fused_op: "FusedOp",
    ops: List[OpDescriptor],
    output_source_map,
    address_slots,
    default_results: List,
    device,
    sem_specs: Tuple[_SemaphoreSpec, ...] = (),
) -> _CacheEntry:
    """Record a fully-initialized cache entry from a freshly-built FusedOp.

    All fields — including ``output_specs``, ``shared_output_map``,
    ``result_reorder``, and ``dispatch_state`` — are computed eagerly so the
    entry is immutable after construction.

    ``address_slots`` must be pre-computed (while addresses are valid) and
    already set on *fused_op*.  Stored in the cache so ``fusion_dispatch_op``
    can refresh all stale addresses from live IO tensors before dispatch.

    ``sem_specs`` stores allocation blueprints for barrier semaphores.
    No live ``GlobalSemaphore`` objects are held in the cache — semaphores
    are allocated ephemerally at each dispatch from these specs.
    """
    desc = fused_op.descriptor
    if desc.custom_program_hash is None:
        desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)

    op_id_to_idx = {id(op): idx for idx, op in enumerate(ops)}
    if output_source_map:
        output_sources: Optional[Tuple[Tuple[int, int], ...]] = tuple(
            (op_id_to_idx[id(op)], t_idx) for op, t_idx in output_source_map
        )
    elif fused_op.output_tensors:
        output_sources = _infer_output_sources_from_merged(fused_op, ops)
    else:
        output_sources = None

    output_specs = None
    shared_output_map: List = []
    result_reorder: List[int] = []

    if output_sources:
        src_outputs = [ops[pi].output_tensors[ti] for pi, ti in output_sources]
        if src_outputs and src_outputs[0] is not None:
            output_specs = [t.spec for t in src_outputs]

            seen_srcs: Dict[Tuple[int, int], int] = {}
            dedup: List[int] = []
            for i, src in enumerate(output_sources):
                if src in seen_srcs:
                    dedup.append(seen_srcs[src])
                else:
                    seen_srcs[src] = i
                    dedup.append(i)
            if any(d != i for i, d in enumerate(dedup)):
                shared_output_map = dedup

            src_id_to_idx: Dict[int, int] = {}
            for src_i, (pi, _ti) in enumerate(output_sources):
                src_id_to_idx[id(ops[pi])] = src_i
            result_reorder = [src_id_to_idx[id(d)] for d in default_results if id(d) in src_id_to_idx]
    elif fused_op.output_tensors:
        output_specs = [t.spec for t in fused_op.output_tensors if t is not None]

    dispatch_state = None
    if output_specs is not None and device is not None:
        dispatch_state = ttnn._ttnn.operations.experimental.FusionDispatchState(
            output_specs,
            shared_output_map,
            result_reorder,
            desc,
            address_slots,
            device,
        )

    return _CacheEntry(
        cached_descriptor=desc,
        sem_specs=sem_specs,
        kernel_labels=fused_op.kernel_labels,
        output_sources=output_sources,
        merged_input_len=len(fused_op.input_tensors),
        address_slots=address_slots,
        result_reorder=result_reorder,
        output_specs=output_specs,
        shared_output_map=shared_output_map,
        dispatch_state=dispatch_state,
    )


def _collect_merged_io_from_ops(entry: _CacheEntry, ops: List) -> Tuple[List, List]:
    """Deduped inputs + outputs for a cache entry (no branch ``.descriptor`` access)."""
    seen_ids: Set[int] = set()
    n_in = entry.merged_input_len
    if n_in is not None:
        all_inputs: List = [None] * n_in
        i = 0
        for op in ops:
            for t in op.input_tensors:
                tid = id(t)
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    all_inputs[i] = t
                    i += 1
        if i != n_in:
            raise ValueError(
                "fusion cache: merged_input_len mismatch — branch inputs changed identity "
                f"since cache write (expected {n_in} deduped inputs, got {i})."
            )
    else:
        all_inputs = []
        for op in ops:
            for t in op.input_tensors:
                tid = id(t)
                if tid not in seen_ids:
                    all_inputs.append(t)
                    seen_ids.add(tid)
    if entry.output_sources:
        all_outputs = [ops[pi].output_tensors[ti] for pi, ti in entry.output_sources]
    else:
        all_outputs = []
    return all_inputs, all_outputs


def _fused_op_from_cache_entry(entry: _CacheEntry, ops: List) -> "FusedOp":
    """Build :class:`FusedOp` from cache; same IO merge as a cold build.

    Fallback path when hidden rebind is not available (legacy cache entries).
    """
    all_inputs, all_outputs = _collect_merged_io_from_ops(entry, ops)
    desc = entry.cached_descriptor
    return FusedOp(
        op=OpDescriptor(desc, all_inputs, all_outputs),
        sem_specs=entry.sem_specs,
        kernel_labels=entry.kernel_labels,
        rebind_output_sources=entry.output_sources,
        branch_ops=tuple(ops),
        address_slots=entry.address_slots,
    )


_BUILD_CACHE_GEN: int = 0


def clear_build_cache() -> None:
    """Clear the fusion build cache, increment the generation counter, and clear all program key caches."""
    global _BUILD_CACHE_GEN
    _BUILD_CACHE.clear()
    _BUILD_CACHE_GEN += 1
    _clear_all_program_key_caches()


def _default_results(items) -> List:
    """Collect default result descriptors from a container's items.

    Used for :class:`Sequential` only (see :func:`_default_results_parallel_branches` for
    :class:`Parallel`).

    - ``Sequential``: the last item's outputs (or expanded ``Parallel`` / nested chain).
    - Nested: recurses, so ``Sequential(stem, Parallel(a, b))`` → ``[a, b]``.
    """
    if not items:
        return []
    last = items[-1]
    if is_op_descriptor(last):
        return [last]
    if isinstance(last, Parallel):
        result = []
        for item in last._items:
            if is_op_descriptor(item):
                result.append(item)
            elif isinstance(item, (Sequential, Parallel)):
                result.extend(_default_results(item._items))
        return result
    if isinstance(last, Sequential):
        return _default_results(last._items)
    return []


def _default_results_parallel_branches(items) -> List:
    """Collect one leaf :class:`OpDescriptor` per top-level branch of a :class:`Parallel`.

    ``Parallel(op_a, op_b)`` with two op descriptors → ``[op_a, op_b]`` (not only ``op_b``).

    A branch may be a :class:`Sequential` chain; then we use :func:`_default_results` for
    that branch's step list (typically the last op in the chain).
    """
    out: List = []
    for item in items:
        if is_op_descriptor(item):
            out.append(item)
        elif isinstance(item, Parallel):
            out.extend(_default_results_parallel_branches(item._items))
        elif isinstance(item, Sequential):
            out.extend(_default_results(item._items))
        else:
            raise TypeError(f"Unsupported Parallel branch type: {type(item)!r}")
    return out


def _detect_internal_edges(items, ops=None) -> List[Tuple[int, int, int, int]]:
    """Detect internal edges at construction time via ``_DeferredOutput`` identity.

    Must be called while ``_DeferredOutput`` instances are still in ``output_tensors``
    (before any ``update()`` triggers materialization).

    Returns list of ``(consumer_op_idx, input_idx, producer_op_idx, output_idx)`` tuples.
    """
    if ops is None:
        ops = _flatten_ops(items)

    # Map each _DeferredOutput id to its source (op_index, output_index)
    output_sources: Dict[int, Tuple[int, int]] = {}
    for i, op in enumerate(ops):
        for j, out in enumerate(op.output_tensors):
            if isinstance(out, _DeferredOutput):
                output_sources[id(out)] = (i, j)

    if not output_sources:
        return []

    edges: List[Tuple[int, int, int, int]] = []
    for i, op in enumerate(ops):
        for j, inp in enumerate(op.input_tensors):
            if isinstance(inp, _DeferredOutput):
                source = output_sources.get(id(inp))
                if source is not None:
                    edges.append((i, j, source[0], source[1]))
    return edges


def _materialize_chain(items, edges: List[Tuple[int, int, int, int]], ops=None) -> None:
    """Materialize deferred descriptors in topological order, connecting internal edges.

    Uses the pre-built ``edges`` list (from ``_detect_internal_edges``) to replace
    ``_DeferredOutput`` inputs with real output tensors after each op materializes.
    """
    if not edges:
        return

    if ops is None:
        ops = _flatten_ops(items)

    for i, op in enumerate(ops):
        # Materialize if ready: _complete_fn set and all inputs are real tensors
        if op._complete_fn is not None and op.program_cache_key is None:
            if all(not isinstance(t, _DeferredOutput) and t is not None for t in op.input_tensors):
                op._materialize()
        elif op._complete_fn is None and any(isinstance(t, _DeferredOutput) for t in op.output_tensors):
            _ = op.descriptor

        # Connect this op's real outputs to downstream deferred inputs
        for consumer_i, inp_j, producer_i, out_j in edges:
            if producer_i == i:
                ops[consumer_i].input_tensors[inp_j] = ops[i].output_tensors[out_j]


def _gather_inputs(ops) -> List:
    """Deduplicate input tensors across ops by identity.

    Raises ``RuntimeError`` if any input slot is ``None``, which means
    ``run()`` was called without a preceding ``update()`` after a
    persistent dispatch cleared the activation slots.
    """
    seen: Set[int] = set()
    inputs: List = []
    for op in ops:
        for i, t in enumerate(op.input_tensors):
            if t is None:
                raise RuntimeError(
                    f"Input slot {i} of descriptor '{op.name}' is None. "
                    f"Call update() on this descriptor before run()."
                )
            tid = id(t)
            if tid not in seen:
                inputs.append(t)
                seen.add(tid)
    return inputs


def _build_run_cache_key(container, ops, device):
    """Build fusion cache key from precomputed container values."""
    branch_keys = tuple(op.program_cache_key for op in ops)
    if device is not None:
        dev_id = _fusion_mesh_runtime_id(device)
    elif ops and ops[0].input_tensors:
        dev_id = _fusion_mesh_runtime_id(ops[0].input_tensors[0].device())
    else:
        dev_id = 0
    return (container._topo_fp, branch_keys, dev_id)


def _cleanup_persistent_ops(ops) -> None:
    """Clear updated activation slots after persistent dispatch.

    Sets activation inputs to ``None`` (zero L1 pinning between calls)
    and clears ``_updated_indices`` / ``output_tensors``.  The next
    ``run()`` requires ``update()`` on every descriptor whose activations
    were cleared; ``_gather_inputs`` enforces this with a ``RuntimeError``.
    """
    for op in ops:
        updated = op._updated_indices
        if updated:
            for idx in updated:
                op.input_tensors[idx] = None
            updated.clear()
            if op.output_tensors:
                op.output_tensors = []


def _allocate_ephemeral_semaphores(device, sem_specs):
    """Allocate fresh barrier semaphores from specs, return (sem_refs, addresses).

    The returned ``sem_refs`` list keeps the ``GlobalSemaphore`` objects alive
    through the dispatch call.  After dispatch completes (command queue
    ordering guarantees the program finishes before deallocation), the refs
    go out of scope and L1 is freed.
    """
    if not sem_specs:
        return [], []
    sems = [ttnn.create_global_semaphore(device, spec.core_ranges, spec.initial_value) for spec in sem_specs]
    addrs = [ttnn.get_global_semaphore_address(s) for s in sems]
    return sems, addrs


def _container_run(container: Any, results, device=None, kernel_dir: Optional[str] = None):
    """Shared implementation for :meth:`Sequential.run` / :meth:`Parallel.run`.

    Three dispatch paths:

    **Persistent hot path** (call 2+ on the same container): ``_cached_entry``
    points to the shared ``_CacheEntry`` whose ``dispatch_state`` allocates
    ephemeral outputs, patches, and dispatches in C++.  Skips cache-key
    computation entirely.  Guarded by ``_cached_entry_gen == _BUILD_CACHE_GEN``
    so that ``clear_build_cache()`` invalidates stale entries without
    needing to track live containers.  After dispatch, updated activation
    slots are cleared (zero L1 pinning between calls); the next ``run()``
    requires ``update()`` on every descriptor whose activations were
    consumed.

    **Inline warm path** (cache hit, first call on this container instance):
    computes the cache key from ``container._topo_fp`` and
    ``container._cached_ops``.  On hit, dispatches via ``dispatch_state``
    and sets ``_cached_entry`` (+ ``_cached_entry_gen``) for subsequent
    hot-path calls.

    **Cold path** (cache miss): full ``build()`` + ``launch()``.
    """
    # ── Persistent hot path ──
    entry = getattr(container, "_cached_entry", None)
    if (
        entry is not None
        and entry.dispatch_state is not None
        and getattr(container, "_cached_entry_gen", -1) == _BUILD_CACHE_GEN
    ):
        inputs = _gather_inputs(container._cached_ops)
        _ephemeral_sems, sem_addrs = _allocate_ephemeral_semaphores(inputs[0].device(), entry.sem_specs)
        outputs = entry.dispatch_state.dispatch(inputs, sem_addrs)
        _cleanup_persistent_ops(container._cached_ops)
        return _filter_results(outputs, container._default_results, results)

    # ── Cache lookup (inline warm OR first persistent call) ──
    ops = container._cached_ops
    cache_key = _build_run_cache_key(container, ops, device)

    entry = _build_cache_get(cache_key)
    if entry is not None:
        if entry.dispatch_state is not None:
            inputs = _gather_inputs(ops)
            _ephemeral_sems, sem_addrs = _allocate_ephemeral_semaphores(inputs[0].device(), entry.sem_specs)
            outputs = entry.dispatch_state.dispatch(inputs, sem_addrs)
            container._cached_entry = entry
            container._cached_entry_gen = _BUILD_CACHE_GEN
            return _filter_results(outputs, container._default_results, results)

        # Fallback warm path (no dispatch_state — e.g. no device at build
        # time): 3-arg dispatch reusing output tensors from branch ops.
        inputs = _gather_inputs(ops)
        if entry.output_sources:
            outputs = [ops[pi].output_tensors[ti] for pi, ti in entry.output_sources]
        else:
            outputs = list(ops[-1].output_tensors) if ops else []
        io_tensors = inputs + outputs
        ttnn._ttnn.operations.experimental.fusion_dispatch_op(io_tensors, entry.cached_descriptor, entry.address_slots)
    else:
        # Cold path: full build (populates _BUILD_CACHE), then launch.
        fused = container.build(device=device, kernel_dir=kernel_dir)
        fused.launch()
        entry = _build_cache_get(cache_key)

    if results is None:
        results = container._default_results

    container._cached_entry = entry
    container._cached_entry_gen = _BUILD_CACHE_GEN
    return [(desc.output_tensors[0] if desc.output_tensors else None) for desc in results]


def _filter_results(outputs, default_results, results):
    """Return *outputs* filtered to the subset the caller requested."""
    if results is not None:
        requested_ids = {id(d) for d in results}
        return [out for d, out in zip(default_results, outputs) if id(d) in requested_ids]
    return outputs


# =============================================================================
# FusedOp
# =============================================================================


class FusedOp:
    """Result of ``Sequential``/``Parallel``.``build()``.

    Holds a merged ``OpDescriptor`` (fused ``ProgramDescriptor`` + IO lists).
    ``semaphores`` keeps build-time ``GlobalSemaphore`` refs alive for the
    initial ``launch()``; ``sem_specs`` stores allocation blueprints so
    subsequent launches can allocate fresh ephemeral semaphores (no persistent
    L1 pinning).

    **Launch:** :meth:`launch` refreshes merged IO from the branch ops stored at
    ``build()`` time, then dispatches (in-place branch tensor updates are picked
    up). Pass ``launch(*branches)`` to refresh from other op objects.

    **Manual refresh:** :meth:`refresh_merged_io` / ``refresh_merged_io_from_*``
    to update merged lists without launching.
    """

    __slots__ = (
        "op",
        "semaphores",
        "sem_specs",
        "kernel_labels",
        "_rebind_output_sources",
        "_branch_ops",
        "_address_slots",
    )

    def __init__(
        self,
        op: OpDescriptor,
        semaphores: Tuple[Any, ...] = (),
        sem_specs: Tuple[_SemaphoreSpec, ...] = (),
        kernel_labels: Tuple[str, ...] = (),
        *,
        rebind_output_sources: Optional[Tuple[Tuple[int, int], ...]] = None,
        branch_ops: Optional[Tuple[Any, ...]] = None,
        address_slots: Any = None,
    ):
        self.op = op
        self.semaphores = semaphores
        self.sem_specs = sem_specs
        self.kernel_labels = kernel_labels
        # Empty tuple is not None but would skip outputs in refresh_merged_io (same bug as missing map).
        if rebind_output_sources is not None and len(rebind_output_sources) == 0:
            rebind_output_sources = None
        self._rebind_output_sources = rebind_output_sources
        self._branch_ops = branch_ops
        self._address_slots = address_slots

    @property
    def descriptor(self):
        return self.op.descriptor

    @property
    def input_tensors(self):
        return self.op.input_tensors

    @property
    def output_tensors(self):
        return self.op.output_tensors

    def launch(self, *branch_ops_override: Any):
        """Dispatch via ``fusion_dispatch_op`` with separate inputs/outputs.

        With no positional arguments, merged IO is refreshed from the branch
        ops captured at ``build()`` before dispatch. With positional arguments,
        those ops are passed to :meth:`refresh_merged_io` instead.
        """
        if branch_ops_override:
            self.refresh_merged_io(list(branch_ops_override))
        elif self._branch_ops is not None:
            self.refresh_merged_io(list(self._branch_ops))
        io_tensors = list(self.input_tensors) + list(self.output_tensors)
        ttnn._ttnn.operations.experimental.fusion_dispatch_op(io_tensors, self.descriptor, self._address_slots)
        return self.output_tensors

    def refresh_merged_io(self, ops: List) -> None:
        """Copy merged IO from *ops* (flatten order) into this fused op's lists in place.

        Branch ``OpDescriptor`` instances must be the
        same objects as at ``build()`` time. Update their ``input_tensors`` /
        ``output_tensors`` before calling.

        Raises:
            ValueError: arity mismatch or invalid single-op case.
        """
        all_inputs: List = []
        seen_ids: Set[int] = set()
        for op in ops:
            for t in op.input_tensors:
                tid = id(t)
                if tid not in seen_ids:
                    all_inputs.append(t)
                    seen_ids.add(tid)

        out_list = self.op.output_tensors
        if self._rebind_output_sources:
            all_outputs = []
            for idx, (pi, ti) in enumerate(self._rebind_output_sources):
                new_out = ops[pi].output_tensors[ti]
                # None = deferred output alloc; keep existing (cached) output.
                all_outputs.append(new_out if new_out is not None else out_list[idx])
        else:
            if len(ops) != 1:
                raise ValueError(
                    "refresh_merged_io: fused op has no multi-branch output map; "
                    "pass exactly one source op (same object(s) as at build time)."
                )
            all_outputs = [new if new is not None else old for new, old in zip(ops[0].output_tensors, out_list)]

        in_list = self.op.input_tensors
        out_list = self.op.output_tensors
        if len(in_list) != len(all_inputs) or len(out_list) != len(all_outputs):
            raise ValueError(
                "refresh_merged_io: arity mismatch — "
                f"fused op expects {len(in_list)} inputs and {len(out_list)} outputs, "
                f"got {len(all_inputs)} and {len(all_outputs)} from source ops."
            )
        in_list[:] = all_inputs
        out_list[:] = all_outputs

    def refresh_merged_io_from_parallel(self, parallel: "Parallel") -> None:
        if not isinstance(parallel, Parallel):
            raise TypeError(f"expected Parallel, got {type(parallel).__name__}")
        self.refresh_merged_io(_flatten_ops(parallel._items))

    def _apply_kernel_dir(self, kernel_dir: str) -> None:
        """Switch kernel sources to file-based, writing files only if they don't exist.

        For single-kernel roles, files are named ``reader.cpp``, ``writer.cpp``,
        ``compute.cpp``.  For multi-kernel roles (parallel branches), filenames
        include op names and core ranges, e.g.
        ``reader_rms_norm_matmul_cores_0x0-3x3.cpp``.
        """
        os.makedirs(kernel_dir, exist_ok=True)

        # Group kernels by RISC type: name -> [(global_idx, kernel)]
        by_type: dict[str, list] = {}
        for idx, kernel in enumerate(self.op.descriptor.kernels):
            risc = _get_risc_type(kernel)
            name = {"riscv_0": "writer", "riscv_1": "reader", "compute": "compute"}.get(risc, "unknown")
            by_type.setdefault(name, []).append((idx, kernel))

        for name, entries in by_type.items():
            for idx, kernel in entries:
                if len(entries) == 1:
                    filename = f"{name}.cpp"
                else:
                    label = self.kernel_labels[idx] if idx < len(self.kernel_labels) else ""
                    core_tag = _core_range_tag(kernel.core_ranges)
                    tag = f"{label}_{core_tag}" if label else core_tag
                    filename = f"{name}_{tag}.cpp"
                filepath = os.path.join(kernel_dir, filename)
                abspath = os.path.abspath(filepath)

                if not os.path.exists(abspath):
                    with open(abspath, "w") as f:
                        f.write(kernel.kernel_source)

                kernel.kernel_source = abspath
                kernel.source_type = ttnn.KernelDescriptor.SourceType.FILE

    def __repr__(self):
        n_kernels = len(self.op.descriptor.kernels) if hasattr(self.op.descriptor, "kernels") else "?"
        return (
            f"FusedOp(kernels={n_kernels}, "
            f"inputs={len(self.op.input_tensors)}, "
            f"outputs={len(self.op.output_tensors)})"
        )


def _register_named_descriptors(container, named_items: dict) -> None:
    """Flatten named descriptors from *named_items* onto *container* as attributes.

    For nested containers (e.g., ``Parallel(q=q, norms=Parallel(a=a, b=b))``),
    child names are hoisted to the top level so ``container.a`` works regardless
    of nesting depth.  Duplicate names raise ``ValueError``.
    """
    seen: Set[str] = set()
    for name, item in named_items.items():
        if name in seen:
            raise ValueError(f"Duplicate descriptor name {name!r}")
        seen.add(name)
        setattr(container, name, item)
        # Hoist child container's named descriptors
        if isinstance(item, (Sequential, Parallel)):
            for child_name in getattr(item, "_descriptor_names", ()):
                if child_name in seen:
                    raise ValueError(f"Duplicate descriptor name {child_name!r}")
                seen.add(child_name)
                setattr(container, child_name, getattr(item, child_name))
    container._descriptor_names = tuple(seen)


class Sequential:
    """A sequence of ops to fuse into a single dispatch.

    Items can be ``OpDescriptor``, ``Sequential``, or ``Parallel`` objects.
    Nested ``Sequential`` items are automatically flattened.

    **Inline mode** (simple, creates descriptors each call)::

        out = Sequential(
            rms_norm=descriptors.rms_norm(tt_x, weight=w, ...),
            mm=descriptors.matmul(tt_x, weight=W, ...),
        ).run()

    **Persistent mode** (fast, reuses descriptors across calls)::

        # Setup (once, activation omitted):
        self.fused = Sequential(
            norm=descriptors.rms_norm(weight=w, ...),
            mm=descriptors.matmul(weight=W, ...),
        )

        # Each forward:
        self.fused.norm.update(new_x)
        [out] = self.fused.run()
    """

    def __init__(self, *items, **named_items):
        all_items = list(items)
        for item in named_items.values():
            all_items.append(item)
        if not all_items:
            raise ValueError("Sequential() requires at least 1 item")
        self._items = all_items
        self._descriptor_names: tuple = ()
        self._cached_ops = _flatten_ops(all_items)
        self._topo_fp = _build_cache_surface_key(all_items, "S")
        self._internal_edges = _detect_internal_edges(all_items, self._cached_ops)
        self._default_results = _default_results(all_items)
        self._cached_entry = None
        _register_named_descriptors(self, named_items)

    def invalidate_run(self) -> None:
        """Reset cached topology info (call after mutating ``_items`` without :meth:`add`)."""
        for op in self._cached_ops:
            op._updated_indices = []
        self._cached_ops = _flatten_ops(self._items)
        self._topo_fp = _build_cache_surface_key(self._items, "S")
        self._internal_edges = _detect_internal_edges(self._items, self._cached_ops)
        self._default_results = _default_results(self._items)
        self._cached_entry = None

    def add(self, item):
        """Append an item.  Returns self for chaining."""
        self._items.append(item)
        self.invalidate_run()
        return self

    def build(self, device=None, kernel_dir: str = None) -> FusedOp:
        """Merge into one ``FusedOp``. Cache hit skips codegen and fills merged IO from branches.

        Args:
            device: Inferred from tensors when omitted.
            kernel_dir: If set, kernel sources are written as files (existing files kept).
        """
        cache_device = device
        if cache_device is None:
            try:
                cache_device = _extract_device(self._items)
            except ValueError:
                cache_device = None
        _materialize_chain(self._items, self._internal_edges, self._cached_ops)
        ops = self._cached_ops
        cache_key = _build_run_cache_key(self, ops, cache_device)
        entry = _build_cache_get(cache_key)
        if entry is not None:
            result = _fused_op_from_cache_entry(entry, ops)
            if kernel_dir is not None:
                result._apply_kernel_dir(kernel_dir)
            return result

        r = self._build_internal(device)
        rebind_src = _make_rebind_output_sources(ops, r.output_source_map)
        fused = FusedOp(
            op=_coerce_mutable_io_opdescriptor(
                OpDescriptor(r.descriptor, list(r.input_tensors), list(r.output_tensors))
            ),
            semaphores=r.semaphores,
            sem_specs=r.sem_specs,
            kernel_labels=r.kernel_labels,
            rebind_output_sources=rebind_src,
            branch_ops=tuple(ops),
        )
        fused.refresh_merged_io(list(ops))
        io_tensors = list(fused.input_tensors) + list(fused.output_tensors)
        fused._address_slots = _compute_address_slots(fused.descriptor, io_tensors, r.sem_addrs)

        _build_cache_put(
            cache_key,
            _cache_build_result(
                fused,
                ops,
                r.output_source_map,
                fused._address_slots,
                self._default_results,
                cache_device,
                sem_specs=r.sem_specs,
            ),
        )

        if kernel_dir is not None:
            fused._apply_kernel_dir(kernel_dir)
        return fused

    def run(self, *, results=None, device=None, kernel_dir: str = None):
        """Dispatch the fused program.  No ``FusedOp`` is retained on the
        container between calls — tensor lifetime is not extended.

        Three dispatch paths, selected automatically:

        1. **Cold path** (first call ever for this topology): runs the full
           codegen pipeline to build a fused ``ProgramDescriptor``, then
           dispatches via ``FusedOp.launch()``.  Populates ``_BUILD_CACHE``
           so subsequent calls with the same topology skip codegen.

        2. **Warm path** (``_BUILD_CACHE`` hit, first call on this container
           instance): gathers inputs from branch descriptors and dispatches
           via the 3-arg ``fusion_dispatch_op`` using the branch ops'
           pre-existing output tensors.  Also creates the C++
           ``FusionDispatchState`` for the hot path and sets
           ``_cached_entry`` on the container.

        3. **Persistent hot path** (call 2+ on the same container): bypasses
           cache-key computation entirely.  ``_cached_entry.dispatch_state``
           allocates ephemeral output tensors, patches the cached descriptor,
           and dispatches — all in a single C++ call.  Updated activation
           slots are cleared after dispatch (zero L1 pinning between calls).

        Inline containers (new each call) always take the warm path since
        ``_cached_entry`` is reset.  Persistent containers (reused across
        calls) take the warm path on the first call, then the hot path on
        all subsequent calls.

        Args:
            results: List of descriptors whose ``output_tensors[0]`` are
                returned. Defaults to the last op's output (for a plain
                chain) or each branch's leaf output (if the chain ends
                in a ``Parallel``).
            device: Inferred from tensors when omitted.
            kernel_dir: If set, kernel sources are written as files.

        Returns:
            List of output tensors, one per descriptor in *results*.
        """
        if self._cached_entry is None:
            _materialize_chain(self._items, self._internal_edges, self._cached_ops)
        return _container_run(self, results, device=device, kernel_dir=kernel_dir)

    def _build_internal(self, device=None):
        """Internal build returning intermediate _BuildResult."""
        from models.experimental.ops.descriptors.fusion.graph import OpGraphBuilder

        if device is None:
            device = _extract_device(self._items)
        nodes = _resolve(self)
        if len(nodes) != 1:
            raise ValueError("Sequential must resolve to a single root node")
        return OpGraphBuilder(nodes[0])._build_internal(device)


class Parallel:
    """Items that run in parallel on disjoint core subsets.

    Each item runs independently on its own cores.  Items can be
    ``OpDescriptor``, ``Sequential``, or ``Parallel`` objects.

    **Inline mode** (simple, creates descriptors each call)::

        tt_q, tt_kv = Parallel(
            q=descriptors.rms_norm(tt_q, weight=qw, ...),
            kv=descriptors.rms_norm(tt_kv, weight=kw, ...),
        ).run()

    **Persistent mode** (fast, reuses descriptors across calls)::

        # Setup (once, activation omitted):
        self.fused = Parallel(
            q=descriptors.rms_norm(weight=qw, ...),
            kv=descriptors.rms_norm(weight=kw, ...),
        )

        # Each forward:
        self.fused.q.update(tt_q)
        self.fused.kv.update(tt_kv)
        tt_q, tt_kv = self.fused.run()

    No ``FusedOp`` is retained on the container between calls — the fusion
    ``_BUILD_CACHE`` provides the fast path without extending tensor lifetime.
    """

    def __init__(self, *items, **named_items):
        all_items = list(items)
        for item in named_items.values():
            all_items.append(item)
        if len(all_items) < 2:
            raise ValueError("Parallel() requires at least 2 items")
        self._items = all_items
        self._descriptor_names: tuple = ()
        self._cached_ops = _flatten_ops(all_items)
        self._topo_fp = _build_cache_surface_key(all_items, "P")
        self._internal_edges = _detect_internal_edges(all_items, self._cached_ops)
        self._default_results = _default_results_parallel_branches(all_items)
        self._cached_entry = None
        _register_named_descriptors(self, named_items)

    def invalidate_run(self) -> None:
        """Reset cached topology info (call after mutating ``_items`` without :meth:`add`)."""
        for op in self._cached_ops:
            op._updated_indices = []
        self._cached_ops = _flatten_ops(self._items)
        self._topo_fp = _build_cache_surface_key(self._items, "P")
        self._internal_edges = _detect_internal_edges(self._items, self._cached_ops)
        self._default_results = _default_results_parallel_branches(self._items)
        self._cached_entry = None

    def add(self, item):
        """Add a branch.  Returns self for chaining."""
        self._items.append(item)
        self.invalidate_run()
        return self

    def build(self, device=None, kernel_dir: str = None) -> FusedOp:
        """Merge branches into one ``FusedOp``. Cache hit fills merged IO from branch lists.

        Args:
            device: Inferred from tensors when omitted.
            kernel_dir: If set, kernel sources are written as files (existing files kept).
        """
        cache_device = device
        if cache_device is None:
            try:
                cache_device = _extract_device(self._items)
            except ValueError:
                cache_device = None
        _materialize_chain(self._items, self._internal_edges, self._cached_ops)
        ops = self._cached_ops
        cache_key = _build_run_cache_key(self, ops, cache_device)
        entry = _build_cache_get(cache_key)
        if entry is not None:
            result = _fused_op_from_cache_entry(entry, ops)
            if kernel_dir is not None:
                result._apply_kernel_dir(kernel_dir)
            return result

        r = self._build_internal(device)
        rebind_src = _make_rebind_output_sources(ops, r.output_source_map)
        fused = FusedOp(
            op=_coerce_mutable_io_opdescriptor(
                OpDescriptor(r.descriptor, list(r.input_tensors), list(r.output_tensors))
            ),
            semaphores=r.semaphores,
            sem_specs=r.sem_specs,
            kernel_labels=r.kernel_labels,
            rebind_output_sources=rebind_src,
            branch_ops=tuple(ops),
        )
        # Compute address_slots AFTER refresh_merged_io so the IO tensor
        # ordering matches what launch() will see (single source of truth).
        fused.refresh_merged_io(list(ops))
        io_tensors = list(fused.input_tensors) + list(fused.output_tensors)
        fused._address_slots = _compute_address_slots(fused.descriptor, io_tensors, r.sem_addrs)

        _build_cache_put(
            cache_key,
            _cache_build_result(
                fused,
                ops,
                r.output_source_map,
                fused._address_slots,
                self._default_results,
                cache_device,
                sem_specs=r.sem_specs,
            ),
        )

        if kernel_dir is not None:
            fused._apply_kernel_dir(kernel_dir)
        return fused

    def run(self, *, results=None, device=None, kernel_dir: str = None):
        """Dispatch the fused program.  No ``FusedOp`` is retained on the
        container between calls — tensor lifetime is not extended.

        See :meth:`Sequential.run` for a full description of the three
        dispatch paths (cold, warm, persistent hot).  The same logic applies
        here via the shared ``_container_run`` implementation.

        Args:
            results: List of descriptors whose ``output_tensors[0]`` are
                returned. Defaults to each branch's leaf output in
                branch order.
            device: Inferred from tensors when omitted.
            kernel_dir: If set, kernel sources are written as files.

        Returns:
            List of output tensors, one per descriptor in *results*.
        """
        if self._cached_entry is None:
            _materialize_chain(self._items, self._internal_edges, self._cached_ops)
        return _container_run(self, results, device=device, kernel_dir=kernel_dir)

    def _build_internal(self, device=None):
        """Internal build returning intermediate _BuildResult."""
        from models.experimental.ops.descriptors.fusion.graph import _merge_build_results

        if device is None:
            device = _extract_device(self._items)
        built = [_build_item(item, device) for item in self._items]
        return _merge_build_results(built)


# =============================================================================
# Internal Helpers
# =============================================================================


def _resolve(item) -> List:
    """Convert a user-facing item into a list of OpNode trees.

    Handles all three types uniformly:
    - ``OpDescriptor`` -> ``[OpNode(op)]``
    - ``Parallel`` -> flat list of children (one OpNode per branch)
    - ``Sequential`` -> single-element list containing a chain of OpNodes
    """
    from models.experimental.ops.descriptors.fusion.graph import OpNode

    if is_op_descriptor(item):
        return [OpNode(item)]

    if isinstance(item, Parallel):
        return [node for child in item._items for node in _resolve(child)]

    if isinstance(item, Sequential):
        # Flatten nested Sequential items
        flat: List = []
        for sub in item._items:
            if isinstance(sub, Sequential):
                flat.extend(sub._items)
            else:
                flat.append(sub)

        if not flat:
            raise ValueError("Sequential() has no items after flattening")

        # Process right-to-left: resolve the last item to get tail nodes,
        # then walk remaining items in reverse, each becoming parent of tail.
        tail = _resolve(flat[-1])

        for sub_item in reversed(flat[:-1]):
            parents = _resolve(sub_item)
            if len(parents) != 1:
                raise ValueError(
                    "Items before a Parallel in a Sequential are not allowed — "
                    "the tree would diverge and can't rejoin.  Place trailing "
                    "items inside each branch instead."
                )
            _set_leaf_children(parents[0], tail)
            tail = parents

        return tail

    if isinstance(item, FusedOp):
        raise TypeError(
            "FusedOp cannot be nested in Sequential/Parallel — "
            "it is the result of build() and has already been fused."
        )

    raise TypeError(f"Unsupported item type: {type(item).__name__}")


def _extract_device(items):
    """Walk item tree, return device from first tensor found."""
    for item in items:
        if is_op_descriptor(item):
            for t in item.input_tensors:
                return t.device()
            for t in item.output_tensors:
                return t.device()
        elif isinstance(item, (Sequential, Parallel)):
            dev = _extract_device(item._items)
            if dev is not None:
                return dev
    raise ValueError(
        "Cannot auto-extract device: no items contain device-backed tensors. "
        "Pass device explicitly to build(device=...)."
    )


def _set_leaf_children(node, children: List):
    """Attach children to the deepest single-child leaf of a node subtree."""
    current = node
    while current.children:
        if len(current.children) != 1:
            raise ValueError("Cannot attach children to a node that already branches")
        current = current.children[0]
    current.children = list(children)


def _build_item(item, device):
    """Build a single item into a _BuildResult."""
    from models.experimental.ops.descriptors.fusion.common import _BuildResult

    if is_op_descriptor(item):
        kpm = tuple([(item, k_idx)] for k_idx in range(len(item.descriptor.kernels)))
        # Source maps for single op (identity mapping).
        # merged_idx == source cb_idx since descriptor is passed through as-is.
        cb_source_map = []
        global_cb_source_map = []
        for cb_idx, cb in enumerate(item.descriptor.cbs):
            if cb.has_buffer():
                cb_source_map.append((cb_idx, item, cb_idx))
            if cb.has_global_circular_buffer():
                global_cb_source_map.append((cb_idx, item, cb_idx))
        output_source_map = [(item, t_idx) for t_idx in range(len(item.output_tensors))]
        return _BuildResult(
            descriptor=item.descriptor,
            input_tensors=item.input_tensors,
            output_tensors=item.output_tensors,
            kernel_phase_map=kpm,
            cb_source_map=cb_source_map,
            global_cb_source_map=global_cb_source_map,
            output_source_map=output_source_map,
        )
    if isinstance(item, (Sequential, Parallel)):
        return item._build_internal(device)
    raise TypeError(f"Unsupported item type: {type(item).__name__}")


def _core_range_tag(core_ranges) -> str:
    """Create a filename-safe tag from a CoreRangeSet, e.g. ``'cores_0x0-3x3'``."""
    parts = []
    for cr in core_ranges.ranges():
        parts.append(f"{cr.start.x}x{cr.start.y}-{cr.end.x}x{cr.end.y}")
    return "cores_" + "_".join(parts)


__all__ = [
    "Sequential",
    "Parallel",
    "FusedOp",
    "clear_build_cache",
]

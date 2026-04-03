# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
High-Level Fusion API: Sequential and Parallel.

Fused execution uses ``patchable_generic_op`` so the device program cache can patch
only tensor-address slots on repeat launches.

**Fusion build cache** (``_BUILD_CACHE``): collision-free tuple key
``(container kind, tree shape, branch program_cache_key / descriptor hash)``.
Cache lookup never accesses :attr:`DeferredOpDescriptor.descriptor`.

The cache stores only the fused ``ProgramDescriptor``, semaphores, kernel labels,
and an output-source map — **no IO tensors**. On a cache hit, a fresh
:class:`FusedOp` is constructed from the cached descriptor and the caller's
current branch ops' tensors. This avoids pinning device buffers in the cache.

The cache key includes **mesh identity** (:meth:`MeshDevice.id` when available,
else Python ``id`` of the device object) from ``build(device=...)`` or inferred
from branch tensors, so entries never cross different open meshes — required
because the device program cache behind ``patchable_generic_op`` is tied to a
specific mesh.

**Steady state:** each ``build()`` call creates new branch descriptors (cheap:
params + hash, no factory), gets a cache hit (reuses the fused
``ProgramDescriptor``), and ``launch()`` dispatches via ``patchable_generic_op``
which patches only changed tensor-address slots.

**Launch path:** :meth:`FusedOp.launch` copies merged IO from the branch ops
captured at ``build()`` before dispatch (so in-place branch tensor updates are
visible). Pass ``launch(*ops)`` to refresh from a different op tuple instead.

**Steady state API:** :meth:`Sequential.run` and :meth:`Parallel.run` call
``build()`` once (until the graph or fusion id changes), then reuse the same
``FusedOp`` for each ``run()`` while still refreshing merged IO every time — the
intended pattern for build-once / launch-many without stale handles.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor, is_op_descriptor
from models.experimental.ops.descriptors.fusion.common import (
    _get_risc_type,
)


# =============================================================================
# Fusion Build Cache
# =============================================================================

# Fused ``ProgramDescriptor`` + metadata, keyed by fusion cache key (collision-free tuple).
# No IO tensors are stored — entries are lightweight and never go stale from
# device buffer deallocation.
_BUILD_CACHE: Dict[tuple, "_CacheEntry"] = {}


@dataclass
class _CacheEntry:
    """Stored in ``_BUILD_CACHE``.  Immutable after construction.

    Contains everything needed to reconstruct a :class:`FusedOp` on cache hit
    without re-running codegen/merge. No tensor references are held.
    """

    cached_descriptor: Any  # ProgramDescriptor — dispatched via patchable_generic_op on hit
    semaphores: tuple  # Keeps GlobalSemaphore L1 alive
    kernel_labels: tuple  # For _apply_kernel_dir file naming
    # (op_idx, tensor_idx) per merged output; None when the fused op has no outputs.
    output_sources: Optional[Tuple[Tuple[int, int], ...]]
    # len(merged input_tensors) after identity dedupe; prealloc on cache hit (no tensor refs).
    merged_input_len: Optional[int] = None


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


def _branch_program_cache_key(op) -> int:
    """Stable branch identity for cache lookup. Never touches ``OpDescriptor.descriptor``."""
    return op.program_cache_key


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


def _build_cache_device_id(items, build_device) -> int:
    """Metal mesh runtime id, or 0 if unknown (no device / no device tensors yet)."""
    if build_device is not None:
        return _fusion_mesh_runtime_id(build_device)
    try:
        return _fusion_mesh_runtime_id(_extract_device(items))
    except ValueError:
        return 0


def _fusion_cache_key_from_ops(items, container_prefix: str, ops: List, device_id: int) -> tuple:
    """Process-local fusion build-cache key from already-flattened ``ops``."""
    surface = _build_cache_surface_key(items, container_prefix)
    return (surface, tuple(_branch_program_cache_key(op) for op in ops), device_id)


def _fusion_cache_key_and_ops(items, container_prefix: str, build_device=None) -> Tuple[tuple, List]:
    """Return ``(fusion_cache_key, flattened_ops)`` for ``Sequential``/``Parallel`` ``._items``."""
    ops = _flatten_ops(items)
    dev_id = _build_cache_device_id(items, build_device)
    cache_key = _fusion_cache_key_from_ops(items, container_prefix, ops, dev_id)
    return cache_key, ops


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


def _cache_build_result(fused_op: "FusedOp", ops: List[OpDescriptor], output_source_map) -> _CacheEntry:
    """Record a slim cache entry from a freshly-built FusedOp (no tensor refs)."""
    # Memoize the descriptor hash so patchable_generic_op skips the full
    # kernel/CB/semaphore walk on every launch (O(1) instead of O(descriptor)).
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

    return _CacheEntry(
        cached_descriptor=fused_op.descriptor,
        semaphores=fused_op.semaphores,
        kernel_labels=fused_op.kernel_labels,
        output_sources=output_sources,
        merged_input_len=len(fused_op.input_tensors),
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
        semaphores=entry.semaphores,
        kernel_labels=entry.kernel_labels,
        rebind_output_sources=entry.output_sources,
        branch_ops=tuple(ops),
    )


def clear_build_cache() -> None:
    """Clear the fusion build cache."""
    _BUILD_CACHE.clear()


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


def _container_run(container: Any, surface_prefix: str, results, device=None, kernel_dir: Optional[str] = None):
    """Shared implementation for :meth:`Sequential.run` / :meth:`Parallel.run`."""
    cache_device = device
    if cache_device is None:
        try:
            cache_device = _extract_device(container._items)
        except ValueError:
            cache_device = None
    cache_key, ops = _fusion_cache_key_and_ops(container._items, surface_prefix, cache_device)
    sig = (cache_key, tuple(id(op) for op in ops), kernel_dir)
    if container._run_fused is None or container._run_signature != sig:
        container._run_fused = container.build(device=device, kernel_dir=kernel_dir)
        container._run_signature = sig
    container._run_fused.launch()
    if results is None:
        if surface_prefix == "P":
            results = _default_results_parallel_branches(container._items)
        else:
            results = _default_results(container._items)
    # Branches may have no DRAM outputs (e.g. GlobalCB-only push); align with results list.
    return [(desc.output_tensors[0] if desc.output_tensors else None) for desc in results]


# =============================================================================
# FusedOp
# =============================================================================


class FusedOp:
    """Result of ``Sequential``/``Parallel``.``build()``.

    Holds a merged ``OpDescriptor`` (fused ``ProgramDescriptor`` + IO lists) and
    keeps references to global semaphores used by the fused program.

    **Launch:** :meth:`launch` refreshes merged IO from the branch ops stored at
    ``build()`` time, then dispatches (in-place branch tensor updates are picked
    up). Pass ``launch(*branches)`` to refresh from other op objects.

    **Manual refresh:** :meth:`refresh_merged_io` / ``refresh_merged_io_from_*``
    to update merged lists without launching.
    """

    __slots__ = (
        "op",
        "semaphores",
        "kernel_labels",
        "_rebind_output_sources",
        "_branch_ops",
    )

    def __init__(
        self,
        op: OpDescriptor,
        semaphores: Tuple[Any, ...] = (),
        kernel_labels: Tuple[str, ...] = (),
        *,
        rebind_output_sources: Optional[Tuple[Tuple[int, int], ...]] = None,
        branch_ops: Optional[Tuple[Any, ...]] = None,
    ):
        self.op = op
        self.semaphores = semaphores
        self.kernel_labels = kernel_labels
        # Empty tuple is not None but would skip outputs in refresh_merged_io (same bug as missing map).
        if rebind_output_sources is not None and len(rebind_output_sources) == 0:
            rebind_output_sources = None
        self._rebind_output_sources = rebind_output_sources
        self._branch_ops = branch_ops

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
        """Enqueue via ``patchable_generic_op`` using merged IO; return outputs.

        With no positional arguments, merged IO is copied from the branch ops
        captured at ``build()`` (so in-place updates to branch tensor lists are
        visible before dispatch). With one or more positional arguments, those ops
        are passed to :meth:`refresh_merged_io` instead.

        On program cache hits, the device program factory patches only runtime-arg
        and CB slots that hold ``io_tensors`` buffer addresses when those addresses
        change (see ``PatchableGenericMeshProgramFactory``).
        """
        if branch_ops_override:
            self.refresh_merged_io(list(branch_ops_override))
        elif self._branch_ops is not None:
            self.refresh_merged_io(list(self._branch_ops))
        io_tensors = list(self.input_tensors) + list(self.output_tensors)
        ttnn._ttnn.operations.experimental.patchable_generic_op(io_tensors, self.descriptor)
        return self.output_tensors

    def refresh_merged_io(self, ops: List) -> None:
        """Copy merged IO from *ops* (flatten order) into this fused op's lists in place.

        Branch ``OpDescriptor`` / ``DeferredOpDescriptor`` instances must be the
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


class Sequential:
    """A sequence of ops to fuse into a single dispatch.

    Items can be ``OpDescriptor``, ``Sequential``, or ``Parallel`` objects.
    Nested ``Sequential`` items are automatically flattened.

    Usage::

        # Inline
        fused = Sequential(op0, op1, op2).build()

        # Incremental
        s = Sequential(op0)
        s.add(op1).add(op2)
        fused = s.build()

        # Composition
        stem = Sequential(op0, op1)
        full = Sequential(stem, op2).build()  # flattened
    """

    def __init__(self, *items):
        if not items:
            raise ValueError("Sequential() requires at least 1 item")
        self._items = list(items)
        self._run_fused: Optional[FusedOp] = None
        self._run_signature: Optional[Tuple] = None

    def invalidate_run(self) -> None:
        """Clear :meth:`run` cache (call after mutating ``_items`` without :meth:`add`)."""
        self._run_fused = None
        self._run_signature = None

    def add(self, item):
        """Append an item.  Returns self for chaining."""
        self.invalidate_run()
        self._items.append(item)
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
        cache_key, ops = _fusion_cache_key_and_ops(self._items, "S", cache_device)
        entry = _BUILD_CACHE.get(cache_key)
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
            kernel_labels=r.kernel_labels,
            rebind_output_sources=rebind_src,
            branch_ops=tuple(ops),
        )

        _BUILD_CACHE[cache_key] = _cache_build_result(fused, ops, r.output_source_map)

        if kernel_dir is not None:
            fused._apply_kernel_dir(kernel_dir)
        return fused

    def run(self, *, results=None, device=None, kernel_dir: str = None):
        """``build()`` once per stable graph, then ``launch()`` each call.

        Args:
            results: List of descriptors whose ``output_tensors[0]`` are
                returned. Defaults to the last op's output (for a plain
                chain) or each branch's leaf output (if the chain ends
                in a ``Parallel``).

        Returns:
            List of output tensors, one per descriptor in *results*.
        """
        return _container_run(self, "S", results, device=device, kernel_dir=kernel_dir)

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

    Branch order is **the order you pass to** ``Parallel(...)`` (and ``.add``).
    ``fused.output_tensors`` follow that same order after ``build()``.

    Usage::

        # Inline
        fused = Parallel(op_a, op_b).build()
        fused.launch()

        # Steady state (same op objects, in-place IO): one call per forward
        parallel = Parallel(op_a, op_b)
        parallel.run()

        # As part of a Sequential
        fused = Sequential(stem, Parallel(branch_a, branch_b)).build()
    """

    def __init__(self, *items):
        if len(items) < 2:
            raise ValueError("Parallel() requires at least 2 items")
        self._items = list(items)
        self._run_fused: Optional[FusedOp] = None
        self._run_signature: Optional[Tuple] = None

    def invalidate_run(self) -> None:
        """Clear :meth:`run` cache (call after mutating ``_items`` without :meth:`add`)."""
        self._run_fused = None
        self._run_signature = None

    def add(self, item):
        """Add a branch.  Returns self for chaining."""
        self.invalidate_run()
        self._items.append(item)
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
        cache_key, ops = _fusion_cache_key_and_ops(self._items, "P", cache_device)
        entry = _BUILD_CACHE.get(cache_key)
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
            kernel_labels=r.kernel_labels,
            rebind_output_sources=rebind_src,
            branch_ops=tuple(ops),
        )

        _BUILD_CACHE[cache_key] = _cache_build_result(fused, ops, r.output_source_map)

        if kernel_dir is not None:
            fused._apply_kernel_dir(kernel_dir)
        return fused

    def run(self, *, results=None, device=None, kernel_dir: str = None):
        """``build()`` once per stable graph, then ``launch()`` each call.

        Args:
            results: List of descriptors whose ``output_tensors[0]`` are
                returned. Defaults to each branch's leaf output in
                branch order.

        Returns:
            List of output tensors, one per descriptor in *results*.
        """
        return _container_run(self, "P", results, device=device, kernel_dir=kernel_dir)

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

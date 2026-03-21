# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
High-Level Fusion API: Sequential and Parallel.

Fused execution uses ``patched_generic_op`` so the device program cache can patch
only tensor-address slots on repeat launches.

**Fusion build cache** (``_BUILD_CACHE``): single stable integer key per
``(container kind, tree shape, branch program_cache_key / descriptor hash)``.
Cache lookup never accesses :attr:`DeferredOpDescriptor.descriptor`.

**Steady state:** reuse branch descriptor objects, update their IO lists in place,
then ``fused.launch()`` (no args) — it refreshes merged IO from the branch refs
captured at ``build()`` and enqueues. Pass ops to ``launch(*branches)`` only for
rare rebinds. Alternatively call :meth:`FusedOp.refresh_merged_io` explicitly.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import DeferredOpDescriptor, OpDescriptor, is_op_descriptor
from models.experimental.ops.descriptors.fusion.common import (
    _get_risc_type,
)


# =============================================================================
# Fusion Build Cache
# =============================================================================

# Fused ``ProgramDescriptor`` + metadata, keyed by ``_fusion_cache_id`` (stable int).
_BUILD_CACHE: Dict[int, "_CacheEntry"] = {}


@dataclass
class _CacheEntry:
    """Stored in _BUILD_CACHE.  Immutable after construction (except cache eviction).

    ``fused_op`` and ``cached_ops`` enable **hidden rebind**: on cache hit,
    ``build()`` patches the cached ops' input tensors from the user's fresh
    ops and reuses the cached output tensors (same device buffers).
    """

    cached_descriptor: Any  # ProgramDescriptor — dispatched via patched_generic_op on hit
    semaphores: tuple  # Keeps GlobalSemaphore L1 alive
    kernel_labels: tuple  # For _apply_kernel_dir file naming
    output_sources: Tuple[Tuple[int, int], ...]  # (op_idx, tensor_idx) for outputs
    fused_op: Optional[Any] = None  # FusedOp — returned directly on hidden-rebind hit
    cached_ops: Optional[Tuple] = None  # Per-position branch ops with stable output tensors


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
    """Stable branch identity for cache lookup. Never touches ``DeferredOpDescriptor.descriptor``."""
    if isinstance(op, DeferredOpDescriptor):
        return op.program_cache_key
    pk = getattr(op, "program_cache_key", None)
    if pk is not None:
        return pk
    # Legacy field name
    legacy = getattr(op, "program_hash", None)
    if legacy is not None:
        return legacy
    return ttnn.compute_program_descriptor_hash(op.descriptor)


def _fusion_hash_from_ops(items, container_prefix: str, ops: List) -> int:
    """Process-local fusion build-cache id from already-flattened ``ops``."""
    surface = _build_cache_surface_key(items, container_prefix)
    return hash((surface, tuple(_branch_program_cache_key(op) for op in ops)))


def _fusion_cache_id(items, container_prefix: str) -> int:
    """Single stable 64-bit fusion build-cache id."""
    return _fusion_hash_from_ops(items, container_prefix, _flatten_ops(items))


def _fusion_cache_id_and_ops(items, container_prefix: str) -> Tuple[int, List]:
    """Return ``(fusion_cache_id, flattened_ops)`` for ``Sequential``/``Parallel`` ``._items``."""
    ops = _flatten_ops(items)
    cache_id = _fusion_hash_from_ops(items, container_prefix, ops)
    return cache_id, ops


def _item_sort_key(item):
    """Diagnostic sort key by branch program identity (does not reorder user ``Parallel`` branches)."""
    if is_op_descriptor(item):
        return (_branch_program_cache_key(item),)
    if isinstance(item, (Sequential, Parallel)):
        return tuple(_branch_program_cache_key(op) for op in _flatten_ops([item]))
    raise TypeError(f"Unsupported item type: {type(item)}")


def _make_rebind_output_sources(ops: List, output_source_map) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Convert merge ``output_source_map`` (op object, tensor_idx) to (op_index, tensor_idx).

    Returns ``None`` when there is no merge map (single-leaf fuse); :meth:`FusedOp.refresh_merged_io`
    then accepts exactly one source op.
    """
    if not output_source_map:
        return None
    op_id_to_idx = {id(op): idx for idx, op in enumerate(ops)}
    return tuple((op_id_to_idx[id(op)], t_idx) for op, t_idx in output_source_map)


def _coerce_mutable_io_opdescriptor(op: OpDescriptor) -> OpDescriptor:
    """Ensure ``input_tensors`` / ``output_tensors`` are lists so in-place rebind works."""
    ins = op.input_tensors if isinstance(op.input_tensors, list) else list(op.input_tensors)
    outs = op.output_tensors if isinstance(op.output_tensors, list) else list(op.output_tensors)
    if ins is op.input_tensors and outs is op.output_tensors:
        return op
    return OpDescriptor(op.descriptor, ins, outs)


def _cache_build_result(fused_op: "FusedOp", ops: List[OpDescriptor], output_source_map) -> _CacheEntry:
    """Record a cache entry from a freshly-built FusedOp."""
    # Memoize the descriptor hash so patched_generic_op skips the full
    # kernel/CB/semaphore walk on every launch (O(1) instead of O(descriptor)).
    desc = fused_op.descriptor
    if desc.custom_program_hash is None:
        desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)

    op_id_to_idx = {id(op): idx for idx, op in enumerate(ops)}
    output_sources = tuple((op_id_to_idx[id(op)], t_idx) for op, t_idx in output_source_map)

    return _CacheEntry(
        cached_descriptor=fused_op.descriptor,
        semaphores=fused_op.semaphores,
        kernel_labels=fused_op.kernel_labels,
        output_sources=output_sources,
        fused_op=fused_op,
        cached_ops=tuple(ops),
    )


def _collect_merged_io_from_ops(entry: _CacheEntry, ops: List) -> Tuple[List, List]:
    """Deduped inputs + outputs for a cache entry (no branch ``.descriptor`` access)."""
    all_inputs: List = []
    seen_ids: Set[int] = set()
    for op in ops:
        for t in op.input_tensors:
            tid = id(t)
            if tid not in seen_ids:
                all_inputs.append(t)
                seen_ids.add(tid)
    all_outputs = [ops[pi].output_tensors[ti] for pi, ti in entry.output_sources]
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


def _hidden_rebind_from_cache(entry: _CacheEntry, fresh_ops: List) -> "FusedOp":
    """Fast cache-hit path: patch cached ops' inputs, reuse cached outputs.

    For each position *i*:
    1. Copy ``input_tensors`` from the user's fresh op into the cached op
       (so ``refresh_merged_io`` picks up current inputs).
    2. Copy the cached op's ``output_tensors`` into the user's fresh op
       (so the user can read results via ``branch.output_tensors[0]``).

    Output tensors are stable device buffers allocated during the first
    (cold) build.  Reuse is safe because outputs are write targets — the
    device overwrites them on each ``launch()``, and command-queue ordering
    prevents overlap.
    """
    cached_ops = entry.cached_ops
    for i, fresh_op in enumerate(fresh_ops):
        cached_ops[i].input_tensors[:] = fresh_op.input_tensors
        fresh_op.output_tensors[:] = cached_ops[i].output_tensors
    entry.fused_op.refresh_merged_io(list(cached_ops))
    return entry.fused_op


def clear_build_cache() -> None:
    """Clear the fusion build cache."""
    _BUILD_CACHE.clear()


# =============================================================================
# FusedOp
# =============================================================================


class FusedOp:
    """Result of ``Sequential``/``Parallel``.``build()``.

    Holds a merged ``OpDescriptor`` (fused ``ProgramDescriptor`` + IO lists) and
    keeps references to global semaphores used by the fused program.

    **Launch:** :meth:`launch` with no arguments refreshes merged IO from the
    branch ``OpDescriptor`` objects stored at ``build()`` time, then enqueues via
    ``patched_generic_op`` (device cache patches address slots). Pass
    ``launch(*branches)`` to refresh from a different op tuple when needed.

    **Refresh IO without launching:** :meth:`refresh_merged_io` (or the
    ``refresh_merged_io_from_*`` helpers) after mutating branch ops' tensor lists.
    """

    __slots__ = (
        "op",
        "semaphores",
        "kernel_labels",
        "_rebind_output_sources",
        "_branch_ops",
        "_changed_io_indices",
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
        self._rebind_output_sources = rebind_output_sources
        self._branch_ops = branch_ops
        self._changed_io_indices = None

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
        """Refresh merged IO from branch ops, enqueue, return outputs.

        With no arguments, uses the branch ops captured at ``build()`` time
        (which ``_hidden_rebind_from_cache`` patches with fresh inputs on
        cache hit).  With arguments, refreshes from the given ops instead.

        ``patched_generic_op`` internally diffs io_tensor addresses against
        the previous dispatch and skips patching unchanged slots (weights,
        reused outputs).  ``_changed_io_indices`` records which slots changed.
        """
        if branch_ops_override:
            self.refresh_merged_io(list(branch_ops_override))
        elif self._branch_ops is not None:
            self.refresh_merged_io(list(self._branch_ops))
        io_tensors = list(self.input_tensors) + list(self.output_tensors)
        _, self._changed_io_indices = ttnn._ttnn.operations.experimental.patched_generic_op(io_tensors, self.descriptor)
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
        if self._rebind_output_sources is not None:
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

    def refresh_merged_io_from_sequential(self, sequential: "Sequential") -> None:
        if not isinstance(sequential, Sequential):
            raise TypeError(f"expected Sequential, got {type(sequential).__name__}")
        self.refresh_merged_io(_flatten_ops(sequential._items))

    def rebind_from_ops(self, ops: List) -> None:
        """Deprecated alias for :meth:`refresh_merged_io`."""
        self.refresh_merged_io(ops)

    def rebind_from_parallel(self, parallel: "Parallel") -> None:
        """Deprecated alias for :meth:`refresh_merged_io_from_parallel`."""
        self.refresh_merged_io_from_parallel(parallel)

    def rebind_from_sequential(self, sequential: "Sequential") -> None:
        """Deprecated alias for :meth:`refresh_merged_io_from_sequential`."""
        self.refresh_merged_io_from_sequential(sequential)

    def dump_kernel_sources(self, output_dir: str) -> None:
        """Write fused kernel sources to reader.cpp, writer.cpp, compute.cpp.

        If multiple kernels share the same RISC type (e.g. two readers for
        different core groups), they are written as reader_0.cpp, reader_1.cpp, etc.

        Args:
            output_dir: Directory to write files into (created if needed).
        """
        os.makedirs(output_dir, exist_ok=True)

        # Group kernels by RISC type
        # riscv_0 = BRISC = writer, riscv_1 = NCRISC = reader
        by_type: dict[str, list] = {}
        for kernel in self.op.descriptor.kernels:
            risc = _get_risc_type(kernel)
            name = {"riscv_0": "writer", "riscv_1": "reader", "compute": "compute"}.get(risc, "unknown")
            by_type.setdefault(name, []).append(kernel)

        for name, kernels in by_type.items():
            for i, kernel in enumerate(kernels):
                if kernel.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE:
                    source = kernel.kernel_source
                else:
                    path = kernel.kernel_source
                    source = ""
                    for base in [os.environ.get("TT_METAL_HOME", ""), ""]:
                        full = os.path.join(base, path) if base else path
                        if os.path.exists(full):
                            with open(full) as f:
                                source = f.read()
                            break
                    if not source:
                        source = f"// Could not read file: {path}\n"

                filename = f"{name}.cpp" if len(kernels) == 1 else f"{name}_{i}.cpp"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w") as f:
                    f.write(source)

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

    def add(self, item):
        """Append an item.  Returns self for chaining."""
        self._items.append(item)
        return self

    def build(self, device=None, kernel_dir: str = None) -> FusedOp:
        """Merge into one ``FusedOp``. Cache hit skips codegen; see module docstring.

        Args:
            device: Inferred from tensors when omitted.
            kernel_dir: If set, kernel sources are written as files (existing files kept).
        """
        cache_id, ops = _fusion_cache_id_and_ops(self._items, "S")
        entry = _BUILD_CACHE.get(cache_id)
        if entry is not None:
            if entry.fused_op is not None and entry.cached_ops is not None:
                result = _hidden_rebind_from_cache(entry, ops)
            else:
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

        _BUILD_CACHE[cache_id] = _cache_build_result(fused, ops, r.output_source_map)

        if kernel_dir is not None:
            fused._apply_kernel_dir(kernel_dir)
        return fused

    def build_launch(self, device=None, kernel_dir: str = None):
        """``build()`` then ``launch()`` (merged IO refreshed from branch ops)."""
        return self.build(device=device, kernel_dir=kernel_dir).launch()

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
        parallel.build_launch()

        # As part of a Sequential
        fused = Sequential(stem, Parallel(branch_a, branch_b)).build()
    """

    def __init__(self, *items):
        if len(items) < 2:
            raise ValueError("Parallel() requires at least 2 items")
        self._items = list(items)

    def add(self, item):
        """Add a branch.  Returns self for chaining."""
        self._items.append(item)
        return self

    def build(self, device=None, kernel_dir: str = None) -> FusedOp:
        """Merge branches into one ``FusedOp``. Cache hit refreshes IO from branch lists only.

        Args:
            device: Inferred from tensors when omitted.
            kernel_dir: If set, kernel sources are written as files (existing files kept).
        """
        cache_id, ops = _fusion_cache_id_and_ops(self._items, "P")
        entry = _BUILD_CACHE.get(cache_id)
        if entry is not None:
            if entry.fused_op is not None and entry.cached_ops is not None:
                result = _hidden_rebind_from_cache(entry, ops)
            else:
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

        _BUILD_CACHE[cache_id] = _cache_build_result(fused, ops, r.output_source_map)

        if kernel_dir is not None:
            fused._apply_kernel_dir(kernel_dir)
        return fused

    def build_launch(self, device=None, kernel_dir: str = None):
        """``build()`` then ``launch()`` (merged IO refreshed from branch ops)."""
        return self.build(device=device, kernel_dir=kernel_dir).launch()

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

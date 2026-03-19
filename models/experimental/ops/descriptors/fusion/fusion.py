# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
High-Level Fusion API: Sequential and Parallel.

Provides the user-facing API for composing operations into fused kernels.
Sequential chains ops linearly; Parallel runs ops on disjoint core subsets.

Usage (linear chain):
    >>> fused = Sequential(op0, op1, op2).build()
    >>> fused.launch()

Usage (branching tree):
    >>> fused = Sequential(stem, Parallel(branch_a, branch_b)).build()
    >>> fused.launch()
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.fusion.common import (
    _get_risc_type,
)
from models.experimental.ops.descriptors.fusion.codegen.args import (
    _get_core_coords_from_ranges,
)


# =============================================================================
# Fusion Build Cache
# =============================================================================

# Caches the built FusedOp descriptor keyed by topology + structural hash.
# On cache hit, deep-copies the descriptor and updates only the RT args
# and CB buffer pointers from the fresh source ops.  generic_op then copies
# these into the cached Program via override_runtime_arguments.
_BUILD_CACHE: Dict[tuple, "_CacheEntry"] = {}


@dataclass
class _CacheHitOverrideSpec:
    """How to override a cached descriptor with fresh op data on cache hit.

    All fields are derived structurally from the builder's output —
    no address matching or id() reverse engineering.  Immutable after
    construction (tuple fields).
    """

    # Per fused kernel: ((op_idx, kernel_idx), ...) — which source kernels'
    # RT args were concatenated into this fused kernel.
    origin_kernel_map: Tuple[Tuple[Tuple[int, int], ...], ...]

    # Per fused kernel: fixed barrier RT arg values (GlobalSemaphore L1 addrs).
    # Constant across cache hits — semaphore refs kept alive in _CacheEntry.
    barrier_suffix: Tuple[Tuple[int, ...], ...]

    # Per fused kernel: ((op_idx, cb_idx, size), ...) — sharded CB rebind
    # sources.  On hit, read fresh buffer_address() from the source op's CB.
    rebind_spec: Tuple[Tuple[Tuple[int, int, int], ...], ...]

    # (merged_cb_idx, op_idx, orig_cb_idx) — which source op CB provides
    # the buffer pointer for each sharded merged CB.
    sharded_cb_map: Tuple[Tuple[int, int, int], ...]

    # (merged_cb_idx, op_idx, orig_cb_idx) — which source op CB provides
    # the GlobalCircularBuffer pointer for each GlobalCB-backed merged CB.
    global_cb_map: Tuple[Tuple[int, int, int], ...]

    # (op_idx, tensor_idx) — which source ops produce the fused output tensors.
    output_sources: Tuple[Tuple[int, int], ...]


@dataclass
class _CacheEntry:
    """Stored in _BUILD_CACHE.  Never mutated after construction."""

    cached_descriptor: Any  # ProgramDescriptor — template for deep copy on hit
    semaphores: tuple  # Keeps GlobalSemaphore L1 alive
    kernel_labels: tuple  # For _apply_kernel_dir file naming
    spec: _CacheHitOverrideSpec  # How to apply fresh ops


def _flatten_ops(items) -> List[OpDescriptor]:
    """Recursively flatten Sequential/Parallel items into an ordered list of OpDescriptors."""
    result = []
    for item in items:
        if isinstance(item, OpDescriptor):
            result.append(item)
        elif isinstance(item, Sequential):
            result.extend(_flatten_ops(item._items))
        elif isinstance(item, Parallel):
            result.extend(_flatten_ops(item._items))
    return result


def _topology_fingerprint(items) -> str:
    """Encode the tree shape of items as a string.

    Distinguishes ``Sequential(A, B, C)`` from ``Sequential(A, Parallel(B, C))``
    even when the flattened op hashes are identical.
    """
    parts = []
    for item in items:
        if isinstance(item, OpDescriptor):
            parts.append("O")
        elif isinstance(item, Sequential):
            parts.append(f"S({_topology_fingerprint(item._items)})")
        elif isinstance(item, Parallel):
            parts.append(f"P({_topology_fingerprint(item._items)})")
    return ",".join(parts)


def _item_sort_key(item):
    """Sort key for Parallel items — normalizes order at construction."""
    h = ttnn.compute_program_descriptor_hash
    if isinstance(item, OpDescriptor):
        return (h(item.descriptor),)
    if isinstance(item, (Sequential, Parallel)):
        return tuple(h(op.descriptor) for op in _flatten_ops([item]))
    raise TypeError(f"Unsupported item type: {type(item)}")


def _cache_key_and_ops(items):
    """Compute a topology-aware hash key and flattened ops list.

    Returns (key, ops).  The key starts with a topology fingerprint
    so that structurally different compositions (e.g. Sequential vs
    Parallel) with the same ops produce different keys.
    """
    ops = _flatten_ops(items)
    h = ttnn.compute_program_descriptor_hash
    topo = _topology_fingerprint(items)
    return (topo, *(h(op.descriptor) for op in ops)), ops


def _extract_barrier_suffix(descriptor) -> Tuple[Tuple[int, ...], ...]:
    """Extract barrier RT arg values per kernel from a freshly-built descriptor.

    Uses named CT args (barrier_rt_offset, rebind_rt_offset) for precise split.
    """
    barrier_suffix = []
    for fused_kernel in descriptor.kernels:
        barrier_offset = None
        rebind_offset = None
        for name, value in fused_kernel.named_compile_time_args:
            if name == "barrier_rt_offset":
                barrier_offset = value
            elif name == "rebind_rt_offset":
                rebind_offset = value

        coords = _get_core_coords_from_ranges(fused_kernel.core_ranges)
        if coords:
            c = coords[0]
            all_args = list(fused_kernel.runtime_args[c.x][c.y])
        else:
            all_args = []

        if barrier_offset is not None:
            if rebind_offset is not None:
                barrier_vals = all_args[barrier_offset:rebind_offset]
            else:
                barrier_vals = all_args[barrier_offset:]
        else:
            barrier_vals = []

        barrier_suffix.append(tuple(barrier_vals))
    return tuple(barrier_suffix)


def _distribute_rebind_to_kernels(rebind_entries, descriptor) -> Tuple[Tuple[Tuple[int, int, int], ...], ...]:
    """Distribute rebind entries to per-kernel specs.

    All fused kernels with a rebind_rt_offset get the full rebind entries.
    Kernels without rebind get empty specs.
    """
    result = []
    for fused_kernel in descriptor.kernels:
        has_rebind = False
        for name, _value in fused_kernel.named_compile_time_args:
            if name == "rebind_rt_offset":
                has_rebind = True
                break
        result.append(rebind_entries if has_rebind else ())
    return tuple(result)


def _cache_build_result(
    fused_op: "FusedOp",
    ops: List[OpDescriptor],
    kernel_phase_map,
    cb_source_map,
    rebind_source_map,
    global_cb_source_map,
    output_source_map,
) -> _CacheEntry:
    """Record metadata from a freshly-built FusedOp for future cache hits.

    All source maps come directly from the builder — no address matching
    or id()-based reverse engineering.

    Args:
        kernel_phase_map: Per fused kernel, list of (OpDescriptor, kernel_index)
            tuples identifying which source phase kernels' RT args were concatenated.
        cb_source_map: [(merged_idx, OpDescriptor, cbs_position)] for sharded CBs.
        rebind_source_map: [(OpDescriptor, cbs_position, size)] for rebind CBs.
        global_cb_source_map: [(merged_idx, OpDescriptor, cbs_position)] for GlobalCBs.
        output_source_map: [(OpDescriptor, tensor_idx)] for output tensors.
    """
    desc = fused_op.descriptor
    op_id_to_idx = {id(op): idx for idx, op in enumerate(ops)}

    # origin_kernel_map: convert OpDescriptor refs → integer indices
    origin_kernel_map = tuple(tuple((op_id_to_idx[id(od)], ki) for od, ki in sources) for sources in kernel_phase_map)

    # barrier_suffix: extract from descriptor (constant across builds)
    barrier_suffix = _extract_barrier_suffix(desc)

    # sharded_cb_map: direct from builder
    sharded_cb_map = tuple((merged_idx, op_id_to_idx[id(op)], orig_idx) for merged_idx, op, orig_idx in cb_source_map)

    # global_cb_map: direct from builder
    global_cb_map = tuple(
        (merged_idx, op_id_to_idx[id(op)], orig_idx) for merged_idx, op, orig_idx in global_cb_source_map
    )

    # rebind entries: convert to (op_idx, cbs_position, size) tuples
    rebind_entries = tuple((op_id_to_idx[id(op)], cbs_pos, size) for op, cbs_pos, size in rebind_source_map)

    # output_sources: direct from builder
    output_sources = tuple((op_id_to_idx[id(op)], t_idx) for op, t_idx in output_source_map)

    spec = _CacheHitOverrideSpec(
        origin_kernel_map=origin_kernel_map,
        barrier_suffix=barrier_suffix,
        rebind_spec=_distribute_rebind_to_kernels(rebind_entries, desc),
        sharded_cb_map=sharded_cb_map,
        global_cb_map=global_cb_map,
        output_sources=output_sources,
    )
    return _CacheEntry(
        cached_descriptor=fused_op.descriptor,
        semaphores=fused_op.semaphores,
        kernel_labels=fused_op.kernel_labels,
        spec=spec,
    )


def _update_cached_descriptor(entry: _CacheEntry, ops: List[OpDescriptor]) -> "FusedOp":
    """Update a cached descriptor from fresh source ops.

    Rebuilds RT args and CB buffer pointers so that generic_op's
    override_runtime_arguments copies the correct values into the
    cached Program.  Uses C++ helpers (~50us total).

    Returns a NEW FusedOp with a deep-copied descriptor.  The cached
    entry is never mutated, which prevents races with in-flight DMA
    when EnqueueMeshWorkload returns before remote devices finish
    reading the host-side descriptor.
    """
    spec = entry.spec

    # Deep-copy via C++ copy constructor (async-safe)
    desc = ttnn.merge_program_descriptors([entry.cached_descriptor])

    # 1. Rebuild per-kernel RT args via C++ helpers
    for ki, fused_kernel in enumerate(desc.kernels):
        fused_kernel.clear_runtime_args()
        for op_idx, k_idx in spec.origin_kernel_map[ki]:
            fused_kernel.append_runtime_args_from(ops[op_idx].descriptor.kernels[k_idx])
        if spec.barrier_suffix[ki]:
            fused_kernel.extend_runtime_args_uniform(spec.barrier_suffix[ki])
        rebind_vals = _recompute_rebind(spec.rebind_spec[ki], ops)
        if rebind_vals:
            fused_kernel.extend_runtime_args_uniform(rebind_vals)

    # 2. Rebuild common_runtime_args
    for ki, fused_kernel in enumerate(desc.kernels):
        common: List[int] = []
        for op_idx, k_idx in spec.origin_kernel_map[ki]:
            src_kernel = ops[op_idx].descriptor.kernels[k_idx]
            try:
                common.extend(list(src_kernel.common_runtime_args))
            except (AttributeError, TypeError):
                pass
        fused_kernel.common_runtime_args = common

    # 3. Update sharded CB buffer pointers
    for merged_cb_idx, op_idx, orig_cb_idx in spec.sharded_cb_map:
        new_cb = ops[op_idx].descriptor.cbs[orig_cb_idx]
        desc.cbs[merged_cb_idx].set_buffer_from_cb(new_cb)

    # 4. Update GlobalCB pointers
    for merged_cb_idx, op_idx, orig_cb_idx in spec.global_cb_map:
        new_cb = ops[op_idx].descriptor.cbs[orig_cb_idx]
        desc.cbs[merged_cb_idx].set_global_circular_buffer_from_cb(new_cb)

    # 5. Build tensor lists
    all_inputs: List = []
    seen_ids: Set[int] = set()
    for op in ops:
        for t in op.input_tensors:
            tid = id(t)
            if tid not in seen_ids:
                all_inputs.append(t)
                seen_ids.add(tid)
    all_outputs = [ops[pi].output_tensors[ti] for pi, ti in spec.output_sources]

    # Return NEW FusedOp — cached entry is never mutated
    return FusedOp(
        op=OpDescriptor(desc, all_inputs, all_outputs),
        semaphores=entry.semaphores,
        kernel_labels=entry.kernel_labels,
    )


def _recompute_rebind(spec_entries, ops: List[OpDescriptor]) -> List[int]:
    """Recompute rebind RT args from new ops' CB buffer addresses.

    Each spec entry is (op_idx, cb_idx, total_size). On cache hit,
    reads the new buffer address from the op's CB descriptor.
    """
    result: List[int] = []
    for op_idx, cb_idx, total_size in spec_entries:
        cb = ops[op_idx].descriptor.cbs[cb_idx]
        addr = cb.buffer_address()
        if addr is not None:
            result.extend([addr, total_size])
    return result


def clear_build_cache() -> None:
    """Clear the fusion build cache."""
    _BUILD_CACHE.clear()


# =============================================================================
# FusedOp
# =============================================================================


class FusedOp:
    """Result of fusing ops via Sequential/Parallel.

    Wraps an ``OpDescriptor`` and adds ``semaphores`` refs that prevent
    GC of GlobalSemaphores whose L1 addresses are baked into runtime args.

    Properties ``descriptor``, ``input_tensors``, and ``output_tensors``
    forward to the underlying ``OpDescriptor``, so ``FusedOp`` is
    duck-type compatible with ``OpDescriptor``.

    Cannot be nested in Sequential/Parallel -- ``_resolve()`` rejects it
    with a TypeError.
    """

    __slots__ = (
        "op",
        "semaphores",
        "kernel_labels",
    )

    def __init__(
        self,
        op: OpDescriptor,
        semaphores: Tuple[Any, ...] = (),
        kernel_labels: Tuple[str, ...] = (),
    ):
        self.op = op
        self.semaphores = semaphores
        self.kernel_labels = kernel_labels

    @property
    def descriptor(self):
        return self.op.descriptor

    @property
    def input_tensors(self):
        return self.op.input_tensors

    @property
    def output_tensors(self):
        return self.op.output_tensors

    def launch(self):
        """Dispatch the fused op via generic_op.

        Returns:
            self.output_tensors (tuple of output tensors)
        """
        io_tensors = list(self.input_tensors) + list(self.output_tensors)
        ttnn.generic_op(io_tensors, self.descriptor)
        return self.output_tensors

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
        """Build the fused op.  Device is auto-extracted from tensors if not provided.

        On repeated calls with the same structural hash, skips the full
        build and patches only RT args and CB buffer pointers (~0.1ms).

        Args:
            device: Target device.  Auto-extracted from tensors if None.
            kernel_dir: Optional directory for file-based kernel sources.
                When set, kernel sources are written to files and the JIT
                compiles from disk instead of in-memory strings.  Existing
                files are NOT overwritten — delete them to force regeneration.
        """
        # Try build cache first (fast path: ~50us)
        key, ops = _cache_key_and_ops(self._items)
        entry = _BUILD_CACHE.get(key)
        if entry is not None:
            result = _update_cached_descriptor(entry, ops)
            if kernel_dir is not None:
                result._apply_kernel_dir(kernel_dir)
            return result

        # Cache miss: full build
        r = self._build_internal(device)
        fused = FusedOp(
            op=OpDescriptor(r.descriptor, r.input_tensors, r.output_tensors),
            semaphores=r.semaphores,
            kernel_labels=r.kernel_labels,
        )

        # Record cache entry for future hits
        if len(ops) > 1:
            _BUILD_CACHE[key] = _cache_build_result(
                fused,
                ops,
                r.kernel_phase_map,
                r.cb_source_map,
                r.rebind_source_map,
                r.global_cb_source_map,
                r.output_source_map,
            )

        if kernel_dir is not None:
            fused._apply_kernel_dir(kernel_dir)
        return fused

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

    Usage::

        # Inline
        fused = Parallel(op_a, op_b).build()
        fused.launch()

        # As part of a Sequential
        fused = Sequential(stem, Parallel(branch_a, branch_b)).build()
    """

    def __init__(self, *items):
        if len(items) < 2:
            raise ValueError("Parallel() requires at least 2 items")
        self._items = sorted(items, key=_item_sort_key)

    def add(self, item):
        """Add a branch.  Returns self for chaining."""
        self._items.append(item)
        return self

    def build(self, device=None, kernel_dir: str = None) -> FusedOp:
        """Build each item independently and merge into one FusedOp.

        Args:
            device: Target device.  Auto-extracted from tensors if None.
            kernel_dir: Optional directory for file-based kernel sources.
                When set, kernel sources are written to files and the JIT
                compiles from disk instead of in-memory strings.  Existing
                files are NOT overwritten — delete them to force regeneration.
        """
        # Try build cache first (fast path)
        key, ops = _cache_key_and_ops(self._items)
        entry = _BUILD_CACHE.get(key)
        if entry is not None:
            result = _update_cached_descriptor(entry, ops)
            if kernel_dir is not None:
                result._apply_kernel_dir(kernel_dir)
            return result

        # Cache miss: full build
        r = self._build_internal(device)
        fused = FusedOp(
            op=OpDescriptor(r.descriptor, r.input_tensors, r.output_tensors),
            semaphores=r.semaphores,
            kernel_labels=r.kernel_labels,
        )

        # Record cache entry for future hits
        if len(ops) > 1:
            _BUILD_CACHE[key] = _cache_build_result(
                fused,
                ops,
                r.kernel_phase_map,
                r.cb_source_map,
                r.rebind_source_map,
                r.global_cb_source_map,
                r.output_source_map,
            )

        if kernel_dir is not None:
            fused._apply_kernel_dir(kernel_dir)
        return fused

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

    if isinstance(item, OpDescriptor):
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
        if isinstance(item, OpDescriptor):
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

    if isinstance(item, OpDescriptor):
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

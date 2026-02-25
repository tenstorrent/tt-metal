# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
High-Level Fusion API: Sequential and Parallel.

Provides the user-facing API for composing operations into fused kernels.
Sequential chains ops linearly; Parallel runs ops on disjoint core subsets.

Usage (linear chain):
    >>> fused = Sequential(op0, op1, op2).build()
    >>> composite.launch([fused])

Usage (branching tree):
    >>> fused = Sequential(stem, Parallel(branch_a, branch_b)).build()
    >>> composite.launch([fused])
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.fusion.common import (
    _get_risc_type,
)
from models.experimental.ops.descriptors.fusion.codegen.args import (
    _get_core_coords_from_ranges,
)


# =============================================================================
# Address-Patching Build Cache
# =============================================================================

# Caches the entire built FusedOp keyed by structural hash of input ops.
# On cache hit, skips all of _build_fused_descriptor and
# OpGraphBuilder._build_internal, only patching RT args and
# CB buffer pointers via C++ helpers.
_PATCH_CACHE: Dict[tuple, "_CacheEntry"] = {}


@dataclass
class _CacheEntry:
    """Metadata for patching a cached FusedOp on cache hit."""

    fused_op: "FusedOp"
    # Per fused kernel: ordered list of (op_idx, kernel_idx_in_op)
    # telling which source kernels contributed to this fused kernel's RT args.
    # op_idx indexes into the flattened ops list from _flatten_ops().
    role_map: List[List[Tuple[int, int]]]
    # Per fused kernel: fixed barrier address values (appended uniformly).
    barrier_suffix: List[List[int]]
    # Per fused kernel: rebind spec entries [(op_idx, cb_idx, total_size)]
    # for recomputing sharded CB address RT args.
    rebind_spec: List[List[Tuple[int, int, int]]]
    # (merged_cb_idx, op_idx, orig_cb_idx) for updating sharded CB buffer ptrs.
    sharded_cb_map: List[Tuple[int, int, int]]
    # Which ops contribute to the FusedOp's output_tensors:
    # [(op_idx, tensor_idx_within_op), ...]
    output_sources: List[Tuple[int, int]]


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


def _cache_key_and_ops(items):
    """Compute a hash key and flattened ops list from the input items.

    Returns (key, ops) to avoid redundant _flatten_ops calls.
    """
    ops = _flatten_ops(items)
    h = ttnn.compute_program_descriptor_hash
    return tuple(h(op.descriptor) for op in ops), ops


def _setup_cache_entry(fused_op: "FusedOp", ops: List[OpDescriptor], kernel_phase_map) -> _CacheEntry:
    """Record metadata from a freshly-built FusedOp for future cache hits.

    Extracts the role_map (which phase kernels map to each fused kernel),
    barrier suffix (fixed barrier addresses), rebind spec (sharded CB info),
    sharded CB map (merged CB indices needing buffer pointer updates), and
    output tensor sources (which phases produce the FusedOp's outputs).

    Args:
        kernel_phase_map: Builder-provided mapping. Per fused kernel, a list
            of (OpDescriptor, kernel_index) tuples identifying which source
            phase kernels' RT args were concatenated into that fused kernel.
    """
    desc = fused_op.descriptor

    # --- Build role_map from builder-provided kernel_phase_map ---
    op_id_to_idx = {id(op): idx for idx, op in enumerate(ops)}
    role_map: List[List[Tuple[int, int]]] = [
        [(op_id_to_idx[id(od)], ki) for od, ki in sources] for sources in kernel_phase_map
    ]

    # --- Extract barrier_suffix and rebind spec per fused kernel ---
    # Uses named CT args (barrier_rt_offset, rebind_rt_offset) for precise split.
    barrier_suffix: List[List[int]] = []
    rebind_spec: List[List[Tuple[int, int, int]]] = []

    for role_idx, fused_kernel in enumerate(desc.kernels):
        # Find barrier_rt_offset and rebind_rt_offset from named CT args
        barrier_offset = None
        rebind_offset = None
        for name, value in fused_kernel.named_compile_time_args:
            if name == "barrier_rt_offset":
                barrier_offset = value
            elif name == "rebind_rt_offset":
                rebind_offset = value

        # Get the full RT args from first core of the fused kernel.
        # RuntimeArgsView supports [x][y] access but not .get() or iteration.
        coords = _get_core_coords_from_ranges(fused_kernel.core_ranges)
        if coords:
            c = coords[0]
            all_args = list(fused_kernel.runtime_args[c.x][c.y])
        else:
            all_args = []

        # Extract barrier portion (constant across builds — semaphore addrs)
        if barrier_offset is not None:
            if rebind_offset is not None:
                barrier_vals = all_args[barrier_offset:rebind_offset]
                rebind_vals = all_args[rebind_offset:]
            else:
                barrier_vals = all_args[barrier_offset:]
                rebind_vals = []
        else:
            barrier_vals = []
            rebind_vals = []

        barrier_suffix.append(barrier_vals)

        # Map rebind values back to source op CBs by address matching.
        # Rebind values are [addr, size, addr, size, ...] pairs.
        role_rebind_entries: List[Tuple[int, int, int]] = []
        for i in range(0, len(rebind_vals), 2):
            addr = rebind_vals[i]
            size = rebind_vals[i + 1]
            found = False
            for op_idx, op in enumerate(ops):
                for cb_idx, cb in enumerate(op.descriptor.cbs):
                    if cb.has_buffer() and cb.buffer_address() == addr and cb.total_size == size:
                        role_rebind_entries.append((op_idx, cb_idx, size))
                        found = True
                        break
                if found:
                    break

        rebind_spec.append(role_rebind_entries)

    # --- Build sharded_cb_map ---
    # Track which merged CB descriptors have buffers and map to source op CBs.
    sharded_cb_map: List[Tuple[int, int, int]] = []
    for merged_idx, merged_cb in enumerate(desc.cbs):
        if not merged_cb.has_buffer():
            continue
        merged_addr = merged_cb.buffer_address()
        found = False
        for op_idx, op in enumerate(ops):
            for cb_idx, cb in enumerate(op.descriptor.cbs):
                if cb.has_buffer() and cb.buffer_address() == merged_addr:
                    sharded_cb_map.append((merged_idx, op_idx, cb_idx))
                    found = True
                    break
            if found:
                break

    # --- Record output tensor sources ---
    output_sources: List[Tuple[int, int]] = []
    for out_t in fused_op.output_tensors:
        out_id = id(out_t)
        found = False
        for op_idx, op in enumerate(ops):
            for t_idx, t in enumerate(op.output_tensors):
                if id(t) == out_id:
                    output_sources.append((op_idx, t_idx))
                    found = True
                    break
            if found:
                break

    return _CacheEntry(
        fused_op=fused_op,
        role_map=role_map,
        barrier_suffix=barrier_suffix,
        rebind_spec=rebind_spec,
        sharded_cb_map=sharded_cb_map,
        output_sources=output_sources,
    )


def _patch_cached(entry: _CacheEntry, ops: List[OpDescriptor]) -> "FusedOp":
    """Patch a cached FusedOp with new tensor addresses from fresh ops.

    Uses C++ helpers for efficient RT arg rebuilding (~50us total).
    """
    fused = entry.fused_op
    desc = fused.descriptor

    # 1. Rebuild per-kernel RT args via C++ helpers
    for role_idx, fused_kernel in enumerate(desc.kernels):
        fused_kernel.clear_runtime_args()
        for op_idx, k_idx in entry.role_map[role_idx]:
            fused_kernel.append_runtime_args_from(ops[op_idx].descriptor.kernels[k_idx])
        if entry.barrier_suffix[role_idx]:
            fused_kernel.extend_runtime_args_uniform(entry.barrier_suffix[role_idx])
        rebind_vals = _recompute_rebind(entry.rebind_spec[role_idx], ops)
        if rebind_vals:
            fused_kernel.extend_runtime_args_uniform(rebind_vals)

    # 2. Rebuild common_runtime_args
    for role_idx, fused_kernel in enumerate(desc.kernels):
        common: List[int] = []
        for op_idx, k_idx in entry.role_map[role_idx]:
            src_kernel = ops[op_idx].descriptor.kernels[k_idx]
            try:
                common.extend(list(src_kernel.common_runtime_args))
            except (AttributeError, TypeError):
                pass
        fused_kernel.common_runtime_args = common

    # 3. Update sharded CB buffer pointers
    for merged_cb_idx, op_idx, orig_cb_idx in entry.sharded_cb_map:
        new_cb = ops[op_idx].descriptor.cbs[orig_cb_idx]
        desc.cbs[merged_cb_idx].set_buffer_from_cb(new_cb)

    # 4. Update tensor lists
    all_inputs: List = []
    seen_ids: Set[int] = set()
    for op in ops:
        for t in op.input_tensors:
            tid = id(t)
            if tid not in seen_ids:
                all_inputs.append(t)
                seen_ids.add(tid)
    all_outputs = [ops[pi].output_tensors[ti] for pi, ti in entry.output_sources]
    fused.op = OpDescriptor(desc, all_inputs, all_outputs)

    return fused


def _recompute_rebind(spec_entries: List[Tuple[int, int, int]], ops: List[OpDescriptor]) -> List[int]:
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
    """Clear the patch cache."""
    _PATCH_CACHE.clear()


# =============================================================================
# Zero-Factory Launch: Tensor Slot Discovery
# =============================================================================


@dataclass
class _TensorSlot:
    """Tracks one external tensor's address positions in the fused descriptor."""

    tensor: Any  # The ttnn.Tensor reference
    address: int  # Current buffer_address()
    is_sharded: bool  # Whether this tensor backs a sharded CB
    input_list_index: int  # Position in fused.input_tensors (-1 if output-only)
    output_list_index: int  # Position in fused.output_tensors (-1 if input-only)
    # Per fused kernel: list of RT arg positions holding this address
    rt_positions: Dict[int, List[int]] = field(default_factory=dict)
    # Per fused kernel: list of common_runtime_args positions
    common_positions: Dict[int, List[int]] = field(default_factory=dict)
    # Merged CB entries: [(merged_cb_idx, cb_buffer_index)] for sharded tensors
    sharded_cb_entries: List[Tuple[int, int]] = field(default_factory=list)


def _discover_tensor_slots(
    fused_op: "FusedOp",
    ops: List[OpDescriptor],
) -> Tuple[List[_TensorSlot], Dict[int, int], Dict[int, int]]:
    """Discover which RT arg positions hold each tensor's buffer address.

    Called once after first build. Scans all fused kernels' RT args (up to
    barrier_rt_offset) for known tensor addresses.

    Returns:
        (slots, id_to_slot_idx, addr_to_slot_idx) where:
        - slots: list of _TensorSlot
        - id_to_slot_idx: maps id(original_tensor) -> slot index
        - addr_to_slot_idx: maps buffer_address -> slot index (prefers input slots)
    """
    desc = fused_op.descriptor

    # Step 1: Collect all unique tensors (inputs + outputs, deduped)
    all_tensors: List[Any] = []
    seen_ids: Set[int] = set()
    for op in ops:
        for t in list(op.input_tensors) + list(op.output_tensors):
            tid = id(t)
            if tid not in seen_ids:
                all_tensors.append(t)
                seen_ids.add(tid)

    # Build index maps for fused input/output tensors
    fused_input_id_to_idx = {id(t): i for i, t in enumerate(fused_op.input_tensors)}
    fused_output_id_to_idx = {id(t): i for i, t in enumerate(fused_op.output_tensors)}

    # Step 2: For each fused kernel, get barrier_rt_offset and read RT args
    kernel_rt_data: List[Tuple[Optional[int], List[int], List[int]]] = []
    for fused_kernel in desc.kernels:
        barrier_offset = None
        for name, value in fused_kernel.named_compile_time_args:
            if name == "barrier_rt_offset":
                barrier_offset = value
                break

        coords = _get_core_coords_from_ranges(fused_kernel.core_ranges)
        if coords:
            c = coords[0]
            all_args = list(fused_kernel.runtime_args[c.x][c.y])
        else:
            all_args = []

        try:
            common_args = list(fused_kernel.common_runtime_args)
        except (AttributeError, TypeError):
            common_args = []

        kernel_rt_data.append((barrier_offset, all_args, common_args))

    # Step 3: For each tensor, discover positions
    slots: List[_TensorSlot] = []
    id_to_slot: Dict[int, int] = {}

    for t in all_tensors:
        try:
            addr = t.buffer_address()
        except Exception:
            continue

        if addr == 0:
            continue

        rt_positions: Dict[int, List[int]] = {}
        common_positions: Dict[int, List[int]] = {}

        for k_idx, (barrier_offset, all_args, common_args) in enumerate(kernel_rt_data):
            # Scan RT args up to barrier_rt_offset
            limit = barrier_offset if barrier_offset is not None else len(all_args)
            positions = [i for i in range(limit) if all_args[i] == addr]
            if positions:
                rt_positions[k_idx] = positions

            # Scan common_runtime_args
            c_positions = [i for i in range(len(common_args)) if common_args[i] == addr]
            if c_positions:
                common_positions[k_idx] = c_positions

        # Check sharded CBs
        is_sharded = False
        sharded_cb_entries: List[Tuple[int, int]] = []
        for merged_idx, merged_cb in enumerate(desc.cbs):
            if merged_cb.has_buffer() and merged_cb.buffer_address() == addr:
                is_sharded = True
                # Get buffer_index from first format descriptor
                fmt_descs = merged_cb.format_descriptors
                buf_idx = fmt_descs[0].buffer_index if fmt_descs else 0
                sharded_cb_entries.append((merged_idx, buf_idx))

        # Only create slot if we found at least one position
        if rt_positions or common_positions or sharded_cb_entries:
            tid = id(t)
            slot = _TensorSlot(
                tensor=t,
                address=addr,
                is_sharded=is_sharded,
                input_list_index=fused_input_id_to_idx.get(tid, -1),
                output_list_index=fused_output_id_to_idx.get(tid, -1),
                rt_positions=rt_positions,
                common_positions=common_positions,
                sharded_cb_entries=sharded_cb_entries,
            )
            id_to_slot[tid] = len(slots)
            slots.append(slot)

    # Build address-to-slot map (prefer input slots for duplicate addresses)
    addr_to_slot: Dict[int, int] = {}
    for idx, slot in enumerate(slots):
        if slot.address not in addr_to_slot:
            addr_to_slot[slot.address] = idx
        elif slot.input_list_index >= 0 and slots[addr_to_slot[slot.address]].input_list_index < 0:
            # Prefer slots that are in fused.input_tensors
            addr_to_slot[slot.address] = idx

    return slots, id_to_slot, addr_to_slot


# =============================================================================
# FusedOp
# =============================================================================


class FusedOp:
    """Result of fusing ops via Sequential/Parallel.

    Wraps an ``OpDescriptor`` and adds ``semaphores`` refs that prevent
    GC of GlobalSemaphores whose L1 addresses are baked into runtime args.

    Properties ``descriptor``, ``input_tensors``, and ``output_tensors``
    forward to the underlying ``OpDescriptor``, so ``FusedOp`` is
    duck-type compatible with ``OpDescriptor`` (e.g. for
    ``composite.launch()``).

    Cannot be nested in Sequential/Parallel -- ``_resolve()`` rejects it
    with a TypeError.
    """

    __slots__ = (
        "op",
        "semaphores",
        "kernel_labels",
        "_tensor_slots",
        "_tensor_id_to_slot",
        "_tensor_addr_to_slot",
        "_io_tensors",
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
        self._tensor_slots: Optional[List[_TensorSlot]] = None
        self._tensor_id_to_slot: Dict[int, int] = {}
        self._tensor_addr_to_slot: Dict[int, int] = {}  # buffer_address -> slot_idx
        self._io_tensors: Optional[List[Any]] = None  # cached for launch()

    @property
    def descriptor(self):
        return self.op.descriptor

    @property
    def input_tensors(self):
        return self.op.input_tensors

    @property
    def output_tensors(self):
        return self.op.output_tensors

    def launch(self, replacements=None):
        """Dispatch the fused op, optionally replacing input/output tensors.

        Args:
            replacements: Optional dict mapping original tensor refs (or int
                indices into input_tensors) to new tensors. Only changed
                addresses are patched. Keys can be:
                - Original tensor references (matched by identity, not equality)
                - Integer indices into self.input_tensors

        Returns:
            self.output_tensors (tuple of output tensors)
        """
        if replacements and self._tensor_slots:
            self._patch_addresses(replacements)
            self._io_tensors = None  # invalidate cached list

        if self._io_tensors is None:
            self._io_tensors = list(self.input_tensors) + list(self.output_tensors)
        ttnn.generic_op(self._io_tensors, self.descriptor)
        return self.output_tensors

    def _patch_addresses(self, replacements):
        """Patch tensor addresses in RT args and CBs for changed tensors."""
        desc = self.descriptor
        input_list = list(self.input_tensors)
        output_list = list(self.output_tensors)
        inputs_changed = False
        outputs_changed = False

        for key, new_tensor in replacements.items():
            # Resolve key to slot index
            if isinstance(key, int):
                # Integer index into input_tensors
                old_tensor = input_list[key]
                slot_idx = self._tensor_id_to_slot.get(id(old_tensor))
                if slot_idx is None:
                    # Fallback: match by address
                    try:
                        slot_idx = self._tensor_addr_to_slot.get(old_tensor.buffer_address())
                    except Exception:
                        pass
            else:
                # Tensor reference — try id() first, fall back to address
                slot_idx = self._tensor_id_to_slot.get(id(key))
                if slot_idx is None:
                    try:
                        slot_idx = self._tensor_addr_to_slot.get(key.buffer_address())
                    except Exception:
                        pass

            if slot_idx is None:
                continue

            slot = self._tensor_slots[slot_idx]
            new_addr = new_tensor.buffer_address()

            if new_addr == slot.address:
                # Address unchanged — still update tensor ref in io lists if needed
                if slot.input_list_index >= 0:
                    input_list[slot.input_list_index] = new_tensor
                    inputs_changed = True
                if slot.output_list_index >= 0:
                    output_list[slot.output_list_index] = new_tensor
                    outputs_changed = True
                # Update id mapping
                old_id = id(slot.tensor)
                if old_id in self._tensor_id_to_slot:
                    del self._tensor_id_to_slot[old_id]
                self._tensor_id_to_slot[id(new_tensor)] = slot_idx
                slot.tensor = new_tensor
                continue

            # Address changed — patch RT args via C++ helper
            for k_idx, positions in slot.rt_positions.items():
                desc.kernels[k_idx].replace_runtime_args_at_positions(positions, new_addr)

            # Patch common_runtime_args
            for k_idx, positions in slot.common_positions.items():
                common = list(desc.kernels[k_idx].common_runtime_args)
                for pos in positions:
                    common[pos] = new_addr
                desc.kernels[k_idx].common_runtime_args = common

            # Patch sharded CBs
            for merged_cb_idx, cb_buf_idx in slot.sharded_cb_entries:
                new_cb = ttnn.cb_descriptor_from_sharded_tensor(cb_buf_idx, new_tensor)
                desc.cbs[merged_cb_idx].set_buffer_from_cb(new_cb)

            # Update slot state
            old_id = id(slot.tensor)
            old_addr = slot.address
            slot.address = new_addr
            slot.tensor = new_tensor

            # Update id mapping
            if old_id in self._tensor_id_to_slot:
                del self._tensor_id_to_slot[old_id]
            self._tensor_id_to_slot[id(new_tensor)] = slot_idx

            # Update address mapping
            if old_addr in self._tensor_addr_to_slot and self._tensor_addr_to_slot[old_addr] == slot_idx:
                del self._tensor_addr_to_slot[old_addr]
            self._tensor_addr_to_slot[new_addr] = slot_idx

            # Update io lists
            if slot.input_list_index >= 0:
                input_list[slot.input_list_index] = new_tensor
                inputs_changed = True
            if slot.output_list_index >= 0:
                output_list[slot.output_list_index] = new_tensor
                outputs_changed = True

        # Rebuild OpDescriptor if any tensors changed
        if inputs_changed or outputs_changed:
            self.op = OpDescriptor(desc, tuple(input_list), tuple(output_list))

    def dump_kernel_sources(self, output_dir: str) -> None:
        """Write fused kernel sources to reader.cpp, writer.cpp, compute.cpp.

        If multiple kernels share the same RISC type (e.g. two readers for
        different core groups), they are written as reader_0.cpp, reader_1.cpp, etc.

        Args:
            output_dir: Directory to write files into (created if needed).
        """
        os.makedirs(output_dir, exist_ok=True)

        # Group kernels by RISC type
        by_type: dict[str, list] = {}
        for kernel in self.op.descriptor.kernels:
            risc = _get_risc_type(kernel)
            # Map internal names to friendly filenames
            if risc == "riscv_0":
                name = "reader"
            elif risc == "riscv_1":
                name = "writer"
            elif risc == "compute":
                name = "compute"
            else:
                name = "unknown"
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
            name = {"riscv_0": "reader", "riscv_1": "writer", "compute": "compute"}.get(risc, "unknown")
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
        # Try patch cache first (fast path: ~50us)
        key, ops = _cache_key_and_ops(self._items)
        entry = _PATCH_CACHE.get(key)
        if entry is not None:
            result = _patch_cached(entry, ops)
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

        # Discover tensor slots for launch() support
        if len(ops) > 1:
            fused._tensor_slots, fused._tensor_id_to_slot, fused._tensor_addr_to_slot = _discover_tensor_slots(
                fused, ops
            )

        # Record cache entry for future hits
        if len(ops) > 1:
            _PATCH_CACHE[key] = _setup_cache_entry(fused, ops, r.kernel_phase_map)

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
        composite.launch([fused])

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
        """Build each item independently and merge into one FusedOp.

        Args:
            device: Target device.  Auto-extracted from tensors if None.
            kernel_dir: Optional directory for file-based kernel sources.
                When set, kernel sources are written to files and the JIT
                compiles from disk instead of in-memory strings.  Existing
                files are NOT overwritten — delete them to force regeneration.
        """
        # Try patch cache first (fast path)
        key, ops = _cache_key_and_ops(self._items)
        entry = _PATCH_CACHE.get(key)
        if entry is not None:
            result = _patch_cached(entry, ops)
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

        # Discover tensor slots for launch() support
        if len(ops) > 1:
            fused._tensor_slots, fused._tensor_id_to_slot, fused._tensor_addr_to_slot = _discover_tensor_slots(
                fused, ops
            )

        # Record cache entry for future hits
        if len(ops) > 1:
            _PATCH_CACHE[key] = _setup_cache_entry(fused, ops, r.kernel_phase_map)

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
        return _BuildResult(
            descriptor=item.descriptor,
            input_tensors=item.input_tensors,
            output_tensors=item.output_tensors,
            kernel_phase_map=kpm,
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

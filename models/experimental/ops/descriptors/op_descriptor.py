# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, NamedTuple, Optional

import ttnn


def core_range_set_fusion_key(core_range_set) -> tuple:
    """Content-based key for :class:`~ttnn.CoreRangeSet` in fusion branch hashes.

    The Python binding may use object-identity hashing; use this tuple when mixing
    core placement into :func:`extend_branch_program_cache_key`.
    """
    return tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in core_range_set.ranges())


def extend_branch_program_cache_key(device_program_hash: int, *extras) -> int:
    """Mix ``compute_program_hash`` with extra factory-only arguments for fusion lookup.

    Device ops sometimes omit arguments from :meth:`compute_program_hash` that still
    affect ``create_descriptor`` (e.g. Layernorm ``core_range_set`` for interleaved
    tensors, passed separately from :class:`~ttnn.LayerNormInputs`). The Python
    fusion build cache must not treat those configs as identical.

    Args:
        device_program_hash: Value from ``*_DeviceOperation.compute_program_hash``
            (typically masked to 64 bits).
        *extras: Hashable values that participate in program identity but are not
            covered by the device hash. For core grids prefer
            :func:`core_range_set_fusion_key` rather than passing ``CoreRangeSet``
            directly.

    Returns:
        Integer for :attr:`DeferredOpDescriptor.program_cache_key`. This may differ
        from ``device_program_hash`` when ``extras`` is non-empty.
    """
    base = int(device_program_hash) & ((1 << 64) - 1)
    if not extras:
        return base
    return hash((base, *extras))


class LazyOutputList:
    """List-like container whose slots are allocated on first read.

    On ``__getitem__``: if the slot is ``None``, calls the ``_alloc_fn`` once to
    fill all slots, then returns the requested one.

    On ``__setitem__`` / slice assignment (``[:] = [...]``): writes directly
    without triggering allocation — hidden rebind patches cached tensors in.
    """

    __slots__ = ("_slots", "_alloc_fn")

    def __init__(self, slots: list, alloc_fn: Optional[Callable] = None):
        self._slots = slots
        self._alloc_fn = alloc_fn

    def _materialize(self):
        if self._alloc_fn is not None:
            self._alloc_fn(self._slots)
            self._alloc_fn = None

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if self._slots[idx] is None and self._alloc_fn is not None:
                self._materialize()
            return self._slots[idx]
        if any(s is None for s in self._slots[idx]) and self._alloc_fn is not None:
            self._materialize()
        return self._slots[idx]

    def __setitem__(self, idx, value):
        self._slots[idx] = value
        self._alloc_fn = None

    def __len__(self):
        return len(self._slots)

    def __iter__(self):
        if any(s is None for s in self._slots) and self._alloc_fn is not None:
            self._materialize()
        return iter(self._slots)

    def __repr__(self):
        return f"LazyOutputList({self._slots!r})"


class OpDescriptor(NamedTuple):
    """
    Eager op: ``ProgramDescriptor`` is already materialized.

    Contains:
    - descriptor: The ProgramDescriptor for the operation
    - input_tensors: All input tensors for the op
    - output_tensors: All output tensors for the op
    """

    descriptor: "ttnn.ProgramDescriptor"
    input_tensors: List["ttnn.Tensor"]
    output_tensors: List["ttnn.Tensor"]
    name: str = ""

    def launch(self):
        """Dispatch via ``generic_op`` (not the fused ``patchable_generic_op`` path)."""
        io_tensors = list(self.input_tensors) + list(self.output_tensors)
        ttnn.generic_op(io_tensors, self.descriptor)
        return self.output_tensors


class DeferredOpDescriptor:
    """Branch op with deferred ``ProgramDescriptor`` materialization.

    ``program_cache_key`` identifies this branch for fusion build-cache lookup. It
    must be computable without calling ``descriptor``. Usually it matches the device
    op's ``compute_program_hash``; when the factory takes side-channel arguments
    that hash omits, use :func:`extend_branch_program_cache_key`.

    The C++ factory runs only when :attr:`descriptor` is first accessed (e.g. fusion
    cache miss, first launch of an eager path, or debugging).
    """

    __slots__ = ("_factory_fn", "_descriptor", "input_tensors", "output_tensors", "name", "program_cache_key")

    def __init__(
        self,
        factory_fn,
        input_tensors,
        output_tensors,
        name: str,
        program_cache_key: int,
    ):
        self._factory_fn = factory_fn
        self._descriptor = None
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.name = name
        self.program_cache_key = program_cache_key

    @property
    def descriptor(self):
        if self._descriptor is None:
            self._descriptor = self._factory_fn()
            self._factory_fn = None
        return self._descriptor

    def launch(self):
        """Dispatch via ``generic_op`` (materializes ``descriptor`` if needed)."""
        io_tensors = list(self.input_tensors) + list(self.output_tensors)
        ttnn.generic_op(io_tensors, self.descriptor)
        return self.output_tensors


def is_op_descriptor(item) -> bool:
    """True if ``item`` is an :class:`OpDescriptor` or :class:`DeferredOpDescriptor`."""
    return isinstance(item, (OpDescriptor, DeferredOpDescriptor))


__all__ = [
    "OpDescriptor",
    "DeferredOpDescriptor",
    "core_range_set_fusion_key",
    "extend_branch_program_cache_key",
    "is_op_descriptor",
]

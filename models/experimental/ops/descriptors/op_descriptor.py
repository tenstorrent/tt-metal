# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, NamedTuple, Optional

import ttnn


class LazyOutputList:
    """List-like container whose slots are allocated on first read.

    On ``__getitem__``: if the slot is ``None``, calls the ``_alloc_fn`` once to
    fill all slots, then returns the requested one.

    On ``__setitem__`` / slice assignment (``[:] = [...]``): writes directly
    without triggering allocation — this is the hidden-rebind path where cached
    output tensors are patched in from outside.
    """

    __slots__ = ("_slots", "_alloc_fn")

    def __init__(self, slots: list, alloc_fn: Optional[Callable] = None):
        self._slots = slots
        self._alloc_fn = alloc_fn

    def _materialize(self):
        if self._alloc_fn is not None:
            self._alloc_fn(self._slots)
            self._alloc_fn = None

    # --- read access triggers lazy alloc ---

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if self._slots[idx] is None and self._alloc_fn is not None:
                self._materialize()
            return self._slots[idx]
        # slice
        if any(s is None for s in self._slots[idx]) and self._alloc_fn is not None:
            self._materialize()
        return self._slots[idx]

    # --- write access never triggers alloc ---

    def __setitem__(self, idx, value):
        self._slots[idx] = value
        # Any external write (rebind or partial patch) disables the allocator
        # so a later read of a different slot can't overwrite this one.
        self._alloc_fn = None

    def __len__(self):
        return len(self._slots)

    def __iter__(self):
        # Iteration reads all slots — trigger alloc if needed.
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
        """Dispatch via ``generic_op`` (not the fused ``patched_generic_op`` path)."""
        io_tensors = list(self.input_tensors) + list(self.output_tensors)
        ttnn.generic_op(io_tensors, self.descriptor)
        return self.output_tensors


class DeferredOpDescriptor:
    """Branch op with deferred ``ProgramDescriptor`` materialization.

    ``program_cache_key`` is a stable, process-independent integer used for fusion
    build-cache lookup. It must be computable without calling ``descriptor``.

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


# Backward compatibility — prefer :class:`DeferredOpDescriptor`.
LazyOpDescriptor = DeferredOpDescriptor


def is_op_descriptor(item) -> bool:
    """True if ``item`` is an :class:`OpDescriptor` or :class:`DeferredOpDescriptor`."""
    return isinstance(item, (OpDescriptor, DeferredOpDescriptor))


__all__ = ["OpDescriptor", "DeferredOpDescriptor", "LazyOpDescriptor", "is_op_descriptor"]

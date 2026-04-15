# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
from typing import Callable, Optional

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
        Integer for :attr:`OpDescriptor.program_cache_key`. This may differ
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


class _DeferredOutput:
    """Sentinel for a deferred output tensor slot.

    Each instance is unique so that Python identity matching (``is``) can
    detect internal edges in Sequential chains::

        norm = rms_norm(weight=w, ...)          # output_tensors = [_DeferredOutput()]
        mm = matmul(input_a=norm.output_tensors[0], input_b=W)
        # mm.input_tensors[0] is norm.output_tensors[0]  → True (same _DeferredOutput)

    ``Sequential.run()`` uses this identity to connect predecessor outputs
    to successor inputs after materialization.
    """

    __slots__ = ()

    def __repr__(self):
        return f"_DeferredOutput(id={id(self):#x})"


class OpDescriptor:
    """Operation descriptor with optional deferred ``ProgramDescriptor`` materialization.

    Three construction patterns:

    **Eager** (descriptor already materialized)::

        op = OpDescriptor(descriptor=prog_desc, input_tensors={...}, output_tensors=[...])

    **Deferred** (factory runs only on first ``.descriptor`` access)::

        op = OpDescriptor(
            factory_fn=lambda: create_descriptor(...),
            input_tensors={...},
            output_tensors=LazyOutputList([None], alloc_fn),
            name="rms_norm",
            program_cache_key=hash_value,
        )

    **Persistent** (created by ``@OpDescriptor.create`` decorator when required
    tensors are missing; materializes on first :meth:`update`)::

        desc = rms_norm(weight=qw, ...)   # input_tensor omitted → persistent
        desc.update(tt_q)                  # materializes hash + factory

    ``program_cache_key`` is always available for fusion build-cache lookup
    without touching ``.descriptor``. For eager ops it is computed from the
    descriptor; for deferred ops it is passed in; for persistent ops it is
    computed on first :meth:`update`.

    ``input_tensors`` may be a ``dict`` — auto-converted to a list with
    ``_input_names`` for named :meth:`update` support.
    """

    __slots__ = (
        "_factory_fn",
        "_descriptor",
        "input_tensors",
        "output_tensors",
        "name",
        "program_cache_key",
        "_input_names",
        "_complete_fn",
        "_fusion_input_setter",
    )

    def __init__(
        self,
        descriptor=None,
        input_tensors=None,
        output_tensors=None,
        name: str = "",
        *,
        factory_fn: Optional[Callable] = None,
        program_cache_key: Optional[int] = None,
        input_names: Optional[tuple] = None,
        complete_fn: Optional[Callable] = None,
    ):
        if complete_fn is not None:
            # Partial descriptor: hash computed lazily on first update().
            self._factory_fn = None
            self._descriptor = None
        elif descriptor is not None and factory_fn is not None:
            raise ValueError("Pass descriptor or factory_fn, not both")
        elif descriptor is None and factory_fn is None:
            raise ValueError("Pass descriptor or factory_fn")
        else:
            self._factory_fn = factory_fn
            self._descriptor = descriptor

        if isinstance(input_tensors, dict):
            self._input_names = {k: i for i, k in enumerate(input_tensors)}
            self.input_tensors = list(input_tensors.values())
        else:
            self.input_tensors = input_tensors if input_tensors is not None else []
            self._input_names = dict(input_names) if input_names else None
        self.output_tensors = output_tensors if output_tensors is not None else []
        self.name = name
        self._complete_fn = complete_fn

        if program_cache_key is not None:
            self.program_cache_key = program_cache_key
        elif descriptor is not None:
            self.program_cache_key = ttnn.compute_program_descriptor_hash(descriptor)
        elif complete_fn is not None:
            self.program_cache_key = None  # computed on first update()
        else:
            raise ValueError("Deferred OpDescriptor requires program_cache_key")

        self._fusion_input_setter = None

    def update(self, *args, **kwargs):
        """Replace input tensors by name or position.

        **Positional** (common single-activation case)::

            desc.update(new_q)

        **Keyword** (multi-input or self-documenting)::

            desc.update(input_tensor=new_q)
            desc.update(input_a=new_a, input_b=new_b)
        """
        # Ultra-fast path: write directly to C++ FusionDispatchState input slots.
        # No Python list update — inputs_ in C++ is the source of truth while
        # _dispatch_state is active.  If invalidate_run() clears the fast path,
        # the next update() falls through to the normal path below.
        setter = self._fusion_input_setter
        if setter is not None:
            state, mappings = setter
            if args:
                for i, t in enumerate(args):
                    state.set_input(mappings[i][0], t)
                return
            elif kwargs:
                names = self._input_names
                if names is not None:
                    for name, t in kwargs.items():
                        idx = names.get(name)
                        if idx is not None:
                            state.set_input(mappings[idx][0], t)
                    return

        if args:
            if kwargs:
                raise ValueError("update(): positional OR keyword arguments, not both")
            for i, t in enumerate(args):
                self.input_tensors[i] = t
        elif kwargs:
            names = self._input_names
            if names is None:
                raise ValueError(
                    "Keyword update requires named inputs. " "This OpDescriptor was created without input_names."
                )
            for name, t in kwargs.items():
                idx = names.get(name)
                if idx is None:
                    raise ValueError(
                        f"Unknown input name {name!r} for {self.name!r} op. " f"Valid names: {sorted(names)}"
                    )
                self.input_tensors[idx] = t

        if self._complete_fn is not None and self.program_cache_key is None:
            if all(t is not None and not isinstance(t, _DeferredOutput) for t in self.input_tensors):
                self._materialize()

    def _materialize(self):
        """Complete a persistent descriptor by calling its ``_complete_fn``.

        Copies ``program_cache_key``, ``_factory_fn``, ``_descriptor``, and
        ``output_tensors`` from the materialized descriptor.  Called by
        :meth:`update` when all inputs are real tensors, or by
        ``_materialize_chain`` during Sequential auto-wiring.
        """
        full = self._complete_fn(self.input_tensors)
        self.program_cache_key = full.program_cache_key
        self._factory_fn = full._factory_fn
        self._descriptor = full._descriptor
        self.output_tensors = full.output_tensors
        self._complete_fn = None

    @property
    def descriptor(self):
        if self._descriptor is None:
            self._descriptor = self._factory_fn()
            self._factory_fn = None
        return self._descriptor

    @property
    def is_deferred(self) -> bool:
        """True if the C++ factory has not yet run."""
        return self._descriptor is None

    def launch(self):
        """Dispatch via ``generic_op`` (materializes ``descriptor`` if needed)."""
        io_tensors = list(self.input_tensors) + list(self.output_tensors)
        ttnn.generic_op(io_tensors, self.descriptor)
        return self.output_tensors

    @staticmethod
    def create(name=""):
        """Decorator for descriptor factory functions.

        Wraps a factory so that it supports both **inline** and **persistent**
        modes automatically.  The factory body always receives real tensors —
        the decorator intercepts calls where a required parameter (no default
        in the signature) is ``None`` and returns a deferred descriptor that
        materializes on first :meth:`update`.

        Tensor inputs for :meth:`update` naming are inferred automatically:
        required parameters (no default) are always included; optional
        parameters are included when they receive a ``Tensor`` value.
        No explicit tensor list needed.

        Usage::

            @OpDescriptor.create(name="rms_norm")
            def rms_norm(input_tensor, weight=None, bias=None, epsilon=1e-12, ...):
                inputs = {"input_tensor": input_tensor}
                if weight is not None: inputs["weight"] = weight
                ...
                return OpDescriptor(factory_fn=..., input_tensors=inputs, ...)

            # Inline — body runs immediately:
            desc = rms_norm(tt_q, weight=qw, ...)

            # Persistent — body deferred until update():
            desc = rms_norm(weight=qw, ...)
            desc.update(tt_q)
        """

        def decorator(fn):
            sig = inspect.signature(fn)
            params = sig.parameters
            # Required params: no default → None means "deferred"
            required = frozenset(
                pname
                for pname, p in params.items()
                if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            )
            # Pre-compute positional index of each required param
            # for fast None-check without sig.bind() on the hot path.
            param_names = list(params.keys())
            required_positions = {t: param_names.index(t) for t in required}

            def _is_pending(val):
                return val is None or isinstance(val, _DeferredOutput)

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                # Fast check: are all required params present and non-pending?
                # _DeferredOutput counts as pending (internal edge, not a real tensor).
                deferred = False
                for tname, pos in required_positions.items():
                    if pos < len(args):
                        if _is_pending(args[pos]):
                            deferred = True
                            break
                    elif _is_pending(kwargs.get(tname)):
                        deferred = True
                        break

                if not deferred:
                    # Inline hot path: call the factory body directly.
                    return fn(*args, **kwargs)

                # Deferred path: use sig.bind() to resolve all arg values by name.
                # Fill in None for missing required params so bind() doesn't raise.
                kw = dict(kwargs)
                for tname in required:
                    if tname not in kw and required_positions[tname] >= len(args):
                        kw[tname] = None
                bound = sig.bind(*args, **kw)
                bound.apply_defaults()
                all_args = bound.arguments

                # Build input list: pending required params (None or _DeferredOutput),
                # plus optional params that received a Tensor value.
                # Non-tensor required params (e.g., begins: List[int]) are excluded.
                # _DeferredOutput inputs preserve identity for Sequential edge detection.
                input_names = []
                inputs = []
                idx = 0
                for pname, val in all_args.items():
                    if _is_pending(val) and pname in required:
                        input_names.append((pname, idx))
                        inputs.append(val)
                        idx += 1
                    elif isinstance(val, ttnn.Tensor):
                        input_names.append((pname, idx))
                        inputs.append(val)
                        idx += 1

                def _complete(final_inputs):
                    kw = dict(all_args)
                    for (pname, _), tensor in zip(input_names, final_inputs):
                        kw[pname] = tensor
                    return wrapper(**kw)

                return OpDescriptor(
                    input_tensors=inputs,
                    output_tensors=[_DeferredOutput()],
                    name=name,
                    complete_fn=_complete,
                    input_names=tuple(input_names),
                )

            return wrapper

        return decorator


def is_op_descriptor(item) -> bool:
    """True if ``item`` is an :class:`OpDescriptor`."""
    return isinstance(item, OpDescriptor)


__all__ = [
    "OpDescriptor",
    "LazyOutputList",
    "_DeferredOutput",
    "core_range_set_fusion_key",
    "extend_branch_program_cache_key",
    "is_op_descriptor",
]

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional

import ttnn

# ---------------------------------------------------------------------------
# Program key cache — LRU helpers
# ---------------------------------------------------------------------------
# Per-factory cache storing (program_cache_key, input_name_map, input_names,
# arg_refs) keyed by a cheap arg fingerprint.  Bounded by LRU eviction to
# prevent unbounded growth.
#
# Keying strategy per arg type:
#   Tensor       → hash(tensor.spec)       (content-based, no pinning)
#   Hashable     → hash(obj)               (content-based, no pinning)
#   id()-based   → id(obj) + strong ref    (pinned until LRU eviction)

_PROGRAM_KEY_CACHE_MAX = int(os.environ.get("TT_METAL_FUSION_PROGRAM_KEY_CACHE_MAX_ENTRIES", "128"))

_PROGRAM_KEY_CACHE_REGISTRY: list = []


def _program_key_cache_get(cache, k):
    v = cache.get(k)
    if v is not None:
        cache.move_to_end(k)
    return v


def _program_key_cache_put(cache, k, v):
    cache[k] = v
    cache.move_to_end(k)
    while len(cache) > _PROGRAM_KEY_CACHE_MAX:
        cache.popitem(last=False)


def _clear_all_program_key_caches():
    for cache in _PROGRAM_KEY_CACHE_REGISTRY:
        cache.clear()


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


@dataclass(frozen=True)
class _BoundOutputSpec:
    """Value key: TensorSpec equality plus the tile flags it omits."""

    spec: Any
    transpose_within_face: bool
    transpose_of_faces: bool

    def __hash__(self):
        # TensorSpec has no content-based Python hash. Equality still checks
        # the whole spec, including memory config and alignment, on collisions.
        return hash((tuple(self.spec.shape), self.spec.dtype, self.spec.layout))


def _fold_bound_output_specs(program_cache_key, bound):
    """Use the same key before and after a factory consumes explicit bindings."""
    if not bound:
        return program_cache_key
    specs = []
    for index, tensor in bound:
        spec = tensor.spec
        tile = spec.tile
        specs.append((index, _BoundOutputSpec(spec, tile.transpose_within_face, tile.transpose_of_faces)))
    # Keep value equality as well as the hash, including both tile flags.
    return (program_cache_key, tuple(specs))


class LazyOutputList:
    """List-like container whose slots are allocated on first read.

    On ``__getitem__``: if the slot is ``None``, calls the ``_alloc_fn`` once to
    fill all slots, then returns the requested one.

    ``__setitem__`` records a caller binding without allocating other slots.
    Assigning ``None`` removes that binding.
    Initial slots, including ``LazyOutputList([tensor])``, are factory outputs,
    not caller bindings. An undecorated eager descriptor's plain ``[tensor]``
    list instead contains caller bindings.
    """

    __slots__ = ("_slots", "_alloc_fn", "_explicit_bindings", "_any_explicit")

    def __init__(self, slots: list, alloc_fn: Optional[Callable] = None):
        self._slots = slots if isinstance(slots, list) else list(slots)
        self._alloc_fn = alloc_fn
        self._explicit_bindings = None
        self._any_explicit = False

    def _materialize(self):
        if self._alloc_fn is not None:
            if self._any_explicit:
                allocated = list(self._slots)
                self._alloc_fn(allocated)
                for index in range(len(allocated)):
                    if index >= len(self._explicit_bindings) or not self._explicit_bindings[index]:
                        self._slots[index] = allocated[index]
            else:
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
        bindings = self._explicit_bindings
        if bindings is None:
            bindings = [False] * len(self._slots)
        else:
            bindings = bindings[: len(self._slots)] + [False] * max(0, len(self._slots) - len(bindings))
        if isinstance(idx, slice):
            value = list(value)
        self._slots[idx] = value
        bindings[idx] = [item is not None for item in value] if isinstance(idx, slice) else value is not None
        self._explicit_bindings = bindings
        self._any_explicit = any(bindings)

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

    Two user-facing construction patterns:

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

    A third **persistent** mode is used internally by the ``@OpDescriptor.create``
    decorator when required tensors are missing at construction time.  It is
    created via the internal ``_persistent()`` classmethod and materializes on
    first :meth:`update`.

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
        "_updated_indices",
        "_descriptor_output_buffers",
        "_descriptor_input_buffers",
        "__weakref__",
    )

    def __init__(
        self,
        descriptor=None,
        input_tensors=None,
        output_tensors=None,
        name: str = "",
        *,
        factory_fn: Optional[Callable] = None,
        program_cache_key: Optional[Hashable] = None,
        input_names: Optional[tuple] = None,
    ):
        if descriptor is not None and factory_fn is not None:
            raise ValueError("Pass descriptor or factory_fn, not both")
        if descriptor is None and factory_fn is None:
            raise ValueError("Pass descriptor or factory_fn")
        self._factory_fn = factory_fn
        self._descriptor = descriptor
        self._complete_fn = None

        if isinstance(input_tensors, dict):
            self._input_names = {k: i for i, k in enumerate(input_tensors)}
            self.input_tensors = list(input_tensors.values())
        else:
            self.input_tensors = input_tensors if input_tensors is not None else []
            self._input_names = dict(input_names) if input_names else None
        self.output_tensors = output_tensors if output_tensors is not None else []
        if descriptor is None and not isinstance(self.output_tensors, LazyOutputList):
            # Keep a factory's mutable list, but do not claim its initial outputs.
            self.output_tensors = LazyOutputList(self.output_tensors)
        self.name = name
        if descriptor is not None and isinstance(self.output_tensors, LazyOutputList):
            if all(slot is not None for slot in self.output_tensors._slots):
                self.output_tensors._alloc_fn = None

        if program_cache_key is not None:
            self.program_cache_key = program_cache_key
        elif descriptor is not None:
            self.program_cache_key = ttnn.compute_program_descriptor_hash(descriptor)
        else:
            raise ValueError("Deferred OpDescriptor requires program_cache_key")

        self._updated_indices = []
        self._descriptor_output_buffers = self._tensor_buffers(self.output_tensors) if descriptor is not None else None
        self._descriptor_input_buffers = self._tensor_buffers(self.input_tensors) if descriptor is not None else None

    @classmethod
    def _persistent(cls, input_tensors, output_tensors, name, complete_fn, input_names):
        """Internal: construct a partial descriptor that materializes on first :meth:`update`."""
        if complete_fn is None:
            raise ValueError("_persistent() requires a complete_fn")
        obj = cls.__new__(cls)
        obj._descriptor = None
        obj._factory_fn = None
        obj._complete_fn = complete_fn
        if isinstance(input_tensors, dict):
            obj._input_names = {k: i for i, k in enumerate(input_tensors)}
            obj.input_tensors = list(input_tensors.values())
        else:
            obj.input_tensors = input_tensors if input_tensors is not None else []
            obj._input_names = dict(input_names) if input_names else None
        obj.output_tensors = output_tensors if output_tensors is not None else []
        obj.name = name
        obj.program_cache_key = None
        obj._updated_indices = []
        obj._descriptor_output_buffers = None
        obj._descriptor_input_buffers = None
        return obj

    def update(self, *args, **kwargs):
        """Replace input tensors by name or position.

        **Positional** (common single-activation case)::

            desc.update(new_q)

        **Keyword** (multi-input or self-documenting)::

            desc.update(input_tensor=new_q)
            desc.update(input_a=new_a, input_b=new_b)

        Updated positions are tracked in ``_updated_indices`` so that
        ``run()`` can clear them after dispatch (zero tensor pinning).
        """
        if args:
            if kwargs:
                raise ValueError("update(): positional OR keyword arguments, not both")
            if len(args) > len(self.input_tensors):
                raise ValueError(
                    f"update(): got {len(args)} positional args but "
                    f"descriptor only has {len(self.input_tensors)} input(s)"
                )
            updated = self._updated_indices
            for i, t in enumerate(args):
                self.input_tensors[i] = t
                updated.append(i)
        elif kwargs:
            names = self._input_names
            if names is None:
                raise ValueError(
                    "Keyword update requires named inputs. " "This OpDescriptor was created without input_names."
                )
            updated = self._updated_indices
            for name, t in kwargs.items():
                idx = names.get(name)
                if idx is None:
                    raise ValueError(
                        f"Unknown input name {name!r} for {self.name!r} op. " f"Valid names: {sorted(names)}"
                    )
                self.input_tensors[idx] = t
                updated.append(idx)

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
        previous = self.output_tensors
        bound = self._snapshot_output_bindings(previous)
        state = self._save_output_slots(previous)
        try:
            full = self._complete_fn(self.input_tensors)
            self._forward_pending_bindings(bound, full)
            descriptor = full.descriptor
            outputs = self._check_output_replacement(full.output_tensors, bound)
        except Exception:
            self._restore_output_slots(previous, state)
            raise
        self.program_cache_key = full.program_cache_key
        self._factory_fn = None
        self._descriptor = descriptor
        self.output_tensors = outputs
        self._descriptor_output_buffers = full._descriptor_output_buffers
        self._descriptor_input_buffers = self._tensor_buffers(self.input_tensors)
        self._complete_fn = None

    @staticmethod
    def _tensor_buffers(tensors):
        # Root IDs can be read without dereferencing descriptor CB pointers.
        # Host, freed and remote-only tensors have no local backing ID.
        slots = tensors._slots if isinstance(tensors, LazyOutputList) else tensors
        buffers = []
        for tensor in slots:
            root = tensor.buffer_unique_id() if isinstance(tensor, ttnn.Tensor) else None
            try:
                buffers.append((root, tensor.buffer_address()) if root is not None else None)
            except RuntimeError:
                # A remote-only view may have a live root but no local buffer.
                # Construction is metadata-only; cold launch validates it later.
                buffers.append(None)
        return tuple(buffers)

    def _validate_cold_outputs(self):
        """Cold launches use the buffers the descriptor was constructed with."""
        descriptor = self.descriptor
        if self._tensor_buffers(self.input_tensors) != self._descriptor_input_buffers:
            raise RuntimeError(
                "An update() after the descriptor was built changed an input buffer and cannot be rebuilt cold; "
                "construct a new descriptor. Cached run() can use updated inputs."
            )
        outputs = self.output_tensors
        slots = list(outputs)
        explicit = outputs._explicit_bindings if isinstance(outputs, LazyOutputList) else None
        current = self._tensor_buffers(self.output_tensors)
        for index, tensor in enumerate(slots):
            if tensor is None:
                raise RuntimeError(
                    "Cold fusion output is empty after cleanup or an unbound None assignment; "
                    "construct a new descriptor before rebuilding."
                )
            if isinstance(outputs, LazyOutputList) and (
                explicit is None or index >= len(explicit) or not explicit[index]
            ):
                continue
            if not isinstance(tensor, ttnn.Tensor):
                continue
            if current[index] is None:
                raise RuntimeError(
                    "Cold fusion output binding has no live local buffer (the tensor or its root may be freed); "
                    "construct a new descriptor with an allocated output tensor."
                )
            original = self._descriptor_output_buffers[index] if index < len(self._descriptor_output_buffers) else None
            if original is None:
                # Eager descriptors may fill an initially empty output slot.
                # Check raw references without dereferencing cached CB pointers.
                if ttnn._ttnn.operations.experimental._descriptor_references_tensor(descriptor, tensor):
                    continue
            if current[index] != original:
                raise RuntimeError(
                    "Cold fusion binding changed after the descriptor was built; this binding takes effect on cached runs. "
                    "For a cold run, bind the output before the op materializes "
                    "or construct a new descriptor for a different output tensor."
                )
        return descriptor

    @staticmethod
    def _snapshot_output_bindings(outputs):
        if isinstance(outputs, LazyOutputList):
            flags = outputs._explicit_bindings or ()
            return [
                (i, t)
                for i, t in enumerate(outputs._slots)
                if i < len(flags) and flags[i] and isinstance(t, ttnn.Tensor)
            ]
        return [(i, t) for i, t in enumerate(outputs) if isinstance(t, ttnn.Tensor)]

    @staticmethod
    def _save_output_slots(outputs):
        if isinstance(outputs, LazyOutputList):
            flags = outputs._explicit_bindings
            return (
                list(outputs._slots),
                None if flags is None else list(flags),
                outputs._any_explicit,
                outputs._alloc_fn,
            )
        return (list(outputs),)

    @staticmethod
    def _restore_output_slots(outputs, state):
        if isinstance(outputs, LazyOutputList):
            outputs._slots[:] = state[0]
            outputs._explicit_bindings, outputs._any_explicit, outputs._alloc_fn = state[1:]
        else:
            outputs[:] = state[0]

    @staticmethod
    def _forward_pending_bindings(bound, full):
        """Supply bindings before a deferred factory allocates its outputs."""
        target = full.output_tensors
        slots = target._slots if isinstance(target, LazyOutputList) else target
        for index, tensor in bound:
            if index >= len(slots):
                continue
            pending = slots[index]
            if isinstance(pending, _DeferredOutput) or (isinstance(target, LazyOutputList) and pending is None):
                target[index] = tensor

    @staticmethod
    def _factory_outputs(outputs):
        """A factory's assignments are allocations, not caller bindings."""
        if not isinstance(outputs, LazyOutputList):
            outputs = LazyOutputList(outputs)
        outputs._explicit_bindings = None
        outputs._any_explicit = False
        return outputs

    @staticmethod
    def _check_output_replacement(outputs, bound):
        """Compare against bindings saved before the factory could mutate its list."""
        slots = outputs._slots if isinstance(outputs, LazyOutputList) else outputs
        if any(i >= len(slots) or slots[i] is not tensor for i, tensor in bound):
            raise RuntimeError(
                "The factory allocates its own outputs; bind output_tensors after the op's first run. "
                "Materialization would replace a binding."
            )
        outputs = OpDescriptor._factory_outputs(outputs)
        for index, tensor in bound:
            outputs[index] = tensor
        return outputs

    @property
    def descriptor(self):
        if self._descriptor is None:
            previous = self.output_tensors
            bound = self._snapshot_output_bindings(previous)
            state = self._save_output_slots(previous)
            try:
                result = self._factory_fn()
                descriptor, outputs = result if isinstance(result, tuple) else (result, previous)
                outputs = self._check_output_replacement(outputs, bound)
            except Exception:
                self._restore_output_slots(previous, state)
                raise
            self.program_cache_key = _fold_bound_output_specs(self.program_cache_key, bound)
            if all(slot is not None for slot in outputs._slots):
                outputs._alloc_fn = None
            self.output_tensors = outputs
            self._descriptor = descriptor
            self._factory_fn = None
            self._descriptor_output_buffers = self._tensor_buffers(self.output_tensors)
            self._descriptor_input_buffers = self._tensor_buffers(self.input_tensors)
        return self._descriptor

    @property
    def is_deferred(self) -> bool:
        """True if the C++ factory has not yet run."""
        return self._descriptor is None

    def launch(self):
        """Dispatch via ``generic_op`` (materializes ``descriptor`` if needed)."""
        d = self._validate_cold_outputs()
        io_tensors = list(self.input_tensors) + list(self.output_tensors)
        ttnn.generic_op(io_tensors, d)
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

            # Per-factory program key cache (LRU): maps a cheap argument
            # fingerprint to a previously computed (program_cache_key,
            # input_name_map, input_names, arg_refs) tuple, skipping the
            # full factory body on repeated inline calls.  Bounded by
            # _PROGRAM_KEY_CACHE_MAX to prevent unbounded growth.
            #
            # Keying: tensors → hash(spec); hashable args → hash(obj);
            # id()-only args → id(obj) + strong ref pinning (see
            # _content_hash / _inline_cache_key_and_refs).
            #
            # Accepted tradeoff: the fingerprint is a proxy for
            # program_cache_key (computed by C++ compute_program_hash).
            # A fingerprint collision on args that produce *different*
            # program_cache_keys would return the wrong cached key —
            # a silent correctness issue.  This requires (1) a hash
            # collision in the fingerprint AND (2) the colliding args
            # to actually change the compiled program.  In practice
            # this is near-impossible for typical decode loops where
            # shapes and configs are fixed.  The tradeoff eliminates
            # a C++ round-trip per inline call (~40 µs).
            _program_key_cache: OrderedDict = OrderedDict()
            _PROGRAM_KEY_CACHE_REGISTRY.append(_program_key_cache)

            def _is_pending(val):
                return val is None or isinstance(val, _DeferredOutput)

            def _content_hash(v):
                """Content-based hash when the type provides one, else ``None``.

                Returns ``None`` for types with only ``object.__hash__``
                (id-based, unsafe across allocator reuse) or unhashable
                types (``__hash__ = None``).  Those args fall back to
                ``id()`` with strong-reference pinning.
                """
                h = type(v).__hash__
                if h is object.__hash__ or h is None:
                    return None
                try:
                    return hash(v)
                except TypeError:
                    return None

            def _inline_cache_key_and_refs(args, kwargs):
                """Cheap fingerprint + list of objects that need pinning.

                Tensors use ``hash(tensor.spec)`` (content-based).  Other
                args use ``hash()`` if the type provides a content-based
                ``__hash__``, otherwise ``id()`` — and those objects are
                collected into ``refs`` for strong-reference pinning in
                the cache entry.

                Iterates kwargs in signature-defined ``param_names`` order
                (deterministic, avoids ``sorted()`` per call).

                Returns ``(key_tuple, refs_tuple)``.
                """
                parts = []
                refs = []
                for i, a in enumerate(args):
                    if isinstance(a, ttnn.Tensor):
                        parts.append((i, hash(a.spec)))
                    else:
                        h = _content_hash(a)
                        if h is not None:
                            parts.append((i, h))
                        else:
                            parts.append((i, id(a)))
                            refs.append(a)
                for k in param_names:
                    if k in kwargs:
                        v = kwargs[k]
                        if isinstance(v, ttnn.Tensor):
                            parts.append((k, hash(v.spec)))
                        else:
                            h = _content_hash(v)
                            if h is not None:
                                parts.append((k, h))
                            else:
                                parts.append((k, id(v)))
                                refs.append(v)
                return tuple(parts), tuple(refs)

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
                    # ── Program key cache: skip full factory on inline hit ──
                    #
                    # The cache key uses a three-tier strategy:
                    #   Tensor       → hash(tensor.spec)   content-based
                    #   Hashable arg → hash(obj)            content-based
                    #   Other arg    → id(obj)              pinned via strong
                    #                                       ref in cache entry
                    #                                       to prevent address
                    #                                       reuse
                    #
                    # ``refs`` collects the id()-keyed objects; stored
                    # alongside the cached value so they stay alive (and
                    # their addresses valid) until the entry is LRU-evicted.
                    ck, refs = _inline_cache_key_and_refs(args, kwargs)
                    hit = _program_key_cache_get(_program_key_cache, ck)
                    if hit is not None:
                        program_cache_key, input_name_map, input_names_tuple, _refs = hit
                        tensors = []
                        for _pname, src in input_name_map:
                            if isinstance(src, int):
                                tensors.append(args[src])
                            else:
                                tensors.append(kwargs[src])

                        shell_outputs = LazyOutputList([_DeferredOutput()])

                        def _lazy_factory():
                            bound = OpDescriptor._snapshot_output_bindings(shell_outputs)
                            full = fn(*args, **kwargs)
                            full.output_tensors = OpDescriptor._factory_outputs(full.output_tensors)
                            OpDescriptor._forward_pending_bindings(bound, full)
                            return full.descriptor, full.output_tensors

                        shell = OpDescriptor(
                            factory_fn=_lazy_factory,
                            input_tensors=tensors,
                            output_tensors=shell_outputs,
                            name=name,
                            program_cache_key=program_cache_key,
                            input_names=input_names_tuple,
                        )
                        return shell

                    # Cache miss — full construction.  The decorator
                    # owns the name; override whatever the factory set.
                    desc = fn(*args, **kwargs)
                    desc.output_tensors = OpDescriptor._factory_outputs(desc.output_tensors)
                    if name:
                        desc.name = name

                    # Populate cache for next call.
                    input_name_map = []
                    if desc._input_names is not None:
                        for pname, _idx in desc._input_names.items():
                            if pname in kwargs:
                                input_name_map.append((pname, pname))
                            else:
                                for pi, p in enumerate(param_names):
                                    if p == pname and pi < len(args):
                                        input_name_map.append((pname, pi))
                                        break
                    input_names_tuple = tuple((pname, idx) for idx, (pname, _) in enumerate(input_name_map))

                    _program_key_cache_put(
                        _program_key_cache, ck, (desc.program_cache_key, input_name_map, input_names_tuple, refs)
                    )
                    return desc

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

                return OpDescriptor._persistent(
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

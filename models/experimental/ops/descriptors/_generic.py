# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Generic descriptor framework — ``expose(DeviceOp)`` one-liner.

Eliminates per-op boilerplate when adding a DeviceOperation to the
descriptors library.  For simple ops (single factory, single output,
3-arg ``create_descriptor``), usage is::

    unary = expose(ttnn.UnaryDeviceOperation)

For ops that need customization (multi-factory, preprocessor, etc.)::

    matmul = expose(
        ttnn.MatmulDeviceOperation,
        output_is_vector=True,
        params_preprocessor=lambda p, i: ttnn.create_matmul_attributes(...),
    )

The returned callable supports **inline** and **persistent** modes
(see :class:`OpDescriptor`).  Inline calls produce a deferred
``OpDescriptor`` with lazy output allocation; persistent calls
(required tensor inputs are ``None``) produce a descriptor that
materializes on first :meth:`~OpDescriptor.update`.
"""

import re
import inspect
from typing import Callable, Optional, Sequence

import ttnn

from models.experimental.ops.descriptors.op_descriptor import (
    OpDescriptor,
    LazyOutputList,
    _DeferredOutput,
    extend_branch_program_cache_key,
)


# ---------------------------------------------------------------------------
# Type and field discovery
# ---------------------------------------------------------------------------

_NB_TYPE_RE = re.compile(r"\w+\(\w+:\s*([\w.]+),\s*\w+:\s*([\w.]+)")


def _resolve_dotted(name):
    """Resolve ``'ttnn._ttnn.operations.matmul.MatmulParams'`` to the class."""
    parts = name.split(".")
    try:
        obj = __import__(parts[0])
    except ImportError:
        return None
    for part in parts[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _types_from_nb_doc(method):
    """Extract param/input types from a nanobind ``__doc__`` string.

    nanobind methods have ``__doc__`` like::

        compute_program_hash(
            operation_attributes: ttnn._ttnn.operations.matmul.MatmulParams,
            tensor_args: ttnn._ttnn.operations.matmul.MatmulInputs) -> int
    """
    doc = getattr(method, "__doc__", None)
    if not doc:
        return None, None
    m = _NB_TYPE_RE.search(doc)
    if not m:
        return None, None
    return _resolve_dotted(m.group(1)), _resolve_dotted(m.group(2))


def _discover_types(device_op, params_override, inputs_override):
    """Extract params/inputs types from bound method signatures.

    Strategy (tried in order for each candidate method):

    1. ``inspect.signature()`` — works for pure-Python functions and some
       nanobind versions.
    2. nanobind ``__doc__`` parsing — nanobind v2 generates docstrings
       with fully-qualified type names.

    Methods tried: ``compute_program_hash``, ``compute_output_specs``,
    ``create_output_tensors``.
    """
    if params_override is not None and inputs_override is not None:
        return params_override, inputs_override

    for method_name in ("compute_program_hash", "compute_output_specs", "create_output_tensors"):
        method = getattr(device_op, method_name, None)
        if method is None:
            continue

        # Strategy 1: inspect.signature (works for regular Python functions)
        try:
            sig = inspect.signature(method)
            sig_params = list(sig.parameters.values())
            if len(sig_params) >= 2:
                p_cls = sig_params[0].annotation
                i_cls = sig_params[1].annotation
                if p_cls is not inspect.Parameter.empty and i_cls is not inspect.Parameter.empty:
                    return params_override or p_cls, inputs_override or i_cls
        except (ValueError, TypeError):
            pass

        # Strategy 2: nanobind __doc__ parsing
        p_cls, i_cls = _types_from_nb_doc(method)
        if p_cls is not None and i_cls is not None:
            return params_override or p_cls, inputs_override or i_cls

    raise TypeError(
        f"Cannot auto-discover param/input types for {device_op.__name__}. "
        f"Pass params_class= and inputs_class= explicitly."
    )


def _discover_fields(cls):
    """Discover settable field names on a nanobind-bound class.

    Tries constructing a default instance first (works for value-member
    structs).  Falls back to class-level introspection for
    reference-member structs that require a constructor argument.
    """
    try:
        instance = cls()
    except TypeError:
        instance = None

    target = instance if instance is not None else cls
    fields = set()
    for attr_name in dir(target):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(target, attr_name)
        except Exception:
            continue
        if callable(attr):
            continue
        fields.add(attr_name)
    return fields


def _auto_build_inputs(inputs_class, tensor_kwargs, inputs_fields):
    """Construct an inputs object from tensor kwargs.

    Value-member structs: default-construct + setattr.
    Reference-member structs: pass first tensor to constructor, setattr rest.
    """
    try:
        inputs = inputs_class()
        for k, v in tensor_kwargs.items():
            if v is not None and k in inputs_fields:
                setattr(inputs, k, v)
        return inputs
    except TypeError:
        first_key = next(iter(tensor_kwargs))
        inputs = inputs_class(tensor_kwargs[first_key])
        for k, v in tensor_kwargs.items():
            if k != first_key and v is not None and k in inputs_fields:
                setattr(inputs, k, v)
        return inputs


def _derive_name(device_op):
    """Derive a snake_case descriptor name from a DeviceOperation class name."""
    raw = device_op.__name__
    for suffix in ("DeviceOperation", "Operation"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            break
    result = []
    for i, c in enumerate(raw):
        if c.isupper() and i > 0:
            result.append("_")
        result.append(c.lower())
    return "".join(result)


def _is_pending(val):
    return val is None or isinstance(val, _DeferredOutput)


# ---------------------------------------------------------------------------
# expose()
# ---------------------------------------------------------------------------


def expose(
    device_op,
    *,
    name: Optional[str] = None,
    params_class=None,
    inputs_class=None,
    required_inputs: Optional[Sequence[str]] = None,
    num_outputs: int = 1,
    output_is_vector: bool = False,
    select_factory: Optional[Callable] = None,
    factory=None,
    params_preprocessor: Optional[Callable] = None,
    validate: Optional[Callable] = None,
    extra_cache_key_fn: Optional[Callable] = None,
    factory_call_override: Optional[Callable] = None,
    inputs_builder: Optional[Callable] = None,
    param_defaults: Optional[dict] = None,
):
    """Create a descriptor function for a DeviceOperation.

    For simple ops this is a true one-liner::

        unary = expose(ttnn.UnaryDeviceOperation)

    Args:
        device_op: Bound DeviceOperation class (e.g. ``ttnn.MatmulDeviceOperation``).
        name: Descriptor name. Derived from *device_op* class name if omitted.
        params_class: Override for the operation-attributes type.
            Auto-discovered from ``compute_program_hash`` signature if omitted.
        inputs_class: Override for the tensor-args type.
            Auto-discovered from ``compute_program_hash`` signature if omitted.
        required_inputs: Tensor input names that must be provided for inline
            mode.  When any are ``None`` at call time, a persistent descriptor
            is returned instead.  Defaults to all discovered input fields.
        num_outputs: Number of output tensor slots (default 1).
        output_is_vector: Pass outputs as a ``list`` to ``create_descriptor``
            (needed for ops returning ``vector<Tensor>``).
        select_factory: Callable ``(params, inputs) -> factory``.  Uses
            ``device_op.select_program_factory`` if not provided.
        factory: Single factory class with ``create_descriptor``.  Use when
            the op has one factory and no ``select_program_factory``.
        params_preprocessor: Callable ``(params, inputs) -> params`` applied
            after params construction (e.g. ``create_matmul_attributes``).
        validate: Callable ``(**kwargs)`` for pre-construction validation.
        extra_cache_key_fn: Callable ``(**kwargs) -> tuple`` of extra
            hashable values mixed into the program cache key.
        factory_call_override: Callable
            ``(factory, params, inputs, output, **extras) -> descriptor``.
            Overrides the default ``factory.create_descriptor(params, inputs, output)``.
        inputs_builder: Callable ``(**tensor_kwargs) -> inputs``.
            Overrides auto-construction of the inputs object.
        param_defaults: Dict mapping field names to default values or
            callables ``(all_kwargs) -> value``.

    Returns:
        A callable that produces :class:`OpDescriptor` instances, supporting
        both inline and persistent construction modes.
    """
    # -- Type discovery --
    p_cls, i_cls = _discover_types(device_op, params_class, inputs_class)

    # -- Field discovery --
    p_fields = _discover_fields(p_cls)
    i_fields = _discover_fields(i_cls)

    # -- Factory resolution --
    _select_factory = select_factory
    _factory = factory
    if _select_factory is None and _factory is None:
        sf = getattr(device_op, "select_program_factory", None)
        if sf is not None:
            _select_factory = sf

    # -- Hash availability (detected once at expose-time) --
    _has_hash = hasattr(device_op, "compute_program_hash")

    # -- Required / optional inputs --
    req = list(required_inputs) if required_inputs is not None else sorted(i_fields)
    opt_inputs = sorted(i_fields - set(req))

    # -- Name --
    op_name = name or _derive_name(device_op)

    # Ordered param names for positional arg resolution
    _ordered = list(req) + opt_inputs + sorted(p_fields - i_fields)

    # ------------------------------------------------------------------
    # The descriptor function
    # ------------------------------------------------------------------

    def descriptor_fn(*args, **kwargs):
        # Merge positional args into kwargs
        all_kw = dict(kwargs)
        for i, val in enumerate(args):
            if i < len(_ordered):
                all_kw[_ordered[i]] = val

        # ── Persistent mode check ──
        if any(_is_pending(all_kw.get(k)) for k in req):
            return _build_persistent(all_kw)

        # ── Inline mode — full construction ──
        return _build_inline(all_kw)

    def _build_inline(all_kw):
        if validate:
            validate(**all_kw)

        # Partition kwargs into tensor inputs, params fields, extras.
        # None-valued tensor inputs are dropped — not set on the C++ inputs
        # object and not tracked in OpDescriptor.input_tensors.
        tensor_kw = {}
        param_kw = {}
        extra_kw = {}
        for k, v in all_kw.items():
            if k in i_fields:
                if v is not None:
                    tensor_kw[k] = v
            elif k in p_fields:
                param_kw[k] = v
            else:
                extra_kw[k] = v

        # Determine device
        device = None
        for v in tensor_kw.values():
            if isinstance(v, ttnn.Tensor):
                device = v.device()
                break
        if device is None:
            raise ValueError(f"{op_name}: no Tensor input found — cannot determine device")

        # Build params
        params = p_cls()
        defaults = param_defaults or {}
        for field, value in param_kw.items():
            if value is None and field in defaults:
                d = defaults[field]
                value = d(all_kw) if callable(d) else d
            if value is not None:
                setattr(params, field, value)

        # Build inputs
        if inputs_builder:
            inputs = inputs_builder(**tensor_kw)
        else:
            inputs = _auto_build_inputs(i_cls, tensor_kw, i_fields)

        # Preprocess params
        if params_preprocessor:
            params = params_preprocessor(params, inputs)

        def _resolve_factory():
            if _select_factory is not None:
                return _select_factory(params, inputs)
            if _factory is not None:
                return _factory
            raise ValueError(f"{op_name}: no factory available. Pass factory= or select_factory= to expose().")

        def _call_factory(fct, out):
            if factory_call_override:
                return factory_call_override(fct, params, inputs, out, **extra_kw)
            return fct.create_descriptor(params, inputs, out)

        if _has_hash:
            # Deferred path — lazy output allocation, factory runs on first .descriptor access
            h = device_op.compute_program_hash(params, inputs)
            extras = extra_cache_key_fn(**all_kw) if extra_cache_key_fn else ()
            if extras:
                program_cache_key = extend_branch_program_cache_key(h, *extras)
            else:
                program_cache_key = int(h) & ((1 << 64) - 1)

            def _alloc(slots):
                result = device_op.create_output_tensors(params, inputs)
                if output_is_vector:
                    for idx, t in enumerate(result):
                        slots[idx] = t
                else:
                    slots[0] = result

            outputs = LazyOutputList([None] * num_outputs, _alloc)

            def _run_factory():
                fct = _resolve_factory()
                out = list(outputs) if output_is_vector else outputs[0]
                return _call_factory(fct, out)

            return OpDescriptor(
                factory_fn=_run_factory,
                input_tensors=tensor_kw,
                output_tensors=outputs,
                program_cache_key=program_cache_key,
            )
        else:
            # Eager path — compute_program_hash not available; create outputs
            # and run factory immediately, derive cache key from the descriptor.
            result = device_op.create_output_tensors(params, inputs)
            if output_is_vector:
                output_list = list(result)
            else:
                output_list = [result]

            fct = _resolve_factory()
            out = output_list if output_is_vector else output_list[0]
            descriptor = _call_factory(fct, out)

            return OpDescriptor(
                descriptor=descriptor,
                input_tensors=tensor_kw,
                output_tensors=output_list,
            )

    def _build_persistent(all_kw):
        # Collect tensor inputs (pending required + present optional)
        input_names = []
        inputs = []
        idx = 0
        for pname in req:
            val = all_kw.get(pname)
            input_names.append((pname, idx))
            inputs.append(val)
            idx += 1
        for pname in opt_inputs:
            val = all_kw.get(pname)
            if isinstance(val, ttnn.Tensor) or isinstance(val, _DeferredOutput):
                input_names.append((pname, idx))
                inputs.append(val)
                idx += 1

        def _complete(final_inputs):
            kw = dict(all_kw)
            for (pname, _), tensor in zip(input_names, final_inputs):
                kw[pname] = tensor
            return _build_inline(kw)

        return OpDescriptor._persistent(
            input_tensors=inputs,
            output_tensors=[_DeferredOutput() for _ in range(num_outputs)],
            name=op_name,
            complete_fn=_complete,
            input_names=tuple(input_names),
        )

    descriptor_fn.__name__ = op_name
    descriptor_fn.__qualname__ = op_name
    return descriptor_fn


__all__ = ["expose"]

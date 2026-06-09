# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import inspect
import weakref
from types import NoneType
from typing import TYPE_CHECKING, Any, overload

import torch
from loguru import logger

import ttnn
from models.tt_dit.utils import tensor

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import ClassVar

_OMITTED = object()


class StateTensor:
    """Helper class to store the state of a tensor during tracing.
    This is mostly useful for tensors that are not part of the trace inputs or outputs.
    """

    def __init__(self) -> None:
        self._data: ttnn.Tensor | None = None

    @property
    def data(self) -> ttnn.Tensor | None:
        return self._data

    def update(
        self,
        value: torch.Tensor | ttnn.Tensor,
        traced: bool,
        dtype: ttnn.DataType = ttnn.bfloat16,
        mesh_axes: list[int] | None = None,
        device: ttnn.Device | None = None,
    ) -> None:
        """Update the state tensor with a new value.
           If the tensor is used during tracing, the value is copied to (or used to initalize) the underlying tensor.

        Args:
            value: The new value to update the state tensor with.
            traced: Whether the tensor is used during tracing.
            dtype: The data type of the value.
            mesh_axes: The mesh axes of the value.
            device: The device of the value.
        """
        if torch.is_tensor(value):
            assert device is not None, "device must be provided if using torch tensor"
            value = tensor.from_torch(value, device=device, mesh_axes=mesh_axes, dtype=dtype)
        if self._data is None or not traced:
            self._data = value
        else:
            ttnn.copy(value, self._data)


class Tracer:
    """Wrapper for capturing and executing a trace of a given function.

    All inputs and outputs of the traced function must be ``ttnn.Tensor`` instances or plain
    Python scalars (``int``, ``float``, ``str``, ``bool``, ``None``), optionally nested in
    tuples, lists, or dicts.

    Important caveats:

    1. Tensors allocated after trace capture may be overwritten during trace execution.
       Host tensors are not affected. Input tensors are copied before trace execution, so
       they can safely be allocated on device if their content is not needed after execution.

    2. The tracer returns the same output tensor objects every time; a subsequent call
       overwrites previous results in place.
    """

    _traces_live: ClassVar[dict[int, int]] = {}

    @overload
    def __init__(
        self,
        function: Callable[..., Any],
        /,
        *,
        device: ttnn.MeshDevice,
        prep_run: bool = True,
        clone_prep_inputs: bool = True,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        function: Callable[..., Any],
        /,
        *,
        devices: Sequence[ttnn.MeshDevice],
        prep_run: bool = True,
        clone_prep_inputs: bool = True,
    ) -> None:
        ...

    def __init__(
        self,
        function: Callable[..., Any],
        /,
        *,
        device: ttnn.MeshDevice | None = None,
        devices: Sequence[ttnn.MeshDevice] | None = None,
        prep_run: bool = True,
        clone_prep_inputs: bool = True,
    ) -> None:
        """Initialize the tracer.

        If the function modifies its input tensors in place, set ``clone_prep_inputs`` to ``True``
        so that preparation runs operate on cloned inputs, leaving the originals intact for trace
        capture.

        Exactly one of ``device`` or ``devices`` must be provided. When ``devices`` is given, a
        trace is captured and executed on each device simultaneously.

        Args:
            function: Function to be traced.
            device: Single device on which to capture and execute the trace.
            devices: Multiple devices on which to capture and execute the trace simultaneously.
            prep_run: Whether to run the function once before capturing the trace.
            clone_prep_inputs: Whether to clone tensor inputs for the preparation run.
        """
        if (device is None) == (devices is None):
            msg = "exactly one of 'device' or 'devices' must be provided"
            raise ValueError(msg)

        if devices is None:
            assert device is not None
            devices = (device,)

        for d in devices:
            if d.id() not in Tracer._traces_live:
                Tracer._traces_live[d.id()] = 0

        self._function = function
        self._devices: tuple[ttnn.MeshDevice, ...] = tuple(devices)
        self._prep_run = prep_run
        self._clone_prep_inputs = clone_prep_inputs
        self._args: tuple[Any, ...] = ()
        self._kwargs: dict[str, Any] = {}
        self._outputs: Any = None
        self._trace_ids: tuple[ttnn.MeshTraceId, ...] | None = None

    def __call__(
        self,
        *args: Any,
        traced: bool = True,
        tracer_cq_id: int = 0,
        tracer_blocking_execution: bool = True,
        tracer_execute_on_capture: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Capture or execute trace.

        In traced mode, the first call captures the trace and subsequent calls execute it. The
        first call's inputs initialize the stored input state; subsequent calls must pass the
        same number of positional arguments and the same set of keyword argument names. Tensor
        inputs are validated for shape/dtype/layout match; non-tensor inputs (scalars, ``None``)
        must compare equal to the captured values. Host tensor inputs are automatically moved
        to the tracer device.

        Args:
            traced: Whether to capture/execute the trace on this call. If ``False``, the wrapped
                function is called directly without tracing.
            tracer_cq_id: Command queue id.
            tracer_blocking_execution: Whether ``ttnn.execute_trace`` should block.
            tracer_execute_on_capture: Whether to execute the trace immediately after capturing it
                on the first call. If ``False``, only the trace is captured and outputs are not
                computed.
            *args: Positional inputs to pass to the wrapped function.
            **kwargs: Named inputs to pass to the wrapped function.

        Returns:
            The outputs of the wrapped function.

        Raises:
            TypeError: If outputs have unsupported types, or if subsequent traced calls do not
                match the captured arity/keyword set.
            Any exception raised by the wrapped function during first invocation.
        """
        if not traced:
            if self._function is None:
                msg = "untraced execution is not possible after the captured function has been released"
                raise RuntimeError(msg)
            if self._trace_ids is not None:
                msg = "untraced execution is not allowed after the trace has been captured"
                raise RuntimeError(msg)

            self._args = args
            self._kwargs = kwargs
            return self._function(*args, **kwargs)

        if tracer_blocking_execution and len(self._devices) != 1:
            tracer_blocking_execution = False
            logger.warning("blocking execution is not supported with multiple devices")

        if self._trace_ids is None:
            self._capture(
                args,
                kwargs,
                cq_id=tracer_cq_id,
                blocking_execution=tracer_blocking_execution,
                execute=tracer_execute_on_capture,
            )
        else:
            self._execute(args, kwargs, cq_id=tracer_cq_id, blocking=tracer_blocking_execution)

        return self._outputs

    def _capture(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        cq_id: int,
        blocking_execution: bool,
        execute: bool,
    ) -> None:
        if self._function is None:
            msg = "tracer cannot be reused after the trace was released"
            raise RuntimeError(msg)

        args = _tree_map(_verify_value, args, path_label="args")
        kwargs = _tree_map(_verify_value, kwargs, path_label="kwargs")
        self._args = _tree_map(self._tensor_to_device, args, path_label="args")
        self._kwargs = _tree_map(self._tensor_to_device, kwargs, path_label="kwargs")

        if self._prep_run:
            if self._clone_prep_inputs:
                prep_args = _tree_map(_clone_tensor, self._args, path_label="args")
                prep_kwargs = _tree_map(_clone_tensor, self._kwargs, path_label="kwargs")
            else:
                prep_args = self._args
                prep_kwargs = self._kwargs

            self._function(*prep_args, **prep_kwargs)
            del prep_args, prep_kwargs

        # capture trace
        logger.debug("capturing trace...")
        trace_ids = tuple(ttnn.begin_trace_capture(d, cq_id=cq_id) for d in self._devices)
        try:
            try:
                outputs = self._function(*self._args, **self._kwargs)
            finally:
                for d, trace_id in zip(self._devices, trace_ids, strict=True):
                    ttnn.end_trace_capture(d, trace_id, cq_id=cq_id)

            outputs = _tree_map(_verify_value, outputs, path_label="outputs")
        except Exception:
            for d, trace_id in zip(self._devices, trace_ids, strict=True):
                ttnn.release_trace(d, trace_id)
            raise

        for d in self._devices:
            Tracer._traces_live[d.id()] += 1
        self._trace_ids = trace_ids
        self._outputs = outputs

        if execute:
            # Trace capture records commands but does not execute them. Execute the trace to
            # actually compute outputs.
            for d, trace_id in zip(self._devices, trace_ids, strict=True):
                ttnn.execute_trace(d, trace_id, cq_id=cq_id, blocking=blocking_execution)

    def _execute(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        cq_id: int,
        blocking: bool,
    ) -> None:
        trace_ids = self._trace_ids
        assert trace_ids is not None

        if len(args) != len(self._args):
            msg = f"expected {len(self._args)} positional args, got {len(args)}"
            raise TypeError(msg)

        if kwargs.keys() != self._kwargs.keys():
            msg = f"expected kwargs {sorted(self._kwargs)}, got {sorted(kwargs)}"
            raise TypeError(msg)

        _tree_map(self._update_input, self._args, args, path_label="args")
        for name, new in kwargs.items():
            _tree_map(self._update_input, self._kwargs[name], new, path_label=f'kwargs["{name}"]')

        for d, trace_id in zip(self._devices, trace_ids, strict=True):
            ttnn.execute_trace(d, trace_id, cq_id=cq_id, blocking=blocking)

    @property
    def trace_captured(self) -> bool:
        """Whether a trace has been captured and is ready for execution."""
        return self._trace_ids is not None

    @property
    def inputs(self) -> dict[int | str, Any]:
        """Stored inputs from the most recent call (empty before the first call).

        Keyed by parameter name (``str``) for keyword inputs, or by index (``int``) for positional
        inputs. After the trace has been captured, tensor entries are the trace's input buffers, so
        the reference is a safe handle to the latest value of that input, which is useful when the
        caller needs to consume an input after trace execution. In untraced mode, tensor entries are
        simply the most recently passed references.
        """
        return {**dict(enumerate(self._args)), **self._kwargs}

    def release_trace(self) -> None:
        """Release the captured trace and clear inputs and outputs."""
        trace_ids = self._trace_ids
        if trace_ids is None:
            return

        self._trace_ids = None
        self._args = ()
        self._kwargs = {}
        self._outputs = None
        for d, trace_id in zip(self._devices, trace_ids, strict=True):
            Tracer._traces_live[d.id()] -= 1
            ttnn.release_trace(d, trace_id)

    def release_function(self) -> None:
        """Drop the reference to the wrapped function.

        This allows resources held by the function to be garbage collected.
        """
        self._function = None

    def _tensor_to_device(self, value: Any, *, path_label: str) -> Any:
        if not isinstance(value, ttnn.Tensor):
            return value

        if value.device() is None:
            if len(self._devices) != 1:
                msg = (
                    f"input '{path_label}' is a host tensor; host tensors are not supported "
                    "during capture when the tracer manages multiple devices"
                )
                raise ValueError(msg)
            return value.to(self._devices[0])

        if any(value.device() == d for d in self._devices):
            return value

        msg = f"input '{path_label}' device {value.device()} does not match any tracer device"
        raise ValueError(msg)

    def _update_input(self, prev: Any, new: Any, *, path_label: str) -> None:
        """Copy ``new`` into the slot ``prev`` in place."""
        if type(new) is not type(prev):
            msg = f"input '{path_label}' type {type(new)} does not match the initial type {type(prev)}"
            raise TypeError(msg)

        if isinstance(new, ttnn.Tensor):
            if new.shape != prev.shape or new.dtype != prev.dtype or new.layout != prev.layout:
                msg = f"input '{path_label}' tensor properties do not match the initial value"
                raise ValueError(msg)

            if new.device() is None:
                ttnn.copy_host_to_device_tensor(new, prev)
            else:
                if new.device() != prev.device():
                    msg = f"input '{path_label}' tensor device does not match the initial device"
                    raise ValueError(msg)

                if new.buffer_address() != prev.buffer_address():
                    ttnn.copy(new, prev)

        elif new != prev:
            msg = f"input '{path_label}' does not match the initial value"
            raise ValueError(msg)


_TRACER_VALID_INPUT_TYPES = (ttnn.Tensor, int, float, str, bool, NoneType)

_MESH_DEVICE_PARAM_NAMES: tuple[str, ...] = ("mesh_device", "device")
"""Parameter names that ``traced_function`` recognises as the Tracer's mesh device, in priority order."""


def _verify_value(value: Any, *, path_label: str) -> Any:
    if not isinstance(value, _TRACER_VALID_INPUT_TYPES):
        msg = f"value '{path_label}' has unsupported type {type(value)}"
        raise TypeError(msg)

    return value


def _is_tracer_valid_value(value: Any) -> bool:
    """Return True if ``value`` can be passed through to a ``Tracer`` as an input.

    Matches ``_TRACER_VALID_INPUT_TYPES`` plus any nesting of those inside ``tuple``,
    ``list``, or ``dict`` (the containers supported by ``Tracer._tree_map``). Used at
    call time by ``traced_function`` to classify each argument as either a tracer input
    or a bindable config value.
    """
    if isinstance(value, _TRACER_VALID_INPUT_TYPES):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_tracer_valid_value(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, _TRACER_VALID_INPUT_TYPES) and _is_tracer_valid_value(v) for k, v in value.items())
    return False


def _clone_tensor(value: Any, *, path_label: str) -> Any:
    """Clone a tensor, passing through non-tensor values unchanged."""
    del path_label
    return ttnn.clone(value) if isinstance(value, ttnn.Tensor) else value


def _tree_map(f: Callable[..., Any], x: Any, /, *xs: Any, path_label: str) -> Any:
    """Apply a function to leaves of nested data structures.

    Recursively traverses nested structures (tuples, lists, dicts) and applies
    the given function to corresponding leaf elements across all input structures.

    Args:
        f: A callable that takes N arguments, where N is the number of input
            structures (1 + len(xs)). Applied to leaf elements.
        x: The first nested data structure to traverse.
        *xs: Additional nested data structures with the same shape as x.
        path_label: String representing the current traversal path (used for error messages).

    Returns:
        A new nested structure with the same shape as the inputs, where each leaf has been
        transformed by applying f to the corresponding leaves from all input structures.

    Raises:
        TypeError: If the input structures don't have matching types at corresponding positions.
        ValueError: If tuples/lists have different lengths or dicts have different keys.
    """
    if not isinstance(x, (tuple, list, dict)):
        return f(x, *xs, path_label=path_label)

    for y in xs:
        if not isinstance(y, type(x)):
            msg = f"types of '{path_label}' should be the same: {type(x)} != {type(y)}"
            raise TypeError(msg)

    if isinstance(x, tuple) and all(isinstance(y, tuple) for y in xs):
        for y in xs:
            if len(x) != len(y):
                msg = f"tuple lengths of '{path_label}' should be the same: {len(x)} != {len(y)}"
                raise ValueError(msg)

        return tuple(
            _tree_map(f, *elts, path_label=f"{path_label}[{i}]") for i, elts in enumerate(zip(x, *xs, strict=True))
        )

    if isinstance(x, list) and all(isinstance(y, list) for y in xs):
        for y in xs:
            if len(x) != len(y):
                msg = f"list lengths of '{path_label}' should be the same: {len(x)} != {len(y)}"
                raise ValueError(msg)

        return [_tree_map(f, *elts, path_label=f"{path_label}[{i}]") for i, elts in enumerate(zip(x, *xs, strict=True))]

    if isinstance(x, dict) and all(isinstance(y, dict) for y in xs):
        for y in xs:
            if x.keys() != y.keys():
                msg = f"dict keys of '{path_label}' should be the same: {x.keys()} != {y.keys()}"
                raise ValueError(msg)
        return {key: _tree_map(f, *(d[key] for d in (x, *xs)), path_label=f'{path_label}["{key}"]') for key in x}

    raise AssertionError  # unreachable


_TRACER_CALL_KWARGS = frozenset(
    name
    for name, param in inspect.signature(Tracer.__call__).parameters.items()
    if param.kind == inspect.Parameter.KEYWORD_ONLY
)


def traced_function(
    _fn: Callable[..., Any] | None = None,
    *,
    device: ttnn.MeshDevice | Callable[..., ttnn.MeshDevice] | None = None,
    inject_mesh_device: bool = False,
    prep_run: bool = True,
    clone_prep_inputs: bool = True,
) -> Any:
    """Decorator that adds optional tracing to any function or method via the ``Tracer`` class.

    Can be applied with or without arguments::

        # Method — device resolved lazily via callable from the bound context (self):
        @traced_function(device=lambda self: self.mesh_device, clone_prep_inputs=False)
        def my_method(self, x: ttnn.Tensor, scale: float) -> ttnn.Tensor: ...

        # Standalone function — device and other non-tracer-valid args are classified
        # dynamically on the first traced call. The parameter must be named `mesh_device`:
        @traced_function(clone_prep_inputs=False)
        def my_function(x, mesh_device, ccl_manager=None, pre_transfer_fn=None): ...

        # Standalone function — `mesh_device` is auto-injected as a wrapper-only kwarg
        # (like `traced`) and never forwarded to the wrapped function:
        @traced_function(inject_mesh_device=True, clone_prep_inputs=False)
        def my_pure_function(x: ttnn.Tensor, scale: float) -> ttnn.Tensor: ...

        # Call without tracing (original function, no overhead):
        result = my_function(x, mesh_device=md)
        result = my_pure_function(x, scale=1.0)
        result = model.my_method(x, scale=1.0)

        # Call with tracing (captures on first call, replays on subsequent calls):
        result = my_function(x, mesh_device=md, traced=True)
        result = my_pure_function(x, scale=1.0, mesh_device=md, traced=True)
        result = model.my_method(x, scale=1.0, traced=True)

        # With optional Tracer call-time kwargs:
        result = my_function(x, mesh_device=md, traced=True, tracer_cq_id=1, tracer_blocking_execution=False)

    The decorated callable gains a ``traced`` keyword argument at call time. When
    ``traced=False`` (the default) the original function is called directly. When
    ``traced=True`` a ``Tracer`` is lazily created on the first call and subsequent
    calls execute the captured trace.

    Three modes for supplying the Tracer's device:

    1. **Context-bound** — ``device=<callable>`` (or literal). The first
       positional argument is treated as a bindable context (typical for
       methods: ``self``), bound away via ``functools.partial``, and — when
       ``device`` is callable — passed through it to resolve the device. One
       ``Tracer`` per unique context in a ``WeakKeyDictionary``.
    2. **Auto-discovered** — ``device=None`` (omitted). The function must
       declare a parameter literally named ``mesh_device`` or ``device``
       (first match in that priority order). On the first traced call,
       each argument is classified at runtime: any value that isn't a
       ``Tracer``-valid input (``ttnn.Tensor``, scalar, ``None``, or a
       nested ``tuple``/``list``/``dict`` of those) is bound into a
       ``functools.partial``. The bind set is frozen after the first call
       and reused. The matched mesh parameter drives the ``Tracer``'s
       device. One ``Tracer`` is cached per unique tuple of bound-value
       identities.
    3. **Injected** — ``inject_mesh_device=True``. The wrapper accepts
       either ``mesh_device=`` or ``device=`` as an auto-added kwarg
       (analogous to ``traced=``), consumes it, and never forwards it to
       the wrapped function. Exactly one of the two may be supplied per
       call. The function itself must *not* declare a ``mesh_device`` or
       ``device`` parameter. Other non-tracer-valid args from the wrapped
       function's signature are still classified and bound on the first
       traced call, exactly as in mode (2).

    Decoration-time validation:
      - ``device=`` and ``inject_mesh_device=True`` are mutually exclusive.
      - ``inject_mesh_device=True`` requires the wrapped function *not* to
        declare a ``mesh_device`` or ``device`` parameter (to avoid ambiguity).
      - Omitting ``device=`` with ``inject_mesh_device=False`` on a function
        that declares neither ``mesh_device`` nor ``device`` raises ``ValueError``.

    Tracer call-time kwargs (``tracer_cq_id``, ``tracer_blocking_execution``,
    ``tracer_execute_on_capture``) are forwarded to the ``Tracer`` when tracing
    and stripped before calling the original function in the untraced path.

    Args:
        _fn: The function to wrap when used without parentheses (``@traced_function``).
        device: Device for tracing. For methods, pass a callable
            (e.g. ``lambda self: self.mesh_device``) so it can be resolved lazily
            from the bound context. Omit entirely for standalone functions that
            declare a ``mesh_device`` parameter or set ``inject_mesh_device=True``.
        inject_mesh_device: If ``True``, the wrapper accepts ``mesh_device=`` as an
            auto-added kwarg and consumes it without forwarding to the wrapped
            function. Mutually exclusive with ``device=``.
        prep_run: Forwarded to ``Tracer.__init__``.
        clone_prep_inputs: Forwarded to ``Tracer.__init__``.
    """

    def _resolve_device(context: Any) -> ttnn.MeshDevice:
        if device is None:
            msg = (
                "device= must be provided to @traced_function. "
                "Pass a ttnn.MeshDevice directly, or a callable that accepts the bound context "
                "and returns one (e.g. device=lambda self: self.mesh_device)."
            )
            raise ValueError(msg)
        return device(context) if callable(device) else device

    if inject_mesh_device and device is not None:
        msg = "@traced_function: inject_mesh_device=True is mutually exclusive with device="
        raise ValueError(msg)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)

        # First matching parameter name (in priority order) is what we'll look up at call time.
        sig_mesh_param = next((n for n in _MESH_DEVICE_PARAM_NAMES if n in sig.parameters), None)

        if inject_mesh_device and sig_mesh_param is not None:
            msg = (
                f"@traced_function: {fn.__qualname__} declares a {sig_mesh_param!r} parameter but "
                "inject_mesh_device=True would also add one. Rename the parameter, or drop "
                "inject_mesh_device=True to use signature-based auto-discovery instead."
            )
            raise ValueError(msg)

        if device is None and not inject_mesh_device and sig_mesh_param is None:
            msg = (
                f"@traced_function: {fn.__qualname__} was decorated without device= and has no "
                f"parameter named any of {_MESH_DEVICE_PARAM_NAMES}. Provide "
                "device=<callable returning a ttnn.MeshDevice> (e.g. lambda self: self.mesh_device) "
                "for methods, declare a 'mesh_device' (or 'device') parameter on the function for "
                "auto-discovery, or pass inject_mesh_device=True to have the wrapper accept one as "
                "an auto-added kwarg."
            )
            raise ValueError(msg)

        _tracers: weakref.WeakKeyDictionary[Any, Tracer] = weakref.WeakKeyDictionary()
        _tracers_auto: dict[tuple[int, ...], Tracer] = {}
        _auto_bind_names: tuple[str, ...] | None = None  # frozen on the first traced call

        def _needs_new_tracer(tracer: Tracer | None) -> bool:
            return tracer is None or (not tracer.trace_captured and tracer._function is None)

        @functools.wraps(fn)
        def wrapper(*args: Any, traced: bool = False, **kwargs: Any) -> Any:
            nonlocal _auto_bind_names

            # When injecting, accept either name but only one at a time. The kwarg is consumed
            # by the wrapper in every path — it is never forwarded to the wrapped function.
            injected_mesh_device: Any = _OMITTED
            if inject_mesh_device:
                supplied = [(n, kwargs.pop(n)) for n in _MESH_DEVICE_PARAM_NAMES if n in kwargs]
                if len(supplied) > 1:
                    msg = (
                        f"@traced_function: pass only one of {_MESH_DEVICE_PARAM_NAMES} as the "
                        f"injected mesh-device kwarg; got "
                        f"{ {n: v for n, v in supplied} !r}"
                    )
                    raise TypeError(msg)
                if supplied:
                    injected_mesh_device = supplied[0][1]

            if not traced:
                for k in _TRACER_CALL_KWARGS:
                    kwargs.pop(k, None)
                return fn(*args, **kwargs)

            # Auto-discovery path: classify args at runtime on the first traced call.
            # Everything that isn't a Tracer-valid value is bound into the partial.
            if device is None:
                tracer_call_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _TRACER_CALL_KWARGS}
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                if _auto_bind_names is None:
                    _auto_bind_names = tuple(
                        name
                        for name in sig.parameters
                        if name in bound.arguments and not _is_tracer_valid_value(bound.arguments[name])
                    )
                    if not inject_mesh_device and sig_mesh_param not in _auto_bind_names:
                        md = bound.arguments.get(sig_mesh_param, None) if sig_mesh_param else None
                        msg = (
                            f"@traced_function: {fn.__qualname__} expected {sig_mesh_param!r} to "
                            f"be a non-tracer-valid value (e.g. ttnn.MeshDevice), got "
                            f"{type(md).__name__}={md!r}"
                        )
                        raise TypeError(msg)

                bind_values = {n: bound.arguments.pop(n) for n in _auto_bind_names}
                if inject_mesh_device:
                    if injected_mesh_device is _OMITTED:
                        msg = (
                            f"@traced_function: {fn.__qualname__} was called with traced=True "
                            f"but none of {_MESH_DEVICE_PARAM_NAMES} were supplied as kwargs "
                            "(required when inject_mesh_device=True)."
                        )
                        raise TypeError(msg)
                    mesh_device_val = injected_mesh_device
                    key = (id(mesh_device_val), *(id(v) for v in bind_values.values()))
                else:
                    mesh_device_val = bind_values[sig_mesh_param]
                    key = tuple(id(v) for v in bind_values.values())

                t = _tracers_auto.get(key)
                if _needs_new_tracer(t):
                    _tracers_auto[key] = Tracer(
                        functools.partial(fn, **bind_values),
                        device=mesh_device_val,
                        prep_run=prep_run,
                        clone_prep_inputs=clone_prep_inputs,
                    )
                return _tracers_auto[key](*bound.args, **bound.kwargs, **tracer_call_kwargs)

            # Context-bound path: first arg is a non-tracer-valid context (e.g. self);
            # bind it away and track one Tracer per instance.
            if args and not isinstance(args[0], _TRACER_VALID_INPUT_TYPES):
                context, rest = args[0], args[1:]
                if _needs_new_tracer(_tracers.get(context)):
                    _tracers[context] = Tracer(
                        functools.partial(fn, context),
                        device=_resolve_device(context),
                        prep_run=prep_run,
                        clone_prep_inputs=clone_prep_inputs,
                    )
                return _tracers[context](*rest, **kwargs)

            # device= was supplied but the first positional argument is a valid Tracer input
            # (or absent), so there's no context to bind. Standalone functions should omit
            # device= and declare a 'mesh_device' parameter for auto-discovery instead.
            msg = (
                f"@traced_function: {fn.__qualname__} was called with traced=True, but device= "
                "was provided at decoration time and the first positional argument is not a "
                "bindable context (it is a tracer-valid type). For standalone functions, omit "
                "device= and declare a 'mesh_device' parameter to use auto-discovery; for "
                "methods, ensure self is passed positionally."
            )
            raise TypeError(msg)

        wrapper._tracers = _tracers  # type: ignore[attr-defined]
        wrapper._tracers_auto = _tracers_auto  # type: ignore[attr-defined]
        return wrapper

    if _fn is not None:
        return decorator(_fn)

    return decorator

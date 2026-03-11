# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import NoneType
from typing import TYPE_CHECKING, Any

from loguru import logger

import ttnn

if TYPE_CHECKING:
    from collections.abc import Callable


class Tracer:
    """Wrapper for capturing and executing a trace of a given function.

    All inputs and outputs of the traced function must be ``ttnn.Tensor`` instances or plain
    Python scalars (``int``, ``float``, ``str``, ``bool``, ``None``), optionally nested in
    tuples, lists, or dicts.
    """

    def __init__(
        self,
        function: Callable[..., Any],
        /,
        *,
        device: ttnn.MeshDevice,
        prep_run: bool = True,
        clone_prep_inputs: bool = True,
    ) -> None:
        """Initialize the tracer.

        If the function modifies its input tensors in place, set ``clone_prep_inputs`` to ``True``
        so that preparation runs operate on cloned inputs, leaving the originals intact for trace
        capture.

        Args:
            function: Function to be traced.
            device: Device on which to capture and execute the trace.
            prep_run: Whether to run the function once before capturing the trace.
            clone_prep_inputs: Whether to clone tensor inputs for the preparation run.
        """
        self._function = function
        self._device = device
        self._prep_run = prep_run
        self._clone_prep_inputs = clone_prep_inputs
        self._args: tuple[Any, ...] = ()
        self._kwargs: dict[str, Any] = {}
        self._outputs: Any = None
        self._trace_id: ttnn.MeshTraceId | None = None

    def __call__(
        self,
        *args: Any,
        tracer_cq_id: int = 0,
        tracer_blocking_execution: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Capture or execute trace.

        On the first call, runs the wrapped function multiple times to capture the trace. On
        subsequent calls, executes the captured trace. On the first call, inputs are used to
        initialize the trace inputs. On subsequent calls, they are used to update the trace inputs.
        Only `ttnn.Tensor` inputs can be changed. Aside from omitting positional inputs to reuse
        previous values, a value of `None` can be passed to reuse the previous value for tensor
        inputs as well.

        Host tensor inputs will automatically be moved to the tracer device for the trace capture
        and execution.

        Executing a trace overwrites any device memory that was used during trace capture. In
        particular, any device tensors that are allocated after the trace was captured may be
        overwritten when the trace is executed, even if they are not inputs or outputs of the trace.
        Host tensors will not be overwritten. Input tensors are copied before trace execution, so
        they can safely be allocated on device if their content is not needed after trace execution.

        Args:
            tracer_cq_id: Command queue id.
            tracer_blocking_execution: Whether `ttnn.execute_trace` should block.
            *args: Positional inputs to pass to the wrapped function.
            **kwargs: Named inputs to pass to the wrapped function. Optional on subsequent calls.

        Returns:
            The outputs of the wrapped function.

        Raises:
            TypeError: If outputs have unsupported types.
            Any exception raised by the wrapped function during first invocation.
        """
        if self._trace_id is None:
            if self._function is None:
                msg = "tracer can not be reused after the trace was released"
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
            trace_id = ttnn.begin_trace_capture(self._device, cq_id=tracer_cq_id)
            try:
                try:
                    outputs = self._function(*self._args, **self._kwargs)
                finally:
                    ttnn.end_trace_capture(self._device, trace_id, cq_id=tracer_cq_id)

                outputs = _tree_map(_verify_value, outputs, path_label="outputs")
            except Exception:
                ttnn.release_trace(self._device, trace_id)
                raise

            # Trace capture records commands but does not execute them. Execute the trace to
            # actually compute outputs.
            ttnn.execute_trace(self._device, trace_id, cq_id=tracer_cq_id, blocking=tracer_blocking_execution)

            # Allow resources referenced by the function to be freed, which might be used to offload
            # weights.
            self._function = None

            self._trace_id = trace_id
            self._outputs = outputs
        else:
            if len(args) > len(self._args):
                msg = f"expected at most {len(self._args)} positional args, got {len(args)}"
                raise TypeError(msg)

            # Pad with None to allow omitting trailing positional args.
            args = args + (None,) * (len(self._args) - len(args))
            _tree_map(self._update_input, self._args, args, path_label="args")

            # kwargs can be omitted entirely to reuse all previous values, but individual
            # entries must be explicitly set to None to preserve them (unlike positional args,
            # _tree_map requires dicts to have matching keys).
            for name, new in kwargs.items():
                if name not in self._kwargs:
                    msg = f"input '{name}' was not in the initial inputs"
                    raise KeyError(msg)

                # None means reuse the previous value entirely.
                if new is not None:
                    _tree_map(self._update_input, self._kwargs[name], new, path_label=f'kwargs["{name}"]')

            ttnn.execute_trace(self._device, self._trace_id, cq_id=tracer_cq_id, blocking=tracer_blocking_execution)

        return self._outputs

    @property
    def trace_captured(self) -> bool:
        """Whether a trace has been captured and is ready for execution."""
        return self._trace_id is not None

    def release_trace(self) -> None:
        """Release the captured trace and clear inputs and outputs."""
        trace_id = self._trace_id

        if trace_id is not None:
            self._trace_id = None
            self._args = ()
            self._kwargs = {}
            self._outputs = None
            ttnn.release_trace(self._device, trace_id)

    def _tensor_to_device(self, value: Any, *, path_label: str) -> Any:
        if not isinstance(value, ttnn.Tensor):
            return value

        if value.device() is None:
            return value.to(self._device)
        if value.device() == self._device:
            return value

        msg = f"input '{path_label}' device {value.device()} does not match tracer device {self._device}"
        raise ValueError(msg)

    def _update_input(self, prev: Any, new: Any, *, path_label: str) -> None:
        if new is None and isinstance(prev, ttnn.Tensor):
            return

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


def _verify_value(value: Any, *, path_label: str) -> Any:
    if not isinstance(value, (ttnn.Tensor, int, float, str, bool, NoneType)):
        msg = f"value '{path_label}' has unsupported type {type(value)}"
        raise TypeError(msg)

    return value


def _clone_tensor(value: Any, *, path_label: str) -> Any:  # noqa: ARG001
    """Clone a tensor, passing through non-tensor values unchanged."""
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

    if isinstance(x, tuple):
        for y in xs:
            if len(x) != len(y):
                msg = f"tuple lengths of '{path_label}' should be the same: {len(x)} != {len(y)}"
                raise ValueError(msg)

        return tuple(
            _tree_map(f, *elts, path_label=f"{path_label}[{i}]") for i, elts in enumerate(zip(x, *xs, strict=True))
        )

    if isinstance(x, list):
        for y in xs:
            if len(x) != len(y):
                msg = f"list lengths of '{path_label}' should be the same: {len(x)} != {len(y)}"
                raise ValueError(msg)

        return [_tree_map(f, *elts, path_label=f"{path_label}[{i}]") for i, elts in enumerate(zip(x, *xs, strict=True))]

    if isinstance(x, dict):
        for y in xs:
            if x.keys() != y.keys():
                msg = f"dict keys of '{path_label}' should be the same: {x.keys()} != {y.keys()}"
                raise ValueError(msg)
        return {key: _tree_map(f, *(d[key] for d in (x, *xs)), path_label=f'{path_label}["{key}"]') for key in x}

    raise AssertionError  # unreachable

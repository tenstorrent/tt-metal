# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import NoneType
from typing import TYPE_CHECKING, Any

import ttnn

if TYPE_CHECKING:
    from collections.abc import Callable


class Tracer:
    """Wrapper for capturing and executing a trace of a given function."""

    def __init__(self, function: Callable[..., Any], /, *, device: ttnn.MeshDevice) -> None:
        """Initialize the tracer.

        Args:
            function: Function to be traced.
            device: Device on which to capture and execute the trace.
        """
        self._function = function
        self._device = device
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

        On the first call, runs the wrapped function twice, once to compile, and once to capture the
        trace. On subsequent calls, executes the captured trace.

        Args:
            tracer_cq_id: Command queue id.
            tracer_blocking_execution: Whether `ttnn.execute_trace` should block.
            *args: Positional inputs to pass to the wrapped function. On the first call, these are
                   used to initialize the trace inputs. On subsequent calls, these are used to
                   update the trace inputs. Only tensor inputs can be changed.
            **kwargs: Named inputs to pass to the wrapped function. On the first call, these are
                      used to initialize the trace inputs. On subsequent calls, these are optional
                      and used to update the trace inputs. Only tensor inputs can be changed.

        Returns:
            The outputs of the wrapped function.

        Raises:
            TypeError: If outputs have unsupported types.
            Any exception raised by the wrapped function during first invocation.
        """
        if self._trace_id is None:
            args = _tree_map(_verify_value, args, path_label="args")
            kwargs = _tree_map(_verify_value, kwargs, path_label="kwargs")
            self._args = _tree_map(self._move_to_device_if_tensor, args, path_label="args")
            self._kwargs = _tree_map(self._move_to_device_if_tensor, kwargs, path_label="kwargs")

            # compile
            self._function(*self._args, **self._kwargs)

            # capture trace
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

            self._trace_id = trace_id
            self._outputs = outputs
        else:
            _tree_map(self._update_input, self._args, args, path_label="args")

            for name, new in kwargs.items():
                if name not in self._kwargs:
                    msg = f"input '{name}' was not in the initial inputs"
                    raise KeyError(msg)
                prev = self._kwargs[name]

                _tree_map(self._update_input, prev, new, path_label=f'kwargs["{name}"]')

            ttnn.execute_trace(self._device, self._trace_id, cq_id=tracer_cq_id, blocking=tracer_blocking_execution)

        return self._outputs

    def release(self) -> None:
        """Release the captured trace and clear inputs and outputs."""
        trace_id = self._trace_id

        if trace_id is not None:
            self._trace_id = None
            self._args = ()
            self._kwargs = {}
            self._outputs = None
            ttnn.release_trace(self._device, trace_id)

    def _move_to_device_if_tensor(self, value: Any, *, path_label: str) -> Any:
        if not isinstance(value, ttnn.Tensor):
            return value

        if value.device() is None:
            return value.to(self._device)
        if value.device() == self._device:
            return value

        msg = f"input '{path_label}' device {value.device()} does not match tracer device {self._device}"
        raise ValueError(msg)

    def _update_input(self, prev: Any, new: Any, *, path_label: str) -> None:
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
                ttnn.copy(new, prev)

        elif new != prev:
            msg = f"input '{path_label}' does not match the initial value"
            raise ValueError(msg)


def _verify_value(value: Any, *, path_label: str) -> Any:
    if not isinstance(value, (ttnn.Tensor, int, float, str, bool, NoneType)):
        msg = f"value '{path_label}' has unsupported type {type(value)}"
        raise TypeError(msg)

    return value


def _tree_map(f: Callable, x: Any, /, *xs: Any, path_label: str) -> Any:
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
        A new nested structure with the same shape as the inputs, where each
        leaf has been transformed by applying f to the corresponding leaves
        from all input structures.

    Raises:
        ValueError: If the input structures don't have matching types at
            corresponding positions, or if dicts have different keys.
    """
    tx = type(x)
    for y in xs:
        if type(y) is not tx:
            msg = f"types of '{path_label}' should be the same: {tx} != {type(y)}"
            raise ValueError(msg)

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

    return f(x, *xs, path_label=path_label)

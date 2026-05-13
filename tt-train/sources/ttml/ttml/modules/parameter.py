# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter and Buffer wrappers for tensor registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


LAZY_PARAMETER_ACCESS_MSG = (
    "This parameter has not been materialized yet. "
    "Build the model inside `with ttml.lazy_init():` and then call "
    "`ttml.materialize_module(model)` before accessing `.tensor` or running forward. "
    "In-place initializers (`ttml.init.uniform_`, etc.) also require a materialized tensor."
)


@dataclass(frozen=True)
class TensorMetadata:
    """Deferred parameter: shape + factory; no device storage until materialization.

    Do not use empty `ttml.autograd.Tensor` as a stand-in; keep allocation in
    `init_fn` until `materialize()` runs.
    """

    shape: tuple[int, ...]
    init_fn: Callable[..., Any]
    mapper: Any | None = None
    requires_grad: bool = True

    def materialize(self, mapper_override: Any | None = None) -> Any:
        """Allocate the autograd tensor using optional mapper override."""
        mapper = self.mapper if mapper_override is None else mapper_override
        return self.init_fn(self.shape, mapper)


class Parameter:
    """Wrapper marking a tensor as a trainable parameter."""

    def __init__(self, tensor: Any) -> None:
        object.__setattr__(self, "tensor", tensor)

    def __getattribute__(self, name: str) -> Any:
        if name == "tensor":
            t = object.__getattribute__(self, "tensor")
            if isinstance(t, TensorMetadata):
                raise RuntimeError(LAZY_PARAMETER_ACCESS_MSG)
            return t
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def peek_tensor(self) -> Any:
        """Return stored tensor or :class:`TensorMetadata` without raising."""
        return object.__getattribute__(self, "tensor")

    @property
    def is_lazy(self) -> bool:
        return isinstance(self.peek_tensor(), TensorMetadata)

    def __repr__(self) -> str:
        inner = self.peek_tensor()
        return f"Parameter({inner})"


class Buffer:
    """Wrapper marking a tensor as a non-trainable buffer."""

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor

    def __repr__(self) -> str:
        return f"Buffer({self.tensor})"

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter and Buffer wrappers for tensor registration."""

from dataclasses import dataclass
from typing import Any, Callable, Tuple


@dataclass(frozen=True)
class TensorMetadata:
    """Describes a tensor's properties without allocating memory.

    Used with deferred initialization: module constructors create TensorMetadata
    Parameters before calling super().__init__(). The root TransformerBase then
    walks the tree and materializes all parameters on device.

    Attributes:
        shape: Full (unsharded) tensor shape.
        init_fn: A closure returned by ttml.init (e.g. ttml.init.uniform(-k, k)).
                 Signature: init_fn(shape) -> Tensor
        requires_grad: Whether the materialized tensor needs gradients.
    """

    shape: Tuple[int, ...]
    init_fn: Callable
    requires_grad: bool = True


class Parameter:
    """Wrapper marking a tensor as a trainable parameter.

    Can hold either:
    - TensorMetadata (before materialization) - describes shape/init_fn.
    - ttml.autograd.Tensor (after materialization).
    """

    def __init__(self, tensor_or_metadata: Any) -> None:
        self.tensor = tensor_or_metadata

    @property
    def is_materialized(self) -> bool:
        """True if the tensor has been allocated on device."""
        return not isinstance(self.tensor, TensorMetadata)

    @property
    def shape(self):
        """Return tensor shape (works for both metadata and materialized tensor)."""
        if isinstance(self.tensor, TensorMetadata):
            return self.tensor.shape
        return tuple(self.tensor.shape())

    def __repr__(self) -> str:
        if isinstance(self.tensor, TensorMetadata):
            return f"Parameter(TensorMetadata(shape={self.tensor.shape}))"
        return f"Parameter({self.tensor})"


class Buffer:
    """Wrapper marking a tensor as a non-trainable buffer."""

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor

    def __repr__(self) -> str:
        return f"Buffer({self.tensor})"

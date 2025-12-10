# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter and Buffer wrapper classes for automatic parameter tracking.

These classes provide a PyTorch-like interface for marking tensors as
trainable parameters or non-trainable buffers.
"""

from typing import Any


class Parameter:
    """A wrapper class for tensors that marks them as trainable parameters.

    Similar to PyTorch's `nn.Parameter`, this class wraps a tensor and marks
    it as a trainable parameter that should be tracked by the module system.

    Example:
        >>> tensor = create_tensor(...)
        >>> param = Parameter(tensor)
        >>> module.weight = param  # Automatically registered as parameter
    """

    def __init__(self, tensor: Any) -> None:
        """Initialize a Parameter wrapper.

        Args:
            tensor: The tensor to wrap as a parameter.
        """
        self.tensor = tensor

    def __repr__(self) -> str:
        """Return string representation of the Parameter."""
        return f"Parameter({self.tensor})"


class Buffer:
    """A wrapper class for tensors that marks them as non-trainable buffers.

    Similar to PyTorch's buffers, this class wraps a tensor and marks it as
    a non-trainable buffer (e.g., running statistics in BatchNorm).

    Example:
        >>> running_mean = create_tensor(...)
        >>> buffer = Buffer(running_mean)
        >>> module.running_mean = buffer  # Automatically registered as buffer
    """

    def __init__(self, tensor: Any) -> None:
        """Initialize a Buffer wrapper.

        Args:
            tensor: The tensor to wrap as a buffer.
        """
        self.tensor = tensor

    def __repr__(self) -> str:
        """Return string representation of the Buffer."""
        return f"Buffer({self.tensor})"

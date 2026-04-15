# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""CPU dispatcher for all TTNN operations.

This dispatcher executes all operations on the CPU without any special optimizations.
"""

import logging

# Configure logger
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None) -> bool:
    """Check if operation can be dispatched with debug logging.

    Args:
        func_name: ATen operation name (e.g., "aten::mul.Tensor")
        args: Positional arguments to the operation
        kwargs: Keyword arguments to the operation

    Returns:
        True if the operation can be dispatched to TTNN, False otherwise
    """
    logger.debug(f"Dispatching {func_name} to CPU.")
    return False


def dispatch_to_ttnn(func_name: str, args, kwargs):
    """Dispatch operation with debug logging.

    Args:
        func_name: ATen operation name (e.g., "aten::mul.Tensor")
        args: Positional arguments to the operation
        kwargs: Keyword arguments to the operation

    Returns:
        Result of the TTNN operation (typically a TorchTTNNTensor)
    """
    raise NotImplementedError("CPU dispatcher does not support TTNN operations.")

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Debug dispatcher with verbose logging for TTNN operations.

This dispatcher wraps the default dispatcher and adds detailed logging
for debugging purposes. Useful for understanding dispatch behavior.
"""

import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None) -> bool:
    """Check if operation can be dispatched with debug logging.

    Args:
        func_name: ATen operation name (e.g., "aten::mul.Tensor")
        args: Positional arguments to the operation
        kwargs: Keyword arguments to the operation

    Returns:
        True if the operation can be dispatched to TTNN, False otherwise
    """
    from models.experimental.tt_symbiote.core.dispatchers.default_dispatcher import (
        can_dispatch_to_ttnn as default_can_dispatch,
    )

    result = default_can_dispatch(func_name, args, kwargs)
    if not result:
        logger.debug(f"  Cannot dispatch {func_name} to TTNN")
    return result


def dispatch_to_ttnn(func_name: str, args, kwargs):
    """Dispatch operation with debug logging.

    Args:
        func_name: ATen operation name (e.g., "aten::mul.Tensor")
        args: Positional arguments to the operation
        kwargs: Keyword arguments to the operation

    Returns:
        Result of the TTNN operation (typically a TorchTTNNTensor)
    """
    from models.experimental.tt_symbiote.core.dispatchers.default_dispatcher import dispatch_to_ttnn as default_dispatch

    try:
        result = default_dispatch(func_name, args, kwargs)
        debug_message = f"  Successfully dispatched {func_name}"
        if hasattr(result, "shape"):
            debug_message += f" with result shape {result.shape}"
        logger.debug(debug_message)
        return result
    except Exception as e:
        logger.error(f"  Failed to dispatch {func_name}: {e}")
        raise

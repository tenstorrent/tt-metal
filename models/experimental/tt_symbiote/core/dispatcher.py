# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN operation dispatch system with pluggable dispatchers.

This module provides a configurable dispatcher system that allows users to select
different dispatcher implementations at runtime, similar to run_config.py.

Basic Usage:
    # Use default dispatcher (automatic)
    from models.experimental.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn, dispatch_to_ttnn

    # Change dispatcher programmatically
    from models.experimental.tt_symbiote.core.dispatcher import set_dispatcher
    set_dispatcher("default")  # or "optimized", "debug", etc.

    # Change via environment variable (takes precedence)
    # export TT_SYMBIOTE_DISPATCHER=default

Available Dispatchers:
    - "default": Standard TTNN dispatcher with all operations

To create a new dispatcher:
    1. Create a module with can_dispatch_to_ttnn() and dispatch_to_ttnn() functions
    2. Register manually:
       from models.experimental.tt_symbiote.core.dispatcher import register_dispatcher
       import my_dispatcher
       register_dispatcher("my_dispatcher", my_dispatcher)
    3. Use it: set_dispatcher("my_dispatcher")

    Or for built-in dispatchers:
    1. Create a file in core/dispatchers/ (e.g., optimized_dispatcher.py)
    2. Register it in dispatcher_config.py's _auto_register_dispatchers()
"""

from models.experimental.tt_symbiote.core.dispatchers import (
    get_active_dispatcher,
    list_available_dispatchers,
    register_dispatcher,
    set_dispatcher,
)

__all__ = [
    "can_dispatch_to_ttnn",
    "dispatch_to_ttnn",
    "set_dispatcher",
    "list_available_dispatchers",
    "register_dispatcher",
]


# ========== Public Dispatcher Interface ==========


def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None) -> bool:
    """Check if operation can be dispatched to TTNN.

    Delegates to the currently active dispatcher implementation.

    Args:
        func_name: ATen operation name (e.g., "aten::mul.Tensor")
        args: Positional arguments to the operation
        kwargs: Keyword arguments to the operation

    Returns:
        True if the operation can be dispatched to TTNN, False otherwise
    """
    dispatcher = get_active_dispatcher()
    return dispatcher.can_dispatch_to_ttnn(func_name, args, kwargs)


def dispatch_to_ttnn(func_name: str, args, kwargs):
    """Dispatch operation to TTNN handler.

    Delegates to the currently active dispatcher implementation.

    Args:
        func_name: ATen operation name (e.g., "aten::mul.Tensor")
        args: Positional arguments to the operation
        kwargs: Keyword arguments to the operation

    Returns:
        Result of the TTNN operation (typically a TorchTTNNTensor)
    """
    dispatcher = get_active_dispatcher()
    return dispatcher.dispatch_to_ttnn(func_name, args, kwargs)

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Configuration system for TTNN dispatchers.

This module provides a pluggable dispatcher system similar to run_config.py,
allowing users to switch between different dispatcher implementations at runtime.
"""

import os
from typing import Any, Dict

# Registry of available dispatchers
_DISPATCHER_REGISTRY: Dict[str, Any] = {}

# Current active dispatcher (default to "default")
_current_dispatcher = None


def register_dispatcher(name: str, dispatcher_module: Any) -> None:
    """Register a dispatcher implementation.

    Args:
        name: Name of the dispatcher (e.g., "DEFAULT", "CPU", "DEBUG")
        dispatcher_module: Module containing can_dispatch_to_ttnn and dispatch_to_ttnn functions
    """
    if not hasattr(dispatcher_module, "can_dispatch_to_ttnn"):
        raise ValueError(f"Dispatcher module must implement can_dispatch_to_ttnn function")
    if not hasattr(dispatcher_module, "dispatch_to_ttnn"):
        raise ValueError(f"Dispatcher module must implement dispatch_to_ttnn function")

    _DISPATCHER_REGISTRY[name] = dispatcher_module


def set_dispatcher(name: str) -> None:
    """Set the active dispatcher.

    Args:
        name: Name of the dispatcher to activate

    Raises:
        ValueError: If dispatcher name is not registered
    """
    global _current_dispatcher

    if name not in _DISPATCHER_REGISTRY:
        raise ValueError(f"Unknown dispatcher '{name}'. Available dispatchers: {list(_DISPATCHER_REGISTRY.keys())}")

    _current_dispatcher = name


def get_active_dispatcher() -> Any:
    """Get the currently active dispatcher module.

    Returns:
        Dispatcher module with can_dispatch_to_ttnn and dispatch_to_ttnn functions
    """
    # Check environment variable first (similar to run_config)
    env_dispatcher = os.environ.get("TT_SYMBIOTE_DISPATCHER", "CPU")
    if env_dispatcher is not None and env_dispatcher in _DISPATCHER_REGISTRY:
        return _DISPATCHER_REGISTRY[env_dispatcher]

    # Fall back to programmatically set dispatcher
    if _current_dispatcher not in _DISPATCHER_REGISTRY:
        raise RuntimeError(
            f"Active dispatcher '{_current_dispatcher}' not registered. "
            f"Available: {list(_DISPATCHER_REGISTRY.keys())}"
        )

    return _DISPATCHER_REGISTRY[_current_dispatcher]


def list_available_dispatchers() -> list[str]:
    """List all registered dispatcher names.

    Returns:
        List of dispatcher names
    """
    return list(_DISPATCHER_REGISTRY.keys())


# Auto-register dispatchers on import
def _auto_register_dispatchers():
    """Automatically register available dispatcher implementations."""
    try:
        from models.experimental.tt_symbiote.core.dispatchers import default_dispatcher

        register_dispatcher("DEFAULT", default_dispatcher)
    except ImportError:
        pass

    # Debug dispatcher with verbose logging
    try:
        from models.experimental.tt_symbiote.core.dispatchers import debug_dispatcher

        register_dispatcher("DEBUG", debug_dispatcher)
    except ImportError:
        pass

    # Debug dispatcher with verbose logging
    try:
        from models.experimental.tt_symbiote.core.dispatchers import cpu_dispatcher

        register_dispatcher("CPU", cpu_dispatcher)
    except ImportError:
        pass

    try:
        from models.experimental.tt_symbiote.core.dispatchers import tensor_operations_dispatcher

        register_dispatcher("TENSOR_OPS", tensor_operations_dispatcher)
    except ImportError:
        pass

    # Add more auto-registrations here as new dispatchers are created
    # try:
    #     from models.experimental.tt_symbiote.core.dispatchers import optimized_dispatcher
    #     register_dispatcher("optimized", optimized_dispatcher)
    # except ImportError:
    #     pass


# Initialize on module load
_auto_register_dispatchers()

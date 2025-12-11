# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for recursively importing symbols from _ttml C++ extension.

This module provides functions to automatically import all symbols from the _ttml
C++ extension module and its submodules into the corresponding Python ttml package
locations, while respecting existing Python implementations.
"""

import importlib
import importlib.util
import inspect
import sys
import types
from typing import Any, Optional, Set


def _should_import_symbol(
    name: str, source_value: Any, target_module: types.ModuleType
) -> bool:
    """
    Determine if a symbol should be imported.

    Args:
        name: Name of the symbol
        source_value: The value of the symbol in the source module
        target_module: The target module to import into

    Returns:
        True if the symbol should be imported, False otherwise
    """
    # Skip private symbols (except __all__)
    if name.startswith("_") and name != "__all__":
        return False

    # Skip module metadata
    if name in (
        "__name__",
        "__file__",
        "__doc__",
        "__package__",
        "__path__",
        "__loader__",
        "__spec__",
        "__cached__",
        "__builtins__",
    ):
        return False

    # Skip if already exists in target
    if hasattr(target_module, name):
        return False

    return True


def _is_submodule(obj: Any) -> bool:
    """
    Check if an object is a module.

    Args:
        obj: Object to check

    Returns:
        True if obj is a module, False otherwise
    """
    return inspect.ismodule(obj)


def _ensure_submodule_exists(
    parent_module: types.ModuleType, submodule_name: str
) -> types.ModuleType:
    """
    Ensure a submodule exists in the parent module, creating it if necessary.

    This function preserves existing Python packages and only creates new modules
    if they don't exist. If a Python package already exists (has __path__), it
    will be preserved and used.

    Args:
        parent_module: The parent module
        submodule_name: Name of the submodule to ensure exists

    Returns:
        The submodule (existing or newly created)
    """
    full_name = f"{parent_module.__name__}.{submodule_name}"

    # First, check if it exists in sys.modules (might be a package that was imported)
    if full_name in sys.modules:
        existing_module = sys.modules[full_name]
        if _is_submodule(existing_module):
            # Use the existing module from sys.modules (preserves Python packages)
            if (
                not hasattr(parent_module, submodule_name)
                or getattr(parent_module, submodule_name) is not existing_module
            ):
                setattr(parent_module, submodule_name, existing_module)
            return existing_module

    # Check if submodule already exists in parent
    if hasattr(parent_module, submodule_name):
        submodule = getattr(parent_module, submodule_name)
        if _is_submodule(submodule):
            # If it's already a module, use it (preserves Python packages)
            # Make sure it's registered in sys.modules
            if full_name not in sys.modules:
                sys.modules[full_name] = submodule
            return submodule

    # Try to import it as a package if it might be one
    # This handles the case where ttml.modules is a Python package
    try:
        spec = importlib.util.find_spec(full_name)
        if spec is not None and spec.loader is not None:
            # It's a real package/module, import it
            imported_module = importlib.import_module(full_name)
            setattr(parent_module, submodule_name, imported_module)
            return imported_module
    except (ImportError, ModuleNotFoundError, ValueError):
        # Not a real package, continue to create a new module
        pass

    # Create a new module only if it doesn't exist
    submodule = types.ModuleType(full_name)
    sys.modules[full_name] = submodule
    setattr(parent_module, submodule_name, submodule)
    return submodule


def _recursive_import_from_ttml(
    source_module: types.ModuleType,
    target_module: types.ModuleType,
    visited: Optional[Set[str]] = None,
) -> None:
    """
    Recursively import all symbols from source_module into target_module,
    skipping symbols that already exist in target_module.

    This function traverses the _ttml C++ extension module structure and
    imports all public symbols into the corresponding Python ttml package
    locations. Python implementations take precedence over C++ symbols.

    Args:
        source_module: The _ttml module or submodule to import from
        target_module: The ttml Python module or submodule to import into
        visited: Set of already visited modules to prevent cycles
    """
    if visited is None:
        visited = set()

    # Track this module to prevent cycles
    source_name = getattr(source_module, "__name__", None)
    if source_name:
        if source_name in visited:
            return
        visited.add(source_name)

    # Get all attributes from source module
    try:
        source_attrs = dir(source_module)
    except (AttributeError, TypeError):
        # If we can't get attributes, skip this module
        return

    # Process each attribute
    for name in source_attrs:
        try:
            source_value = getattr(source_module, name)
        except (AttributeError, TypeError):
            # Skip if we can't get the attribute
            continue

        # Check if we should import this symbol
        if not _should_import_symbol(name, source_value, target_module):
            continue

        # Handle submodules recursively
        if _is_submodule(source_value):
            try:
                # Check if target already has this as a module
                target_has_module = hasattr(target_module, name)
                target_submodule = None

                if target_has_module:
                    existing = getattr(target_module, name)
                    if _is_submodule(existing):
                        # Check if it's a Python package (has __path__)
                        if hasattr(existing, "__path__"):
                            # It's a Python package - preserve it and import symbols into it
                            target_submodule = existing
                        else:
                            # It's a regular module - we can use it
                            target_submodule = existing

                # If we don't have a target submodule yet, create/ensure it exists
                if target_submodule is None:
                    target_submodule = _ensure_submodule_exists(target_module, name)

                # Recursively import from the submodule
                _recursive_import_from_ttml(source_value, target_submodule, visited)
            except (AttributeError, TypeError, ValueError) as e:
                # Skip if we can't create or access the submodule
                # This can happen with some special modules
                continue
        else:
            # Import regular symbols
            try:
                setattr(target_module, name, source_value)
            except (TypeError, AttributeError):
                # Skip if we can't set the attribute (e.g., read-only)
                continue

    # Handle __all__ if present in source
    if hasattr(source_module, "__all__"):
        source_all = getattr(source_module, "__all__")
        if isinstance(source_all, (list, tuple)):
            # Merge with target's __all__ if it exists
            if hasattr(target_module, "__all__"):
                target_all = list(getattr(target_module, "__all__"))
                # Add items from source that aren't already in target
                for item in source_all:
                    if item not in target_all and hasattr(target_module, item):
                        target_all.append(item)
                setattr(target_module, "__all__", tuple(target_all))
            else:
                # Create __all__ from source, filtered by what actually exists
                filtered_all = [
                    item for item in source_all if hasattr(target_module, item)
                ]
                if filtered_all:
                    setattr(target_module, "__all__", tuple(filtered_all))

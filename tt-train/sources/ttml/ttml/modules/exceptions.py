# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions for module-related errors."""


class ModuleError(Exception):
    """Base exception for module-related errors."""

    pass


class DuplicateNameError(ModuleError):
    """Raised when attempting to register a tensor or module with a name that already exists."""

    pass


class NameNotFoundError(ModuleError):
    """Raised when attempting to override a tensor or module with a name that doesn't exist."""

    pass


class UninitializedModuleError(ModuleError):
    """Raised when attempting to register a None module."""

    pass

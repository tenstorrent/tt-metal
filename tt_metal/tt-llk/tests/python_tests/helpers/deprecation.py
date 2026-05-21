# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
import warnings


def deprecated(message: str = ""):
    """Decorator to mark functions as deprecated.

    Usage:
        @deprecated("message")
        def function(...):
            ...

    When the decorated function is called, a DeprecationWarning is emitted
    once per call site.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. {message}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator

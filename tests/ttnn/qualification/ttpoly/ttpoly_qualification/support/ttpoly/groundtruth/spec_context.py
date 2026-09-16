# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Explicit, read-only specification location for the canonical reference.

Packaged callers scope the complete scoring operation, including domain helpers,
to their bundled activation JSON directory. No repository fallback is performed.
The context is task-local; cached providers from different roots cannot mix.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
from pathlib import Path

_ROOT = ContextVar("groundtruth_activation_spec_root", default=None)


def current_spec_root():
    return _ROOT.get()


def spec_cache_key(name):
    root = current_spec_root()
    return name if root is None else (str(root), name)


@contextmanager
def activation_spec_root(path):
    """Use an existing immutable JSON directory for all reference lookups."""
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"activation specification root is not a directory: {root}")
    token = _ROOT.set(root)
    try:
        yield root
    finally:
        _ROOT.reset(token)


def scoped_spec_cache(function):
    """Memoize a spec reader without leaking results across explicit roots."""

    @lru_cache(maxsize=None)
    def cached(root, name):
        return function(name)

    @wraps(function)
    def wrapped(name):
        return cached(current_spec_root(), name)

    wrapped.cache_clear = cached.cache_clear
    wrapped.cache_info = cached.cache_info
    return wrapped

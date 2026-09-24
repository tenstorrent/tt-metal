# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gate-2 instrumentation: count calls on the graduated stub instances that sit inside the chain.

`track(name, obj, methods)` swaps obj's class for a cached subclass whose listed methods bump
counts[name] before delegating, so every instance tracked under one name shares one type (a stack of
60 tracked blocks is still a list of same-typed elements) and isinstance() still holds. Nothing here
calls a stub; the counts only move when the real forward calls them.
"""
from __future__ import annotations

from collections import defaultdict

_SUBCLASS_CACHE = {}


class InvocationTracker:
    def __init__(self):
        self.counts = defaultdict(int)
        self.instances = defaultdict(int)
        self.stub_modules = {}  # name -> dotted module name of the graduated stub (a name, not the module:
        # keeping module objects on the pipeline lets object walkers descend into module globals)

    def _subclass(self, cls, name, methods):
        key = (cls, name, tuple(methods), id(self))
        sub = _SUBCLASS_CACHE.get(key)
        if sub is not None:
            return sub
        counts = self.counts
        ns = {}
        for m in methods:
            base = getattr(cls, m)

            def _wrap(base=base):
                def counted(inst, *a, **k):
                    counts[name] += 1
                    return base(inst, *a, **k)

                counted.__name__ = base.__name__
                counted.__doc__ = base.__doc__
                return counted

            ns[m] = _wrap()
        sub = type(cls.__name__, (cls,), ns)
        sub.__module__ = cls.__module__
        sub.__qualname__ = cls.__qualname__
        _SUBCLASS_CACHE[key] = sub
        return sub

    def track(self, name, obj, methods=("__call__",), stub_module=None):
        obj.__class__ = self._subclass(type(obj), name, methods)
        self.instances[name] += 1
        if stub_module is not None:
            self.stub_modules[name] = getattr(stub_module, "__name__", str(stub_module))
        return obj

    def reset(self):
        for k in list(self.counts):
            self.counts[k] = 0

    def snapshot(self):
        return {k: int(self.counts.get(k, 0)) for k in self.instances}

    def missing(self, names):
        return [n for n in names if self.counts.get(n, 0) == 0]

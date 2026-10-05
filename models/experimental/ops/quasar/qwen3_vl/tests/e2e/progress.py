# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Append one line per ttnn op start/end so a hang points at the op that never finished."""
import time
from contextlib import contextmanager


def _name(operation):
    return getattr(operation, "python_fully_qualified_name", None) or getattr(operation, "__name__", repr(operation))


def _describe(x):
    shape = getattr(x, "shape", None)
    if shape is None:
        return type(x).__name__
    parts = [f"shape={list(shape)}", f"dtype={getattr(x, 'dtype', '?')}"]
    mc = getattr(x, "memory_config", None)
    if callable(mc):
        try:
            m = mc()
            parts.append(f"mem={m.memory_layout}/{m.buffer_type}")
        except Exception:
            pass
    return "(" + ", ".join(parts) + ")"


class ProgressLog:
    def __init__(self, path):
        self.path = path
        self.stage = "-"
        self._open = []
        self._f = open(path, "a", buffering=1)

    def _line(self, operation, args):
        return f"{_name(operation)} stage={self.stage} " + " ".join(_describe(a) for a in args)

    def pre(self, operation, args, kwargs):
        line = self._line(operation, args)
        self._open.append(line)
        self._f.write(f"{time.time():.3f} PRE {line}\n")

    def post(self, operation, args, kwargs, output):
        if self._open:
            self._open.pop()
        self._f.write(f"{time.time():.3f} POST {self._line(operation, args)}\n")

    def last_unfinished(self):
        return self._open[-1] if self._open else None

    @staticmethod
    def hooks_active():
        import ttnn

        return not ttnn.CONFIG.enable_fast_runtime_mode

    @contextmanager
    def installed(self):
        import ttnn

        with ttnn.register_pre_operation_hook(self.pre), ttnn.register_post_operation_hook(self.post):
            yield self

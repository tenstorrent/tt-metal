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
        self.capture_prefix = None  # debug: keep top-level op outputs of stages starting with this
        self.captured = []  # (index, op name, stage, host tensor)
        self._capturing = False

    def _line(self, operation, args, kwargs):
        named = [f"{k}={_describe(v)}" for k, v in kwargs.items() if hasattr(v, "shape")]
        return f"{_name(operation)} stage={self.stage} " + " ".join([_describe(a) for a in args] + named)

    def pre(self, operation, args, kwargs):
        line = self._line(operation, args, kwargs)
        self._open.append(line)
        self._f.write(f"{time.time():.3f} PRE {line}\n")

    def post(self, operation, args, kwargs, output):
        if self._open:
            self._open.pop()
        self._f.write(f"{time.time():.3f} POST {self._line(operation, args, kwargs)}\n")
        if (
            self.capture_prefix
            and not self._open
            and not self._capturing
            and self.stage.startswith(self.capture_prefix)
        ):
            self._capture(operation, output)

    def _capture(self, operation, output):
        import ttnn

        from models.experimental.ops.quasar.qwen3_vl.tests.e2e.recorder import to_host

        self._capturing = True  # reading back runs ttnn ops, whose hooks must not capture again
        try:
            for t in output if isinstance(output, (list, tuple)) else [output]:
                if isinstance(t, ttnn.Tensor) and t.storage_type() == ttnn.StorageType.DEVICE and t.is_allocated():
                    self.captured.append((len(self.captured), _name(operation), self.stage, to_host(t)))
        finally:
            self._capturing = False

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

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Drain the device profiler all through a profiled run, from its first op.

Each core's profiler buffer only empties when the host reads it (ttnn.ReadDeviceProfiler). The
generated perf test reads every TT_PERF_FLUSH_EVERY ops, but only inside its measured forward:
everything before it -- building the model, uploading weights, the adapter's setup, the per-stage
pass -- ran with no read at all. WH Galaxy, 2026-09-28, on a freshly reset board: the first read of
the process found every buffer of 32 chips full (all 11,520 sites, 32 chips x 72 cores x 5 RISCs,
once each), and every marker recorded before it was gone before the forward began.

So the tool loads this module into the profiled pytest (make_run_profiled adds `-p <this module>`)
and it wraps every ttnn FastOperation for the whole session. No model or test is edited.

TWO CADENCES, because a read is not free. Reading all 32 chips every TT_PERF_FLUSH_EVERY (4) ops from
the first op spent 353 reader threads' worth of CPU and 40+ minutes before the model had even
finished loading (2026-09-28, the same Galaxy). The session only has to keep the buffer from filling,
and its size is known: TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT programs per core. So the session
reads every capacity_cadence() ops; the measured regions -- the per-stage pass here, the test's own
wrapper around its forward -- keep the fine cadence, and their reads reset the session's count. A
buffer the heal grows (probes.choose_marker_drop_remedy) widens the session interval with it. The device is learned from the ops themselves, by shape, the
way stage_marks finds the pipeline. Outside a profiling run it does nothing.
"""

from __future__ import annotations

import os
import types


def is_device(v) -> bool:
    """A ttnn device or mesh, recognised by shape rather than by name: it answers get_num_devices()."""
    return callable(getattr(v, "get_num_devices", None))


def _device_of(args, kwargs, result):
    for v in (kwargs.get("device"), result, *args):
        if is_device(v):
            return v
        dev = getattr(v, "device", None)
        if callable(dev):
            try:
                d = dev()
            except Exception:  # noqa: BLE001 -- a host tensor has no device to give
                continue
            if is_device(d):
                return d
    return None


def _profiling_env() -> dict:
    try:
        from .probes import PROFILING_ENV

        return dict(PROFILING_ENV)
    except Exception:  # noqa: BLE001
        return {}


def profiling() -> bool:
    """Is this process a tracy profiling run (the env make_run_profiled gives it)?"""
    env = _profiling_env()
    return bool(env) and all(os.environ.get(k) == v for k, v in env.items())


def cadence() -> int:
    """Read the profiler every this many ops; 0 when the run carries no cadence."""
    try:
        from .probes import PERF_FLUSH_EVERY_ENV

        return max(0, int(os.environ.get(PERF_FLUSH_EVERY_ENV) or 0))
    except Exception:  # noqa: BLE001
        return 0


# An op dispatches at least one program per chip; composites dispatch several. The session interval
# assumes at most this many on average, so a full interval fills a quarter of what the buffer holds.
_PROGRAMS_PER_OP_HEADROOM = 4


def capacity_cadence() -> int:
    """Ops between session reads: the buffer's program capacity over the per-op headroom."""
    try:
        from .probes import _DEFAULT_SUPPORT_COUNT, _SUPPORT_COUNT_ENV

        support = int(os.environ.get(_SUPPORT_COUNT_ENV) or _DEFAULT_SUPPORT_COUNT)
    except Exception:  # noqa: BLE001
        return 0
    return max(1, support // _PROGRAMS_PER_OP_HEADROOM)


class FastOperation:
    """A drained ttnn op that is still found BY TYPE NAME. Every wrapper around ttnn ops -- this
    module's nested drains, and the op wrapper in every generated perf test -- selects ops by
    type(op).__name__ == "FastOperation"; a plain function in their place would make each later
    wrapper wrap nothing, and the test's forward would silently lose its drain. Attributes are the
    original op's."""

    def __init__(self, op, call):
        self._tt_op, self._tt_call = op, call

    def __call__(self, *a, **k):
        return self._tt_call(*a, **k)

    def __getattr__(self, name):
        if name.startswith("_tt_"):  # not yet set (a copy mid-construction): never recurse
            raise AttributeError(name)
        return getattr(self._tt_op, name)


_OP_TYPE_NAME = FastOperation.__name__  # the type name every ttnn op wrapper selects by


class ProfilerDrain:
    """Wrap every FastOperation ttnn exposes so the profiler is read every cadence() ops.

    Drains nest: one opened inside another (the per-stage pass inside the session) wraps on top, and
    every read -- by any of them, or by the test's own wrapper -- resets every open drain's count, so
    an interval already read is never read again. `final_read` reads once on exit, which a pass that
    holds an open device wants and a session ending after the device closed does not."""

    def __init__(self, ttnn, device=None, final_read=True, every=None):
        self._ttnn, self._device, self._final = ttnn, device, final_read
        self._read_fn = getattr(ttnn, "ReadDeviceProfiler", None)  # before anything is wrapped
        self._on = profiling() and callable(self._read_fn)
        self._every = cadence() if every is None else max(0, int(every))
        self._orig: list = []
        self._reading = False
        self._count = 0  # ops since the drain opened, across every op it wrapped

    def read(self):
        if not self._on or self._device is None or self._reading:
            return
        self._reading = True
        try:
            self._read_fn(self._device)
        except Exception:  # noqa: BLE001 -- e.g. inside a trace capture: the next read catches up
            pass
        finally:
            self._reading = False
            self._count = 0

    def _wrap(self, fn):
        def inner(*a, **k):
            # Checked BEFORE the op, so a read someone else made right after the previous op (the
            # test's own wrapper drains at the same cadence) resets the count first: one read per
            # interval, never two.
            if self._count >= self._every:
                self.read()
            r = fn(*a, **k)
            d = _device_of(a, k, r)
            if d is not None:
                self._device = d  # the device the ops are running on now, never a stale one
            self._count += 1
            return r

        return inner

    def _watch_reads(self, fn):
        def read(*a, **k):
            self._count = 0  # any read empties the buffers this drain is counting toward
            return fn(*a, **k)

        return read

    def __enter__(self):
        if not (self._on and self._every):
            return self
        mods = [self._ttnn] + [
            v
            for v in vars(self._ttnn).values()
            if isinstance(v, types.ModuleType) and v.__name__.startswith(self._ttnn.__name__ + ".")
        ]
        for mod in mods:
            for n in dir(mod):
                op = getattr(mod, n, None)
                if type(op).__name__ == _OP_TYPE_NAME:
                    self._orig.append((mod, n, op))
                    setattr(mod, n, FastOperation(op, self._wrap(op)))
        if self._orig and callable(self._read_fn):
            self._orig.append((self._ttnn, "ReadDeviceProfiler", self._read_fn))
            self._ttnn.ReadDeviceProfiler = self._watch_reads(self._read_fn)
        return self

    def __exit__(self, *exc):
        if self._orig:
            for mod, n, op in reversed(self._orig):
                setattr(mod, n, op)
            self._orig = []
        if self._final:
            self.read()
        return False


# --- pytest plugin: `-p <this module>` on the profiled run -----------------------------------------
_session = None


def pytest_configure(config):
    global _session
    if not profiling():
        return
    try:
        import ttnn
    except Exception:  # noqa: BLE001 -- no ttnn, nothing to drain
        return
    _session = ProfilerDrain(ttnn, final_read=False, every=capacity_cadence()).__enter__()


def pytest_unconfigure(config):
    global _session
    if _session is not None:
        _session.__exit__(None, None, None)
        _session = None

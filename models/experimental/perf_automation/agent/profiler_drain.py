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
and it wraps every ttnn FastOperation for the whole session, at the cadence the profiling run
carries. No model or test is edited. The device is learned from the ops themselves, by shape, the
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


class ProfilerDrain:
    """Wrap every FastOperation ttnn exposes so the profiler is read every cadence() ops.

    One instance at a time wraps: a drain opened while another is wrapping (the per-stage pass inside
    the session-wide one) only adds its explicit read() calls. `final_read` reads once on exit, which
    a pass that holds an open device wants and a session ending after the device closed does not."""

    _wrapping = False

    def __init__(self, ttnn, device=None, final_read=True):
        self._ttnn, self._device, self._final = ttnn, device, final_read
        self._read_fn = getattr(ttnn, "ReadDeviceProfiler", None)  # before anything is wrapped
        self._on = profiling() and callable(self._read_fn)
        self._every = cadence()
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
        if not (self._on and self._every) or ProfilerDrain._wrapping:
            return self
        mods = [self._ttnn] + [
            v
            for v in vars(self._ttnn).values()
            if isinstance(v, types.ModuleType) and v.__name__.startswith(self._ttnn.__name__ + ".")
        ]
        for mod in mods:
            for n in dir(mod):
                op = getattr(mod, n, None)
                if type(op).__name__ == "FastOperation":
                    self._orig.append((mod, n, op))
                    setattr(mod, n, self._wrap(op))
        if self._orig and callable(self._read_fn):
            self._orig.append((self._ttnn, "ReadDeviceProfiler", self._read_fn))
            self._ttnn.ReadDeviceProfiler = self._watch_reads(self._read_fn)
        ProfilerDrain._wrapping = bool(self._orig)
        return self

    def __exit__(self, *exc):
        if self._orig:
            for mod, n, op in reversed(self._orig):
                setattr(mod, n, op)
            self._orig = []
            ProfilerDrain._wrapping = False
        if self._final:
            self.read()
        return False


# --- pytest plugin: `-p <this module>` on the profiled run -----------------------------------------
_session = None


def pytest_configure(config):
    global _session
    if not (profiling() and cadence()):
        return
    try:
        import ttnn
    except Exception:  # noqa: BLE001 -- no ttnn, nothing to drain
        return
    _session = ProfilerDrain(ttnn, final_read=False).__enter__()


def pytest_unconfigure(config):
    global _session
    if _session is not None:
        _session.__exit__(None, None, None)
        _session = None

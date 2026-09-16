# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side ttnn op profiler.

Tracy op-level profiling is blocked on the 5B pipeline (the per-core 12000-marker
device buffer overflows and the host<->device join then fails). For an *eager*
section like the Wan VAE decode the interesting question is answerable without it:

  * ``mode="dispatch"``  -- wrap every ttnn op and record host wall time with no
    synchronisation. Because dispatch is async this measures how long the host
    spends *issuing* work. If that sum approaches the section wall time, the
    section is host-bound and tracing (which replays a captured command stream)
    is the right fix.
  * ``mode="sync"``      -- synchronise the mesh after every op, so each op's
    measurement includes the device executing it. Serialises the pipeline, so the
    total overstates the true device time, but the *ranking* is by device cost.

Both are pure monkeypatching of module-level ttnn callables; nothing in the model
is modified, and the patch is reverted on exit.
"""

import sys
import time
import types

import ttnn

# Device/session management, config objects and pure-python helpers: wrapping these
# measures nothing useful and in some cases (synchronize) would recurse.
_DENY = {
    "synchronize_device",
    "close_mesh_device",
    "open_mesh_device",
    "close_device",
    "open_device",
    "CoreCoord",
    "CoreGrid",
    "CoreRange",
    "CoreRangeSet",
    "MeshShape",
    "MeshCoordinate",
    "distributed_context_get_rank",
    "init_device_compute_kernel_config",
    "create_sharded_memory_config",
    "ConcatMesh2dToTensor",
    "ShardTensor2dMesh",
    "ReplicateTensorToMesh",
    "get_memory_config",
    "dump_device_profiler",
}

_MODULES = ("", "experimental", "transformer", "distributed")


def _iter_targets():
    for modname in _MODULES:
        mod = ttnn if modname == "" else getattr(ttnn, modname, None)
        if mod is None:
            continue
        for name in dir(mod):
            if name.startswith("_") or name in _DENY:
                continue
            try:
                obj = getattr(mod, name)
            except Exception:
                continue
            if isinstance(obj, (type, types.ModuleType)):
                continue
            if not callable(obj):
                continue
            yield mod, modname, name, obj


class HostOpProfile:
    """Accumulated per-op and per-callsite timings for one profiled region."""

    def __init__(self, mode, wall, by_op, by_site):
        self.mode = mode
        self.wall = wall
        self.by_op = by_op  # name -> [count, total_s]
        self.by_site = by_site  # (name, site) -> [count, total_s]

    @property
    def total(self):
        return sum(v[1] for v in self.by_op.values())

    @property
    def count(self):
        return sum(v[0] for v in self.by_op.values())

    def _pct(self, x):
        return 100.0 * x / self.wall if self.wall else 0.0

    def top_ops(self, n=30):
        return sorted(self.by_op.items(), key=lambda kv: -kv[1][1])[:n]

    def top_sites(self, n=35):
        return sorted(self.by_site.items(), key=lambda kv: -kv[1][1])[:n]

    def report(self, title=""):
        lines = [f"=== host op profile [{self.mode}] {title} ==="]
        lines.append(
            f"wall={self.wall:.3f}s  instrumented_sum={self.total:.3f}s "
            f"({self._pct(self.total):.1f}% of wall)  ops={self.count}"
        )
        lines.append(f"{'op':<46}{'count':>8}{'total_s':>12}{'%wall':>9}{'us/call':>11}")
        for name, (cnt, tot) in self.top_ops():
            lines.append(f"{name:<46}{cnt:>8}{tot:>12.4f}{self._pct(tot):>8.1f}%{1e6 * tot / cnt:>11.1f}")
        lines.append("--- by callsite ---")
        lines.append(f"{'op @ callsite':<74}{'count':>8}{'total_s':>12}{'%wall':>9}")
        for (name, site), (cnt, tot) in self.top_sites():
            lines.append(f"{name + ' @ ' + site:<74}{cnt:>8}{tot:>12.4f}{self._pct(tot):>8.1f}%")
        return "\n".join(lines)


def _callsite(depth=6):
    """Nearest frame inside models/tt_dit, so the site points at model code."""
    try:
        f = sys._getframe(2)
    except ValueError:
        return "?"
    for _ in range(depth):
        if f is None:
            return "?"
        fn = f.f_code.co_filename
        if "tt_dit" in fn and "_ttnn_host_profiler" not in fn:
            return f"{fn.rsplit('/', 1)[-1]}:{f.f_lineno}"
        f = f.f_back
    return "?"


class profile_ttnn_ops:
    """Context manager. ``mode`` is "dispatch" or "sync"; "sync" needs ``device``."""

    def __init__(self, mode="dispatch", device=None):
        assert mode in ("dispatch", "sync"), mode
        if mode == "sync":
            assert device is not None, "sync mode needs the mesh device"
        self.mode = mode
        self.device = device
        self._orig = []
        self.result = None

    def __enter__(self):
        by_op = {}
        by_site = {}
        self._by_op, self._by_site = by_op, by_site
        sync = self.mode == "sync"
        device = self.device
        perf = time.perf_counter

        for mod, modname, name, fn in _iter_targets():
            key = f"{modname + '.' if modname else ''}{name}"

            def make(fn=fn, key=key):
                def wrapper(*args, **kwargs):
                    t0 = perf()
                    out = fn(*args, **kwargs)
                    if sync:
                        ttnn.synchronize_device(device)
                    dt = perf() - t0
                    rec = by_op.get(key)
                    if rec is None:
                        by_op[key] = [1, dt]
                    else:
                        rec[0] += 1
                        rec[1] += dt
                    sk = (key, _callsite())
                    rec = by_site.get(sk)
                    if rec is None:
                        by_site[sk] = [1, dt]
                    else:
                        rec[0] += 1
                        rec[1] += dt
                    return out

                return wrapper

            try:
                setattr(mod, name, make())
            except Exception:
                continue
            self._orig.append((mod, name, fn))

        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        wall = time.perf_counter() - self._t0
        for mod, name, fn in self._orig:
            try:
                setattr(mod, name, fn)
            except Exception:
                pass
        self._orig = []
        self.result = HostOpProfile(self.mode, wall, self._by_op, self._by_site)
        return False

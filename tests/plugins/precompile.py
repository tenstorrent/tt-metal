# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Precompile: build the kernels a test session needs before or while it runs, in a child process on a mock device.

Loaded with ``-p tests.plugins.precompile`` (``scripts/run_safe_pytest.sh`` does it). The pytest process the user
started (the parent) collects, decides whether precompile pays off, and then runs its tests exactly as plain pytest
would. Before that it starts a second pytest (the child) over the same tests:

1. A short helper process opens the real cluster once and writes its descriptor (chips, harvesting, links).
2. The child runs with ``TT_METAL_MOCK_CLUSTER_DESC_PATH`` set to that descriptor, so every device it opens is a mock
   of this machine: same arch, harvesting and dispatch configuration, no hardware behind it. It runs the tests
   normally; with the build sink of a ``ttnn.graph.CompileSession`` on, every program compiled for the mock device
   queues its kernels, and the session's workers build them into the on-disk kernel cache.
3. The parent's real run finds those kernels in the cache.

What the separation buys:
  - Nothing the dry run does to Python state (stand-ins, memoized results, environment) reaches the real run: it
    happens in another process.
  - The dry run cannot touch the hardware: its devices are mocks.
  - While a body runs, the child refuses Python file writes outside its own directory, which holds a private temp
    directory it gets as TMPDIR (a Python audit hook). Results computed from fake or stand-in values never reach a
    golden, reference or weight cache on disk that the real run would read, the shared /tmp included.
The kernel cache is the only thing passed from child to parent. A kernel the child misses or builds wrong is compiled
by the real run when it needs it, as without precompile.

Knobs (environment):
  PRECOMPILE_WAIT=1               start the real run only after the child has finished building (default 0: start it
                                  right away; the child keeps building alongside and is stopped when the real run ends)
  PRECOMPILE_DECIDE=auto|on       auto (default): on only for more than PRECOMPILE_NARROW_MAX_IDS (default 3) collected
                                  tests outside _SKIP_PATHS; on: always, still without those
  PRECOMPILE_WORKERS=N            child build threads (0 = all cores)
  PRECOMPILE_JIT_SERVER=h:p[,h:p] child builds go to JIT compile server(s); after the first refused or lost connection,
                                  or a reply slower than TT_METAL_JIT_SERVER_TIMEOUT_S (default 240), the rest build
                                  locally. The child then sets TT_METAL_JIT_PREPROCESS=1 so the real run can reuse what
                                  the servers build.
  PRECOMPILE_REMOTE_MIN_KERNELS=N use the server only once N kernels are queued (default 10)
  PRECOMPILE_CHILD_IDLE=0|1       run the child at SCHED_IDLE (default 1, or 0 with PRECOMPILE_WAIT=1)
  PRECOMPILE_CHILD_STALL_S=S      stop the child when it finishes no test for S seconds (default 900)
  PRECOMPILE_CHILD_MAX_ATTEMPTS=N start a child that ended early again from the next test, up to N children in
                                  all (default 20)
  PRECOMPILE_CHILD_DIR=dir        where the child's log, descriptor and result go (default: a new temp dir)
  PRECOMPILE_READONLY_DISK=0      let dry-run bodies write files (debug)
  PRECOMPILE_VERIFY=1             check the child's binaries instead of using them: the child builds into a cache of
                                  its own, the real run compiles as without precompile, and afterwards every ELF both
                                  built is compared by its loaded sections (verify_* fields; differences are listed)
  PRECOMPILE_FAKE_DATA=0          real torch data in the child (default 1: fake, see _FakeDataMode)
  PRECOMPILE_FAKE_BUDGET=N        with fake data, a body stops after N passing values with no new program; an op run
                                  on zero-filled stand-ins counts as PRECOMPILE_ZERO_FILL_COST plus one per
                                  thousand elements filled (defaults 10000 and 200)

Tests marked ``precompile_real_data`` always run with real data in the child. Tests under _SKIP_PATHS never run in the
child: their kernels compile in the real run, as without precompile.

The child's dry run differs from a plain run in ways that only cost or save it time; the real run is untouched:
  - Asserts in test code are soft (see _SoftAssert): the condition runs, a failure is counted and the body goes on.
  - No failure tracebacks (pytest re-reads the source file for each), and the heap left by collection is frozen, so
    a test's own gc.collect() does not walk it.
  - Host reference math, numeric verifiers and torch-to-device conversion are shape-only stand-ins (_stand_ins).
  - Trace capture stops the body; graph capture runs on the mock device.

The plugin prints one ``PRECOMPILE_RESULT:`` line of key=value fields for harnesses to parse.
"""

from __future__ import annotations

import ast
import builtins
import collections
import contextlib
import dataclasses
import gc
import glob
import importlib.abc
import importlib.machinery
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import pytest

_ROLE_ENV = "PRECOMPILE_ROLE"
_RESULT_ENV = "PRECOMPILE_CHILD_RESULT"
_PROGRESS_ENV = "PRECOMPILE_CHILD_PROGRESS"
_IDS_ENV = "PRECOMPILE_CHILD_IDS"
_IS_CHILD = os.environ.get(_ROLE_ENV) == "child"

_WAIT = os.environ.get("PRECOMPILE_WAIT", "0") == "1"
_DECIDE = os.environ.get("PRECOMPILE_DECIDE", "auto")
_NARROW_MAX_IDS = int(os.environ.get("PRECOMPILE_NARROW_MAX_IDS", "3"))
_WORKERS = int(os.environ.get("PRECOMPILE_WORKERS", "0"))
_JIT_SERVER = [e for e in os.environ.get("PRECOMPILE_JIT_SERVER", "").split(",") if e]
_REMOTE_MIN_KERNELS = int(os.environ.get("PRECOMPILE_REMOTE_MIN_KERNELS", "10"))
_STALL_S = float(os.environ.get("PRECOMPILE_CHILD_STALL_S", "900"))
_MAX_ATTEMPTS = int(os.environ.get("PRECOMPILE_CHILD_MAX_ATTEMPTS", "20"))
_READONLY_DISK = os.environ.get("PRECOMPILE_READONLY_DISK", "1") == "1"
_VERIFY = os.environ.get("PRECOMPILE_VERIFY") == "1"
_DESCRIPTOR_TIMEOUT_S = float(os.environ.get("PRECOMPILE_DESCRIPTOR_TIMEOUT_S", "300"))
# The child at SCHED_IDLE (see _child_preexec) leaves the cores to the real run it runs alongside. In wait mode nothing
# runs alongside it.
_CHILD_IDLE = os.environ.get("PRECOMPILE_CHILD_IDLE", "0" if _WAIT else "1") == "1"
# Fake test data in the child (see _FakeDataMode): torch tensors a body creates, and what it reads back from the mock
# device, carry shape and dtype but no values, so reference and comparison math costs next to nothing. A body that
# looks at a value before it has built a program runs again with real data.
_FAKE_DATA = os.environ.get("PRECOMPILE_FAKE_DATA", "1") == "1"
# Bodies of one test function that may need real data before the rest of it runs with real data straight away.
_FAKE_DATA_TRIES = int(os.environ.get("PRECOMPILE_FAKE_DATA_TRIES", "2"))
# After its first program, a body that takes this many passing values without building another one stops (see
# _FakeDataMode._spend); an op run on zero-filled stand-ins costs _ZERO_FILL_COST of them plus one per thousand
# elements filled, about its time relative to a passing value.
_FAKE_BUDGET = int(os.environ.get("PRECOMPILE_FAKE_BUDGET", "10000"))
_ZERO_FILL_COST = int(os.environ.get("PRECOMPILE_ZERO_FILL_COST", "200"))
_REAL_DATA_MARKER = "precompile_real_data"

# Tests the child leaves out, by node id prefix (relative to the rootdir). Each entry was measured to cost more in the
# child than it saves the real run, for the reason given; precompile is off when every selected test is under one.
_SKIP_PATHS = (
    # Quantization scales are read from the inputs before the first device op, so every body runs again with real
    # data: the child takes about half the real run's time for a fifth of its compile.
    "tests/ttnn/nightly/unit_tests/operations/eltwise/test_quantization.py",
    # Most of the child's time goes to host conversion of large cache tensors: 130 s for 27 s of compile.
    "tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py",
    # A few seconds of tests: starting the child costs more than the compile it saves.
    "tests/ttnn/unit_tests/operations/point_to_point/",
)

# Options of the parent's command line the child must not inherit: they would stop it early, write the parent's
# reports, or open a debugger. Options taking a value are matched with and without "=value".
_DROP_FLAGS = {"--pdb", "--trace", "--lf", "--last-failed", "--ff", "--failed-first", "--sw", "--stepwise", "--nf"}
_DROP_WITH_VALUE = {"--pdbcls", "--junitxml", "--junit-xml", "--resultlog", "--report-log"}


def _holds_tt_device():
    """True when this process has a Tenstorrent device node open, i.e. it already opened the cluster."""
    try:
        fds = os.listdir("/proc/self/fd")
    except OSError:
        return False
    for fd in fds:
        try:
            if os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/"):
                return True
        except OSError:
            continue
    return False


# ----------------------------------------------------------------------------------------------------------------
# Child: stand-ins for host work whose values mean nothing in the dry run
# ----------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _DryRunStats:
    bodies: int = 0
    skipped_capture: int = 0
    skipped_socket: int = 0
    shallow: int = 0
    fallback: int = 0


_STATS = _DryRunStats()


_HOST_DTYPES = {}


def _from_torch_dtype(real_from_torch, torch_dtype):
    """The ttnn dtype real_from_torch picks for torch_dtype, read from a one-element host conversion."""
    import torch

    if torch_dtype not in _HOST_DTYPES:
        _HOST_DTYPES[torch_dtype] = real_from_torch(torch.zeros(1, dtype=torch_dtype)).dtype
    return _HOST_DTYPES[torch_dtype]


def _mark_stand_in(replacement, original):
    # A fresh function per install: one stand-in serves several names (randn and rand, layer_norm and
    # group_norm), and each name must restore to its own original.
    def stand_in(*args, **kwargs):
        return replacement(*args, **kwargs)

    stand_in._precompile_original = original
    return stand_in


class _AttrPatch:
    """Reversibly set ``obj.attr = build(original)``."""

    def __init__(self, obj, attr, build):
        self._obj, self._attr, self._build = obj, attr, build

    def __enter__(self):
        self._orig = getattr(self._obj, self._attr)
        setattr(self._obj, self._attr, _mark_stand_in(self._build(self._orig), self._orig))
        return self

    def __exit__(self, *exc):
        setattr(self._obj, self._attr, self._orig)
        return False


# (home_module, attr) -> (original, [modules exposing it]); memoizes the sys.modules walk across bodies.
_REBIND_TARGET_CACHE = {}


class _RebindPatch:
    """Reversibly rebind every module attribute that ``is`` ``home_module.attr``, so helpers imported with
    ``from x import y`` are reached too. No-op if the home module isn't imported or lacks the attribute."""

    def __init__(self, home_module, attr, build):
        self._home_module, self._attr, self._build = home_module, attr, build
        self._orig = None
        self._patched = []

    def __enter__(self):
        self._patched = []
        home = sys.modules.get(self._home_module)
        self._orig = getattr(home, self._attr, None) if home is not None else None
        # Already a stand-in: the same function is exposed under another home module and an earlier patch
        # rebound it. Patching again would record the stand-in as the original.
        if self._orig is None or hasattr(self._orig, "_precompile_original"):
            self._orig = None
            return self
        replacement = _mark_stand_in(self._build(self._orig), self._orig)
        key = (self._home_module, self._attr)
        cached = _REBIND_TARGET_CACHE.get(key)
        if cached is not None and cached[0] is self._orig:
            targets = cached[1]
        else:
            targets = []
            for m in list(sys.modules.values()):
                try:
                    if getattr(m, self._attr, None) is self._orig:
                        targets.append(m)
                except Exception:
                    pass
            _REBIND_TARGET_CACHE[key] = (self._orig, targets)
        for m in targets:
            setattr(m, self._attr, replacement)
        self._patched = targets
        return self

    def __exit__(self, *exc):
        for m in self._patched:
            setattr(m, self._attr, self._orig)
        return False


def _try_shallow(alloc):
    try:
        t = alloc()
        _STATS.shallow += 1
        return t
    except Exception:
        _STATS.fallback += 1
        return None


def _reference_op_patches():
    """Host reference compute -> shape-correct zeros. Programs depend on shapes and config, not values.
    Random inputs stay real: zeros would break host code that derives parameters from the data, such as a
    quantization scale of (max - min) / range."""
    import torch
    import torch.nn.functional as F

    real_matmul = torch.matmul
    real_sdpa = getattr(F, "scaled_dot_product_attention", None)

    def _ntuple(n, x):
        return tuple(x) if isinstance(x, (tuple, list)) else (x,) * n

    def _fast_conv(input, weight, stride, padding, dilation, n):
        s, p, d = _ntuple(n, stride), _ntuple(n, padding), _ntuple(n, dilation)
        spatial = [(input.shape[i - n] + 2 * p[i] - d[i] * (weight.shape[i - n] - 1) - 1) // s[i] + 1 for i in range(n)]
        return input.new_zeros((input.shape[0], weight.shape[0], *spatial))

    def _fast_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return _fast_conv(input, weight, stride, padding, dilation, 2)

    def _fast_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return _fast_conv(input, weight, stride, padding, dilation, 3)

    def _fast_norm(input, *a, **k):
        return torch.zeros_like(input)

    def _fast_matmul(a, b):
        try:
            shp = tuple(real_matmul(a.to("meta"), b.to("meta")).shape)
            return torch.zeros(shp, dtype=getattr(a, "dtype", torch.bfloat16), device=getattr(a, "device", "cpu"))
        except Exception:
            return real_matmul(a, b)

    def _fast_bmm(a, b, *x, **k):
        return _fast_matmul(a, b)

    def _fast_tensor_matmul(self, other):
        return _fast_matmul(self, other)

    def _fast_sdpa(query, key, value, *a, **k):
        try:
            return torch.zeros((*query.shape[:-1], value.shape[-1]), dtype=query.dtype, device=query.device)
        except Exception:
            return real_sdpa(query, key, value, *a, **k)

    def _fast_manual_seed(seed):
        # torch.manual_seed also seeds accelerator backends, recording a Python stack for each; the CPU
        # generator is all host RNG needs.
        return torch.random.default_generator.manual_seed(int(seed))

    patches = [
        _AttrPatch(torch, "manual_seed", lambda _orig: _fast_manual_seed),
        _AttrPatch(F, "conv2d", lambda _orig: _fast_conv2d),
        _AttrPatch(F, "conv3d", lambda _orig: _fast_conv3d),
        _AttrPatch(F, "layer_norm", lambda _orig: _fast_norm),
        _AttrPatch(F, "group_norm", lambda _orig: _fast_norm),
        _AttrPatch(torch, "matmul", lambda _orig: _fast_matmul),
        _AttrPatch(torch, "bmm", lambda _orig: _fast_bmm),
        _AttrPatch(torch.Tensor, "matmul", lambda _orig: _fast_tensor_matmul),
        _AttrPatch(torch.Tensor, "__matmul__", lambda _orig: _fast_tensor_matmul),
    ]
    if real_sdpa is not None:
        patches.append(_AttrPatch(F, "scaled_dot_product_attention", lambda _orig: _fast_sdpa))
    return patches


def _verifier_patches():
    """Numeric verifiers -> no-ops. A verifier failing on a meaningless readback would abort the rest of a
    multi-op body and lose its ops."""

    def _pcc(*a, **k):
        return (True, 0.999999)

    def _assert_numeric_metrics(*a, **k):
        return (True, "precompile-stub")

    def _matches(*a, **k):
        return (True, "precompile-stub")

    def _true(*a, **k):
        return True

    def _none(*a, **k):
        return None

    import torch

    return [
        _RebindPatch("tests.ttnn.utils_for_testing", "comp_pcc", lambda _orig: _pcc),
        _RebindPatch("models.common.utility_functions", "comp_pcc", lambda _orig: _pcc),
        _RebindPatch("tests.ttnn.utils_for_testing", "assert_numeric_metrics", lambda _orig: _assert_numeric_metrics),
        _RebindPatch("ttnn.comparison", "comp_ulp", lambda _orig: _matches),
        _RebindPatch("tt_py_test_utils_common.utility_functions", "comp_allclose", lambda _orig: _matches),
        _RebindPatch("tt_py_test_utils_common.utility_functions", "comp_equal", lambda _orig: _matches),
        _AttrPatch(torch, "equal", lambda _orig: _true),
        _AttrPatch(torch, "allclose", lambda _orig: _true),
        _AttrPatch(torch.testing, "assert_close", lambda _orig: _none),
    ]


def _boundary_patches():
    """ttnn <-> torch boundary -> shape-only device allocation; program build needs only the spec.
    Mesh mappers, custom tiles and host or meta tensors fall back to the real path."""
    import ttnn

    def _build_from_torch(real_from_torch):
        def _shallow_from_torch(
            tensor,
            dtype=None,
            *,
            spec=None,
            tile=None,
            layout=None,
            device=None,
            memory_config=None,
            mesh_mapper=None,
            **kw,
        ):
            if (
                tensor is not None
                and device is not None
                and mesh_mapper is None
                and tile is None
                and not getattr(tensor, "is_meta", False)
            ):

                def _alloc():
                    if spec is not None:
                        dt, lay, mc = spec.dtype, spec.layout, spec.memory_config
                        shp = ttnn.Shape(tuple(spec.shape))
                    else:
                        dt = dtype or _from_torch_dtype(real_from_torch, tensor.dtype)
                        lay = layout or ttnn.ROW_MAJOR_LAYOUT
                        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
                        shp = ttnn.Shape(tuple(tensor.shape))
                    return ttnn.allocate_tensor_on_device(shp, dt, lay, device, mc)

                t = _try_shallow(_alloc)
                if t is not None:
                    return t
            return real_from_torch(
                tensor,
                dtype,
                spec=spec,
                tile=tile,
                layout=layout,
                device=device,
                memory_config=memory_config,
                mesh_mapper=mesh_mapper,
                **kw,
            )

        return _shallow_from_torch

    def _build_torch2tt(real_t2tt):
        interleaved_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED)

        def _shallow_torch2tt(
            py_tensor, tt_device, tt_layout=ttnn.TILE_LAYOUT, tt_memory_config=interleaved_mc, tt_dtype=ttnn.bfloat16
        ):
            if tt_device is not None and not getattr(py_tensor, "is_meta", False):

                def _alloc():
                    size = list(py_tensor.size())
                    while len(size) < 4:
                        size.insert(0, 1)
                    return ttnn.allocate_tensor_on_device(
                        ttnn.Shape(tuple(size)), tt_dtype, tt_layout, tt_device, tt_memory_config
                    )

                t = _try_shallow(_alloc)
                if t is not None:
                    return t
            return real_t2tt(py_tensor, tt_device, tt_layout, tt_memory_config, tt_dtype)

        return _shallow_torch2tt

    return [
        _AttrPatch(ttnn, "from_torch", _build_from_torch),
        _RebindPatch("models.common.utility_functions", "torch2tt_tensor", _build_torch2tt),
    ]


class _DryRunCaptureStop(Exception):
    """A dry-run body reached trace or graph capture or a host-device socket transfer, which need real dispatch."""


def _capture_guard_patches():
    """Trace capture -> stop the body there: replaying a trace needs the binaries a mock device never loads. Ops built
    before it are already queued. Graph capture runs on the mock device like any other run."""

    def _build(name):
        def _stop(*a, **k):
            _STATS.skipped_capture += 1
            raise _DryRunCaptureStop(f"precompile dry run: stopped at {name}")

        return lambda _orig: _stop

    return [
        _RebindPatch("ttnn", "begin_trace_capture", _build("begin_trace_capture")),
        _RebindPatch("ttnn", "execute_trace", _build("execute_trace")),
    ]


def _socket_guard_patches():
    """Host-device socket transfers -> stop the body there. With dispatch blocked the device never sends or acks,
    and the wait is a blocking C++ call that pytest-timeout cannot interrupt; the socket op's program is already
    queued."""
    import ttnn

    def _build(name):
        def _stop(*a, **k):
            _STATS.skipped_socket += 1
            raise _DryRunCaptureStop(f"precompile dry run: stopped at {name}")

        return lambda _orig: _stop

    patches = []
    for cls_name, methods in (
        ("H2DSocket", ("write", "write_tensor", "barrier")),
        ("D2HSocket", ("read", "read_tensor", "barrier")),
    ):
        cls = getattr(ttnn, cls_name, None)
        for method in methods:
            if cls is not None and hasattr(cls, method):
                patches.append(_AttrPatch(cls, method, _build(f"{cls_name}.{method}")))
    return patches


@contextlib.contextmanager
def _stand_ins():
    with contextlib.ExitStack() as stack:
        for patch in (
            _capture_guard_patches()
            + _socket_guard_patches()
            + _reference_op_patches()
            + _verifier_patches()
            + _boundary_patches()
        ):
            stack.enter_context(patch)
        yield


def _cache_root():
    return os.environ.get("TT_METAL_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "tt-metal-cache")


def _kernel_build_dirs():
    """One entry per kernel binary in the kernel cache (<cache>/<build>/kernels/<name>/<hash>)."""
    root = _cache_root()
    dirs = set()
    for depth in ("", "*", os.path.join("*", "*")):
        dirs.update(glob.glob(os.path.join(root, depth, "kernels", "*", "*")))
    return dirs


def _loaded_sections(path):
    """{name: bytes} of an ELF's sections that occupy memory and have contents (SHF_ALLOC, not NOBITS): what reaches
    the device. Debug info and symbol tables, which name the cache directory the ELF was built in, are left out."""
    import struct

    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF":
        return {"<not an ELF>": data}
    is64, end = data[4] == 2, "<" if data[5] == 1 else ">"
    if is64:
        shoff, shentsize, shnum, shstrndx = struct.unpack_from(end + "Q", data, 0x28)[0], *struct.unpack_from(
            end + "HHH", data, 0x3A
        )
        layout, fields = end + "IIQQQQ", (0, 1, 2, 4, 5)
    else:
        shoff, shentsize, shnum, shstrndx = struct.unpack_from(end + "I", data, 0x20)[0], *struct.unpack_from(
            end + "HHH", data, 0x2E
        )
        layout, fields = end + "IIIIII", (0, 1, 2, 4, 5)
    headers = []
    for i in range(shnum):
        raw = struct.unpack_from(layout, data, shoff + i * shentsize)
        headers.append(tuple(raw[k] for k in fields))  # name, type, flags, offset, size
    strtab = headers[shstrndx]
    names = data[strtab[3] : strtab[3] + strtab[4]]
    sections = {}
    for name, sh_type, flags, offset, size in headers:
        if flags & 0x2 and sh_type != 8:  # SHF_ALLOC, not SHT_NOBITS
            sections[names[name : names.index(b"\0", name)].decode()] = data[offset : offset + size]
    return sections


def _verify_binaries(child_root, real_root):
    """Compare every ELF both caches hold at the same relative path (firmware and kernels, keyed by build key and
    kernel hash) by its loaded sections. Returns (compared, differing paths)."""
    compared, differing = 0, []
    for path in glob.glob(os.path.join(child_root, "**", "*.elf"), recursive=True):
        rel = os.path.relpath(path, child_root)
        real = os.path.join(real_root, rel)
        if not os.path.exists(real):
            continue
        compared += 1
        try:
            if _loaded_sections(path) != _loaded_sections(real):
                differing.append(rel)
        except (OSError, ValueError, IndexError) as e:
            differing.append(f"{rel} (unreadable: {e})")
    return compared, sorted(differing)


def _emit_result(**fields):
    line = " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in fields.items())
    print(f"PRECOMPILE_RESULT: {line}", flush=True)


def _stats_dict(stats):
    return {
        "programs_added": stats.programs_added,
        "programs_built": stats.programs_built,
        "kernels_queued": stats.kernels_queued,
        "kernels_built": stats.kernels_built,
        "kernels_preempted": stats.kernels_preempted,
        "kernels_remote": stats.kernels_remote,
        "kernels_remote_fallback": stats.kernels_remote_fallback,
        "kernels_failed": stats.kernels_failed,
        "kernels_dropped": stats.kernels_dropped,
        "kernels_pending": stats.kernels_pending,
        "kernels_running": stats.kernels_running,
        "busy_s": stats.busy_seconds,
    }


# ----------------------------------------------------------------------------------------------------------------
# Child: the dry run
# ----------------------------------------------------------------------------------------------------------------

_SESSION = None
_BODY_ACTIVE = False
_STOP_SITES = collections.Counter()
_WRITES_REFUSED = collections.Counter()
_FAILED_AT_SAFE = 0
_WRITE_ALLOWED_ROOTS = ()


class _DryRunWriteRefused(Exception):
    """A dry-run body tried to write a file outside the child's own directory."""


_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _is_write(event, args):
    if event == "open":
        path, mode, flags = args
        if isinstance(path, int):
            return None
        if mode is not None:
            return path if any(c in mode for c in "wax+") else None
        return path if flags is not None and flags & _WRITE_FLAGS else None
    if event in ("os.remove", "os.rmdir", "os.mkdir", "os.truncate"):
        return args[0]
    if event in ("os.rename", "os.replace"):
        return args[1]
    return None


def _audit_hook(event, args):
    """Refuse writes outside the child's own directory (its private TMPDIR included) while a dry-run body runs. Audit hooks cannot be removed, which is
    fine here: this process exists only for the dry run."""
    if not _BODY_ACTIVE:
        return
    try:
        path = _is_write(event, args)
    except (TypeError, ValueError):
        return
    if path is None:
        return
    _refuse_write(path, event)


def _refuse_write(path, what):
    """Raise _DryRunWriteRefused when a body is running and path is outside the roots the child may write."""
    if not _BODY_ACTIVE:
        return
    real = os.path.realpath(os.fsdecode(path))
    if any(real == root or real.startswith(root + os.sep) for root in _WRITE_ALLOWED_ROOTS):
        return
    _WRITES_REFUSED[real] += 1
    raise _DryRunWriteRefused(f"precompile dry run: refused {what} on {real}")


def _guard_native_writers(owner, names):
    """Writers that open their file in native code, out of the audit hook's sight (torch.save to a path, ttnn's
    flatbuffer dumps, safetensors), go through the same check. Each takes the file name as its first file argument."""
    for name, file_arg in names:
        real = getattr(owner, name, None)
        if real is None or hasattr(real, "_precompile_original"):
            continue

        def guarded(*args, _real=real, _name=name, _i=file_arg, **kwargs):
            target = args[_i] if len(args) > _i else None
            if isinstance(target, (str, bytes, os.PathLike)):
                _refuse_write(target, _name)
            return _real(*args, **kwargs)

        guarded._precompile_original = real
        setattr(owner, name, guarded)


def _stop_site(excinfo):
    """'<ExceptionType> at <test file>:<line>' for the innermost frame in a test file, else the innermost frame."""
    tb = excinfo.tb
    site = None
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        if site is None or (("/tests/" in filename or "/models/" in filename) and "/tests/plugins/" not in filename):
            site = f"{os.path.relpath(filename)}:{tb.tb_lineno}"
        tb = tb.tb_next
    return f"{excinfo.type.__name__} at {site}"


# Soft asserts. An assert in test code checks values that mean nothing here (mock device memory, fake data), and a
# failing one stops the body before its later ops build their programs. Test modules, conftests and test helpers are
# compiled with every ``assert cond, msg`` turned into ``with __precompile_soft_assert__: <check>(cond)``: the
# condition still runs, so an op inside it builds its program, and a failure is counted and dropped. pytest's own
# assert rewriting is off in the child (--assert=plain). Library code (ttnn, torch) keeps its asserts.

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_SOFT_ASSERT_ROOTS = tuple(os.path.join(_REPO, d) + os.sep for d in ("tests", "models"))
_SOFT_ASSERT_SKIP = os.path.join(_REPO, "tests", "plugins") + os.sep
_SOFT_ASSERT_NAME = "__precompile_soft_assert__"


def _soft_assert_file(path):
    if not path or not path.endswith(".py"):
        return False
    path = os.path.realpath(path)
    if path.startswith(_SOFT_ASSERT_SKIP):
        return False
    if path.startswith(_SOFT_ASSERT_ROOTS):
        return True
    return os.path.basename(path) == "conftest.py" and path.startswith(_REPO + os.sep)


class _SoftAssert:
    """Entered around the condition of a rewritten assert. Inside, fake data a body looks at gives a passing value even
    before its first program: the value decides only the check, not what the body builds."""

    def __init__(self):
        self.depth = 0
        self.failed = 0

    def check(self, cond):
        if not cond:
            self.failed += 1

    def __enter__(self):
        self.depth += 1

    def __exit__(self, exc_type, exc, tb):
        self.depth -= 1
        if exc_type is not None and issubclass(exc_type, Exception):
            self.failed += 1
            return True
        return False


_SOFT_ASSERT = _SoftAssert()


class _SoftAssertRewriter(ast.NodeTransformer):
    def visit_Assert(self, node):
        check = ast.Attribute(ast.Name(_SOFT_ASSERT_NAME, ast.Load()), "check", ast.Load())
        body = ast.copy_location(ast.Expr(ast.Call(check, [node.test], [])), node)
        return ast.copy_location(ast.With([ast.withitem(ast.Name(_SOFT_ASSERT_NAME, ast.Load()))], [body]), node)


class _SoftAssertLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Always from source: bytecode an earlier plain run cached has the asserts.
        tree = _SoftAssertRewriter().visit(ast.parse(self.get_data(self.path), self.path))
        return compile(ast.fix_missing_locations(tree), self.path, "exec", dont_inherit=True)


class _SoftAssertFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or type(spec.loader) is not importlib.machinery.SourceFileLoader:
            return None
        if not _soft_assert_file(spec.origin):
            return None
        spec.loader = _SoftAssertLoader(fullname, spec.origin)
        return spec


def _install_soft_asserts():
    """At plugin import, which -p puts before the conftests and test modules are imported."""
    setattr(builtins, _SOFT_ASSERT_NAME, _SOFT_ASSERT)
    sys.meta_path.insert(0, _SoftAssertFinder())


if _IS_CHILD:
    _install_soft_asserts()


def _child_configure(config):
    global _WRITE_ALLOWED_ROOTS
    # A failed body's traceback is rendered before any hook can drop it, re-reading the test's source file each time;
    # the child's failures mean nothing.
    config.option.tbstyle = "no"
    sys.dont_write_bytecode = True
    # Only places the real run never reads: the child's directory and the private temp directory the parent made it
    # (TMPDIR, so tempfile works as usual). Not the shared /tmp or the kernel cache: a test that caches a reference
    # there would hand the real run results computed from fake data. Kernel builds are written by C++, which this
    # hook does not see.
    roots = {tempfile.gettempdir(), "/dev/null", "/proc"}
    if os.environ.get("PRECOMPILE_CHILD_DIR"):
        roots.add(os.environ["PRECOMPILE_CHILD_DIR"])
    _WRITE_ALLOWED_ROOTS = tuple(sorted(os.path.realpath(r) for r in roots))
    if _READONLY_DISK:
        sys.addaudithook(_audit_hook)
        import torch
        import torch.serialization

        _guard_native_writers(torch, [("save", 1)])
        _guard_native_writers(torch.serialization, [("save", 1)])
        try:
            import safetensors.torch

            _guard_native_writers(safetensors.torch, [("save_file", 1)])
        except ImportError:
            pass


def _child_collection_modifyitems(config, collected, items):
    # The parent wrote the node ids it selected, in its order. Selection (-k, -m, pytest-split's --splits/--group) is
    # already applied there, and selecting again here would cut that list down (a split of a split), so the child
    # takes the parent's tests from everything it collected, whatever the other selection hooks kept. A test whose
    # id differs between the two collections is left to the real run.
    with open(os.environ[_IDS_ENV]) as f:
        wanted = {line.rstrip("\n"): i for i, line in enumerate(f) if line.strip()}
    kept = [item for item in collected if item.nodeid in wanted]
    kept.sort(key=lambda item: wanted[item.nodeid])
    items[:] = kept


def _child_collection_finish(session):
    # Fixtures that gc.collect() around every test (models/conftest.py) would walk everything import and collection
    # left on the heap each time, ~0.1 s a test; frozen, those objects are skipped.
    gc.collect()
    gc.freeze()


def _child_runtestloop_before(session):
    global _SESSION
    import ttnn

    if _READONLY_DISK:
        # ttnn.dump_tensor and ttnn.as_tensor's weight cache write through these.
        _guard_native_writers(ttnn._ttnn.tensor, [("dump_tensor_flatbuffer", 0), ("dump_overlapped_tensors", 0)])

    _SESSION = ttnn.graph.CompileSession(
        workers=_WORKERS, remote_endpoints=_JIT_SERVER, remote_min_kernels=_REMOTE_MIN_KERNELS
    )
    _SESSION.set_build_sink(True)
    session.config._precompile_t0 = time.monotonic()


def _child_runtestloop_after(session):
    t0 = session.config._precompile_t0
    walk_s = time.monotonic() - t0
    _SESSION.set_build_sink(False)
    at_walk_end = _SESSION.stats()
    _SESSION.wait()
    build_s = time.monotonic() - t0
    _SESSION.close()
    stats = _SESSION.stats()
    result = {
        "items": len(session.items),
        "bodies": _STATS.bodies,
        "skipped_capture": _STATS.skipped_capture,
        "skipped_socket": _STATS.skipped_socket,
        "shallow": _STATS.shallow,
        "fallback": _STATS.fallback,
        "stopped": sum(_STOP_SITES.values()),
        "writes_refused": sum(_WRITES_REFUSED.values()),
        "soft_asserts_failed": _SOFT_ASSERT.failed,
        "walk_s": walk_s,
        "build_s": build_s,
        "kernels_queued_at_walk_end": at_walk_end.kernels_queued,
        "stats": _stats_dict(stats),
        "top_stop_sites": _STOP_SITES.most_common(25),
        "top_refused_writes": _WRITES_REFUSED.most_common(10),
        "fake_data": int(_FAKE_DATA),
        **{f"fake_{k}": v for k, v in _FAKE_STATS.items() if not k.startswith("rerun: ")},
        "fake_rerun_reasons": [
            (k[len("rerun: ") :], v) for k, v in _FAKE_STATS.most_common() if k.startswith("rerun: ")
        ][:15],
    }
    with open(os.environ[_RESULT_ENV], "w") as f:
        json.dump(result, f, indent=1)


_FAKE_STATS = collections.Counter()
_NEEDS_REAL = collections.Counter()


class _FakeDataNeeded(BaseException):
    """A body looked at fake data before it built a program, so the values may decide what it builds. A BaseException
    so that a test's own ``except Exception`` doesn't swallow it."""


class _FakeBudgetStop(BaseException):
    """A body spent its fake-data budget without building a new program (see _FakeDataMode._spend). A BaseException
    so that neither a test's ``except Exception`` nor a soft assert swallows it."""


def _names_meta_tensor(e):
    """True for an error raised because a meta tensor (fake data) reached code that needs values."""
    return re.search(r"\bmeta\b", str(e), re.IGNORECASE) is not None


def _programs_added():
    return _SESSION.stats().programs_added


def _has_meta(x, tensor_type, depth=3):
    """A meta tensor within ``depth`` container levels. Cheaper than a pytree flatten for an op on a long list of
    numbers (torch.tensor(<100k floats>)), which is most of what a body runs before and after its device ops."""
    if isinstance(x, tensor_type):
        return x.is_meta
    if depth and isinstance(x, (list, tuple)):
        for y in x:
            if y is None or isinstance(y, (int, float, bool, complex, str)):
                continue
            if _has_meta(y, tensor_type, depth - 1):
                return True
        return False
    if depth and isinstance(x, dict):
        return any(_has_meta(v, tensor_type, depth - 1) for v in x.values())
    return False


class _FakeDataMode:
    """Torch function mode for one dry-run body: tensors carry shape and dtype only (torch's meta device).

    - Tensor factories without an explicit device (randn, zeros, empty, arange, the *_like forms, ...) make meta
      tensors; torch.tensor keeps its literal values. An op with any meta argument takes its real CPU arguments
      along to meta, so reference and comparison math only infers shapes.
    - ttnn.to_torch of a device tensor gives a meta tensor without reading the mock device, whose contents mean
      nothing; ttnn.from_torch and the ttnn.Tensor constructor take uninitialized host memory of the same shape.
    - Looking at a value (item, bool, tolist, numpy, ...) or writing fake data into a real tensor: before the body
      has built a program this raises _FakeDataNeeded and the body runs again with real data, since the values may
      set its parameters. After, it is almost always a check of the result, so it gets a passing value and the body
      goes on to its later ops. Inside a soft assert it gets a passing value either way.
    - An op with no meta implementation (boolean-mask indexing, nonzero, unique): before the first program, real data
      as above; after, it runs on zero-filled CPU copies and its result goes back to meta.
    - Passing values and zero-filled ops are paid from a budget that a new program refills; a body that runs out
      (an element-by-element compare loop) stops there.
    """

    _FACTORIES = (
        "randn", "rand", "randint", "randperm", "normal", "empty", "empty_strided", "zeros", "ones", "full",
        "arange", "linspace", "logspace", "eye",
        "randn_like", "rand_like", "randint_like", "empty_like", "zeros_like", "ones_like", "full_like",
    )  # fmt: skip
    _INPLACE_DUNDERS = {"__setitem__"} | {
        f"__i{op}__"
        for op in (
            "add",
            "sub",
            "mul",
            "matmul",
            "truediv",
            "floordiv",
            "mod",
            "pow",
            "and",
            "or",
            "xor",
            "lshift",
            "rshift",
        )
    }
    _OBSERVERS = (
        "item",
        "tolist",
        "numpy",
        "__bool__",
        "__float__",
        "__int__",
        "__index__",
        "__complex__",
        "__array__",
    )

    def __init__(self):
        import torch
        from torch.overrides import TorchFunctionMode

        outer = self

        class _Mode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return outer._dispatch(func, args, kwargs or {})

        self._torch = torch
        self._mode = _Mode()
        self._factories = {getattr(torch, n) for n in self._FACTORIES if hasattr(torch, n)}
        self._observers = {getattr(torch.Tensor, n) for n in self._OBSERVERS if hasattr(torch.Tensor, n)}
        self._observer_names = set(self._OBSERVERS)
        self.programs0 = 0
        self._budget_programs = None
        self._budget_left = 0
        self._patches = []

    def _dispatch(self, func, args, kwargs):
        torch = self._torch
        if func in self._factories and kwargs.get("device") is None:
            kwargs = dict(kwargs, device="meta")
            kwargs.pop("generator", None)
            kwargs.pop("pin_memory", None)
            _FAKE_STATS["fake_tensors"] += 1
            return func(*args, **kwargs)
        if not _has_meta(args, torch.Tensor) and not _has_meta(kwargs, torch.Tensor):
            return func(*args, **kwargs)
        flat, spec = torch.utils._pytree.tree_flatten((args, kwargs))
        name = getattr(func, "__name__", "")
        if func in self._observers or name in self._observer_names:
            return self._observe(func, name, args)
        target = kwargs.get("out", args[0] if args else None)
        writes_target = (
            "out" in kwargs or name in self._INPLACE_DUNDERS or (name.endswith("_") and not name.startswith("_"))
        )
        if writes_target and isinstance(target, torch.Tensor) and not target.is_meta:
            # Fake data into a real tensor: its contents would be wrong without any later read noticing.
            self._need_real(f"{name} of fake data into a real tensor")
            _FAKE_STATS["skipped_writes"] += 1
            return target
        flat = [
            x.to("meta") if isinstance(x, torch.Tensor) and not x.is_meta and x.device.type == "cpu" else x
            for x in flat
        ]
        args, kwargs = torch.utils._pytree.tree_unflatten(flat, spec)
        try:
            return func(*args, **kwargs)
        except NotImplementedError:
            # No meta implementation. Before the first program the values may decide what the body builds, and the
            # exception reruns it with real data.
            if _programs_added() == self.programs0:
                raise
        fakes = [x for x in flat if isinstance(x, torch.Tensor) and x.is_meta]
        self._spend(_ZERO_FILL_COST + sum(x.numel() for x in fakes) // 1000)
        _FAKE_STATS["zero_filled_ops"] += 1
        flat = [torch.zeros(tuple(x.shape), dtype=x.dtype) if any(x is f for f in fakes) else x for x in flat]
        args, kwargs = torch.utils._pytree.tree_unflatten(flat, spec)
        out = func(*args, **kwargs)  # the mode is off while its __torch_function__ runs
        return torch.utils._pytree.tree_map(lambda t: t.to("meta") if isinstance(t, torch.Tensor) else t, out)

    def _need_real(self, why):
        if _programs_added() == self.programs0:
            raise _FakeDataNeeded(why)

    def _spend(self, cost):
        """Checks after the last program mean nothing here (the real run makes them), but they can run for minutes: an
        element-by-element compare loop, or a chunked comparison of zero-filled stand-ins. A body that spends
        _FAKE_BUDGET without building a new program stops."""
        programs = _programs_added()
        if programs != self._budget_programs:
            self._budget_programs, self._budget_left = programs, _FAKE_BUDGET
        self._budget_left -= cost
        if self._budget_left < 0 and programs != self.programs0:
            raise _FakeBudgetStop(f"fake-data budget of {_FAKE_BUDGET} spent without a new program")

    def _observe(self, func, name, args):
        torch = self._torch
        if not _SOFT_ASSERT.depth:
            self._need_real(f"Tensor.{name}")
        self._spend(1)
        _FAKE_STATS["passing_values"] += 1
        t = args[0]
        if name in ("__bool__",):
            return True
        if name in ("__float__",):
            return 1.0
        if name in ("__int__", "__index__"):
            return 1
        if name == "__complex__":
            return 1 + 0j
        ones = torch.ones(tuple(t.shape), dtype=t.dtype, device="cpu")
        if name == "item":
            return ones.item()
        if name == "tolist":
            return ones.tolist()
        return ones.float().numpy() if t.dtype == torch.bfloat16 else ones.numpy()

    def _ttnn_patches(self):
        import ttnn

        torch = self._torch

        def _build_to_torch(real):
            def to_torch(tensor, dtype=None, *, torch_rank=None, mesh_composer=None, device=None, cq_id=None):
                if mesh_composer is None and ttnn.is_tensor_storage_on_device(tensor):
                    out = torch.empty(
                        tuple(tensor.shape), dtype=dtype or ttnn.ttnn_dtype_to_torch_dtype(tensor.dtype), device="meta"
                    )
                    while torch_rank is not None and out.dim() > torch_rank:
                        out = out.squeeze(0)
                    _FAKE_STATS["fake_readbacks"] += 1
                    return out
                return real(
                    tensor, dtype, torch_rank=torch_rank, mesh_composer=mesh_composer, device=device, cq_id=cq_id
                )

            return to_torch

        def _build_from_torch(inner):
            def from_torch(tensor=None, *args, **kwargs):
                if isinstance(tensor, torch.Tensor) and tensor.is_meta:
                    # Uninitialized host memory of the same shape: the shape-only stand-in allocates on the device
                    # without reading it, and anything else converts values that mean nothing on a mock device.
                    tensor = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu")
                return inner(tensor, *args, **kwargs)

            return from_torch

        def _build_tensor_init(real_init):
            def __init__(self, *args, **kwargs):
                if args and isinstance(args[0], torch.Tensor) and args[0].is_meta:
                    args = (torch.empty(tuple(args[0].shape), dtype=args[0].dtype, device="cpu"),) + args[1:]
                return real_init(self, *args, **kwargs)

            return __init__

        return [
            _AttrPatch(ttnn, "to_torch", _build_to_torch),
            _AttrPatch(ttnn, "from_torch", _build_from_torch),
            _AttrPatch(ttnn.Tensor, "__init__", _build_tensor_init),
        ]

    def __enter__(self):
        self.programs0 = _programs_added()
        self._patches = self._ttnn_patches()
        for p in self._patches:
            p.__enter__()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        self._mode.__exit__(*exc)
        for p in reversed(self._patches):
            p.__exit__(*exc)
        return False


def _child_runtest_call_wrapper(item):
    global _BODY_ACTIVE
    import ttnn

    _STATS.bodies += 1
    try:
        with _stand_ins():
            _BODY_ACTIVE = True
            try:
                if not _FAKE_DATA:
                    return (yield)
                if item.get_closest_marker(_REAL_DATA_MARKER) is not None:
                    _FAKE_STATS["real_by_marker"] += 1
                    return (yield)
                # Within this run: a test function whose bodies keep needing real data skips the fake attempt.
                func = item.nodeid.split("::")[0] + "::" + getattr(item, "originalname", item.name)
                if _NEEDS_REAL[func] >= _FAKE_DATA_TRIES:
                    _FAKE_STATS["real_first"] += 1
                    return (yield)
                rerun = None
                mode = _FakeDataMode()
                try:
                    with mode:
                        return (yield)
                except _FakeBudgetStop:
                    _FAKE_STATS["budget_stops"] += 1
                    return None
                except _FakeDataNeeded as e:
                    rerun = str(e)
                except Exception as e:
                    # Fake data a consumer couldn't take (a meta tensor reaching C++, a data-dependent shape). Skips
                    # and xfails are BaseExceptions and pass through. After the body's first program an exception
                    # usually comes from a check on meaningless values, and the body just stops there; one that names a
                    # meta tensor is fake data that slipped past the mode (a helper thread runs without it), and the
                    # body's later programs would be lost, so it runs again with real data too.
                    after = _programs_added() != mode.programs0
                    if after and not _names_meta_tensor(e):
                        raise
                    frame = traceback.extract_tb(e.__traceback__)[-1]
                    where = f"{os.path.basename(frame.filename)}:{frame.lineno}"
                    when = "after" if after else "before"
                    rerun = f"{type(e).__name__} at {where} {when} the first program: {str(e).splitlines()[0][:100] if str(e) else ''}"
                if rerun is not None:
                    # The fake attempt's tensors went with its traceback, so the real attempt's buffers get the
                    # addresses the real run gives them.
                    _NEEDS_REAL[func] += 1
                    _FAKE_STATS["real_reruns"] += 1
                    _FAKE_STATS["rerun: " + rerun] += 1
                    item.runtest()
                    return None
            finally:
                _BODY_ACTIVE = False
    finally:
        if ttnn.graph.is_graph_capture_active():
            # A body that stopped inside its capture would leave every later test capturing.
            ttnn.graph.end_graph_capture()
            _FAKE_STATS["graph_captures_ended"] += 1


def _child_makereport(item, call, rep):
    if call.when == "call" and call.excinfo is not None and rep.outcome == "failed":
        _STOP_SITES[_stop_site(call.excinfo)] += 1


def _child_logstart(nodeid):
    progress = os.environ.get(_PROGRESS_ENV)
    if progress:
        # The test being run: if this child dies, the next one starts after it. The parent's stall guard also
        # watches this file's modification time.
        with open(progress, "w") as f:
            f.write(nodeid)


def _child_logfinish(nodeid):
    progress = os.environ.get(_PROGRESS_ENV)
    if progress:
        os.utime(progress, None)
        # Every kernel queued so far is on disk: a child started after this test loses nothing if this one dies.
        # A build that failed since the last safe point may have failed for a reason the next child won't have
        # (out of file descriptors, disk full), so the safe point moves only over a stretch with no failures.
        global _FAILED_AT_SAFE
        stats = _SESSION.stats() if _SESSION is not None else None
        if stats is not None and stats.kernels_pending == 0 and stats.kernels_running == 0:
            if stats.kernels_failed == _FAILED_AT_SAFE:
                with open(progress + ".safe", "w") as f:
                    f.write(nodeid)
            _FAILED_AT_SAFE = stats.kernels_failed


# ----------------------------------------------------------------------------------------------------------------
# Parent: decide, start the child, run the real tests
# ----------------------------------------------------------------------------------------------------------------


def _child_preexec():
    """Runs in the child before exec.

    A parent killed outright (safe pytest's hang handling, SIGKILL) takes the child with it, although the child has
    its own process group. The child runs every test body again, torch references included, so at normal priority
    it competes with the real run for the cores. SCHED_IDLE, inherited by its threads and by the compilers it
    starts, gives it only the CPU time the real run leaves free; a kernel it hasn't built yet the real run builds
    itself, as it would without precompile.
    """
    import ctypes

    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    if _CHILD_IDLE:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))


class _Child:
    """The dry-run child and its supervisor thread. A child that ends without writing its result (a crash, a stall
    or pytest-timeout's os._exit) loses the kernels it had queued but not built, so the next child starts from the
    last test after which its build queue was empty, and leaves out every test a child has died in. One bad test
    costs that test, not every test after it."""

    def __init__(self, config, ids):
        self.dir = os.environ.get("PRECOMPILE_CHILD_DIR") or tempfile.mkdtemp(prefix="precompile_")
        os.makedirs(self.dir, exist_ok=True)
        self.descriptor = os.path.join(self.dir, "cluster_descriptor.yaml")
        self.descriptor_s = None
        self.progress = os.path.join(self.dir, "progress")
        self.log = os.path.join(self.dir, "child.log")
        self.config = config
        self.ids = list(ids)
        self.attempts = []  # (result path, exit code)
        self.crashed = []  # (node id, why)
        self.proc = None
        self.thread = None
        self.stopping = False
        self.t_start = None
        self.t_end = None
        self.outcome = "not-started"

    def capture_descriptor(self):
        """Open the real cluster once, in a process of its own that exits before any test opens a device.

        If this process already holds the chips (a test module asked for the device count at import), a helper
        would wait on the chip lock forever. The cluster exists here already, so it is read in this process.
        """
        t = time.monotonic()
        if _holds_tt_device():
            import ttnn

            shutil.copyfile(ttnn.cluster.serialize_cluster_descriptor(), self.descriptor)
            self.descriptor_s = time.monotonic() - t
            return
        script = "import shutil, sys, ttnn; shutil.copyfile(ttnn.cluster.serialize_cluster_descriptor(), sys.argv[1])"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script, self.descriptor],
                cwd=str(self.config.invocation_params.dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=_DESCRIPTOR_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            tail = "\n".join(out.splitlines()[-20:])
            raise RuntimeError(f"cluster descriptor capture timed out after {_DESCRIPTOR_TIMEOUT_S:.0f}s:\n{tail}")
        self.descriptor_s = time.monotonic() - t
        if proc.returncode != 0 or not os.path.exists(self.descriptor):
            tail = "\n".join(proc.stdout.splitlines()[-20:])
            raise RuntimeError(f"cluster descriptor capture failed (exit {proc.returncode}):\n{tail}")

    def args(self, junit):
        out = []
        skip_value = False
        for arg in self.config.invocation_params.args:
            if skip_value:
                skip_value = False
                continue
            name = arg.split("=", 1)[0]
            if arg in _DROP_FLAGS:
                continue
            if name in _DROP_WITH_VALUE:
                skip_value = "=" not in arg
                continue
            out.append(arg)
        if not any(a == "tests.plugins.precompile" or a.endswith("-ptests.plugins.precompile") for a in out):
            out += ["-p", "tests.plugins.precompile"]
        # Later options win: never stop at a failure, keep JUnit away from the parent's file, leave asserts to the
        # soft-assert loader.
        out += ["--maxfail=0", f"--junitxml={junit}", "-p", "no:cacheprovider", "--assert=plain"]
        return out

    def start(self):
        self.t_start = time.monotonic()
        self.outcome = "running"
        self.thread = threading.Thread(target=self._supervise, name="precompile-child", daemon=True)
        self.thread.start()

    def _launch(self, attempt, ids):
        ids_path = os.path.join(self.dir, f"ids.{attempt}.txt")
        result = os.path.join(self.dir, f"result.{attempt}.json")
        with open(ids_path, "w") as f:
            f.writelines(nodeid + "\n" for nodeid in ids)
        for path in (self.progress, self.progress + ".safe"):
            with open(path, "w"):
                pass
        # The child's temp files stay out of the shared temp directory the real run uses (see _child_configure).
        private_tmp = os.path.join(self.dir, "tmp")
        os.makedirs(private_tmp, exist_ok=True)
        env = dict(os.environ)
        env.update(
            {
                _ROLE_ENV: "child",
                _RESULT_ENV: result,
                _PROGRESS_ENV: self.progress,
                _IDS_ENV: ids_path,
                "TMPDIR": private_tmp,
                "TMP": private_tmp,
                "TEMP": private_tmp,
                "PRECOMPILE_CHILD_DIR": self.dir,
                "TT_METAL_MOCK_CLUSTER_DESC_PATH": self.descriptor,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        # The child's builds go to PRECOMPILE_JIT_SERVER through its session, never through the runtime.
        env.pop("TT_METAL_JIT_SERVER_ENABLE", None)
        if _VERIFY:
            env["TT_METAL_CACHE"] = os.path.join(self.dir, "verify_cache")
        if _JIT_SERVER:
            # The real run is another process and reuses a server-built ELF only through the dependency sidecar
            # that preprocess-and-ship writes next to it; without it every such kernel is compiled again.
            env.setdefault("TT_METAL_JIT_PREPROCESS", "1")
        with open(self.log, "a") as log:
            log.write(f"\n===== precompile child attempt {attempt}: {len(ids)} test(s) =====\n")
            log.flush()
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "pytest", *self.args(os.path.join(self.dir, f"junit.{attempt}.xml"))],
                cwd=str(self.config.invocation_params.dir),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=_child_preexec,
            )
        return result

    def _supervise(self):
        remaining = list(self.ids)
        died_in = set()
        attempt = 0
        while remaining and not self.stopping:
            result = self._launch(attempt, remaining)
            stalled = False
            while self.proc.poll() is None:
                if self.stopping:
                    break
                if time.time() - os.path.getmtime(self.progress) > _STALL_S:
                    stalled = True
                    self._kill()
                    break
                time.sleep(0.5)
            self.proc.wait()
            self.attempts.append((result, self.proc.returncode))
            # pytest exits 0 or 1 when it ran every test; anything else (an internal error, a signal) ended it early,
            # even if it wrote a result on the way out.
            if self.stopping or (self.proc.returncode in (0, 1) and os.path.exists(result)):
                break
            # Ended early: start again after the last safe point, without the tests children died in.
            with open(self.progress) as f:
                current = f.read().strip()
            with open(self.progress + ".safe") as f:
                safe = f.read().strip()
            why = "stalled" if stalled else f"exit {self.proc.returncode}"
            if current not in remaining or attempt + 1 >= _MAX_ATTEMPTS:
                self.crashed.append((current or "<before the first test>", why + ", not resumed"))
                break
            self.crashed.append((current, why))
            died_in.add(current)
            start = remaining.index(safe) + 1 if safe in remaining else 0
            remaining = [nodeid for nodeid in remaining[start:] if nodeid not in died_in]
            attempt += 1
        self.t_end = time.monotonic()
        if not self.stopping:
            self.outcome = "done" if not self.crashed else f"done_after_{len(self.crashed)}_restart(s)"

    def _kill(self):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def wait(self):
        if self.thread is not None:
            self.thread.join()

    def stop(self, why):
        if not self.running():
            return
        self.stopping = True
        if self.proc is not None and self.proc.poll() is None:
            self._kill()
        self.thread.join()
        self.t_end = time.monotonic()
        self.outcome = why

    def read_result(self):
        """The attempts' results added up."""
        total = {}
        stats = collections.Counter()
        stop_sites = collections.Counter()
        refused = collections.Counter()
        reruns = collections.Counter()
        for path, _ in self.attempts:
            try:
                with open(path) as f:
                    r = json.load(f)
            except (OSError, ValueError):
                continue
            for k, v in r.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    total[k] = total.get(k, 0) + v
            stats.update(r.get("stats", {}))
            stop_sites.update(dict(map(tuple, r.get("top_stop_sites", []))))
            refused.update(dict(map(tuple, r.get("top_refused_writes", []))))
            reruns.update(dict(map(tuple, r.get("fake_rerun_reasons", []))))
        if not total:
            return None
        total["stats"] = dict(stats)
        total["top_stop_sites"] = stop_sites.most_common(25)
        total["top_refused_writes"] = refused.most_common(10)
        total["fake_rerun_reasons"] = reruns.most_common(15)
        return total


_CHILD = None
_ENABLED = False
_REASON = ""
_CHILD_IDS = []  # the selected tests the child runs: all but those under _SKIP_PATHS


def _parent_decide(session):
    global _ENABLED, _REASON, _CHILD_IDS
    _CHILD_IDS = [item.nodeid for item in session.items if not item.nodeid.startswith(_SKIP_PATHS)]
    n = len(_CHILD_IDS)
    skipped = len(session.items) - n
    if session.config.option.collectonly:
        _ENABLED, _REASON = False, "--collect-only"
    elif not session.items:
        _ENABLED, _REASON = False, "no tests collected"
    elif n == 0:
        _ENABLED, _REASON = False, f"all {skipped} test(s) are in its skip list"
    elif _DECIDE == "on":
        _ENABLED, _REASON = True, f"forced, {n} test(s)"
    elif n > _NARROW_MAX_IDS:
        _ENABLED, _REASON = True, f"auto: {n} tests collected"
    else:
        _ENABLED, _REASON = False, f"auto: {n} test(s) collected, at most {_NARROW_MAX_IDS}"
    if _ENABLED and skipped:
        _REASON += f"; {skipped} in its skip list compile in the real run"
    print(f"\nPRECOMPILE: {'on' if _ENABLED else 'off'} ({_REASON})", flush=True)


def _parent_before_real_run(session):
    global _CHILD
    if not _ENABLED:
        return
    if os.environ.pop("TT_METAL_JIT_SERVER_ENABLE", None) is not None:
        # The runtime's remote path sends every kernel that has no reuse sidecar, which is every kernel built locally,
        # to the server again. The child's builds use PRECOMPILE_JIT_SERVER.
        print("PRECOMPILE: ignoring TT_METAL_JIT_SERVER_ENABLE; the real run compiles locally", flush=True)
    _CHILD = _Child(session.config, _CHILD_IDS)
    try:
        _CHILD.capture_descriptor()
    except RuntimeError as e:
        print(f"PRECOMPILE: off, {e}", flush=True)
        _CHILD.outcome = "no-descriptor"
        return
    _CHILD.start()
    print(
        f"PRECOMPILE: dry run started on a mock of this machine (descriptor {_CHILD.descriptor_s:.1f}s); "
        f"log {_CHILD.log}",
        flush=True,
    )
    if _WAIT:
        _CHILD.wait()
        print(f"PRECOMPILE: dry run {_CHILD.outcome} after {_CHILD.t_end - _CHILD.t_start:.1f}s", flush=True)
        for nodeid, why in _CHILD.crashed:
            print(f"PRECOMPILE:   child ended early ({why}) in {nodeid}", flush=True)
    session.config._precompile_before_real = _kernel_build_dirs()
    session.config._precompile_t_real = time.monotonic()


def _parent_after_real_run(session):
    if _CHILD is None or _CHILD.thread is None:
        return
    real_s = time.monotonic() - session.config._precompile_t_real
    # Binaries that appeared during the real run. Exact in wait mode; in stream mode it includes the child's.
    real_new = len(_kernel_build_dirs() - session.config._precompile_before_real)
    child_running_at_end = _CHILD.running()
    _CHILD.stop("stopped at real-run end")
    child = _CHILD.read_result() or {}
    stats = child.get("stats", {})
    verify = {}
    if _VERIFY:
        compared, differing = _verify_binaries(os.path.join(_CHILD.dir, "verify_cache"), _cache_root())
        verify = {
            "verify_compared": compared,
            "verify_identical": compared - len(differing),
            "verify_differ": len(differing),
        }
    _emit_result(
        mode="wait" if _WAIT else "stream",
        route="server" if _JIT_SERVER else "local",
        items=len(session.items),
        child_items=len(_CHILD.ids),
        child_outcome=_CHILD.outcome.replace(" ", "_"),
        descriptor_s=_CHILD.descriptor_s or 0.0,
        child_s=(_CHILD.t_end or time.monotonic()) - _CHILD.t_start,
        walk_s=float(child.get("walk_s", -1)),
        build_s=float(child.get("build_s", -1)),
        real_s=real_s,
        real_new_binaries=real_new,
        child_running_at_real_end=int(child_running_at_end),
        bodies=child.get("bodies", -1),
        stopped=child.get("stopped", -1),
        writes_refused=child.get("writes_refused", -1),
        soft_asserts_failed=child.get("soft_asserts_failed", -1),
        skipped_capture=child.get("skipped_capture", -1),
        skipped_socket=child.get("skipped_socket", -1),
        shape_only=child.get("shallow", -1),
        child_attempts=len(_CHILD.attempts),
        child_crashed=len(_CHILD.crashed),
        **{k: v for k, v in child.items() if k.startswith("fake_") and isinstance(v, (int, float))},
        **stats,
        **verify,
    )
    if verify.get("verify_differ"):
        print("PRECOMPILE: binaries the child built differently from the real run:", flush=True)
        for rel in differing[:50]:
            print(f"PRECOMPILE:   {rel}", flush=True)
    if child.get("fake_rerun_reasons"):
        print("PRECOMPILE: why bodies reran with real data (most common first):", flush=True)
        for why, count in child["fake_rerun_reasons"][:10]:
            print(f"PRECOMPILE:   {count:6d}  {why}", flush=True)
    for nodeid, why in _CHILD.crashed:
        print(f"PRECOMPILE: child ended early ({why}) in {nodeid}", flush=True)
    if child.get("top_stop_sites"):
        print("PRECOMPILE: where dry-run bodies stopped (most common first):", flush=True)
        for site, count in child["top_stop_sites"][:10]:
            print(f"PRECOMPILE:   {count:6d}  {site}", flush=True)
    if child.get("top_refused_writes"):
        print("PRECOMPILE: refused dry-run writes:", flush=True)
        for path, count in child["top_refused_writes"]:
            print(f"PRECOMPILE:   {count:6d}  {path}", flush=True)


# ----------------------------------------------------------------------------------------------------------------
# Hooks
# ----------------------------------------------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers", f"{_REAL_DATA_MARKER}: the precompile mock child runs this test with real torch data, never fake"
    )
    if _IS_CHILD:
        _child_configure(config)


@pytest.hookimpl(wrapper=True)
def pytest_collection_modifyitems(config, items):
    collected = list(items)
    result = yield
    if _IS_CHILD:
        _child_collection_modifyitems(config, collected, items)
    return result


def pytest_collection_finish(session):
    if _IS_CHILD:
        _child_collection_finish(session)
    else:
        _parent_decide(session)


@pytest.hookimpl(wrapper=True)
def pytest_runtestloop(session):
    if session.config.option.collectonly or (session.testsfailed and not _IS_CHILD):
        return (yield)
    if _IS_CHILD:
        _child_runtestloop_before(session)
        try:
            return (yield)
        finally:
            _child_runtestloop_after(session)
    _parent_before_real_run(session)
    try:
        return (yield)
    finally:
        _parent_after_real_run(session)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    if not _IS_CHILD:
        return (yield)
    return (yield from _child_runtest_call_wrapper(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    if _IS_CHILD:
        _child_makereport(item, call, rep)
    return rep


def pytest_runtest_logstart(nodeid, location):
    if _IS_CHILD:
        _child_logstart(nodeid)


def pytest_runtest_logfinish(nodeid, location):
    if _IS_CHILD:
        _child_logfinish(nodeid)


def pytest_unconfigure(config):
    # A parent stopped by -x or Ctrl-C still stops its child.
    if not _IS_CHILD and _CHILD is not None:
        _CHILD.stop("stopped at exit")

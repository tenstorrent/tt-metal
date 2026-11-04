# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Precompile for any ttnn pytest suite.

Loaded with ``-p tests.plugins.precompile`` (``scripts/run_safe_pytest.sh --precompile`` does it), the
plugin runs the session twice:

1. A dry run. Every test body runs once with dispatch blocked (NO_DISPATCH graph capture): ops build
   their programs, nothing runs on the device. Each program is handed to a ``ttnn.graph.CompileSession``,
   whose worker threads start building its kernels into the on-disk kernel cache right away, in test
   order. Values are meaningless here, so failures are neutralized at report time and host-side
   reference work is replaced by shape-only stand-ins.
2. The real run, which compiles nothing that was already built. A kernel still queued is built by
   the test that needs it; one being built is waited for.

Knobs (environment):
  PRECOMPILE_WAIT=1               wait for the compile queue to empty before the real run
                                  (default 0: start the real run as soon as the dry run ends)
  PRECOMPILE_WORKERS=N            build threads (0 = all cores)
  PRECOMPILE_JIT_SERVER=h:p[,h:p] send kernels to JIT compile server(s)
  PRECOMPILE_REMOTE_MIN_KERNELS=N use the server only once N kernels are queued (default 10)
  PRECOMPILE_REAL_ALLOC=0         mock buffer addresses in the dry run (default 1: real allocator, so
                                  kernels that bake addresses into compile-time args match the real run)
  PRECOMPILE_FAST_COLLECT=0       keep real torch reference work in the dry run (debug)
  PRECOMPILE_DRY_RUN_ONLY=1       stop after the dry run and the compile (no real run)

Not warmed by the dry run, so they compile inline in the real run: tests that drive trace or graph
capture (skipped in the dry run), programs chosen from values read back from the device, and ops after a
point where a body throws.

The plugin prints one ``PRECOMPILE_RESULT:`` line of key=value fields for harnesses to parse.
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import inspect
import os
import sys
import time

import pytest

_WAIT = os.environ.get("PRECOMPILE_WAIT", "0") == "1"
_WORKERS = int(os.environ.get("PRECOMPILE_WORKERS", "0"))
_JIT_SERVER = [e for e in os.environ.get("PRECOMPILE_JIT_SERVER", "").split(",") if e]
_REMOTE_MIN_KERNELS = int(os.environ.get("PRECOMPILE_REMOTE_MIN_KERNELS", "10"))
_REAL_ALLOC = os.environ.get("PRECOMPILE_REAL_ALLOC", "1") == "1"
_FAST_COLLECT = os.environ.get("PRECOMPILE_FAST_COLLECT", "1") == "1"
_DRY_RUN_ONLY = os.environ.get("PRECOMPILE_DRY_RUN_ONLY") == "1"

# A body that drives trace or graph capture needs real dispatch and allocation; it is skipped in the dry run.
_CAPTURE_MARKERS = (
    "begin_trace_capture",
    "capture_trace",
    "execute_trace",
    "begin_graph_capture",
    "begin_mesh_trace_capture",
)


@dataclasses.dataclass
class _DryRunStats:
    bodies: int = 0
    swallowed: int = 0
    skipped_capture: int = 0
    shallow: int = 0
    fallback: int = 0


_STATS = _DryRunStats()
_SESSION = None  # ttnn.graph.CompileSession while the plugin's test loop runs
_DRY_RUN = False  # True while the dry run is in progress
_DRY_RUN_BODY_KEY = pytest.StashKey()


_HOST_DTYPES = {}


def _from_torch_dtype(real_from_torch, torch_dtype):
    """The ttnn dtype real_from_torch picks for torch_dtype, read from a one-element host conversion."""
    import torch

    if torch_dtype not in _HOST_DTYPES:
        _HOST_DTYPES[torch_dtype] = real_from_torch(torch.zeros(1, dtype=torch_dtype)).dtype
    return _HOST_DTYPES[torch_dtype]


def _mark_stand_in(replacement, original):
    try:
        replacement._precompile_original = original
    except (AttributeError, TypeError):
        pass
    return replacement


# Names the stand-ins are installed under; _restore_leaked_stand_ins looks for leaked copies of these.
_STAND_IN_NAMES = (
    "randn",
    "rand",
    "manual_seed",
    "conv2d",
    "conv3d",
    "layer_norm",
    "group_norm",
    "matmul",
    "bmm",
    "scaled_dot_product_attention",
    "comp_pcc",
    "assert_numeric_metrics",
    "from_torch",
    "torch2tt_tensor",
)


def _restore_leaked_stand_ins():
    """A module first imported inside a dry-run body can bind a stand-in (``from x import comp_pcc``) that the
    patches never saw, so it would keep it into the real run. Put the originals back everywhere."""
    restored = 0
    for module in list(sys.modules.values()):
        for name in _STAND_IN_NAMES:
            try:
                value = getattr(module, name, None)
                original = getattr(value, "_precompile_original", None)
            except Exception:
                continue
            if original is not None:
                setattr(module, name, original)
                restored += 1
    return restored


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
        home = sys.modules.get(self._home_module)
        self._orig = getattr(home, self._attr, None) if home is not None else None
        if self._orig is None:
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
    """Host reference compute -> shape-correct zeros. Programs depend on shapes and config, not values."""
    import torch
    import torch.nn.functional as F

    real_matmul = torch.matmul
    real_sdpa = getattr(F, "scaled_dot_product_attention", None)

    def _fast_randn(*size, **kw):
        if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
            size = tuple(size[0])
        kw.pop("generator", None)
        return torch.zeros(*size, **kw)

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
        _AttrPatch(torch, "randn", lambda _orig: _fast_randn),
        _AttrPatch(torch, "rand", lambda _orig: _fast_randn),
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

    return [
        _RebindPatch("tests.ttnn.utils_for_testing", "comp_pcc", lambda _orig: _pcc),
        _RebindPatch("models.common.utility_functions", "comp_pcc", lambda _orig: _pcc),
        _RebindPatch("tests.ttnn.utils_for_testing", "assert_numeric_metrics", lambda _orig: _assert_numeric_metrics),
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


@contextlib.contextmanager
def _stand_ins():
    with contextlib.ExitStack() as stack:
        if _FAST_COLLECT:
            for patch in _reference_op_patches() + _verifier_patches() + _boundary_patches():
                stack.enter_context(patch)
        yield


def _drives_capture(item) -> bool:
    fn = getattr(item, "function", None)
    if fn is None:
        return False
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return any(m in src for m in _CAPTURE_MARKERS)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """In the dry run, run the body inside a collect window: dispatch blocked, programs go to the session."""
    if not _DRY_RUN:
        return (yield)
    import ttnn

    if _drives_capture(item) or ttnn.graph.is_graph_capture_active():
        _STATS.skipped_capture += 1
        pytest.skip("precompile dry run: drives trace or graph capture")
    _STATS.bodies += 1
    item.stash[_DRY_RUN_BODY_KEY] = True
    _SESSION.begin_collect(real_alloc=_REAL_ALLOC)
    try:
        with _stand_ins():
            # A body exception (expected: asserts on meaningless outputs) propagates like any failure, so
            # makereport-time hooks that release the device tensors a failing frame holds still run.
            return (yield)
    finally:
        _SESSION.end_collect()


@pytest.hookimpl(tryfirst=True, wrapper=True)
def pytest_runtest_makereport(item, call):
    """Flip a failed dry-run body's report to passed. The exception itself (call.excinfo) is untouched, so hooks
    that act on it, such as releasing the device tensors a failing frame holds, still see it."""
    rep = yield
    if call.when != "call" or _DRY_RUN_BODY_KEY not in item.stash:
        return rep
    del item.stash[_DRY_RUN_BODY_KEY]
    if call.excinfo is not None and rep.outcome == "failed" and isinstance(call.excinfo.value, Exception):
        _STATS.swallowed += 1
        rep.outcome = "passed"
        rep.longrepr = None
    return rep


def _set_short_tracebacks(config, on: bool):
    """pytest renders a failure's traceback before any wrapper can discard it (~12 ms each); the dry run
    discards them all, so only the one-line summary is built."""
    if on:
        config._precompile_saved_tbstyle = config.getoption("tbstyle", "auto")
        config.option.tbstyle = "no"
    elif hasattr(config, "_precompile_saved_tbstyle"):
        config.option.tbstyle = config._precompile_saved_tbstyle
        del config._precompile_saved_tbstyle


def _kernel_build_dirs():
    """One entry per kernel binary in the kernel cache (<cache>/<build>/kernels/<name>/<hash>)."""
    root = os.environ.get("TT_METAL_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "tt-metal-cache")
    dirs = set()
    for depth in ("", "*", os.path.join("*", "*")):
        dirs.update(glob.glob(os.path.join(root, depth, "kernels", "*", "*")))
    return dirs


def _emit_result(**fields):
    line = " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in fields.items())
    print(f"PRECOMPILE_RESULT: {line}", flush=True)


def _stats_dict(stats):
    return {
        "programs_added": stats.programs_added,
        "programs_deduped": stats.programs_deduped,
        "kernels_queued": stats.kernels_queued,
        "kernels_built": stats.kernels_built,
        "kernels_preempted": stats.kernels_preempted,
        "kernels_remote": stats.kernels_remote,
        "kernels_failed": stats.kernels_failed,
        "kernels_dropped": stats.kernels_dropped,
        "kernels_pending": stats.kernels_pending,
        "kernels_running": stats.kernels_running,
        "busy_s": stats.busy_seconds,
    }


def _dry_run_item(item, nextitem):
    """runtestprotocol without reporting. It bypasses pytest_runtest_protocol, where pytest-timeout arms its
    timer, so arm it here the same way."""
    from _pytest.runner import runtestprotocol

    hooks = item.config.pluginmanager.hook
    settings = None
    timeout_plugin = item.config.pluginmanager.get_plugin("timeout")
    if timeout_plugin is not None:
        try:
            settings = timeout_plugin._get_item_settings(item)
            if not (settings.timeout and settings.timeout > 0):
                settings = None
        except Exception:
            settings = None
    if settings is not None:
        hooks.pytest_timeout_set_timer(item=item, settings=settings)
    try:
        runtestprotocol(item, log=False, nextitem=nextitem)
    finally:
        if settings is not None:
            hooks.pytest_timeout_cancel_timer(item=item)


def _run_items(session, items):
    n = len(items)
    for i, item in enumerate(items):
        nextitem = items[i + 1] if i + 1 < n else None
        item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
        if session.shouldfail:
            raise session.Failed(session.shouldfail)
        if session.shouldstop:
            raise session.Interrupted(session.shouldstop)


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    """Dry run -> (optionally wait for the compile queue) -> real run, in this one session."""
    global _SESSION, _DRY_RUN
    if session.testsfailed and not session.config.option.continue_on_collection_errors:
        raise session.Interrupted(
            f"{session.testsfailed} error{'s' if session.testsfailed != 1 else ''} during collection"
        )
    if session.config.option.collectonly:
        return True

    import ttnn

    if os.environ.pop("TT_METAL_JIT_SERVER_ENABLE", None) is not None:
        # The real run must compile through the in-process build table the session shares; the remote compile
        # path in ProgramImpl::compile bypasses it. Use PRECOMPILE_JIT_SERVER for the session's builds instead.
        print("PRECOMPILE: ignoring TT_METAL_JIT_SERVER_ENABLE; the real run compiles locally", flush=True)

    items = session.items
    n = len(items)
    _SESSION = ttnn.graph.CompileSession(
        workers=_WORKERS, remote_endpoints=_JIT_SERVER, remote_min_kernels=_REMOTE_MIN_KERNELS
    )
    t0 = time.monotonic()
    dry_run_s = None
    real_s = 0.0
    wait_s = 0.0
    real_run_compiled = -1
    at_real_start = None
    try:
        # Dry run. log=False keeps it out of reporting, JUnit and -x accounting. nextitem=None on the last
        # item tears every fixture down, so the real run starts from fresh fixtures.
        _DRY_RUN = True
        _set_short_tracebacks(session.config, True)
        saved_usepdb = session.config.option.usepdb
        session.config.option.usepdb = False  # every dry-run failure is expected
        try:
            for i, item in enumerate(items):
                _SESSION.set_priority(i)
                _dry_run_item(item, items[i + 1] if i + 1 < n else None)
        finally:
            session.config.option.usepdb = saved_usepdb
            _set_short_tracebacks(session.config, False)
            _DRY_RUN = False
            leaked = _restore_leaked_stand_ins()
            if leaked:
                print(f"PRECOMPILE: restored {leaked} stand-in(s) bound by modules imported during the dry run")
        t1 = time.monotonic()
        dry_run_s = t1 - t0
        at_dry_run_end = _SESSION.stats()
        print(
            f"\nPRECOMPILE: dry run over {n} item(s) took {dry_run_s:.1f}s: {at_dry_run_end.programs_added} "
            f"programs, {at_dry_run_end.kernels_queued} kernels queued, {at_dry_run_end.kernels_built} built so far "
            f"(swallowed {_STATS.swallowed}, skipped-capture {_STATS.skipped_capture}, "
            f"shape-only {_STATS.shallow}, fallback {_STATS.fallback})",
            flush=True,
        )
        if _WAIT or _DRY_RUN_ONLY:
            _SESSION.wait()
            wait_s = time.monotonic() - t1
            print(f"PRECOMPILE: compile queue empty after another {wait_s:.1f}s", flush=True)
        at_real_start = _SESSION.stats()
        if not _DRY_RUN_ONLY:
            before_real = _kernel_build_dirs()
            t2 = time.monotonic()
            _run_items(session, items)
            real_s = time.monotonic() - t2
            # Kernels the real run compiled itself: binary directories that appeared during it, less one for each
            # session job that started meanwhile (a job creates its directory when it starts, so jobs already
            # running at the start are in the snapshot). Approximate if another process shares the kernel cache.
            new_binaries = len(_kernel_build_dirs() - before_real)
            started_during = at_real_start.kernels_pending - _SESSION.stats().kernels_pending
            real_run_compiled = max(0, new_binaries - started_during)
            print(f"PRECOMPILE: real run took {real_s:.1f}s", flush=True)
    finally:
        at_end = _SESSION.stats()
        _SESSION.close()
        closed = _SESSION.stats()
        _SESSION = None
        _emit_result(
            mode="dry_run_only" if _DRY_RUN_ONLY else ("wait" if _WAIT else "stream"),
            route="server" if _JIT_SERVER else "local",
            items=n,
            bodies=_STATS.bodies,
            swallowed=_STATS.swallowed,
            skipped_capture=_STATS.skipped_capture,
            dry_run_s=dry_run_s if dry_run_s is not None else time.monotonic() - t0,
            wait_s=wait_s,
            real_s=real_s,
            pending_at_real_start=at_real_start.kernels_pending if at_real_start else -1,
            real_run_compiled=real_run_compiled,
            running_at_end=at_end.kernels_running,
            **_stats_dict(closed),
        )
    return True

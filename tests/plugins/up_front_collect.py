# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Up-front precompile collector for any ttnn pytest suite.

A call-phase hookwrapper runs every test body under NO_DISPATCH, stashing and deduping
its ops via the C++ collector (``ttnn.graph.up_front_*``), then JIT-compiles the distinct
set once, in parallel, at session end — warming the on-disk cache (``TT_METAL_CACHE``).
A hookwrapper (vs re-invoking tests) reuses each test's own fixtures, so any suite works.
``begin_collect(clear=False)`` wraps only the call phase (device setup stays real) and
accumulates across tests.

TWO PASSES: under NO_DISPATCH each body runs neutered (addr-0 buffers), so its asserts
fail and are swallowed — pass 1 only collects + compiles. Re-run for the real, warm results:

    # pass 1 — collect + parallel-compile (warms the cache). Loading the plugin IS the opt-in.
    # PYTHONPATH=$PWD makes the local `tests` package win over any foreign checkout on sys.path.
    TT_METAL_CACHE=/tmp/c  PYTHONPATH="$PWD" pytest -p tests.plugins.up_front_collect <tests>
    # pass 2 — real run, warm (plugin not loaded)
    TT_METAL_CACHE=/tmp/c  pytest <tests>

Knobs:  UP_FRONT_COLLECT_WORKERS=N (0 => hardware_concurrency) ·
        UP_FRONT_COLLECT_DEVICE_ID=N (device for the session-end compile, default 0)

Cold-compiled in pass 2 instead (NO_DISPATCH can't capture them faithfully): tests driving
trace/graph capture (skipped); ops after a mid-body tensor readback; ops whose program is
chosen from live allocator state (they see empty L1 and may collect a different variant).

Every collect-time substitution is one of two reversible primitives (_AttrPatch /
_RebindPatch), grouped into declarative sets and entered under one ExitStack — see
_collect_window.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import os
import sys

import pytest

# Active whenever loaded (-p tests.plugins.up_front_collect); disable a forced load with pytest's
# own -p no:up_front_collect.
_WORKERS = int(os.environ.get("UP_FRONT_COLLECT_WORKERS", "0"))  # 0 => hardware_concurrency
_DEVICE_ID = int(os.environ.get("UP_FRONT_COLLECT_DEVICE_ID", "0"))
# Collect with REAL buffer addresses (dispatch still blocked) instead of addr-0 mocking, so
# address-baked kernels (pool reader, move) warm. Costs ~the real run's peak memory; use when it fits.
_REAL_ALLOC = os.environ.get("UP_FRONT_REAL_ALLOC") == "1"
# Fast collect (default): swap host reference work for shape-correct stand-ins (randn/conv/matmul/
# layer_norm -> zeros, comp_pcc -> no-op) and allocate ttnn.from_torch shape-only. Values are
# irrelevant under NO_DISPATCH and programs depend only on op shapes/config, so collection is
# unchanged. =0 is a debug escape hatch: run the body with real torch (slow).
_FAST_COLLECT = os.environ.get("UP_FRONT_FAST_COLLECT", "1") == "1"
# Collect/dedup only, skip the session-end compile (isolates body-run cost when benchmarking).
_NO_COMPILE = os.environ.get("UP_FRONT_COLLECT_NO_COMPILE") == "1"

# A body is skipped (cold-compiles in pass 2) if its source names any of these: trace/graph capture
# needs the real dispatch + alloc that NO_DISPATCH blocks.
_CAPTURE_MARKERS = (
    "begin_trace_capture",
    "capture_trace",
    "execute_trace",
    "begin_graph_capture",
    "begin_mesh_trace_capture",
)


@dataclasses.dataclass
class CollectStats:
    """Session counters printed at the end (shallow/fallback = shape-only boundary hits vs fallbacks)."""

    bodies: int = 0
    swallowed: int = 0
    skipped_capture: int = 0
    shallow: int = 0
    fallback: int = 0
    xpass_strict: int = 0
    other_failures: int = 0


_STATS = CollectStats()


def _emit_result(status, reason, *, unique, programs, errors, pytest_exit):
    """Emit the stable, machine-readable warm-pass result consumed by run_safe_pytest.sh."""
    print(
        f"UP_FRONT_COLLECT_RESULT: status={status} reason={reason} "
        f"unique={unique} programs={programs} errors={errors} pytest_exit={int(pytest_exit)} "
        f"xpass_strict={_STATS.xpass_strict} other_failures={_STATS.other_failures}",
        flush=True,
    )


def _torch_to_ttnn_dtype(torch_dtype):
    import torch
    import ttnn

    return {
        torch.float32: ttnn.float32,
        torch.float64: ttnn.float32,
        torch.bfloat16: ttnn.bfloat16,
        torch.float16: ttnn.bfloat16,
        torch.int32: ttnn.uint32,
        torch.int64: ttnn.uint32,
        torch.int16: ttnn.uint16,
        torch.uint8: ttnn.uint8,
        torch.bool: ttnn.uint8,
    }.get(torch_dtype, ttnn.bfloat16)


# Patch primitives: every collect-time substitution is one of these two, entered together under one
# ExitStack (see _collect_window) for uniform automatic restore.


class _AttrPatch:
    """Reversibly set ``obj.attr = build(original)``. For attribute-reached callables —
    torch.randn, ttnn.from_torch, torch.Tensor.__matmul__ (``build`` may wrap the original or ignore it)."""

    def __init__(self, obj, attr, build):
        self._obj, self._attr, self._build = obj, attr, build

    def __enter__(self):
        self._orig = getattr(self._obj, self._attr)
        setattr(self._obj, self._attr, self._build(self._orig))
        return self

    def __exit__(self, *exc):
        setattr(self._obj, self._attr, self._orig)
        return False


# (home_module, attr) -> (orig_callable, [modules exposing it]). Memoizes the
# sys.modules walk in _RebindPatch across bodies (see __enter__).
_REBIND_TARGET_CACHE = {}


class _RebindPatch:
    """Reversibly rebind EVERY module attr that ``is`` ``home_module.attr`` to ``build(original)``.

    Reaches helpers pulled in via ``from x import y`` (patching the home module alone wouldn't).
    No-op if the home module isn't imported or lacks the attr; all matches share the one original.
    """

    def __init__(self, home_module, attr, build):
        self._home_module, self._attr, self._build = home_module, attr, build
        self._orig = None
        self._patched = []

    def __enter__(self):
        home = sys.modules.get(self._home_module)
        self._orig = getattr(home, self._attr, None) if home is not None else None
        if self._orig is None:
            return self
        replacement = self._build(self._orig)
        # The set of modules exposing this attr (via `from x import y`) is stable
        # across test bodies — modules are imported once, before the first body.
        # Walking all of sys.modules on every body was the dominant capture cost
        # (~14ms/body: 5 rebind patches x thousands of modules). Memoize the
        # target list keyed by (home_module, attr) + the original's identity;
        # re-walk only if the original object changed (re-import). A module
        # imported *after* the cache is built just keeps its real attr for that
        # body — harmless (the op is already stashed before the verifier runs).
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


def _try_shallow(stats, alloc):
    """Run a shape-only ttnn allocation; tally it. Returns the tensor, or None on failure
    (caller then falls back to the real ttnn path)."""
    try:
        t = alloc()
        stats.shallow += 1
        return t
    except Exception:
        stats.fallback += 1
        return None


# Declarative patch sets, built lazily so torch/ttnn imports stay deferred.


def _reference_op_patches():
    """Host reference compute -> shape-correct zeros. Drops the wasted golden-reference + RNG cost;
    collection is unchanged (programs depend on shapes/config, not values). ttnn ops are untouched."""
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
        # n-D conv: out = (in + 2p - d(k-1) - 1)//s + 1 per spatial dim. Output (N, C_out, *spatial).
        s, p, d = _ntuple(n, stride), _ntuple(n, padding), _ntuple(n, dilation)
        spatial = [(input.shape[i - n] + 2 * p[i] - d[i] * (weight.shape[i - n] - 1) - 1) // s[i] + 1 for i in range(n)]
        return input.new_zeros((input.shape[0], weight.shape[0], *spatial))

    def _fast_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return _fast_conv(input, weight, stride, padding, dilation, 2)

    def _fast_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return _fast_conv(input, weight, stride, padding, dilation, 3)

    def _fast_norm(input, *a, **k):
        return torch.zeros_like(input)  # layer_norm / rms_norm / group_norm output shape == input shape

    def _fast_matmul(a, b):
        # Shape via meta inference (no data, broadcasting stays exact); no GEMM compute.
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
        # Output = (*query.shape[:-1], value.shape[-1]) for EVERY variant — MHA / MQA / GQA, causal,
        # masked: query's batch/heads/seq are kept, only the head dim becomes value's. mask / scale /
        # dropout / is_causal don't change shape. Fall back to the real op if the rule ever doesn't fit.
        try:
            return torch.zeros((*query.shape[:-1], value.shape[-1]), dtype=query.dtype, device=query.device)
        except Exception:
            return real_sdpa(query, key, value, *a, **k)

    patches = [
        _AttrPatch(torch, "randn", lambda _orig: _fast_randn),
        _AttrPatch(torch, "rand", lambda _orig: _fast_randn),
        _AttrPatch(F, "conv2d", lambda _orig: _fast_conv2d),
        _AttrPatch(F, "conv3d", lambda _orig: _fast_conv3d),
        _AttrPatch(F, "layer_norm", lambda _orig: _fast_norm),
        _AttrPatch(F, "group_norm", lambda _orig: _fast_norm),
        _AttrPatch(torch, "matmul", lambda _orig: _fast_matmul),
        _AttrPatch(torch, "bmm", lambda _orig: _fast_bmm),
        _AttrPatch(torch.Tensor, "matmul", lambda _orig: _fast_tensor_matmul),
        _AttrPatch(torch.Tensor, "__matmul__", lambda _orig: _fast_tensor_matmul),
    ]
    if real_sdpa is not None:  # added in torch 2.0; guard so loading never fails on older torch
        patches.append(_AttrPatch(F, "scaled_dot_product_attention", lambda _orig: _fast_sdpa))
    return patches


def _verifier_patches():
    """Host numeric verifiers -> no-ops: a verifier failing on the addr-0 readback would abort the
    rest of a multi-op body and lose its ops (results are meaningless under NO_DISPATCH anyway).
    Covers BOTH verifier families: the legacy model/ttnn `comp_pcc`/`assert_numeric_metrics` (two
    home modules) AND the registry-model golden harness `eval.metrics.check_output` — the latter is
    the dominant collect cost (it does `ttnn.to_torch` on the garbage output then ULP-p99/PCC/median
    on full shape-sized tensors, ~10ms/body, scaling with shape). The op is already stashed before
    check_output runs, so no-op'ing it is safe and skips the readback + metrics entirely."""

    def _pcc(*a, **k):
        return (True, 0.999999)

    def _assert_numeric_metrics(*a, **k):
        return (True, "collect-stub")

    def _noop_check(*a, **k):
        return None  # collect discards the verdict; output is addr-0 garbage under NO_DISPATCH

    return [
        _RebindPatch("tests.ttnn.utils_for_testing", "comp_pcc", lambda _orig: _pcc),
        _RebindPatch("models.common.utility_functions", "comp_pcc", lambda _orig: _pcc),
        _RebindPatch("tests.ttnn.utils_for_testing", "assert_numeric_metrics", lambda _orig: _assert_numeric_metrics),
        _RebindPatch("eval.metrics", "check_output", lambda _orig: _noop_check),
    ]


def _boundary_patches(stats):
    """ttnn<->torch boundary -> shape-only device allocation: program build needs only the spec, so
    from_torch / torch2tt_tensor skip the host tilize/convert/copy (the dominant collect cost on
    weight-heavy bodies). Not-safely-shape-only cases (mesh_mapper, custom tile, host/meta) fall back."""
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
                        dt = dtype or _torch_to_ttnn_dtype(tensor.dtype)
                        lay = layout or ttnn.ROW_MAJOR_LAYOUT
                        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
                        shp = ttnn.Shape(tuple(tensor.shape))
                    return ttnn.allocate_tensor_on_device(shp, dt, lay, device, mc)

                t = _try_shallow(stats, _alloc)
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
        # torch2tt_tensor builds device tensors outside from_torch; same shape-only lever.
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

                t = _try_shallow(stats, _alloc)
                if t is not None:
                    return t
            return real_t2tt(py_tensor, tt_device, tt_layout, tt_memory_config, tt_dtype)

        return _shallow_torch2tt

    return [
        _AttrPatch(ttnn, "from_torch", _build_from_torch),
        _RebindPatch("models.common.utility_functions", "torch2tt_tensor", _build_torch2tt),
    ]


def _collect_patches(stats):
    """The patch set active during the collect window (empty unless fast collect is on)."""
    patches = []
    if _FAST_COLLECT:
        patches += _reference_op_patches()
        patches += _verifier_patches()
        patches += _boundary_patches(stats)
    return patches


@contextlib.contextmanager
def _collect_window(stats):
    """Enter every active patch under one ExitStack, run the body, restore on exit. With no active
    patches (UP_FRONT_FAST_COLLECT=0) it's a transparent pass-through (body runs with real torch)."""
    with contextlib.ExitStack() as stack:
        for patch in _collect_patches(stats):
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
    """Run each body under one NO_DISPATCH collect window (call phase only)."""
    import ttnn

    if _drives_capture(item):
        _STATS.skipped_capture += 1
        return (yield)  # runs normally; cold-compiles in pass 2

    _STATS.bodies += 1
    # UP_FRONT_LOG_SWALLOWED=1: per-body trace of throw + ops-stashed-before-throw (a throw BEFORE
    # stashing is a real miss; one on the addr-0 readback after stashing is fine).
    _log = os.environ.get("UP_FRONT_LOG_SWALLOWED")
    n_before = ttnn.graph.up_front_num_collected() if _log else 0
    swallowed = False
    exc_info = ""
    ttnn.graph.up_front_begin_collect(clear=False, real_alloc=_REAL_ALLOC)  # accumulate; wraps ONLY the body
    try:
        with _collect_window(_STATS):
            return (yield)  # ops stash into the collector as the body runs
    except Exception as e:
        # Expected under NO_DISPATCH: a readback/assert on an addr-0 output, after the program was
        # already stashed. Swallow so pass 1 stays green (its results are meaningless).
        _STATS.swallowed += 1
        swallowed = True
        exc_info = f"{type(e).__name__}: {str(e)[:140]}"
        return None
    finally:
        ttnn.graph.up_front_end_collect()
        if _log:
            n_after = ttnn.graph.up_front_num_collected()
            print(
                f"UP_FRONT_BODY: {'SWALLOWED' if swallowed else 'ok       '} "
                f"stashed={n_after - n_before:<3} {item.nodeid}" + (f"  -> {exc_info}" if swallowed else ""),
                flush=True,
            )


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    """Separate strict XPASS from failures that make a warm pass unusable.

    The collector intentionally swallows call-time exceptions because values under
    NO_DISPATCH are meaningless.  A strict-xfail test whose expected exception was
    swallowed consequently reaches pytest as XPASS(strict), which pytest reports as
    a failure even though collection and compilation were successful.
    """
    if not report.failed:
        return
    if report.when == "call" and "[XPASS(strict)]" in str(report.longrepr):
        _STATS.xpass_strict += 1
    else:
        _STATS.other_failures += 1


def pytest_sessionstart(session):
    import ttnn

    ttnn.graph.up_front_clear()  # clean slate before the session


def pytest_sessionfinish(session, exitstatus):
    import ttnn

    n_unique = ttnn.graph.up_front_num_unique()
    n_collected = ttnn.graph.up_front_num_collected()
    print(
        f"\nUP_FRONT_COLLECT: {n_collected} ops stashed across {_STATS.bodies} bodies "
        f"-> {n_unique} unique programs "
        f"(swallowed {_STATS.swallowed}, skipped-capture {_STATS.skipped_capture})",
        flush=True,
    )
    if _FAST_COLLECT:
        print(
            f"UP_FRONT_COLLECT: fast-shallow from_torch — shape-only={_STATS.shallow} " f"fallback={_STATS.fallback}",
            flush=True,
        )
    if n_unique == 0:
        print("UP_FRONT_COLLECT: nothing to compile", flush=True)
        _emit_result("ok", "nothing_to_compile", unique=0, programs=0, errors=0, pytest_exit=exitstatus)
        return
    if _NO_COMPILE:
        print(f"UP_FRONT_COLLECT: NO_COMPILE set — collected {n_unique}, skipping compile", flush=True)
        _emit_result("skipped", "no_compile", unique=n_unique, programs=0, errors=0, pytest_exit=exitstatus)
        return

    device = None
    created = False
    n_prog = 0
    n_err = 0
    result_status = "failed"
    result_reason = "exception"
    try:
        try:
            device = ttnn.GetDefaultDevice()
        except Exception:
            device = None
        if device is None:
            device = ttnn.CreateDevice(device_id=_DEVICE_ID)
            created = True
        print(f"UP_FRONT_COLLECT: compile device = {'FRESH' if created else 'REUSED live'}", flush=True)
        try:
            device.enable_program_cache()
        except Exception:
            pass
        n_prog, n_err, used, wall = ttnn.graph.up_front_compile(device, _WORKERS, True)
        print(
            f"UP_FRONT_COLLECT: compiled {n_prog} programs in {wall:.1f}s "
            f"(workers={used}, errors={n_err}) -> on-disk JIT cache warm",
            flush=True,
        )
        if n_err == 0 and n_prog == n_unique:
            result_status = "ok"
            result_reason = "ok"
        elif n_err:
            result_reason = "compile_errors"
        else:
            result_reason = "incomplete"
    except Exception as e:  # best-effort: never break the session
        print(f"UP_FRONT_COLLECT: compile skipped (best-effort) — {e!r}", flush=True)
    finally:
        if created and device is not None:
            try:
                ttnn.close_device(device)
            except Exception:
                pass
    _emit_result(
        result_status,
        result_reason,
        unique=n_unique,
        programs=n_prog,
        errors=n_err,
        pytest_exit=exitstatus,
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Up-front precompile collector for any ttnn pytest suite.

Drives the op-agnostic C++ collector (``ttnn.graph.up_front_*``) across a whole
pytest session via a call-phase hookwrapper: every op of every test body is
stashed and deduped, then the distinct set is JIT-compiled once, in parallel, at
session end — warming the on-disk JIT cache (``TT_METAL_CACHE``) so a subsequent
real run finds its kernels already built.

A hookwrapper is used (rather than re-invoking each test) because pytest runs each
body through its own fixture machinery, so this works for arbitrary tests without
knowing their signatures. ``begin_collect(clear=False)`` wraps ONLY the call phase,
so NO_DISPATCH never spans fixture setup/teardown (the device is created for real in
setup); ``clear=False`` makes the collector ACCUMULATE across tests.

TWO PASSES: under NO_DISPATCH each body runs "neutered" (mocked addr-0 buffers,
nothing dispatched), so its asserts fail and are swallowed — pass 1 only collects +
compiles. Re-run the SAME suite over the SAME on-disk cache for the real, now-warm
results:

    # pass 1 — collect every op + parallel-compile the distinct set (warms the cache).
    # Loading the plugin with -p IS the opt-in; it is active whenever loaded.
    # (run from the repo root; PYTHONPATH=$PWD ensures the local `tests` package wins
    #  over any foreign tt-metal checkout on an inherited PYTHONPATH)
    TT_METAL_CACHE=/tmp/c  PYTHONPATH="$PWD" pytest -p tests.plugins.up_front_collect <tests>
    # pass 2 — real run, warm (plugin not loaded)
    TT_METAL_CACHE=/tmp/c  pytest <tests>

Knobs:  UP_FRONT_COLLECT_WORKERS=N (0 => hardware_concurrency) ·
        UP_FRONT_COLLECT_DEVICE_ID=N (device for the session-end compile, default 0)

Limitations:
  * Tests that drive trace/graph capture are SKIPPED — NO_DISPATCH blocks the real
    dispatch + buffer allocation that recording needs. They cold-compile in pass 2.
  * A body that reads a tensor back / branches on values mid-body breaks on addr-0:
    we capture the ops before that point; the rest cold-compile in pass 2.
  * Ops that pick their program from live allocator state see empty L1 under
    NO_DISPATCH and may collect a different variant than the real run uses.

Collect-time substitutions are all expressed as one of two reversible primitives
(_AttrPatch / _RebindPatch), grouped into declarative sets (reference ops, verifiers,
boundary) and entered together under a single ExitStack — see _collect_window.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import os
import sys

import pytest

# The plugin is active whenever it is loaded (``-p tests.plugins.up_front_collect``); loading it IS
# the opt-in. To disable a load that some config forces, use pytest's own ``-p no:up_front_collect``.
_WORKERS = int(os.environ.get("UP_FRONT_COLLECT_WORKERS", "0"))  # 0 => hardware_concurrency
_DEVICE_ID = int(os.environ.get("UP_FRONT_COLLECT_DEVICE_ID", "0"))
# UP_FRONT_REAL_ALLOC=1: collect with REAL buffer addresses (dispatch still blocked) instead of
# addr-0 mocking, so address-baked / address-branched kernels (pool reader, move fwd/bwd) warm.
# Costs real device memory (~the real run's peak) — only use when the model fits.
_REAL_ALLOC = os.environ.get("UP_FRONT_REAL_ALLOC") == "1"
# Fast collect (default): in the collect window (results thrown away under NO_DISPATCH) replace
# expensive HOST-side torch work — torch.randn -> zeros, the torch *reference* conv/matmul/layer_norm
# -> shape-correct zeros, comp_pcc -> no-op — and allocate ttnn.from_torch SHAPE-ONLY (skip the host
# tilize/convert/copy; the dominant collect cost on weight-heavy models). Values are irrelevant under
# NO_DISPATCH and collected programs depend only on ttnn op shapes/config, so this does NOT change what
# is collected; the shape-only boundary falls back to the real path for anything it can't safely fake.
# UP_FRONT_FAST_COLLECT=0 is a debug escape hatch: run the body with real torch (full-fidelity, slow).
_FAST_COLLECT = os.environ.get("UP_FRONT_FAST_COLLECT", "1") == "1"
# UP_FRONT_COLLECT_NO_COMPILE=1: collect/dedup only, skip the session-end parallel compile. Used to
# isolate the body-run cost from the compile floor when benchmarking.
_NO_COMPILE = os.environ.get("UP_FRONT_COLLECT_NO_COMPILE") == "1"

# Bodies that drive trace/graph capture can't run under NO_DISPATCH (it blocks the
# dispatch + alloc that recording needs). Heuristic: scan the test function source.
_CAPTURE_MARKERS = (
    "begin_trace_capture",
    "capture_trace",
    "execute_trace",
    "begin_graph_capture",
    "begin_mesh_trace_capture",
)


@dataclasses.dataclass
class CollectStats:
    """Session counters printed at the end. `bodies/swallowed/skipped_capture` are hook-level;
    `shallow/fallback` count the shape-only boundary's hits vs real-path fallbacks."""

    bodies: int = 0
    swallowed: int = 0
    skipped_capture: int = 0
    shallow: int = 0
    fallback: int = 0


_STATS = CollectStats()


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


# ---------------------------------------------------------------------------
# Patch primitives. Every collect-time substitution is one of these two,
# entered together under a single ExitStack (see _collect_window) so each has
# uniform, automatic restore instead of a hand-rolled try/finally.
# ---------------------------------------------------------------------------


class _AttrPatch:
    """Reversibly replace ``obj.attr`` with ``build(original)`` for the collect window.

    For callables reached by attribute access on a stable object — e.g. ``torch.randn``,
    ``ttnn.from_torch``, ``torch.Tensor.__matmul__``. ``build`` receives the original (some
    replacements wrap it; reference stand-ins ignore it).
    """

    def __init__(self, obj, attr, build):
        self._obj, self._attr, self._build = obj, attr, build

    def __enter__(self):
        self._orig = getattr(self._obj, self._attr)
        setattr(self._obj, self._attr, self._build(self._orig))
        return self

    def __exit__(self, *exc):
        setattr(self._obj, self._attr, self._orig)
        return False


class _RebindPatch:
    """Reversibly rebind EVERY module attribute that ``is`` ``home_module.attr`` to ``build(original)``.

    Needed for helpers pulled into test modules via ``from x import y`` — patching the home module
    alone wouldn't reach those local copies, so we rebind every reference. No-op if the home module
    isn't imported or lacks the attribute. All matches share one original, so restore is to that.
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
        for m in list(sys.modules.values()):
            try:
                if getattr(m, self._attr, None) is self._orig:
                    self._patched.append(m)
                    setattr(m, self._attr, replacement)
            except Exception:
                pass
        return self

    def __exit__(self, *exc):
        for m in self._patched:
            setattr(m, self._attr, self._orig)
        return False


def _try_shallow(stats, alloc):
    """Run a shape-only ttnn allocation, tallying shallow vs fallback. Returns the allocated
    tensor, or None if it raised — in which case the caller falls back to the real ttnn path."""
    try:
        t = alloc()
        stats.shallow += 1
        return t
    except Exception:
        stats.fallback += 1
        return None


# ---------------------------------------------------------------------------
# Declarative patch sets, built lazily so torch/ttnn imports stay deferred.
# ---------------------------------------------------------------------------


def _reference_op_patches():
    """Host reference compute -> cheap shape-correct stand-ins.

    Under NO_DISPATCH the values are thrown away and the collected programs depend only on ttnn op
    shapes/config, so zeroing the inputs / torch reference conv / matmul / layer_norm does not change
    WHAT is collected — it just drops the wasted golden-reference + RNG cost. ttnn ops are untouched.
    """
    import torch
    import torch.nn.functional as F

    real_matmul = torch.matmul

    def _fast_randn(*size, **kw):
        if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
            size = tuple(size[0])
        kw.pop("generator", None)
        return torch.zeros(*size, **kw)

    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    def _fast_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        sH, sW = _pair(stride)
        pH, pW = _pair(padding)
        dH, dW = _pair(dilation)
        N, C_out = input.shape[0], weight.shape[0]
        H, W = input.shape[-2], input.shape[-1]
        kH, kW = weight.shape[-2], weight.shape[-1]
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        return input.new_zeros((N, C_out, H_out, W_out))

    def _fast_layer_norm(input, *a, **k):
        return torch.zeros_like(input)  # layer_norm / rms_norm output shape == input shape

    def _fast_matmul(a, b):
        # Host golden reference matmul/@/.matmul()/bmm -> shape-correct zeros, NO GEMM compute.
        # Output shape via meta inference (no data) so all broadcasting rules stay exact. Only the
        # host reference uses these; ttnn.matmul is untouched, so the collected programs are unchanged.
        try:
            shp = tuple(real_matmul(a.to("meta"), b.to("meta")).shape)
            return torch.zeros(shp, dtype=getattr(a, "dtype", torch.bfloat16), device=getattr(a, "device", "cpu"))
        except Exception:
            return real_matmul(a, b)

    def _fast_bmm(a, b, *x, **k):
        return _fast_matmul(a, b)

    def _fast_tensor_matmul(self, other):
        return _fast_matmul(self, other)

    return [
        _AttrPatch(torch, "randn", lambda _orig: _fast_randn),
        _AttrPatch(torch, "rand", lambda _orig: _fast_randn),
        _AttrPatch(F, "conv2d", lambda _orig: _fast_conv2d),
        _AttrPatch(F, "layer_norm", lambda _orig: _fast_layer_norm),
        _AttrPatch(torch, "matmul", lambda _orig: _fast_matmul),
        _AttrPatch(torch, "bmm", lambda _orig: _fast_bmm),
        _AttrPatch(torch.Tensor, "matmul", lambda _orig: _fast_tensor_matmul),
        _AttrPatch(torch.Tensor, "__matmul__", lambda _orig: _fast_tensor_matmul),
    ]


def _verifier_patches():
    """Host numeric verifiers -> no-ops.

    A verifier failing on the addr-0/zeros readback aborts the rest of a multi-op body and loses its
    ops; results are meaningless under NO_DISPATCH anyway. Rebound across modules because tests pull
    these in via ``from ... import``. comp_pcc has two possible homes; either (or both) is covered.
    """

    def _pcc(*a, **k):
        return (True, 0.999999)

    def _assert_numeric_metrics(*a, **k):
        return (True, "collect-stub")

    return [
        _RebindPatch("tests.ttnn.utils_for_testing", "comp_pcc", lambda _orig: _pcc),
        _RebindPatch("models.common.utility_functions", "comp_pcc", lambda _orig: _pcc),
        _RebindPatch("tests.ttnn.utils_for_testing", "assert_numeric_metrics", lambda _orig: _assert_numeric_metrics),
    ]


def _boundary_patches(stats):
    """ttnn<->torch boundary -> SHAPE-ONLY device allocation.

    Program build needs only the spec, so ttnn.from_torch and torch2tt_tensor skip the host
    tilize/convert/copy (the dominant collect cost on weight-heavy bodies). Anything not safely
    shape-only (mesh_mapper, custom tile, host/meta tensors) falls back to the real path.
    """
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
        # torch2tt_tensor builds device tensors outside from_torch (ttnn.Tensor(t).to(...)); same lever.
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
    """Enter every active patch under one ExitStack, run the body, restore all on exit (LIFO).

    With no active patches (UP_FRONT_FAST_COLLECT=0) this is a transparent pass-through, so the body
    runs with real torch.
    """
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
        return (yield)  # let it run normally; it cold-compiles in pass 2

    _STATS.bodies += 1
    # UP_FRONT_LOG_SWALLOWED=1: per-body diagnostic — which tests threw under NO_DISPATCH and
    # how many ops each stashed before the throw (reveals coverage: a body that throws BEFORE
    # stashing its op is a real miss; one that throws on the addr-0 readback after stashing is fine).
    _log = os.environ.get("UP_FRONT_LOG_SWALLOWED")
    n_before = ttnn.graph.up_front_num_collected() if _log else 0
    swallowed = False
    exc_info = ""
    ttnn.graph.up_front_begin_collect(clear=False, real_alloc=_REAL_ALLOC)  # accumulate; wraps ONLY the body
    try:
        with _collect_window(_STATS):
            return (yield)  # body runs with the active collect-time patches; ttnn ops still stash
    except Exception as e:
        # Expected under NO_DISPATCH: a readback/assert on an addr-0 output. The
        # program was already stashed by the funnel before the failure. Swallow so
        # pass 1 stays green (its results are meaningless — pass 2 is the real run).
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


def pytest_sessionstart(session):
    import ttnn

    ttnn.graph.up_front_clear()  # clean slate before accumulating across the session


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
        return
    if _NO_COMPILE:
        print(f"UP_FRONT_COLLECT: NO_COMPILE set — collected {n_unique}, skipping compile", flush=True)
        return

    device = None
    try:
        device = ttnn.CreateDevice(device_id=_DEVICE_ID)
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
    except Exception as e:  # best-effort: never break the session
        print(f"UP_FRONT_COLLECT: compile skipped (best-effort) — {e!r}", flush=True)
    finally:
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception:
                pass

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.bringup.block_select_ttnn:block_max_bf16``.

Every test calls the operation through the ``block_max_bf16_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (``block_max_bf16(block_max, *, factor=1, n=None, n_tensor=None, output_tensor=None,
memory_config=None, max_grid=None, strip_major=False)``, README / docstring of the package):

* Fold (``strip_major=False``): block_max [.., R, NI] fp32 ROW_MAJOR DRAM interleaved holding bf16 values ->
  [.., R, NI / factor] bf16 ROW_MAJOR, column b = max of input columns [b factor, (b + 1) factor), exact. ``factor``
  a power of two in [1, 16]; NI a multiple of 8 x factor. Exactly one of ``n`` (runtime value: one program for every
  n) / ``n_tensor`` ([1] uint32 ROW_MAJOR DRAM, read on the device). Only the first ``n`` output columns (rounded up
  to 8) are written; the rest of a caller's ``output_tensor`` is untouched. ``output_tensor`` (bf16 ROW_MAJOR, the
  output shape, DRAM interleaved) is written in place and returned; else a new DRAM (or ``memory_config``) tensor.
* Strip-major (``strip_major=True``, ``factor`` 1): block_max [.., S NB / 8, 256] bf16 ROW_MAJOR DRAM ->
  the required ``output_tensor`` [.., 32 S, NB] bf16 (NB a multiple of 8): each strip's [NB, 32] transposed (exact),
  the first ``n`` blocks (rounded up to 8) written.
* ``max_grid``: cores (0, 0) .. (max_grid - 1), capped at the device grid (a CoreCoord or an (x, y) pair).
* Refusals (source: AssertionError; a native port raises RuntimeError, so both are accepted): factor not a power of
  two in [1, 16]; block_max not fp32 (fold) / bf16 (strip) ROW_MAJOR DRAM interleaved; NI not a multiple of 8 x
  factor; not exactly one of n / n_tensor; output_tensor of the wrong dtype / layout / shape or not DRAM interleaved;
  strip-major with factor != 1, width != 256, no output_tensor, output NB % 8 != 0 or rows % 32 != 0, input rows !=
  S NB / 8.
* Programs: built once per (mode, factor, n-vs-n_tensor, max_grid, shapes, memory configs); addresses and the value
  of ``n`` / the contents of ``n_tensor`` change on a hit. A call adds one program-cache entry on a miss and none on
  a hit.

All values are compared byte-exact against an independent torch reference (block_max_ref.py).
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys

import pytest
import ttnn
from loguru import logger

# --------------------------------------------------------------------------- route selection (written by the driver)

SOURCE_ENTRY = "ttnn.bringup.block_select_ttnn:block_max_bf16"
MODE_ENV = "TT_PRE_MIGRATION_MODE"
ENTRY_ENV = "TT_PRE_MIGRATION_ENTRY"


def _resolve(spec):
    module_name, sep, name = spec.partition(":")
    if not sep or not module_name or not name:
        raise pytest.UsageError(f"{ENTRY_ENV} must be module.path:callable, got {spec!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, name, None)
    if fn is None or not callable(fn):
        raise pytest.UsageError(f"{spec} does not resolve to a callable")
    return module, fn


def _selection():
    mode = os.environ.get(MODE_ENV, "source") or "source"
    entry = os.environ.get(ENTRY_ENV, "")
    if mode == "source":
        if entry and entry != SOURCE_ENTRY:
            raise pytest.UsageError(f"source mode runs {SOURCE_ENTRY} only, but {ENTRY_ENV}={entry!r}")
        return mode, SOURCE_ENTRY
    if mode == "native":
        if not entry:
            raise pytest.UsageError(f"native mode requires {ENTRY_ENV}=module.path:callable")
        if entry == SOURCE_ENTRY:
            raise pytest.UsageError(f"native mode needs the op's registered entry, not the source path {SOURCE_ENTRY}")
        return mode, entry
    raise pytest.UsageError(f"{MODE_ENV} must be 'source' or 'native', got {mode!r}")


def _generic_op_owners():
    """Every (owner, attribute) through which the tested call path can reach ``ttnn.generic_op``."""
    original = ttnn.generic_op
    owners = [(ttnn, "generic_op")]
    for name, module in list(sys.modules.items()):
        if module is None or module is ttnn:
            continue
        if name.startswith("ttnn.bringup.block_select_ttnn") or name.startswith("ttnn._ttnn.operations.generic"):
            if getattr(module, "generic_op", None) is original:
                owners.append((module, "generic_op"))
    return original, owners


@contextlib.contextmanager
def _patched_generic_op(replacement):
    original, owners = _generic_op_owners()
    patched = []
    try:
        for owner, attr in owners:
            try:
                setattr(owner, attr, replacement(original))
            except (AttributeError, TypeError):
                continue
            patched.append((owner, attr))
        yield
    finally:
        for owner, attr in reversed(patched):
            setattr(owner, attr, original)


class _Entry:
    """The single call point for the operation under test. ``last_dispatches`` is how many
    ``ttnn.generic_op`` calls the last source call made (0 on the native route)."""

    def __init__(self, mode, spec, module, fn):
        self.mode = mode
        self.spec = spec
        self.module = module
        self.fn = fn
        self.calls = 0
        self.generic_dispatches = 0
        self.last_dispatches = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.mode == "native":
            return self._call_native(*args, **kwargs)
        return self._call_source(*args, **kwargs)

    def _call_source(self, *args, **kwargs):
        seen = []

        def spy(original):
            def forward(*a, **k):
                seen.append(1)
                return original(*a, **k)

            return forward

        with _patched_generic_op(spy):
            try:
                return self.fn(*args, **kwargs)
            finally:
                self.last_dispatches = len(seen)
                self.generic_dispatches += len(seen)

    def _call_native(self, *args, **kwargs):
        def forbid(_original):
            def raise_(*_a, **_k):
                raise AssertionError("native block_max_bf16 must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def block_max_bf16_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"block_max_bf16 acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch  # noqa: E402

from tests.ttnn.unit_tests.operations.block_max_bf16.pre_migration.block_max_ref import (  # noqa: E402
    SENTINEL_FILL,
    assert_bytes_equal,
    check_written,
    downs,
    fold_input,
    fold_ref,
    strip_device_view,
    strip_input,
    strip_ref,
    up,
    up_n,
    written,
)

# One single-chip device for the whole module: the source keeps its prepared programs per device, so the device is
# not reopened between tests; cache tests start from an emptied program cache (``fresh_cache``).
pytestmark = pytest.mark.use_module_device

REFUSED = (AssertionError, RuntimeError)  # source asserts; a native op's TT_FATAL is a RuntimeError


def _cache_delta(device, fn):
    """Program-cache entries added by ``fn()`` alone (setup and readback stay outside)."""
    before = device.num_program_cache_entries()
    result = fn()
    return result, device.num_program_cache_entries() - before


def _addr(t):
    return t.buffer_address()


# --------------------------------------------------------------------------- fold: valid cases (one per planner branch)
# id: (shape, factor (None: the default), n-kind, n, max_grid, output: "caller" | "alloc" | "alloc_memcfg")
# CAST_CHUNK of the source is 2048 input values a work unit: widths below, at and above it (with a partial last unit);
# rows fewer than, and more than, any device's core count (idle cores / uneven split); 2-D, 4-D and batched 4-D.
FOLD_CASES = {
    "f1_r1_w64_min_work": ([1, 1, 1, 64], 1, "value", 64, None, "caller"),
    "f2_r5_w256_n13_round8": ([1, 1, 5, 256], 2, "value", 13, None, "caller"),
    "f4_r40_w2048_one_full_unit": ([1, 1, 40, 2048], 4, "value", 512, None, "caller"),
    "f8_r300_w4608_multi_unit_partial_tensor": ([1, 1, 300, 4608], 8, "tensor", 500, None, "caller"),
    "f16_batched_2x3x7_w1024_tensor_n3": ([2, 3, 7, 1024], 16, "tensor", 3, None, "caller"),
    "f4_2d_r24_w8192_alloc_memcfg": ([24, 8192], 4, "value", 2048, None, "alloc_memcfg"),
    "fdefault_r8_w512_alloc": ([1, 1, 8, 512], None, "value", 512, None, "alloc"),
    "f1_r6_w128_n1": ([1, 1, 6, 128], 1, "value", 1, None, "caller"),
    "f4_r9_w2304_grid1x1": ([1, 1, 9, 2304], 4, "value", 100, ttnn.CoreCoord(1, 1), "caller"),
    "f2_r17_w1024_grid3x2_tensor": ([1, 1, 17, 1024], 2, "tensor", 300, ttnn.CoreCoord(3, 2), "caller"),
    "f8_r33_w512_grid_tuple_2x2": ([1, 1, 33, 512], 8, "value", 64, (2, 2), "caller"),
    "f1_r4_w256_grid_capped_999": ([1, 1, 4, 256], 1, "value", 200, ttnn.CoreCoord(999, 999), "caller"),
}


def _fold_kwargs(factor, n_kind, n, max_grid, device):
    kw = {}
    if factor is not None:
        kw["factor"] = factor
    if n_kind == "value":
        kw["n"] = n
    else:
        kw["n_tensor"] = up_n(n, device)
    if max_grid is not None:
        kw["max_grid"] = max_grid
    return kw


@pytest.mark.parametrize("case", list(FOLD_CASES), ids=list(FOLD_CASES))
def test_fold_values(device, block_max_bf16_op, case):
    shape, factor, n_kind, n, max_grid, out_kind = FOLD_CASES[case]
    F = factor or 1
    x = fold_input(shape, seed=len(case))
    want = fold_ref(x, F)
    out_shape = list(want.shape)
    m = written(n, out_shape[-1])
    xt = up(x, device, ttnn.float32)
    kw = _fold_kwargs(factor, n_kind, n, max_grid, device)
    prev = None
    if out_kind == "caller":
        prev = torch.full(out_shape, SENTINEL_FILL, dtype=torch.bfloat16)
        ot = up(prev, device, ttnn.bfloat16)
        kw["output_tensor"] = ot
    elif out_kind == "alloc_memcfg":
        kw["memory_config"] = ttnn.DRAM_MEMORY_CONFIG

    got = block_max_bf16_op(xt, **kw)

    # metadata of the returned tensor
    assert list(got.shape) == out_shape
    assert got.dtype == ttnn.bfloat16
    assert got.layout == ttnn.ROW_MAJOR_LAYOUT
    mc = got.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    if out_kind == "caller":
        assert _addr(got) == _addr(ot), "the caller's output_tensor must be written in place and returned"
    else:
        assert _addr(got) != _addr(xt), "a new output must not alias the input"
    # values: written columns exact, the caller's columns past n untouched
    check_written(got, want, m, prev, case)
    # the input is not modified
    for d in downs(xt):
        assert_bytes_equal(d.reshape(x.shape).to(torch.bfloat16), x.to(torch.bfloat16), f"{case} input")


# --------------------------------------------------------------------------- strip-major: valid cases
# id: (S, NB, n-kind, n, max_grid). STRIP_CHUNK of the source is 128 blocks a work unit: NB below, at and above it
# with a partial last unit; work units fewer and more than the cores; a single core.
STRIP_CASES = {
    "s1_nb64_partial_unit": (1, 64, "value", 64, None),
    "s2_nb128_one_unit_tensor_n50": (2, 128, "tensor", 50, None),
    "s3_nb264_multi_unit_partial": (3, 264, "value", 264, None),
    "s16_nb2176_many_units_n2000": (16, 2176, "value", 2000, None),
    "s4_nb1024_grid1x1_n1000": (4, 1024, "value", 1000, ttnn.CoreCoord(1, 1)),
    "s2_nb16_tensor_n1": (2, 16, "tensor", 1, None),
    "s5_nb512_grid2x3_tensor_n77": (5, 512, "tensor", 77, ttnn.CoreCoord(2, 3)),
}


@pytest.mark.parametrize("case", list(STRIP_CASES), ids=list(STRIP_CASES))
def test_strip_values(device, block_max_bf16_op, case):
    S, NB, n_kind, n, max_grid = STRIP_CASES[case]
    x = strip_input([1, 1], S, NB, seed=len(case))
    want = strip_ref(x, S, NB)
    m = written(n, NB)
    xt = up(strip_device_view(x, S, NB), device, ttnn.bfloat16)
    prev = torch.full(list(want.shape), SENTINEL_FILL, dtype=torch.bfloat16)
    ot = up(prev, device, ttnn.bfloat16)
    kw = _fold_kwargs(None, n_kind, n, max_grid, device)

    got = block_max_bf16_op(xt, strip_major=True, output_tensor=ot, **kw)

    assert list(got.shape) == list(want.shape)
    assert got.dtype == ttnn.bfloat16 and got.layout == ttnn.ROW_MAJOR_LAYOUT
    assert _addr(got) == _addr(ot)
    check_written(got, want, m, prev, case)
    for d in downs(xt):
        assert_bytes_equal(d.reshape(x.shape), x, f"{case} input")


# --------------------------------------------------------------------------- refusals
def _fold_call_args(device, kind):
    """(block_max, kwargs) of one refused fold call."""
    x = fold_input([1, 1, 4, 256], seed=1)
    good = lambda: up(x, device, ttnn.float32)  # noqa: E731
    out_ok = lambda: up(torch.zeros(1, 1, 4, 64, dtype=torch.bfloat16), device, ttnn.bfloat16)  # noqa: E731
    if kind == "factor_3":
        return good(), {"factor": 3, "n": 8}
    if kind == "factor_32":
        return good(), {"factor": 32, "n": 8}
    if kind == "factor_0":
        return good(), {"factor": 0, "n": 8}
    if kind == "input_bf16":
        return up(x, device, ttnn.bfloat16), {"factor": 4, "n": 8}
    if kind == "input_tile":
        return up(fold_input([1, 1, 32, 256], 1), device, ttnn.float32, layout=ttnn.TILE_LAYOUT), {"factor": 4, "n": 8}
    if kind == "input_l1":
        return up(x, device, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG), {"factor": 4, "n": 8}
    if kind == "width_not_multiple_of_8f":
        return up(fold_input([1, 1, 4, 96], 1), device, ttnn.float32), {"factor": 8, "n": 8}
    if kind == "n_and_n_tensor":
        return good(), {"factor": 4, "n": 8, "n_tensor": up_n(8, device)}
    if kind == "neither_n":
        return good(), {"factor": 4}
    if kind == "output_wrong_shape":
        return good(), {"factor": 4, "n": 8, "output_tensor": up(torch.zeros(1, 1, 4, 128), device, ttnn.bfloat16)}
    if kind == "output_wrong_dtype":
        return good(), {"factor": 4, "n": 8, "output_tensor": up(torch.zeros(1, 1, 4, 64), device, ttnn.float32)}
    if kind == "output_l1":
        o = up(torch.zeros(1, 1, 4, 64), device, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        return good(), {"factor": 4, "n": 8, "output_tensor": o}
    if kind == "output_ok":  # (not a refusal: the valid neighbour of the output cases)
        return good(), {"factor": 4, "n": 8, "output_tensor": out_ok()}
    raise ValueError(kind)


FOLD_REFUSALS = {
    "factor_3": "factor",
    "factor_32": "factor",
    "factor_0": "factor",
    "input_bf16": "block_max",
    "input_tile": "block_max",
    "input_l1": "DRAM interleaved",
    "width_not_multiple_of_8f": "multiple of 8",
    "n_and_n_tensor": "exactly one of n",
    "neither_n": "exactly one of n",
    "output_wrong_shape": "output_tensor",
    "output_wrong_dtype": "output_tensor",
    "output_l1": "DRAM interleaved",
}


@pytest.mark.parametrize("kind", list(FOLD_REFUSALS), ids=list(FOLD_REFUSALS))
def test_fold_refusals(device, block_max_bf16_op, expect_error, kind):
    xt, kw = _fold_call_args(device, kind)
    with expect_error(REFUSED, FOLD_REFUSALS[kind]):
        block_max_bf16_op(xt, **kw)


def _strip_call_args(device, kind):
    S, NB = 2, 64
    x = strip_input([1, 1], S, NB, seed=2)
    good = lambda: up(strip_device_view(x, S, NB), device, ttnn.bfloat16)  # noqa: E731
    out = lambda shape: up(torch.zeros(shape), device, ttnn.bfloat16)  # noqa: E731
    if kind == "factor_2":
        return good(), {"factor": 2, "n": 8, "output_tensor": out([1, 1, 64, 64])}
    if kind == "width_128":
        return up(x.reshape(1, 1, 32, 128), device, ttnn.bfloat16), {"n": 8, "output_tensor": out([1, 1, 64, 64])}
    if kind == "no_output_tensor":
        return good(), {"n": 8}
    if kind == "output_nb_12":
        return up(torch.zeros(1, 1, 8, 256), device, ttnn.bfloat16), {"n": 8, "output_tensor": out([1, 1, 32, 12])}
    if kind == "output_rows_48":
        return good(), {"n": 8, "output_tensor": out([1, 1, 48, 64])}
    if kind == "input_rows_mismatch":
        return good(), {"n": 8, "output_tensor": out([1, 1, 96, 64])}
    if kind == "input_fp32":
        return up(strip_device_view(x, S, NB).float(), device, ttnn.float32), {
            "n": 8,
            "output_tensor": out([1, 1, 64, 64]),
        }
    if kind == "input_l1":
        t = up(strip_device_view(x, S, NB), device, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        return t, {"n": 8, "output_tensor": out([1, 1, 64, 64])}
    if kind == "neither_n":
        return good(), {"output_tensor": out([1, 1, 64, 64])}
    if kind == "n_and_n_tensor":
        return good(), {"n": 8, "n_tensor": up_n(8, device), "output_tensor": out([1, 1, 64, 64])}
    if kind == "output_fp32":
        return good(), {"n": 8, "output_tensor": up(torch.zeros(1, 1, 64, 64), device, ttnn.float32)}
    raise ValueError(kind)


STRIP_REFUSALS = {
    "factor_2": "factor",
    "width_128": "width",
    "no_output_tensor": "output_tensor",
    "output_nb_12": "output",
    "output_rows_48": "output",
    "input_rows_mismatch": "input rows",
    "input_fp32": "block_max",
    "input_l1": "DRAM interleaved",
    "neither_n": "exactly one of n",
    "n_and_n_tensor": "exactly one of n",
    "output_fp32": "output",
}


@pytest.mark.parametrize("kind", list(STRIP_REFUSALS), ids=list(STRIP_REFUSALS))
def test_strip_refusals(device, block_max_bf16_op, expect_error, kind):
    xt, kw = _strip_call_args(device, kind)
    with expect_error(REFUSED, STRIP_REFUSALS[kind]):
        block_max_bf16_op(xt, strip_major=True, **kw)


# --------------------------------------------------------------------------- cache sequences
# Every program-cache delta is measured around the op call only. A miss adds one entry (the op's single device
# program); a hit adds none. Hits are additionally proven by fresh live buffers (asserted distinct addresses), fresh
# data and changed runtime values, each checked against the reference.


def _fold_setup(device, shape, F, n, seed, n_kind="value"):
    x = fold_input(shape, seed)
    want = fold_ref(x, F)
    prev = torch.full(list(want.shape), SENTINEL_FILL, dtype=torch.bfloat16)
    xt = up(x, device, ttnn.float32)
    ot = up(prev, device, ttnn.bfloat16)
    kw = {"factor": F, "output_tensor": ot}
    if n_kind == "value":
        kw["n"] = n
    else:
        kw["n_tensor"] = up_n(n, device)
    return xt, kw, want, prev


def test_cache_fold_miss_then_hits_fresh_buffers_and_n(fresh_cache, block_max_bf16_op):
    """A -> new A -> new A: same metadata, new input/output buffers (kept alive), new data, a new ``n`` each call
    (a runtime value: never part of the key)."""
    device = fresh_cache
    shape, F = [1, 1, 50, 4608], 4
    keep = []
    for i, n in enumerate((1000, 37, 1152)):
        xt, kw, want, prev = _fold_setup(device, shape, F, n, seed=100 + i)
        got, delta = _cache_delta(device, lambda: block_max_bf16_op(xt, **kw))
        assert delta == (1 if i == 0 else 0), f"call {i}: program-cache delta {delta}"
        check_written(got, want, written(n, want.shape[-1]), prev, f"call {i}")
        for old_x, old_o in keep:
            assert _addr(xt) != _addr(old_x) and _addr(got) != _addr(old_o), "buffers must be fresh live allocations"
        keep.append((xt, got))


def test_cache_fold_n_value_to_n_tensor_and_back(fresh_cache, block_max_bf16_op):
    """n value -> n_tensor -> n value -> n_tensor: reading the length on the device is a different program (its own
    miss once); afterwards both are hits, with the value / tensor contents changing on every call."""
    device = fresh_cache
    shape, F = [1, 1, 20, 2048], 2
    seq = [("value", 1024, 1), ("tensor", 300, 1), ("value", 9, 0), ("tensor", 1, 0), ("tensor", 1024, 0)]
    for i, (kind, n, expected) in enumerate(seq):
        xt, kw, want, prev = _fold_setup(device, shape, F, n, seed=200 + i, n_kind=kind)
        got, delta = _cache_delta(device, lambda: block_max_bf16_op(xt, **kw))
        assert delta == expected, f"step {i} ({kind} {n}): program-cache delta {delta}, expected {expected}"
        check_written(got, want, written(n, want.shape[-1]), prev, f"step {i}")


def test_cache_fold_n_tensor_rewritten_in_place(fresh_cache, block_max_bf16_op):
    """The same ``n_tensor`` buffer, its contents rewritten between calls: the device reads the current value."""
    device = fresh_cache
    shape, F = [1, 1, 12, 1024], 1
    nt = up_n(512, device)
    for i, n in enumerate((512, 40, 1000)):
        if i:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.tensor([n], dtype=torch.int32).view(1, 1, 1, 1), dtype=ttnn.uint32), nt
            )
        x = fold_input(shape, seed=300 + i)
        want = fold_ref(x, F)
        prev = torch.full(list(want.shape), SENTINEL_FILL, dtype=torch.bfloat16)
        xt, ot = up(x, device, ttnn.float32), up(prev, device, ttnn.bfloat16)
        got, delta = _cache_delta(device, lambda: block_max_bf16_op(xt, factor=F, n_tensor=nt, output_tensor=ot))
        assert delta == (1 if i == 0 else 0)
        check_written(got, want, written(n, want.shape[-1]), prev, f"n_tensor={n}")


def test_cache_fold_factor_a_b_a(fresh_cache, block_max_bf16_op):
    """factor 4 -> 16 -> 4 (same input shape): factor shapes the kernel's fold, so a new program, then reuse."""
    device = fresh_cache
    shape = [1, 1, 30, 2048]
    for i, (F, expected) in enumerate(((4, 1), (16, 1), (4, 0), (16, 0))):
        xt, kw, want, prev = _fold_setup(device, shape, F, 64, seed=400 + i)
        got, delta = _cache_delta(device, lambda: block_max_bf16_op(xt, **kw))
        assert delta == expected, f"factor {F} step {i}: delta {delta}"
        check_written(got, want, written(64, want.shape[-1]), prev, f"factor {F}")


def test_cache_fold_max_grid_a_b_a(fresh_cache, block_max_bf16_op):
    """default grid -> max_grid 2x2 -> default -> 2x2: the core set is program-defining."""
    device = fresh_cache
    shape, F = [1, 1, 40, 1024], 8
    seq = [(None, 1), (ttnn.CoreCoord(2, 2), 1), (None, 0), (ttnn.CoreCoord(2, 2), 0)]
    for i, (mg, expected) in enumerate(seq):
        xt, kw, want, prev = _fold_setup(device, shape, F, 128, seed=500 + i)
        if mg is not None:
            kw["max_grid"] = mg
        got, delta = _cache_delta(device, lambda: block_max_bf16_op(xt, **kw))
        assert delta == expected, f"max_grid {mg} step {i}: delta {delta}"
        check_written(got, want, written(128, want.shape[-1]), prev, f"max_grid {mg}")


def test_cache_fold_output_absent_present_absent(fresh_cache, block_max_bf16_op):
    """allocated output -> caller's output -> allocated: the same DRAM interleaved output either way, so only the
    output address changes (a hit); each output holds its own call's result."""
    device = fresh_cache
    shape, F, n = [1, 1, 16, 512], 2, 256
    outs = []
    for i, (caller, expected) in enumerate(((False, 1), (True, 0), (False, 0), (True, 0))):
        xt, kw, want, prev = _fold_setup(device, shape, F, n, seed=600 + i)
        if not caller:
            kw.pop("output_tensor")
        got, delta = _cache_delta(device, lambda: block_max_bf16_op(xt, **kw))
        assert delta == expected, f"step {i} (caller output {caller}): delta {delta}"
        if caller:
            assert _addr(got) == _addr(kw["output_tensor"])
        check_written(got, want, written(n, want.shape[-1]), prev if caller else None, f"step {i}")
        outs.append(got)
    assert len({_addr(o) for o in outs}) == len(outs)


def test_cache_fold_shape_change_and_return(fresh_cache, block_max_bf16_op):
    """rows 64 -> rows 7 -> rows 64: returning to the first shape reuses its program; the middle shape is correct
    (its own program on the source; a port that keeps rows out of the key may hit, so it is not pinned)."""
    device = fresh_cache
    F, n = 4, 512
    for i, (rows, expected) in enumerate(((64, 1), (7, None), (64, 0))):
        xt, kw, want, prev = _fold_setup(device, [1, 1, rows, 2048], F, n, seed=700 + i)
        got, delta = _cache_delta(device, lambda: block_max_bf16_op(xt, **kw))
        if expected is None:
            assert delta in (0, 1), f"rows {rows}: delta {delta}"
        else:
            assert delta == expected, f"rows {rows}: delta {delta}"
        check_written(got, want, written(n, want.shape[-1]), prev, f"rows {rows}")


def test_cache_valid_invalid_valid(fresh_cache, block_max_bf16_op, expect_error):
    """A warmed program still refuses invalid input (factor 3, then both n and n_tensor), and the next valid call is
    a correct hit."""
    device = fresh_cache
    shape, F = [1, 1, 10, 1024], 4
    xt, kw, want, prev = _fold_setup(device, shape, F, 200, seed=800)
    got, delta = _cache_delta(device, lambda: block_max_bf16_op(xt, **kw))
    assert delta == 1
    check_written(got, want, written(200, want.shape[-1]), prev, "warm")
    xt2, kw2, _, _ = _fold_setup(device, shape, F, 200, seed=801)
    with expect_error(REFUSED, "factor"):
        block_max_bf16_op(xt2, **{**kw2, "factor": 3})
    with expect_error(REFUSED, "exactly one of n"):
        block_max_bf16_op(xt2, **kw2, n_tensor=up_n(8, device))
    xt3, kw3, want3, prev3 = _fold_setup(device, shape, F, 72, seed=802)
    got3, delta = _cache_delta(device, lambda: block_max_bf16_op(xt3, **kw3))
    assert delta == 0
    check_written(got3, want3, written(72, want3.shape[-1]), prev3, "after refusals")


def test_cache_strip_miss_hits_and_n_kinds(fresh_cache, block_max_bf16_op):
    """strip-major: miss, then hits on fresh buffers with a new n each call; n_tensor its own program once; and
    a fold call of the same session does not reuse a strip program."""
    device = fresh_cache
    S, NB = 3, 520
    keep = []
    seq = [("value", 520, 1), ("value", 9, 0), ("value", 130, 0), ("tensor", 256, 1), ("tensor", 1, 0), ("value", 7, 0)]
    for i, (kind, n, expected) in enumerate(seq):
        x = strip_input([1, 1], S, NB, seed=900 + i)
        want = strip_ref(x, S, NB)
        prev = torch.full(list(want.shape), SENTINEL_FILL, dtype=torch.bfloat16)
        xt, ot = up(strip_device_view(x, S, NB), device, ttnn.bfloat16), up(prev, device, ttnn.bfloat16)
        kw = {"n": n} if kind == "value" else {"n_tensor": up_n(n, device)}
        got, delta = _cache_delta(device, lambda: block_max_bf16_op(xt, strip_major=True, output_tensor=ot, **kw))
        assert delta == expected, f"step {i} ({kind} {n}): delta {delta}, expected {expected}"
        check_written(got, want, written(n, NB), prev, f"strip step {i}")
        for old_x, old_o in keep:
            assert _addr(xt) != _addr(old_x) and _addr(ot) != _addr(old_o)
        keep.append((xt, ot))
    # a fold call is a different program
    xt, kw, want, prev = _fold_setup(device, [1, 1, 8, 256], 1, 256, seed=999)
    got, delta = _cache_delta(device, lambda: block_max_bf16_op(xt, **kw))
    assert delta == 1
    check_written(got, want, 256, prev, "fold after strip")


def test_dispatch_route(device, block_max_bf16_op):
    """The selected route runs (source: the Python package's prepared-program dispatch; native: the C++ op with
    ttnn.generic_op forbidden by the fixture) and its result feeds a consumer op correctly."""
    shape, F, n = [1, 1, 8, 1024], 4, 256
    x = fold_input(shape, seed=1000)
    want = fold_ref(x, F)
    got = block_max_bf16_op(up(x, device, ttnn.float32), factor=F, n=n)
    assert block_max_bf16_op.last_dispatches >= 0
    # consumer: a TTNN op reading the result (tilize -> row max) agrees with the reference
    tiled = ttnn.to_layout(got, ttnn.TILE_LAYOUT)
    red = ttnn.max(tiled, dim=-1, keepdim=True)
    for d in downs(red):
        assert_bytes_equal(d.reshape(1, 1, 8, 1), want.amax(-1, keepdim=True), "consumer max")

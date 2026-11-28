# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.bringup.fp8_rows_ttnn:unpack_fp8_rows``.

Every test calls the operation through the ``unpack_fp8_rows_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (the public callable ``ttnn.bringup.fp8_rows_ttnn.unpack_fp8_rows``)::

    unpack_fp8_rows(rows, out, *, row_lo, row_hi, batch=0, out_row0=None, max_grid=None,
                    whole_grid=False, common_values=False) -> out

- ``rows``: [B, .., R, 192] uint8 ROW_MAJOR in DRAM (interleaved or ND-sharded), one 192-byte row a page:
  [e4m3 x 128 | fp32 scale little endian | ..]. Leading dims flattened are the batch index.
- ``out``: [.., M, 128] bfloat8_b TILE, DRAM interleaved (M = every dim but the last, flattened); written in place
  at rows [out_row0, out_row0 + row_hi - row_lo) (out_row0 defaults to row_lo); every other row untouched; the call
  returns ``out`` (same buffer). ``rows`` is never written.
- Values: e4m3 x scale in fp32, rounded to nearest even onto the bfloat8_b grid (16 values a group). Exact against
  the float64 reference (the golden suite's ``_bfp8(_decode(rows))``, tolerance: equality).
- On a mesh every chip runs the same range on its own rows.
- Refusals (AssertionError): rows not uint8 / ROW_MAJOR / width 192 ("rows must be uint8 ROW_MAJOR"); out not
  bfloat8_b / TILE / width 128 ("out must be bfloat8_b TILE"); rows outside DRAM ("rows must be in DRAM"); out not
  DRAM interleaved ("out must be DRAM interleaved"); batch outside [0, B) or not 0 <= row_lo <= row_hi <= R
  ("outside"); row_lo, row_hi - row_lo or out_row0 not a multiple of 32 ("whole tile rows only"); out rows past M
  ("past its"). An empty range (row_lo == row_hi) returns ``out`` without dispatching.
- Programs (README / CHANGELOG): one program per (shapes, memory configs, core count, common_values). Cores =
  min(cores of max_grid, 32-row bands), or every core of max_grid with ``whole_grid``. row_lo, row_hi, batch and
  out_row0 are per-run values (no new program once the core count is fixed); buffers are patched per call.
  max_grid accepts a CoreCoord or an (x, y) tuple and is clipped to the device grid.
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

SOURCE_ENTRY = "ttnn.bringup.fp8_rows_ttnn:unpack_fp8_rows"
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
        if name.startswith("ttnn.bringup.fp8_rows_ttnn") or name.startswith("ttnn._ttnn.operations.generic"):
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
                raise AssertionError("native unpack_fp8_rows must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def unpack_fp8_rows_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"unpack_fp8_rows acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch  # noqa: E402

import unpack_fp8_rows_pre_migration_helpers as H  # noqa: E402

# One new program on a miss: the op dispatches one device operation per call (README: "always prepared", one
# program per shapes / memory configs / core count), so a miss adds exactly one program-cache entry and a hit none.
MISS, HIT = 1, 0


def _counted(mesh, op, *args, **kwargs):
    """Call the op, returning (result, program-cache entries it added). Only the op runs between the two counts."""
    before = mesh.num_program_cache_entries()
    result = op(*args, **kwargs)
    return result, mesh.num_program_cache_entries() - before


def _mc(mesh, kind, rank):
    if kind == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    return H.nd_config(mesh, shard_rows=int(kind[2:]), rank=rank)


def _grid(mesh, spec):
    if isinstance(spec, str) and spec == "cap":
        return H.grid_cap(mesh)
    return spec


def _setup(mesh, rows_shape, out_shape, mc_kind="dram", seed=0, salt=0):
    chips = H.num_chips(mesh)
    rows_host = H.make_rows(chips, rows_shape, seed)
    prior = H.fill(chips, out_shape, salt)
    rows = H.rows_to_device(mesh, rows_host, _mc(mesh, mc_kind, len(rows_shape)))
    out = H.out_to_device(mesh, prior)
    return rows_host, prior, rows, out


def _check_result(result, out, out_shape):
    """The op returns ``out`` itself: same buffer, same metadata."""
    assert H.addr(result) == H.addr(out), "unpack_fp8_rows must return the caller's out buffer"
    assert result.dtype == ttnn.bfloat8_b and result.layout == ttnn.TILE_LAYOUT
    assert tuple(result.shape)[-len(out_shape) :] == tuple(out_shape)
    mc = result.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


# --------------------------------------------------------------------------- API, output and memory effects


def test_defaults_output_and_untouched(unpack_mesh, unpack_fp8_rows_op):
    """Defaults batch=0 and out_row0=row_lo; returns out (in place, same address, metadata); rows outside the range
    and the rows tensor untouched; a downstream consumer (typecast to bfloat16, exact for bfloat8_b values) reads it."""
    mesh = unpack_mesh
    rows_shape, out_shape = (2, 1, 256, H.ROW), (1, 1, 256, H.VALUES)
    rows_host, prior, rows, out = _setup(mesh, rows_shape, out_shape, seed=1, salt=1)
    result = unpack_fp8_rows_op(rows, out, row_lo=64, row_hi=192)
    logger.info(f"source dispatches through ttnn.generic_op: {unpack_fp8_rows_op.last_dispatches}")
    _check_result(result, out, out_shape)
    exp = H.expected_out(prior, rows_host, 64, 192, 0, 64)
    H.assert_exact(H.to_host(result), exp, "defaults")
    assert torch.equal(H.to_host(rows), rows_host), "rows must not be written"
    bf16 = ttnn.typecast(result, ttnn.bfloat16)
    H.assert_exact(H.to_host(bf16), exp, "typecast consumer")


def test_empty_range_is_noop(fresh_cache, unpack_fp8_rows_op):
    """row_lo == row_hi: returns out unchanged and builds / dispatches nothing."""
    mesh = fresh_cache
    rows_host, prior, rows, out = _setup(mesh, (1, 1, 128, H.ROW), (1, 1, 128, H.VALUES), seed=2, salt=2)
    result, added = _counted(mesh, unpack_fp8_rows_op, rows, out, row_lo=64, row_hi=64, out_row0=32)
    assert added == 0, f"an empty range must not build a program (added {added})"
    assert H.addr(result) == H.addr(out)
    H.assert_exact(H.to_host(out), prior, "empty range")


# --------------------------------------------------------------------------- planner branches (one program each)

# (id, rows shape, rows memory, out shape, row_lo, row_hi, batch, out_row0, max_grid, whole_grid, common_values)
BRANCHES = [
    ("one_band_min", (1, 1, 32, H.ROW), "dram", (1, 1, 32, H.VALUES), 0, 32, None, None, None, False, False),
    ("nd_offset_batch_outrow", (3, 1, 512, H.ROW), "nd16", (1, 1, 640, H.VALUES), 64, 448, 1, 128, "cap", False, False),
    (
        "uneven_split_tuple_grid",
        (1, 1, 1280, H.ROW),
        "dram",
        (1, 1, 1280, H.VALUES),
        32,
        1216,
        0,
        None,
        (2, 3),
        False,
        False,
    ),
    (
        "many_bands_per_core_nd8",
        (2, 1, 2048, H.ROW),
        "nd8",
        (1, 1, 2048, H.VALUES),
        0,
        1600,
        1,
        384,
        ttnn.CoreCoord(2, 2),
        False,
        False,
    ),
    ("whole_grid_idle_cores", (2, 1, 256, H.ROW), "nd16", (1, 1, 256, H.VALUES), 96, 192, 1, 0, "cap", True, False),
    (
        "common_values_whole_grid",
        (1, 1, 2048, H.ROW),
        "dram",
        (1, 1, 2048, H.VALUES),
        512,
        1696,
        0,
        None,
        "cap",
        True,
        True,
    ),
    (
        "common_values_core_per_band",
        (1, 1, 512, H.ROW),
        "nd16",
        (1, 1, 512, H.VALUES),
        0,
        160,
        None,
        None,
        None,
        False,
        True,
    ),
    (
        "full_range_last_batch_end_of_out",
        (2, 1, 256, H.ROW),
        "dram",
        (1, 1, 512, H.VALUES),
        0,
        256,
        1,
        256,
        "cap",
        False,
        False,
    ),
    ("rows_rank3_out_rank2", (2, 256, H.ROW), "dram", (256, H.VALUES), 64, 192, 1, 32, "cap", False, False),
    (
        "out_row_crosses_leading_dim",
        (1, 1, 256, H.ROW),
        "dram",
        (1, 2, 256, H.VALUES),
        0,
        64,
        None,
        224,
        "cap",
        False,
        False,
    ),
]


@pytest.mark.parametrize(
    "rows_shape, mc_kind, out_shape, row_lo, row_hi, batch, out_row0, max_grid, whole_grid, common_values",
    [pytest.param(*c[1:], id=c[0]) for c in BRANCHES],
)
def test_planner_branch(
    unpack_mesh,
    unpack_fp8_rows_op,
    rows_shape,
    mc_kind,
    out_shape,
    row_lo,
    row_hi,
    batch,
    out_row0,
    max_grid,
    whole_grid,
    common_values,
):
    """Each core-count / layout / option branch, exact against the reference, rows outside untouched, and the same
    call again (a cache hit) reproduces it on the same buffers."""
    mesh = unpack_mesh
    rows_host, prior, rows, out = _setup(mesh, rows_shape, out_shape, mc_kind, seed=sum(rows_shape) + row_lo, salt=3)
    kwargs = dict(row_lo=row_lo, row_hi=row_hi)
    if batch is not None:
        kwargs["batch"] = batch
    if out_row0 is not None:
        kwargs["out_row0"] = out_row0
    if max_grid is not None:
        kwargs["max_grid"] = _grid(mesh, max_grid)
    if whole_grid:
        kwargs["whole_grid"] = True
    if common_values:
        kwargs["common_values"] = True
    result = unpack_fp8_rows_op(rows, out, **kwargs)
    _check_result(result, out, out_shape)
    exp = H.expected_out(prior, rows_host, row_lo, row_hi, batch or 0, row_lo if out_row0 is None else out_row0)
    H.assert_exact(H.to_host(out), exp, "first call")
    result, added = _counted(mesh, unpack_fp8_rows_op, rows, out, **kwargs)
    assert added == HIT, f"the same call again must reuse its program (added {added})"
    H.assert_exact(H.to_host(out), exp, "repeated call (determinism)")


# --------------------------------------------------------------------------- program cache sequences


def test_cache_fresh_buffers_and_runtime_values(fresh_cache, unpack_fp8_rows_op):
    """A -> new A -> new A: one miss, then hits on fresh live rows/out buffers (distinct addresses, new data) with
    changed per-run values (row_lo, batch, out_row0 at the same band count): every output uses the current inputs,
    output buffer and values. Interleaved and ND-sharded rows (separate programs) each."""
    mesh = fresh_cache
    rows_shape, out_shape = (3, 1, 512, H.ROW), (1, 1, 640, H.VALUES)
    grid = H.grid_cap(mesh)
    for mc_kind in ("nd16", "dram"):
        live = []
        calls = [(64, 448, 1, 128), (0, 384, 2, 256), (128, 512, 0, 0)]  # 12 bands each: one core count
        for i, (lo, hi, b, o0) in enumerate(calls):
            rows_host, prior, rows, out = _setup(mesh, rows_shape, out_shape, mc_kind, seed=100 + i, salt=10 + i)
            for _, _, r_prev, o_prev in live:
                assert H.addr(rows) != H.addr(r_prev) and H.addr(out) != H.addr(o_prev), "buffers must be fresh"
            live.append((rows_host, prior, rows, out))
            result, added = _counted(
                mesh, unpack_fp8_rows_op, rows, out, row_lo=lo, row_hi=hi, batch=b, out_row0=o0, max_grid=grid
            )
            assert added == (MISS if i == 0 else HIT), f"{mc_kind} call {i}: added {added}"
            _check_result(result, out, out_shape)
            H.assert_exact(H.to_host(out), H.expected_out(prior, rows_host, lo, hi, b, o0), f"{mc_kind} call {i}")
        # earlier outputs were not touched by later calls
        rows_host, prior, rows, out = live[0]
        H.assert_exact(H.to_host(out), H.expected_out(prior, rows_host, 64, 448, 1, 128), f"{mc_kind} call 0 kept")


def test_cache_band_count_programs(fresh_cache, unpack_fp8_rows_op):
    """Default mode: a range below the grid's core count runs a core a band, one program per band count (1 -> 3 -> 1:
    miss, miss, hit); ranges of at least the grid's cores share one program (cores + 5 bands: miss; 2 * cores + 3
    bands at another offset: hit). In place on one out, each step exact and the rest of out kept."""
    mesh = fresh_cache
    grid = H.grid_cap(mesh)
    cores = H.grid_cores(mesh, grid)
    big1, big2 = cores + 5, 2 * cores + 3
    r = 32 * (big2 + 4)
    rows_host, prior, rows, out = _setup(mesh, (1, 1, r, H.ROW), (1, 1, r, H.VALUES), seed=21, salt=4)
    exp = prior
    steps = [  # (row_lo, bands, out_row0, expected entries added)
        (0, 1, 32, MISS),
        (64, 3, 128, MISS),
        (160, 1, 0, HIT),
        (32, big1, 64, MISS),
        (96, big2, 32, HIT),
    ]
    for lo, nb, o0, want in steps:
        hi = lo + 32 * nb
        _, added = _counted(mesh, unpack_fp8_rows_op, rows, out, row_lo=lo, row_hi=hi, out_row0=o0, max_grid=grid)
        assert added == want, f"[{lo}, {hi}) -> {o0}: added {added}, expected {want}"
        exp = H.expected_out(exp, rows_host, lo, hi, 0, o0)
        H.assert_exact(H.to_host(out), exp, f"[{lo}, {hi}) -> {o0}")


def test_cache_whole_grid_one_program(fresh_cache, unpack_fp8_rows_op):
    """whole_grid: ONE program for every row count (1, 3, 37 bands and back to 1: miss, hit, hit, hit), on fresh
    out buffers, bytes equal to the reference; common_values with whole_grid is its own single program likewise."""
    mesh = fresh_cache
    grid = H.grid_cap(mesh)
    rows_shape, out_shape = (2, 1, 2048, H.ROW), (1, 1, 2048, H.VALUES)
    rows_host, _, rows, _ = _setup(mesh, rows_shape, out_shape, "nd16", seed=31)
    chips = H.num_chips(mesh)
    for cv in (False, True):
        for i, (lo, nb) in enumerate(((0, 1), (96, 3), (512, 37), (1024, 1))):
            prior = H.fill(chips, out_shape, 20 + i)
            out = H.out_to_device(mesh, prior)
            hi = lo + 32 * nb
            _, added = _counted(
                mesh,
                unpack_fp8_rows_op,
                rows,
                out,
                row_lo=lo,
                row_hi=hi,
                batch=1,
                max_grid=grid,
                whole_grid=True,
                common_values=cv,
            )
            assert added == (MISS if i == 0 else HIT), f"common_values={cv} {nb} bands: added {added}"
            H.assert_exact(H.to_host(out), H.expected_out(prior, rows_host, lo, hi, 1, lo), f"cv={cv} {nb} bands")


def test_cache_option_transitions(fresh_cache, unpack_fp8_rows_op):
    """A -> B -> A over each program-defining option, then back (hit): common_values off -> on -> off; rows memory
    interleaved -> ND-sharded -> interleaved; max_grid (2, 3) tuple -> CoreCoord(2, 3) (the same grid: hit) -> a
    max_grid beyond the device grid, clipped to it, same as max_grid=None (the same cores)."""
    mesh = fresh_cache
    chips = H.num_chips(mesh)
    rows_shape, out_shape = (2, 1, 1024, H.ROW), (1, 1, 1024, H.VALUES)
    rows_host = H.make_rows(chips, rows_shape, 41)
    rows_i = H.rows_to_device(mesh, rows_host)
    rows_n = H.rows_to_device(mesh, rows_host, H.nd_config(mesh, 16))
    grid = H.grid_cap(mesh)
    g = mesh.compute_with_storage_grid_size()
    wg = dict(max_grid=grid, whole_grid=True)
    steps = [  # (rows, kwargs, expected entries added)
        (rows_i, dict(wg, common_values=False), MISS),
        (rows_i, dict(wg, common_values=True), MISS),
        (rows_i, dict(wg, common_values=False), HIT),
        (rows_n, dict(wg), MISS),
        (rows_i, dict(wg), HIT),
        (rows_n, dict(wg), HIT),
        (rows_i, dict(max_grid=(2, 3)), MISS),
        (rows_i, dict(max_grid=ttnn.CoreCoord(2, 3)), HIT),
        # the device's whole grid: the capped grid's program when the device grid fits inside the cap
        (rows_i, dict(max_grid=None, whole_grid=True), MISS if H.grid_cores(mesh) != H.grid_cores(mesh, grid) else HIT),
        (rows_i, dict(max_grid=(g.x + 7, g.y + 7), whole_grid=True), HIT),
    ]
    for i, (rows, kwargs, want) in enumerate(steps):
        lo, b = (i % 4) * 64, i % 2
        hi = lo + 640  # 20 bands
        prior = H.fill(chips, out_shape, 50 + i)
        out = H.out_to_device(mesh, prior)
        _, added = _counted(mesh, unpack_fp8_rows_op, rows, out, row_lo=lo, row_hi=hi, batch=b, **kwargs)
        assert added == want, f"step {i} {kwargs}: added {added}, expected {want}"
        H.assert_exact(H.to_host(out), H.expected_out(prior, rows_host, lo, hi, b, lo), f"step {i} {kwargs}")


def test_cache_valid_invalid_valid(fresh_cache, unpack_fp8_rows_op, expect_error):
    """A warmed program still refuses an invalid range (nothing built, out untouched), and the next valid call hits
    and is correct."""
    mesh = fresh_cache
    rows_host, prior, rows, out = _setup(mesh, (1, 1, 256, H.ROW), (1, 1, 256, H.VALUES), seed=51, salt=5)
    grid = H.grid_cap(mesh)
    _, added = _counted(mesh, unpack_fp8_rows_op, rows, out, row_lo=0, row_hi=128, max_grid=grid)
    assert added == MISS
    exp = H.expected_out(prior, rows_host, 0, 128, 0, 0)
    before = mesh.num_program_cache_entries()
    with expect_error(AssertionError, "whole tile rows only"):
        unpack_fp8_rows_op(rows, out, row_lo=16, row_hi=144, max_grid=grid)
    with expect_error(AssertionError, "outside"):
        unpack_fp8_rows_op(rows, out, row_lo=128, row_hi=384, max_grid=grid)
    assert mesh.num_program_cache_entries() == before, "a refused call must not build a program"
    H.assert_exact(H.to_host(out), exp, "after refusals")
    _, added = _counted(mesh, unpack_fp8_rows_op, rows, out, row_lo=128, row_hi=256, out_row0=0, max_grid=grid)
    assert added == HIT
    H.assert_exact(H.to_host(out), H.expected_out(exp, rows_host, 128, 256, 0, 0), "valid after invalid")


# --------------------------------------------------------------------------- refusals


def _bad_inputs(mesh, which):
    """(rows, out) with one property broken; every other property valid."""
    chips = H.num_chips(mesh)
    rows_host = H.make_rows(chips, (2, 1, 64, H.ROW), 61)
    prior = H.fill(chips, (1, 1, 64, H.VALUES), 6)
    if which == "rows_dtype":
        rows = ttnn.from_torch(
            rows_host.float().reshape(chips * 2, 1, 64, H.ROW),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if chips > 1 else None,
        )
    elif which == "rows_width":
        rows = H.rows_to_device(mesh, rows_host[..., :128].contiguous())
    elif which == "rows_l1":
        rows = H.rows_to_device(mesh, rows_host, ttnn.L1_MEMORY_CONFIG)
    else:
        rows = H.rows_to_device(mesh, rows_host)
    if which == "out_dtype":
        out = ttnn.from_torch(
            prior.reshape(chips, 64, H.VALUES),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if chips > 1 else None,
        )
    elif which == "out_width":
        out = H.out_to_device(mesh, torch.cat([prior, prior], dim=-1))
    elif which == "out_l1":
        out = H.out_to_device(mesh, prior, ttnn.L1_MEMORY_CONFIG)
    elif which == "out_is_rows":
        out = rows
    else:
        out = H.out_to_device(mesh, prior)
    return rows, out


# (id, broken input, kwargs, message the refusal carries)
REFUSALS = [
    ("rows_not_uint8", "rows_dtype", dict(row_lo=0, row_hi=32), "rows must be uint8 ROW_MAJOR"),
    ("rows_width_128", "rows_width", dict(row_lo=0, row_hi=32), "rows must be uint8 ROW_MAJOR"),
    ("rows_in_l1", "rows_l1", dict(row_lo=0, row_hi=32), "rows must be in DRAM"),
    ("out_not_bfloat8_b", "out_dtype", dict(row_lo=0, row_hi=32), "out must be bfloat8_b TILE"),
    ("out_width_256", "out_width", dict(row_lo=0, row_hi=32), "out must be bfloat8_b TILE"),
    ("out_is_rows", "out_is_rows", dict(row_lo=0, row_hi=32), "out must be bfloat8_b TILE"),
    ("out_in_l1", "out_l1", dict(row_lo=0, row_hi=32), "out must be DRAM interleaved"),
    ("batch_past_end", None, dict(row_lo=0, row_hi=32, batch=2), "outside"),
    ("batch_negative", None, dict(row_lo=0, row_hi=32, batch=-1), "outside"),
    ("row_hi_past_rows", None, dict(row_lo=32, row_hi=96), "outside"),
    ("row_lo_after_row_hi", None, dict(row_lo=64, row_hi=32), "outside"),
    ("row_lo_negative", None, dict(row_lo=-32, row_hi=32), "outside"),
    ("row_lo_misaligned", None, dict(row_lo=16, row_hi=48), "whole tile rows only"),
    ("range_misaligned", None, dict(row_lo=0, row_hi=48), "whole tile rows only"),
    ("out_row0_misaligned", None, dict(row_lo=0, row_hi=32, out_row0=8), "whole tile rows only"),
    ("out_rows_past_end", None, dict(row_lo=0, row_hi=64, out_row0=32), "past its"),
]


@pytest.mark.parametrize("which, kwargs, message", [pytest.param(*c[1:], id=c[0]) for c in REFUSALS])
def test_refusal(unpack_mesh, unpack_fp8_rows_op, expect_error, which, kwargs, message):
    """Every current validation rule refuses with AssertionError and its diagnostic; out is left untouched."""
    mesh = unpack_mesh
    rows, out = _bad_inputs(mesh, which)
    before = H.to_host(out) if which is None else None
    with expect_error(AssertionError, message):
        unpack_fp8_rows_op(rows, out, **kwargs)
    if before is not None:
        assert torch.equal(H.to_host(out), before), "a refused call must not write out"


def test_limits_just_inside(unpack_mesh, unpack_fp8_rows_op):
    """The valid edges of each range rule: last batch, row_hi == R, out rows ending at M."""
    mesh = unpack_mesh
    rows_host, prior, rows, out = _setup(mesh, (2, 1, 64, H.ROW), (1, 1, 64, H.VALUES), seed=61, salt=6)
    unpack_fp8_rows_op(rows, out, row_lo=32, row_hi=64, batch=1, out_row0=32)
    exp = H.expected_out(prior, rows_host, 32, 64, 1, 32)
    H.assert_exact(H.to_host(out), exp, "edges")
    unpack_fp8_rows_op(rows, out, row_lo=0, row_hi=64, batch=1, out_row0=0)
    H.assert_exact(H.to_host(out), H.expected_out(exp, rows_host, 0, 64, 1, 0), "whole rows, whole out")

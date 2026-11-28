# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.bringup.ckv_bfp4_ttnn:ckv_bfp4_rows``.

Every test calls the operation through the ``ckv_bfp4_rows_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (``ckv_bfp4_rows(src, rows, src_row0, n, row0, *, max_grid=None) -> None``):
    * ``src``: bf16 ROW_MAJOR, last dim 512, any rank (rows flattened over the leading dims), DRAM interleaved or
      ND-sharded. ``rows``: uint8 ROW_MAJOR, last dim 320, any rank (flattened likewise), written in place.
    * Rows ``[src_row0, src_row0 + n)`` of ``src`` -> 320-byte bfloat4_b records (32 shared exponents twice, then 256
      nibble bytes) at ``[row0, row0 + n)`` of ``rows``, byte-exact to tt-metal's host packer rule (round to nearest
      even, clamp at 7); every other row of ``rows`` and all of ``src`` untouched. Returns None.
    * ``n % 32 == 0`` (0 is a no-op), ``0 <= src_row0``, ``src_row0 + n <= src rows``, ``0 <= row0``,
      ``row0 + n <= table rows``; anything else is refused (source: AssertionError; port: RuntimeError).
    * ``max_grid`` ((x, y) or ttnn.CoreCoord): the cores used, inside that rectangle; default the whole grid.
    * Program reuse: one program per (device, src geometry incl. its shard spec, table geometry, max_grid); src_row0,
      n and row0 and every buffer address are per-call values (no new program for them).
    * Not promised (not tested): L1 tensors, host tensors, other dtypes. On a mesh, the same values on every chip.
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

SOURCE_ENTRY = "ttnn.bringup.ckv_bfp4_ttnn:ckv_bfp4_rows"
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
        if name.startswith("ttnn.bringup.ckv_bfp4_ttnn") or name.startswith("ttnn._ttnn.operations.generic"):
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
                raise AssertionError("native ckv_bfp4_rows must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def ckv_bfp4_rows_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"ckv_bfp4_rows acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch  # noqa: E402

from tests.ttnn.unit_tests.operations.ckv_bfp4_rows.pre_migration.ckv_rows_reference import (  # noqa: E402
    FILL,
    RECORD_BYTES,
    WIDTH,
    check_table,
    encode_rows,
    make_rows,
    nd_sharded_dram,
    src_host,
    table_host,
    to_src,
    to_table,
)

# Refusals: the source refuses with AssertionError (Python asserts); a C++ port refuses with RuntimeError (TT_FATAL).
# Both are the contract's refusal; the message keys below are what the source's messages carry.
REFUSED = (AssertionError, RuntimeError)


def _result_ok(result, table):
    """The op writes ``table`` in place and returns None (the source's contract). A port that returns the table it
    wrote (the same buffer) keeps that contract; any other return value does not."""
    if result is None:
        return True
    return isinstance(result, ttnn.Tensor) and result.buffer_address() == table.buffer_address()


def _entries(device):
    return device.num_program_cache_entries()


def _assert_miss(op, delta, what):
    # Native: one registered device operation, one program per miss. Source: the program is prepared once
    # (ttnn.bringup.prepare_program) and dispatched by run_prepared, outside the device program cache, so a source
    # miss adds 0 or 1 entries; what matters there is that hits add none.
    if op.mode == "native":
        assert delta == 1, f"{what}: a program-defining change must build exactly one program, got {delta}"
    else:
        assert delta in (0, 1), f"{what}: source miss added {delta} program cache entries"


def _assert_hit(delta, what):
    assert delta == 0, f"{what}: expected a program cache hit (no new entry), got {delta} new entries"


def _call(op, device, src, table, src_row0, n, row0, **kwargs):
    """Call the op and return (result, program-cache delta), measured around the op only."""
    before = _entries(device)
    result = op(src, table, src_row0, n, row0, **kwargs)
    return result, _entries(device) - before


# Planner branches reached through the public inputs: bands of 32 rows split over the grid by core index.
# (id, src shape, src layout, table shape, src_row0, n, row0, max_grid)
BRANCH_CASES = [
    # one band, every other core idle; smallest call
    pytest.param((1, 1, 64, WIDTH), "interleaved", (1, 1, 64, RECORD_BYTES), 0, 32, 0, None, id="one_band_idle_cores"),
    # 20 bands < cores, offsets in both tensors; a 4-D slab whose call crosses a batch boundary (rows 1024 apart)
    pytest.param(
        (4, 1, 1024, WIDTH),
        "interleaved",
        (1, 1, 1024, RECORD_BYTES),
        1056 - 64,
        640,
        96,
        None,
        id="slab_offset_20_bands",
    ),
    # 256 bands > cores: several bands a core, uneven split on any production grid
    pytest.param(
        (4, 1, 4096, WIDTH),
        "interleaved",
        (1, 1, 8192 + 1024, RECORD_BYTES),
        4096,
        8192,
        0,
        None,
        id="many_bands_per_core",
    ),
    # max_grid (x, y) = 4 cores, 7 bands: uneven split (2, 2, 2, 1)
    pytest.param(
        (1, 1, 512, WIDTH), "interleaved", (1, 1, 512, RECORD_BYTES), 64, 224, 32, (2, 2), id="max_grid_2x2_uneven"
    ),
    # max_grid as ttnn.CoreCoord: 3 cores, 10 bands, record rows past the source rows
    pytest.param(
        (1, 1, 320, WIDTH), "interleaved", (1, 1, 1024, RECORD_BYTES), 0, 320, 640, "coord_3x1", id="max_grid_coord_3x1"
    ),
    # ranks: a 2-D source, a 4-D table whose rows flatten over its leading dims (row0 crosses a [.., 512, 320] slice)
    pytest.param(
        (2048, WIDTH), "interleaved", (2, 1, 512, RECORD_BYTES), 1000, 64, 480, None, id="rank2_src_rank4_table"
    ),
    # both ranges end exactly at their tensors' last row (the limits, just inside)
    pytest.param(
        (1, 1, 256, WIDTH), "interleaved", (1, 1, 160, RECORD_BYTES), 160, 96, 64, None, id="ranges_end_at_last_row"
    ),
    # ND-sharded source, [1, 1, 16, 512] shards round-robin over the DRAM banks: a band spans two shards
    pytest.param(
        (4, 1, 1024, WIDTH), "nd16", (1, 1, 2048, RECORD_BYTES), 1024 + 32, 1280, 256, None, id="nd_sharded_16"
    ),
    # ND-sharded source with band-sized shards and a capped grid
    pytest.param(
        (2, 1, 512, WIDTH), "nd32", (1, 1, 512, RECORD_BYTES), 256, 512, 0, (3, 2), id="nd_sharded_32_max_grid"
    ),
]


def _mc(device, layout):
    if layout == "interleaved":
        return ttnn.DRAM_MEMORY_CONFIG
    return nd_sharded_dram(device, shard_rows=int(layout[2:]))


def _mg(max_grid):
    return ttnn.CoreCoord(3, 1) if max_grid == "coord_3x1" else max_grid


def _total(shape):
    return int(torch.Size(shape).numel()) // shape[-1]


@pytest.mark.parametrize(("src_shape", "layout", "table_shape", "src_row0", "n", "row0", "max_grid"), BRANCH_CASES)
def test_records_match_host_rule(device, ckv_bfp4_rows_op, src_shape, layout, table_shape, src_row0, n, row0, max_grid):
    x = make_rows(_total(src_shape), seed=11 + n + row0)
    src = to_src(device, x, src_shape, _mc(device, layout))
    table = to_table(device, table_shape)
    spec_before = (list(table.shape), table.dtype, table.layout, table.memory_config(), table.buffer_address())
    kwargs = {} if max_grid is None else {"max_grid": _mg(max_grid)}
    result = ckv_bfp4_rows_op(src, table, src_row0, n, row0, **kwargs)
    assert _result_ok(result, table), f"unexpected return value {result!r}"
    # written in place: the same tensor, spec and buffer
    assert (list(table.shape), table.dtype, table.layout, table.memory_config(), table.buffer_address()) == spec_before
    check_table(table, x, src_row0, n, row0, label=str(src_shape))
    # the source is read only
    for got in src_host(src):
        assert torch.equal(got, x), "the source rows were modified"
    # deterministic: the same call again gives the same bytes
    again = to_table(device, table_shape)
    ckv_bfp4_rows_op(src, again, src_row0, n, row0, **kwargs)
    for a, b in zip(table_host(table), table_host(again)):
        assert torch.equal(a, b), "a repeated call wrote different bytes"


def test_n_zero_is_a_no_op(device, ckv_bfp4_rows_op):
    # n == 0 is valid (0 % 32 == 0); nothing is written, nothing dispatched, no program built.
    x = make_rows(64, seed=3)
    src = to_src(device, x, (1, 1, 64, WIDTH))
    table = to_table(device, (1, 1, 64, RECORD_BYTES))
    result, delta = _call(ckv_bfp4_rows_op, device, src, table, 0, 0, 64)  # row0 == total rows: legal when n == 0
    assert _result_ok(result, table)
    assert ckv_bfp4_rows_op.last_dispatches == 0
    assert delta == 0, "an n == 0 call must not build a program"
    for got in table_host(table):
        assert bool((got == FILL).all()), "an n == 0 call wrote the table"


def test_overwrites_existing_records(device, ckv_bfp4_rows_op):
    # A table that already holds records (a growing history): a second call over part of them replaces exactly
    # [row0, row0 + n) and keeps the rest.
    x = make_rows(1024, seed=5)
    y = make_rows(1024, seed=6)
    src_x = to_src(device, x, (1, 1, 1024, WIDTH))
    src_y = to_src(device, y, (1, 1, 1024, WIDTH))
    table = to_table(device, (1, 1, 512, RECORD_BYTES), fill=0)
    ckv_bfp4_rows_op(src_x, table, 0, 512, 0)
    ckv_bfp4_rows_op(src_y, table, 320, 96, 128)
    expected = encode_rows(x[:512])
    expected[128:224] = encode_rows(y[320:416])
    for got in table_host(table):
        assert torch.equal(got, expected)


# --------------------------------------------------------------------------- refusals (the support contract)


# (id, builder of (src, table, src_row0, n, row0), message regex)
def _r_src_dtype(device):
    return (
        ttnn.from_torch(torch.zeros(1, 1, 64, WIDTH), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        to_table(device, (1, 1, 64, RECORD_BYTES)),
        0,
        32,
        0,
    )


def _r_src_layout(device):
    return (
        ttnn.from_torch(torch.zeros(1, 1, 64, WIDTH), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        to_table(device, (1, 1, 64, RECORD_BYTES)),
        0,
        32,
        0,
    )


def _r_src_width(device):
    return (
        ttnn.from_torch(torch.zeros(1, 1, 64, 256), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        to_table(device, (1, 1, 64, RECORD_BYTES)),
        0,
        32,
        0,
    )


def _r_table_dtype(device):
    table = ttnn.from_torch(
        torch.zeros(1, 1, 64, RECORD_BYTES), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    return (to_src(device, make_rows(64, 1), (1, 1, 64, WIDTH)), table, 0, 32, 0)


def _r_table_width(device):
    return (to_src(device, make_rows(64, 1), (1, 1, 64, WIDTH)), to_table(device, (1, 1, 64, 288)), 0, 32, 0)


def _r_table_layout(device):
    table = ttnn.from_torch(
        torch.zeros(1, 1, 64, RECORD_BYTES, dtype=torch.uint8), dtype=ttnn.uint8, layout=ttnn.TILE_LAYOUT, device=device
    )
    return (to_src(device, make_rows(64, 1), (1, 1, 64, WIDTH)), table, 0, 32, 0)


def _valid(device, src_row0, n, row0):
    return (
        to_src(device, make_rows(64, 1), (1, 1, 64, WIDTH)),
        to_table(device, (1, 1, 64, RECORD_BYTES)),
        src_row0,
        n,
        row0,
    )


REFUSALS = [
    pytest.param(_r_src_dtype, r"DataType|dtype|bfloat16", id="src_not_bfloat16"),
    pytest.param(_r_src_layout, r"Layout|layout|ROW_MAJOR", id="src_tile_layout"),
    pytest.param(_r_src_width, r"Shape|shape|width|512", id="src_width_not_512"),
    pytest.param(_r_table_dtype, r"DataType|dtype|uint8", id="table_not_uint8"),
    pytest.param(_r_table_layout, r"Layout|layout|ROW_MAJOR", id="table_tile_layout"),
    pytest.param(_r_table_width, r"Shape|shape|width|320", id="table_width_not_320"),
    pytest.param(lambda d: _valid(d, 0, 16, 0), r"multiple of 32", id="n_not_multiple_of_32"),
    pytest.param(lambda d: _valid(d, 32, 64, 0), r"source rows", id="src_range_past_end"),
    pytest.param(lambda d: _valid(d, 0, 64, 32), r"record rows", id="record_range_past_end"),
]


@pytest.mark.parametrize(("build", "message"), REFUSALS)
def test_refusals(device, ckv_bfp4_rows_op, expect_error, build, message):
    src, table, src_row0, n, row0 = build(device)
    with expect_error(REFUSED, message):
        ckv_bfp4_rows_op(src, table, src_row0, n, row0)


NEGATIVE = [
    pytest.param(-32, 32, 0, r"source rows|incompatible function arguments|src_row0", id="negative_src_row0"),
    pytest.param(0, 32, -32, r"record rows|incompatible function arguments|row0", id="negative_row0"),
    pytest.param(0, -32, 0, r"multiple of 32|incompatible function arguments|\bn\b", id="negative_n"),
]


@pytest.mark.parametrize(("src_row0", "n", "row0", "message"), NEGATIVE)
def test_negative_values_refused(device, ckv_bfp4_rows_op, expect_error, src_row0, n, row0, message):
    # Negative offsets / counts are outside the contract. The source asserts; a port taking unsigned integers may
    # refuse them at argument conversion (TypeError) instead.
    src, table, _, _, _ = _valid(device, 0, 32, 0)
    with expect_error(REFUSED + (TypeError,), message):
        ckv_bfp4_rows_op(src, table, src_row0, n, row0)
    for got in table_host(table):
        assert bool((got == FILL).all()), "a refused call wrote the table"


def test_warm_call_still_refuses_then_recovers(device, ckv_bfp4_rows_op, expect_error):
    # valid -> invalid -> valid on one warmed program: the range checks run on every call (src_row0, n and row0 are
    # per-call values, never part of the cached program), and the next valid call is correct.
    x = make_rows(256, seed=21)
    src = to_src(device, x, (1, 1, 256, WIDTH))
    table = to_table(device, (1, 1, 128, RECORD_BYTES))
    ckv_bfp4_rows_op(src, table, 0, 64, 0)
    check_table(table, x, 0, 64, 0)
    with expect_error(REFUSED, r"record rows"):
        ckv_bfp4_rows_op(src, table, 0, 96, 64)
    with expect_error(REFUSED, r"source rows"):
        ckv_bfp4_rows_op(src, table, 224, 64, 0)
    with expect_error(REFUSED, r"multiple of 32"):
        ckv_bfp4_rows_op(src, table, 0, 48, 0)
    for got in table_host(table):  # the refused calls wrote nothing
        assert torch.equal(got[:64], encode_rows(x[:64])) and bool((got[64:] == FILL).all())
    fresh = to_table(device, (1, 1, 128, RECORD_BYTES))
    ckv_bfp4_rows_op(src, fresh, 160, 96, 32)
    check_table(fresh, x, 160, 96, 32)


# --------------------------------------------------------------------------- program cache

# Program-defining (per the op's contract: one program per (mesh, src geometry, rows geometry, max_grid)): the source's
# memory layout (interleaved / ND-sharded and its shard shape) and the core grid (max_grid). Runtime values, never a
# new program: src_row0, n, row0 and every buffer address ("no program a row count or offset").


def test_cache_runtime_values_and_fresh_buffers_hit(device, ckv_bfp4_rows_op):
    device.enable_program_cache()
    device.clear_program_cache()
    src_shape, table_shape = (2, 1, 1024, WIDTH), (1, 1, 2048, RECORD_BYTES)
    calls = [(0, 32, 0), (1024 + 32, 640, 96), (64, 1920, 128), (2048 - 96, 96, 2048 - 96), (512, 1024, 1024)]
    keep = []  # every source and table stays alive: each call sees new live buffers
    first = True
    for i, (src_row0, n, row0) in enumerate(calls):
        x = make_rows(_total(src_shape), seed=100 + i)  # new data every call: a stale read fails the comparison
        src = to_src(device, x, src_shape)
        table = to_table(device, table_shape)
        for s, t in keep:
            assert src.buffer_address() != s.buffer_address() and table.buffer_address() != t.buffer_address()
        keep.append((src, table))
        result, delta = _call(ckv_bfp4_rows_op, device, src, table, src_row0, n, row0)
        assert _result_ok(result, table)
        if first:
            _assert_miss(ckv_bfp4_rows_op, delta, "first call")
            first = False
        else:
            _assert_hit(delta, f"call {i} (src_row0={src_row0}, n={n}, row0={row0}, new buffers)")
        check_table(table, x, src_row0, n, row0, label=f"call {i}")
    # A -> A on the very same (first) buffers: other values, rows written earlier kept
    src, table = keep[0]
    result, delta = _call(ckv_bfp4_rows_op, device, src, table, 32, 64, 256)
    _assert_hit(delta, "same buffers, new values")
    x0 = make_rows(_total(src_shape), seed=100)
    got = table_host(table)[0]
    assert torch.equal(got[256:320], encode_rows(x0[32:96])) and torch.equal(got[0:32], encode_rows(x0[0:32]))


def test_cache_program_fields_miss_then_return_hits(device, ckv_bfp4_rows_op):
    device.enable_program_cache()
    device.clear_program_cache()
    src_shape, table_shape = (1, 1, 1024, WIDTH), (1, 1, 1024, RECORD_BYTES)
    x = make_rows(1024, seed=31)
    src_i = to_src(device, x, src_shape)
    src_nd = to_src(device, x, src_shape, nd_sharded_dram(device, 16))

    def run(src, src_row0, n, row0, label, **kwargs):
        table = to_table(device, table_shape)
        result, delta = _call(ckv_bfp4_rows_op, device, src, table, src_row0, n, row0, **kwargs)
        assert _result_ok(result, table)
        check_table(table, x, src_row0, n, row0, label=label)
        return delta

    _assert_miss(ckv_bfp4_rows_op, run(src_i, 0, 256, 0, "A"), "A: interleaved, full grid")
    _assert_hit(run(src_i, 96, 512, 64, "A again"), "A again (other values)")
    # A -> B (max_grid: other cores and split) -> A
    _assert_miss(ckv_bfp4_rows_op, run(src_i, 0, 256, 0, "B", max_grid=(2, 2)), "B: max_grid (2, 2)")
    _assert_hit(run(src_i, 32, 320, 32, "B again", max_grid=(2, 2)), "B again")
    _assert_hit(run(src_i, 64, 128, 0, "back to A"), "back to A")
    # A -> C (ND-sharded source: other reader addressing) -> A
    _assert_miss(ckv_bfp4_rows_op, run(src_nd, 0, 256, 0, "C"), "C: ND-sharded source")
    _assert_hit(run(src_nd, 160, 640, 96, "C again"), "C again")
    _assert_hit(run(src_i, 0, 1024, 0, "back to A"), "back to A after C")
    # n == 0 between warmed calls: no program, nothing written
    table = to_table(device, table_shape)
    result, delta = _call(ckv_bfp4_rows_op, device, src_i, table, 0, 0, 0)
    assert delta == 0 and _result_ok(result, table)
    assert all(bool((t == FILL).all()) for t in table_host(table))


def test_cache_other_table_geometry_is_correct(device, ckv_bfp4_rows_op):
    # A table of another size (a longer history) may or may not need its own program (the source keys on the table's
    # geometry; a port need not): at most one new program, correct records, and the original geometry still hits.
    device.enable_program_cache()
    device.clear_program_cache()
    x = make_rows(512, seed=41)
    src = to_src(device, x, (1, 1, 512, WIDTH))
    small = to_table(device, (1, 1, 256, RECORD_BYTES))
    _, d0 = _call(ckv_bfp4_rows_op, device, src, small, 0, 256, 0)
    _assert_miss(ckv_bfp4_rows_op, d0, "first table geometry")
    check_table(small, x, 0, 256, 0)
    large = to_table(device, (2, 1, 1024, RECORD_BYTES))
    _, d1 = _call(ckv_bfp4_rows_op, device, src, large, 64, 448, 1500)
    assert d1 in (0, 1), f"another table geometry built {d1} programs"
    check_table(large, x, 64, 448, 1500)
    small2 = to_table(device, (1, 1, 256, RECORD_BYTES))
    _, d2 = _call(ckv_bfp4_rows_op, device, src, small2, 256, 128, 128)
    _assert_hit(d2, "the first table geometry again")
    check_table(small2, x, 256, 128, 128)

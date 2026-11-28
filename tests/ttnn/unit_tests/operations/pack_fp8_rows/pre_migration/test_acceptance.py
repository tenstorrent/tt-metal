# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.bringup.fp8_rows_ttnn:pack_fp8_rows``.

Every test calls the operation through the ``pack_fp8_rows_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (``pack_fp8_rows(e4m3, scale, *, output_tensor=None, memory_config=None, max_grid=None)``):
    e4m3   [.., M, 128] FP8_E4M3 ROW_MAJOR, DRAM interleaved
    scale  [.., M, 1]   FLOAT32  ROW_MAJOR, DRAM interleaved, the same leading dims
    ->     [.., M, 192] UINT8 ROW_MAJOR DRAM interleaved; row m = [e4m3 row m (128 B) | scale m, fp32 LE (4 B) | 60 zero
           bytes], and all 192 bytes zero where the row's 128 values are all +-0 (the scale too).
    output_tensor  the caller's buffer of exactly that spec, written and returned as is (the same object).
    memory_config  the allocated output's memory config (default DRAM interleaved; anything else is refused).
    max_grid       ttnn.CoreCoord or (x, y): cores from (0, 0) .. (max_grid - 1), clamped to the device grid.
    Refusals (AssertionError today, the golden suite asserts the type): e4m3 not FP8_E4M3 ROW_MAJOR; scale not FLOAT32
    ROW_MAJOR; width not 128 / scale width not 1; leading dims differ; e4m3 / scale / output not DRAM interleaved;
    output_tensor not uint8 ROW_MAJOR of the output shape. The options are keyword-only.
    Program reuse (README: "built once per shape, memory configs and grid; the three addresses patched a call"): the
    shape and max_grid's effective core set select the program; buffer addresses and data are per-call; whether the
    output is allocated or the caller's does not change the program (same spec).
    Planner (README): rows split over ceil(M / 16) cores up to the grid, staged 16 rows a block; the cases cover one
    row, one full block, a partial final block, multiple blocks a core, uneven splits and a 1x1 grid.

Source note: the source dispatches through ``ttnn.bringup.run_prepared`` (a prepared program), not ``ttnn.generic_op``,
so ``last_dispatches`` is 0 on both routes; the tests do not assert on it.
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

SOURCE_ENTRY = "ttnn.bringup.fp8_rows_ttnn:pack_fp8_rows"
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
                raise AssertionError("native pack_fp8_rows must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def pack_fp8_rows_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"pack_fp8_rows acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch  # noqa: E402

import pack_fp8_rows_pre_migration_helpers as H  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def _check(out, e, s, expected, *, e_before=None, s_before=None):
    """Output spec, every byte against the independent reference, inputs untouched."""
    H.assert_output_spec(out, e)
    H.assert_rows_equal(H.output_rows(out), expected)
    if e_before is not None:
        assert torch.equal(H.e4m3_rows(e), e_before), "e4m3 input modified"
    if s_before is not None:
        assert torch.equal(H.scale_rows(s), s_before), "scale input modified"


# --------------------------------------------------------------------------------------------------------- planner
# One case per planner branch reached through the public inputs (rows = product of the leading dims, 16 rows a block,
# ceil(rows / 16) cores up to the grid).
PLANNER_CASES = [
    pytest.param([1, 1, 1, 128], None, id="m1_single_row"),
    pytest.param([1, 1, 16, 128], None, id="m16_one_full_block"),
    pytest.param([1, 1, 17, 128], None, id="m17_partial_final_block"),
    pytest.param([1, 1, 320, 128], None, id="m320_twenty_cores"),
    pytest.param([100, 128], None, id="rank2_m100"),
    pytest.param([3, 50, 128], None, id="rank3_b3_m50"),
    pytest.param([2, 3, 37, 128], None, id="rank4_b2x3_m37"),
    pytest.param([1, 1, 200, 128], (1, 1), id="grid1x1_tuple_m200_multi_block"),
    pytest.param([1, 1, 1000, 128], ttnn.CoreCoord(2, 3), id="grid2x3_coord_m1000_uneven"),
    pytest.param([1, 1, 333, 128], (100, 100), id="grid_oversize_clamped_m333"),
    pytest.param([1, 1, 4100, 128], None, id="m4100_blocks_exceed_cores"),
]


@pytest.mark.parametrize(("shape", "max_grid"), PLANNER_CASES)
def test_planner_branches(device, pack_fp8_rows_op, shape, max_grid):
    e, s, expected = H.make_inputs(shape, device, seed=sum(shape))
    e_before, s_before = H.e4m3_rows(e), H.scale_rows(s)
    kwargs = {} if max_grid is None else {"max_grid": max_grid}
    out = pack_fp8_rows_op(e, s, **kwargs)
    _check(out, e, s, expected, e_before=e_before, s_before=s_before)
    # a repeated call on the same inputs (the reused program) gives the same bytes into a new output
    out2 = pack_fp8_rows_op(e, s, **kwargs)
    assert out2.buffer_address() != out.buffer_address()
    _check(out2, e, s, expected)


def test_special_rows_and_scales(device, pack_fp8_rows_op):
    """Zero-row detection is on the values only (+0 / -0 mixed), scales are copied bit-exactly (negative, -0.0, huge,
    tiny, fp32 subnormal), and the extreme e4m3 values +-448 and the smallest normal survive."""
    rows = 48
    gen = torch.Generator().manual_seed(7)
    vals = H.random_e4m3_bytes(rows, gen)
    vals[0] = 0x7E  # all +448
    vals[1] = 0xFE  # all -448
    vals[2] = 0x08  # all the smallest normal
    vals[3] = H.E4M3_POS_ZERO  # zero row, scale 3.0: zeroed
    vals[4] = H.E4M3_NEG_ZERO  # -0 row, scale -1.0: zeroed
    vals[5] = torch.where(torch.arange(128) % 2 == 0, H.E4M3_POS_ZERO, H.E4M3_NEG_ZERO).to(torch.uint8)  # mixed zeros
    vals[6] = H.E4M3_POS_ZERO
    vals[6, 64] = 0x80 | 0x38  # one -1.0 in the middle: kept
    vals[16:32] = H.E4M3_POS_ZERO  # a whole zero block
    scales = torch.linspace(0.5, 9.5, rows).reshape(rows, 1)
    scales[3, 0] = 3.0
    scales[4, 0] = -1.0
    scales[7, 0] = -0.0  # zero scale, nonzero values: written as is
    scales[8, 0] = 3.0e38
    scales[9, 0] = 1.0e-4 / 448.0  # the cast's clamp floor
    scales[10, 0] = float(torch.tensor([1], dtype=torch.int32).view(torch.float32))  # fp32 subnormal
    scales[11, 0] = -2.5e-30
    e = H.to_device_e4m3(vals.reshape(1, 1, rows, 128), device)
    s = H.to_device_scale(scales.reshape(1, 1, rows, 1), device)
    expected = H.reference_rows(H.e4m3_rows(e), H.scale_rows(s))
    out = pack_fp8_rows_op(e, s)
    _check(out, e, s, expected)
    got = H.output_rows(out)[0]
    for r in (3, 4, 5, *range(16, 32)):
        assert (got[r] == 0).all(), f"zero row {r} not written all zero"
    assert (got[6] != 0).any() and (got[7, :128] != 0).any()
    assert torch.equal(got[7, 128:132], torch.tensor([0.0 * -1], dtype=torch.float32).view(torch.uint8))
    assert (got[:, H.PAD_OFF :] == 0).all(), "padding bytes not zero"


# --------------------------------------------------------------------------------------------------------- output
def test_preallocated_dirty_output(device, pack_fp8_rows_op):
    """The caller's buffer (dirty with 0xA5) is written in full (values, scales, 60 zero bytes, zero rows) and returned
    as is, at the same address."""
    shape = [1, 1, 77, 128]
    e, s, expected = H.make_inputs(shape, device, seed=11)
    out = H.preallocated_output(shape, device, fill=0xA5)
    addr = out.buffer_address()
    ret = pack_fp8_rows_op(e, s, output_tensor=out)
    assert ret is out, "output_tensor must be returned as is"
    assert ret.buffer_address() == addr
    _check(ret, e, s, expected)


def test_memory_config_dram_explicit(device, pack_fp8_rows_op):
    shape = [1, 1, 40, 128]
    e, s, expected = H.make_inputs(shape, device, seed=12)
    out = pack_fp8_rows_op(e, s, memory_config=DRAM)
    _check(out, e, s, expected)


def test_output_keeps_other_tensors(device, pack_fp8_rows_op):
    """The op writes only its own output: an earlier output and a neighbouring buffer stay as they were."""
    shape = [1, 1, 64, 128]
    e1, s1, exp1 = H.make_inputs(shape, device, seed=21)
    out1 = pack_fp8_rows_op(e1, s1)
    guard = H.preallocated_output(shape, device, fill=0x3C)
    e2, s2, exp2 = H.make_inputs(shape, device, seed=22)
    out2 = pack_fp8_rows_op(e2, s2)
    _check(out2, e2, s2, exp2)
    _check(out1, e1, s1, exp1)
    assert (H.output_rows(guard) == 0x3C).all(), "an unrelated buffer was written"


# --------------------------------------------------------------------------------------------------------- cache
# Program-defining: the shape (row count) and the effective core set (max_grid). Per call: the three buffer addresses
# and the data. Not in the key: whether output_tensor is passed (same spec) and max_grid's spelling (tuple vs CoreCoord).
# One program per selected configuration (a single prepared data-movement program): a miss adds exactly 1 entry.


def _call_counted(op, entries, *args, **kwargs):
    n0 = entries()
    out = op(*args, **kwargs)
    return out, entries() - n0


def test_cache_fresh_buffers_and_output_transitions(device, program_cache, entries, pack_fp8_rows_op):
    """A miss, then hits on fresh live inputs/outputs with new data; allocated -> caller's -> allocated output."""
    shape = [1, 1, 96, 128]
    keep = []
    e, s, exp = H.make_inputs(shape, device, seed=31)
    out, d = _call_counted(pack_fp8_rows_op, entries, e, s)
    assert d == 1, f"first call: expected one new program, got {d}"
    _check(out, e, s, exp)
    keep.append((e, s, out, exp))
    for i, use_out in enumerate([False, True, False, True]):
        e, s, exp = H.make_inputs(shape, device, seed=32 + i)
        kwargs = {}
        if use_out:
            kwargs["output_tensor"] = H.preallocated_output(shape, device, fill=0x5A + i)
        out, d = _call_counted(pack_fp8_rows_op, entries, e, s, **kwargs)
        assert d == 0, f"call {i + 2} (output_tensor={'caller' if use_out else 'allocated'}): expected a hit, got {d}"
        if use_out:
            assert out is kwargs["output_tensor"]
        _check(out, e, s, exp)
        for pe, ps, po, _ in keep:
            assert e.buffer_address() != pe.buffer_address()
            assert s.buffer_address() != ps.buffer_address()
            assert out.buffer_address() != po.buffer_address()
        keep.append((e, s, out, exp))
    for pe, ps, po, pexp in keep:  # every output still holds its own call's rows
        H.assert_rows_equal(H.output_rows(po), pexp, "retained output")


def test_cache_program_fields_and_return(device, program_cache, entries, pack_fp8_rows_op):
    """A -> B (rows) -> A -> C (max_grid 1x1) -> A, and max_grid spelled as a tuple or a CoreCoord."""
    a, b = [1, 1, 320, 128], [1, 1, 96, 128]
    seq = [
        (a, {}, 1, "A first"),
        (b, {}, 1, "B: other row count"),
        (a, {}, 0, "A again"),
        (a, {"max_grid": (1, 1)}, 1, "C: 1x1 grid"),
        (a, {"max_grid": ttnn.CoreCoord(1, 1)}, 0, "C as a CoreCoord"),
        (a, {}, 0, "A once more"),
        (b, {}, 0, "B again"),
    ]
    for i, (shape, kwargs, want, what) in enumerate(seq):
        e, s, exp = H.make_inputs(shape, device, seed=40 + i)
        out, d = _call_counted(pack_fp8_rows_op, entries, e, s, **kwargs)
        assert d == want, f"{what}: expected {want} new programs, got {d}"
        _check(out, e, s, exp)


def test_cache_valid_invalid_valid(device, program_cache, entries, pack_fp8_rows_op, expect_error):
    """A warmed op still refuses a bad output buffer, adds no program for it, and the next valid call hits."""
    shape = [1, 1, 32, 128]
    e, s, exp = H.make_inputs(shape, device, seed=51)
    out, d = _call_counted(pack_fp8_rows_op, entries, e, s)
    assert d == 1
    _check(out, e, s, exp)
    bad = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 128]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, device, DRAM)
    n0 = entries()
    with expect_error(AssertionError, "output_tensor must be uint8 ROW_MAJOR"):
        pack_fp8_rows_op(e, s, output_tensor=bad)
    assert entries() == n0
    e2, s2, exp2 = H.make_inputs(shape, device, seed=52)
    out2, d = _call_counted(pack_fp8_rows_op, entries, e2, s2)
    assert d == 0
    _check(out2, e2, s2, exp2)


# --------------------------------------------------------------------------------------------------------- refusals
def _bf16_rm(shape, device, mem=DRAM):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem,
    )


def _f32(shape, device, layout=ttnn.ROW_MAJOR_LAYOUT, mem=DRAM):
    return ttnn.from_torch(
        torch.ones(shape, dtype=torch.float32), dtype=ttnn.float32, layout=layout, device=device, memory_config=mem
    )


def _e4m3(shape, device, mem=DRAM):
    """FP8_E4M3 ROW_MAJOR of any shape (the width included: the refusal cases need widths other than 128)."""
    gen = torch.Generator().manual_seed(1)
    vals = H.random_e4m3_bytes(H.rows_of(shape), gen)[:, : shape[-1]]
    return H.to_device_e4m3(vals.reshape(shape), device, mem)


def _u8(shape, device, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, mem=DRAM):
    return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, device, mem)


M = 32
E = [1, 1, M, 128]
S = [1, 1, M, 1]
O = [1, 1, M, 192]

# (id, build(device) -> (args, kwargs), message regex)
REFUSALS = [
    ("e4m3_bf16", lambda d: ((_bf16_rm(E, d), _f32(S, d)), {}), "e4m3 must be FP8_E4M3 ROW_MAJOR"),
    ("e4m3_is_float32_scale", lambda d: ((_f32(S, d), _f32(S, d)), {}), "e4m3 must be FP8_E4M3 ROW_MAJOR"),
    ("scale_bf16", lambda d: ((_e4m3(E, d), _bf16_rm([1, 1, M, 2], d)), {}), "scale must be FLOAT32 ROW_MAJOR"),
    (
        "scale_tile",
        lambda d: ((_e4m3(E, d), _f32(S, d, layout=ttnn.TILE_LAYOUT)), {}),
        "scale must be FLOAT32 ROW_MAJOR",
    ),
    ("e4m3_width_64", lambda d: ((_e4m3([1, 1, M, 64], d), _f32(S, d)), {}), "one scale a row"),
    ("scale_width_2", lambda d: ((_e4m3(E, d), _f32([1, 1, M, 2], d)), {}), "one scale a row"),
    ("rows_differ", lambda d: ((_e4m3([1, 1, 48, 128], d), _f32([1, 1, 80, 1], d)), {}), r"48[\s\S]*80"),
    ("rank_differs", lambda d: ((_e4m3([48, 128], d), _f32([1, 48, 1], d)), {}), "48"),
    ("e4m3_l1", lambda d: ((_e4m3(E, d, mem=L1), _f32(S, d)), {}), "e4m3 must be DRAM interleaved"),
    ("scale_l1", lambda d: ((_e4m3(E, d), _f32(S, d, mem=L1)), {}), "scale must be DRAM interleaved"),
    (
        "out_uint32",
        lambda d: ((_e4m3(E, d), _f32(S, d)), {"output_tensor": _u8(O, d, dtype=ttnn.uint32)}),
        "output_tensor must be uint8 ROW_MAJOR",
    ),
    (
        "out_shape",
        lambda d: ((_e4m3(E, d), _f32(S, d)), {"output_tensor": _u8([1, 1, M + 1, 192], d)}),
        "output_tensor must be uint8 ROW_MAJOR",
    ),
    (
        "out_tile",
        lambda d: ((_e4m3(E, d), _f32(S, d)), {"output_tensor": _u8(O, d, layout=ttnn.TILE_LAYOUT)}),
        "output_tensor must be uint8 ROW_MAJOR",
    ),
    (
        "out_l1",
        lambda d: ((_e4m3(E, d), _f32(S, d)), {"output_tensor": _u8(O, d, mem=L1)}),
        "output must be DRAM interleaved",
    ),
    (
        "memory_config_l1",
        lambda d: ((_e4m3(E, d), _f32(S, d)), {"memory_config": L1}),
        "output must be DRAM interleaved",
    ),
]


@pytest.mark.parametrize(("build", "message"), [pytest.param(b, m, id=i) for i, b, m in REFUSALS])
def test_refusals(device, pack_fp8_rows_op, expect_error, build, message):
    args, kwargs = build(device)
    out = kwargs.get("output_tensor")
    before = (
        H.output_rows(out)
        if out is not None and out.dtype == ttnn.uint8 and out.layout == ttnn.ROW_MAJOR_LAYOUT
        else None
    )
    with expect_error(AssertionError, message):
        pack_fp8_rows_op(*args, **kwargs)
    if before is not None and list(out.shape) == O and out.memory_config().buffer_type == ttnn.BufferType.DRAM:
        assert torch.equal(H.output_rows(out), before), "a refused call wrote the output buffer"


def test_options_are_keyword_only(device, pack_fp8_rows_op, expect_error):
    e, s = _e4m3(E, device), _f32(S, device)
    out = _u8(O, device)
    with expect_error(TypeError, "argument"):
        pack_fp8_rows_op(e, s, out)

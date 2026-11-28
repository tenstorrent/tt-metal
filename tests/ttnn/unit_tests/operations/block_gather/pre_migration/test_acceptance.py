# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.bringup.block_select_ttnn:block_gather``.

Every test calls the operation through the ``block_gather_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (``block_gather(scores, ids, *, block, output_tensor=None, memory_config=None, max_grid=None)``):
    scores [.., R, W] bf16 ROW_MAJOR DRAM interleaved, ids [.., R, K] uint32 ROW_MAJOR DRAM interleaved, equal
    leading dims -> [.., R, K block] bf16 ROW_MAJOR DRAM interleaved: out[r, j block + o] = scores[r, ids[r, j] block
    + o]; an id >= W / block (0xFFFFFFFF included) gives block x -inf. Exact (bit-compared against torch).
    ``block`` a power of two in [1, 32]; W a multiple of 32. ``output_tensor`` (bf16 ROW_MAJOR [.., R, K block], DRAM
    interleaved) is written and returned as the same object. ``memory_config`` must leave the output DRAM interleaved.
    ``max_grid`` (tuple or CoreCoord) caps the core rectangle, clamped to the device grid. Inputs are not modified.
Refusals (source: ``AssertionError``; a native port raises ``RuntimeError`` from TT_FATAL, so either type is
accepted, the message key phrase is asserted): block not a power of two <= 32; scores / ids / output not DRAM
interleaved ROW_MAJOR of their dtype; W not a multiple of 32; leading dims differing; output_tensor of a wrong
dtype / layout / shape.
Program reuse: one program per (shapes, memory configs, block, max_grid); data, buffer addresses and the presence of
a caller's output_tensor change on a hit (no value enters the key).
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

SOURCE_ENTRY = "ttnn.bringup.block_select_ttnn:block_gather"
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
                raise AssertionError("native block_gather must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def block_gather_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"block_gather acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch  # noqa: E402

from block_gather_pm_ref import (  # noqa: E402
    SENT,
    bits_equal,
    make_ids,
    make_scores,
    mismatch,
    ref_block_gather,
    to_dev,
    to_host,
)

pytestmark = pytest.mark.use_module_device

REFUSED = (AssertionError, RuntimeError)  # source asserts; a native TT_FATAL raises RuntimeError


def _entries(device):
    return device.num_program_cache_entries()


def _inputs(device, lead, W, K, B, seed):
    g = torch.Generator().manual_seed(seed)
    s = make_scores((*lead, W), g)
    ids = make_ids(lead, K, W // B, g)
    return s, ids, to_dev(s, device, ttnn.bfloat16), to_dev(ids, device, ttnn.uint32)


def _check_out(out, s, ids, B, *, device):
    """Output metadata and values against the torch reference; inputs untouched."""
    want = ref_block_gather(s, ids, B)
    assert list(out.shape) == list(want.shape), (list(out.shape), list(want.shape))
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc
    assert out.storage_type() == ttnn.StorageType.DEVICE
    (got,) = to_host(out)
    assert bits_equal(got, want), mismatch(got, want)
    return want


def _check_inputs_untouched(st, it, s, ids):
    (gs,) = to_host(st)
    assert bits_equal(gs, s), "scores modified"
    (gi,) = to_host(it)
    assert torch.equal(gi.to(torch.int64) & 0xFFFFFFFF, ids & 0xFFFFFFFF), "ids modified"


# --------------------------------------------------------------------------- planner branches (one case each)
# (lead dims, W, K, block, max_grid). K selects how a row's blocks split into work units (K / 4 or K / 2 blocks when
# that is a multiple of 16, else whole rows); block 32 is the 64-byte-read-straight-to-output path; rows vs cores
# covers idle cores, an even and an uneven split; max_grid caps (and is clamped to) the grid.
BRANCHES = [
    pytest.param((1, 1, 64), 4096, 64, 8, None, id="k64_quarters_b8"),
    pytest.param((1, 1, 40), 1024, 32, 4, None, id="k32_halves_b4"),
    pytest.param((1, 1, 33), 512, 16, 2, None, id="k16_whole_rows_b2"),
    pytest.param((1, 1, 17), 256, 24, 4, None, id="k24_whole_rows_b4"),
    pytest.param((1, 1, 50), 8192, 128, 32, None, id="b32_direct_k128"),
    pytest.param((1, 1, 9), 1024, 8, 32, None, id="b32_k8"),
    pytest.param((1, 1, 20), 2080, 64, 1, None, id="b1_w2080_not_x64"),
    pytest.param((1, 1, 8), 1024, 64, 16, None, id="b16"),
    pytest.param((1, 1, 1), 512, 64, 8, None, id="one_row_idle_cores"),
    pytest.param((1, 1, 1000), 1024, 64, 8, None, id="rows_exceed_cores_uneven"),
    pytest.param((48,), 512, 32, 8, None, id="rank2"),
    pytest.param((3, 10), 1024, 64, 4, None, id="rank3"),
    pytest.param((2, 3, 16), 2048, 64, 16, None, id="rank4_batched"),
    pytest.param((1, 1, 6), 32, 4, 32, None, id="w32_single_block"),
    pytest.param((1, 1, 100), 4096, 64, 8, (2, 3), id="max_grid_tuple_2x3"),
    pytest.param((1, 1, 37), 1024, 64, 32, "corecoord_1x1", id="max_grid_corecoord_1x1"),
    pytest.param((1, 1, 70), 1024, 32, 8, (1000, 1000), id="max_grid_clamped"),
]


def _grid_arg(mg):
    return ttnn.CoreCoord(1, 1) if mg == "corecoord_1x1" else mg


@pytest.mark.parametrize(("lead", "W", "K", "B", "max_grid"), BRANCHES)
def test_branch(device, block_gather_op, lead, W, K, B, max_grid):
    s, ids, st, it = _inputs(device, lead, W, K, B, seed=W + K + B)
    kwargs = {"block": B}
    if max_grid is not None:
        kwargs["max_grid"] = _grid_arg(max_grid)
    out = block_gather_op(st, it, **kwargs)
    _check_out(out, s, ids, B, device=device)
    _check_inputs_untouched(st, it, s, ids)
    # repeated call (prepared hit) on the same inputs: a new output, same values
    out2 = block_gather_op(st, it, **kwargs)
    assert out2.buffer_address() != out.buffer_address()
    _check_out(out2, s, ids, B, device=device)


# --------------------------------------------------------------------------- API / output arguments


def test_preallocated_output(device, block_gather_op):
    """output_tensor: written in full (a stale fill is gone), returned as the same object and buffer."""
    s, ids, st, it = _inputs(device, (1, 1, 30), 2048, 64, 8, seed=11)
    stale = torch.full((1, 1, 30, 64 * 8), 7.0, dtype=torch.bfloat16)
    ot = to_dev(stale, device, ttnn.bfloat16)
    addr = ot.buffer_address()
    out = block_gather_op(st, it, block=8, output_tensor=ot)
    assert out is ot
    assert out.buffer_address() == addr
    _check_out(out, s, ids, 8, device=device)
    _check_inputs_untouched(st, it, s, ids)


def test_explicit_dram_memory_config(device, block_gather_op):
    s, ids, st, it = _inputs(device, (1, 1, 24), 1024, 32, 16, seed=12)
    out = block_gather_op(st, it, block=16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _check_out(out, s, ids, 16, device=device)


def test_ids_patterns(device, block_gather_op):
    """Every id edge on one row each: none, nb - 1, nb, 2**31, 0xFFFFFFFE, 0, repeated, descending."""
    W, K, B, R = 1024, 64, 8, 8
    nb = W // B
    g = torch.Generator().manual_seed(21)
    s = make_scores((1, 1, R, W), g)
    ids = torch.zeros((1, 1, R, K), dtype=torch.int64)
    ids[..., 0, :] = SENT
    ids[..., 1, :] = nb - 1
    ids[..., 2, :] = nb
    ids[..., 3, :] = 2**31
    ids[..., 4, :] = 0xFFFFFFFE
    ids[..., 5, :] = 0
    ids[..., 6, :] = torch.arange(K) % 3
    ids[..., 7, :] = nb - 1 - torch.arange(K)
    st, it = to_dev(s, device, ttnn.bfloat16), to_dev(ids, device, ttnn.uint32)
    out = block_gather_op(st, it, block=B)
    want = _check_out(out, s, ids, B, device=device)
    assert torch.isneginf(want[..., 0, :].float()).all() and torch.isneginf(want[..., 4, :].float()).all()


# --------------------------------------------------------------------------- refusals


def _refusal_inputs(device, *, W=256, K=16, R=8, lead=(1, 1), sdtype=ttnn.bfloat16, idtype=ttnn.uint32, **kw):
    g = torch.Generator().manual_seed(3)
    s = torch.randn(*lead, R, W, generator=g).to(torch.bfloat16)
    ids = torch.randint(0, 8, (*lead, R, K), generator=g)
    st = to_dev(s, device, sdtype, layout=kw.get("slayout", ttnn.ROW_MAJOR_LAYOUT), memory_config=kw.get("smem"))
    it = to_dev(ids, device, idtype, layout=kw.get("ilayout", ttnn.ROW_MAJOR_LAYOUT), memory_config=kw.get("imem"))
    return st, it


@pytest.mark.parametrize("block", [0, 3, 6, 64, -4], ids=["b0", "b3", "b6", "b64", "bneg4"])
def test_refuses_block(device, block_gather_op, expect_error, block):
    st, it = _refusal_inputs(device)
    with expect_error(REFUSED, "power of two"):
        block_gather_op(st, it, block=block)


REFUSALS = [
    pytest.param({"sdtype": ttnn.float32}, {}, "scores", id="scores_fp32"),
    pytest.param({"slayout": ttnn.TILE_LAYOUT, "R": 32}, {}, "scores", id="scores_tile"),
    pytest.param({"smem": ttnn.L1_MEMORY_CONFIG}, {}, "scores", id="scores_l1"),
    pytest.param({"idtype": ttnn.bfloat16}, {}, "ids", id="ids_bf16"),
    pytest.param({"idtype": ttnn.int32}, {}, "ids", id="ids_int32"),
    pytest.param({"imem": ttnn.L1_MEMORY_CONFIG}, {}, "ids", id="ids_l1"),
    pytest.param({"W": 48}, {}, "multiple of 32", id="width_not_x32"),
    pytest.param({}, {"memory_config": ttnn.L1_MEMORY_CONFIG}, "output", id="memory_config_l1"),
]


@pytest.mark.parametrize(("inp", "kwargs", "message"), REFUSALS)
def test_refuses_input(device, block_gather_op, expect_error, inp, kwargs, message):
    st, it = _refusal_inputs(device, **inp)
    with expect_error(REFUSED, message):
        block_gather_op(st, it, block=16, **kwargs)


def test_refuses_leading_dims_mismatch(device, block_gather_op, expect_error):
    g = torch.Generator().manual_seed(4)
    st = to_dev(torch.randn(1, 1, 64, 256, generator=g).to(torch.bfloat16), device, ttnn.bfloat16)
    it = to_dev(torch.randint(0, 8, (1, 1, 32, 16), generator=g), device, ttnn.uint32)
    with expect_error(REFUSED, "(?i)shape"):
        block_gather_op(st, it, block=32)


OUT_REFUSALS = [
    pytest.param((1, 1, 8, 16 * 16 - 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, None, "output", id="out_width"),
    pytest.param((1, 1, 16, 16 * 16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, None, "output", id="out_rows"),
    pytest.param((1, 1, 8, 16 * 16), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, None, "output", id="out_fp32"),
    pytest.param((1, 1, 32, 16 * 16), ttnn.bfloat16, ttnn.TILE_LAYOUT, None, "output", id="out_tile"),
    pytest.param(
        (1, 1, 8, 16 * 16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG, "output", id="out_l1"
    ),
]


@pytest.mark.parametrize(("shape", "dtype", "layout", "mem", "message"), OUT_REFUSALS)
def test_refuses_output_tensor(device, block_gather_op, expect_error, shape, dtype, layout, mem, message):
    R = shape[2] if layout == ttnn.TILE_LAYOUT else 8
    st, it = _refusal_inputs(device, R=R)
    ot = to_dev(torch.zeros(shape, dtype=torch.bfloat16), device, dtype, layout=layout, memory_config=mem)
    with expect_error(REFUSED, message):
        block_gather_op(st, it, block=16, output_tensor=ot)


# --------------------------------------------------------------------------- program cache sequences
# The program depends on (shapes, memory configs, block, max_grid) only: buffer addresses, scores / ids values and
# whether the output is the caller's are patched on a hit. block is a compile-time quantity of the kernel (block
# bytes a read), so a different block is a different program; max_grid changes the cores the program runs on.
# Counts are read around the operation call only (inputs are uploaded before, outputs read back after).


def _call_counted(device, op, *args, **kwargs):
    before = _entries(device)
    out = op(*args, **kwargs)
    return out, _entries(device) - before


def test_cache_hit_fresh_buffers_and_values(device, block_gather_op):
    """A (miss) -> A on new buffers with new data (hit) -> again (hit); every output follows the current inputs."""
    device.clear_program_cache()
    lead, W, K, B = (1, 1, 64), 4096, 64, 8
    live = []
    for i in range(3):
        s, ids, st, it = _inputs(device, lead, W, K, B, seed=100 + i)
        out, delta = _call_counted(device, block_gather_op, st, it, block=B)
        assert delta == (1 if i == 0 else 0), (i, delta)
        _check_out(out, s, ids, B, device=device)
        live.append((st, it, out))
    for k in range(3):  # scores, ids, output: every call ran on its own live buffers
        addrs = [live[j][k].buffer_address() for j in range(3)]
        assert len(set(addrs)) == 3, addrs


def test_cache_block_change_and_return(device, block_gather_op):
    """block 8 (miss) -> block 16 (miss) -> block 8 (hit) -> block 16 (hit), same scores / ids shapes."""
    device.clear_program_cache()
    lead, W, K = (1, 1, 32), 2048, 64
    expected = [1, 1, 0, 0]
    for i, B in enumerate((8, 16, 8, 16)):
        s, ids, st, it = _inputs(device, lead, W, K, B, seed=200 + i)
        out, delta = _call_counted(device, block_gather_op, st, it, block=B)
        assert delta == expected[i], (i, B, delta)
        _check_out(out, s, ids, B, device=device)


def test_cache_max_grid_change_and_return(device, block_gather_op):
    """default grid (miss) -> max_grid 2x3 (miss) -> default (hit) -> 2x3 (hit)."""
    device.clear_program_cache()
    lead, W, K, B = (1, 1, 48), 1024, 64, 8
    expected = [1, 1, 0, 0]
    for i, mg in enumerate((None, (2, 3), None, (2, 3))):
        s, ids, st, it = _inputs(device, lead, W, K, B, seed=300 + i)
        kwargs = {"block": B} if mg is None else {"block": B, "max_grid": mg}
        out, delta = _call_counted(device, block_gather_op, st, it, **kwargs)
        assert delta == expected[i], (i, mg, delta)
        _check_out(out, s, ids, B, device=device)


def test_cache_shape_change_and_return(device, block_gather_op):
    """A (miss) -> more rows (correct) -> wider scores (correct) -> A on fresh buffers (hit)."""
    device.clear_program_cache()
    B, K = 8, 64
    shapes = [((1, 1, 40), 1024), ((1, 1, 77), 1024), ((1, 1, 40), 2048), ((1, 1, 40), 1024)]
    for i, (lead, W) in enumerate(shapes):
        s, ids, st, it = _inputs(device, lead, W, K, B, seed=400 + i)
        out, delta = _call_counted(device, block_gather_op, st, it, block=B)
        if i == 0:
            assert delta == 1, delta
        if i == 3:
            assert delta == 0, delta
        _check_out(out, s, ids, B, device=device)


@pytest.mark.parametrize("first", ["absent", "present"])
def test_cache_output_tensor_transitions(device, block_gather_op, first):
    """absent -> present -> absent (and present -> absent -> present): one program, the output always current."""
    device.clear_program_cache()
    lead, W, K, B = (1, 1, 24), 2048, 32, 16
    order = ["absent", "present", "absent"] if first == "absent" else ["present", "absent", "present"]
    keep = []
    for i, mode in enumerate(order):
        s, ids, st, it = _inputs(device, lead, W, K, B, seed=500 + i)
        kwargs = {"block": B}
        ot = None
        if mode == "present":
            ot = to_dev(torch.full((*lead, K * B), 3.0, dtype=torch.bfloat16), device, ttnn.bfloat16)
            kwargs["output_tensor"] = ot
        out, delta = _call_counted(device, block_gather_op, st, it, **kwargs)
        assert delta == (1 if i == 0 else 0), (i, mode, delta)
        if ot is not None:
            assert out is ot
        _check_out(out, s, ids, B, device=device)
        keep.append(out)
    addrs = [t.buffer_address() for t in keep]
    assert len(set(addrs)) == len(addrs), addrs


def test_cache_output_tensor_reused_across_calls(device, block_gather_op):
    """The same caller buffer for two different inputs (a hit): the second call overwrites the first's result."""
    device.clear_program_cache()
    lead, W, K, B = (1, 1, 16), 1024, 64, 4
    ot = to_dev(torch.zeros((*lead, K * B), dtype=torch.bfloat16), device, ttnn.bfloat16)
    for i in range(2):
        s, ids, st, it = _inputs(device, lead, W, K, B, seed=600 + i)
        out, delta = _call_counted(device, block_gather_op, st, it, block=B, output_tensor=ot)
        assert delta == (1 if i == 0 else 0), (i, delta)
        assert out is ot
        _check_out(out, s, ids, B, device=device)


def test_cache_valid_invalid_valid(device, block_gather_op, expect_error):
    """A warmed program still refuses a bad block and a bad output_tensor; the next valid call hits and is right."""
    device.clear_program_cache()
    lead, W, K, B = (1, 1, 32), 1024, 64, 8
    s, ids, st, it = _inputs(device, lead, W, K, B, seed=700)
    out, delta = _call_counted(device, block_gather_op, st, it, block=B)
    assert delta == 1
    _check_out(out, s, ids, B, device=device)
    with expect_error(REFUSED, "power of two"):
        block_gather_op(st, it, block=12)
    bad = to_dev(torch.zeros((*lead, K * B - 32), dtype=torch.bfloat16), device, ttnn.bfloat16)
    with expect_error(REFUSED, "output"):
        block_gather_op(st, it, block=B, output_tensor=bad)
    s2, ids2, st2, it2 = _inputs(device, lead, W, K, B, seed=701)
    out2, delta = _call_counted(device, block_gather_op, st2, it2, block=B)
    assert delta == 0, delta
    _check_out(out2, s2, ids2, B, device=device)


# --------------------------------------------------------------------------- integration


def test_consumer_reads_output(device, block_gather_op):
    """The output feeds a stock op (ttnn.max over the gathered row) and matches torch."""
    g = torch.Generator().manual_seed(800)
    s = torch.randn(1, 1, 32, 1024, generator=g).to(torch.bfloat16)
    ids = torch.randint(0, 1024 // 8, (1, 1, 32, 64), generator=g)  # all in range: finite maxima
    st, it = to_dev(s, device, ttnn.bfloat16), to_dev(ids, device, ttnn.uint32)
    out = block_gather_op(st, it, block=8)
    want = _check_out(out, s, ids, 8, device=device)
    tiled = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    m = ttnn.to_torch(ttnn.max(tiled, dim=-1, keepdim=True))
    assert torch.equal(m.float().reshape(-1), want.float().amax(-1).reshape(-1))

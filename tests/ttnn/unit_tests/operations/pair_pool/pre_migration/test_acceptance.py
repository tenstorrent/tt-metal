# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.bringup.pair_pool_ttnn:pair_pool``.

Every test calls the operation through the ``pair_pool_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (README, the callable's docstring and its refusals; no planner/descriptor code is read):
    pair_pool(x, *, output_tensor=None, memory_config=None, max_grid=None) -> ttnn.Tensor
    x [.., R, 2 hd] float32 TILE DRAM interleaved = [kv | gate] per row, rank 2..4 (leading dims flattened with
    the rows), R % 64 == 0, hd % 32 == 0  ->  out [.., R / 2, hd] float32 TILE interleaved:
        out[t] = kv[2t] w0 + kv[2t+1] w1,  (w0, w1) = softmax(gate[2t], gate[2t+1])  per column.
    output_tensor: the caller's buffer, exactly the output spec (float32, TILE, shape, interleaved; DRAM or L1),
        written and returned (same buffer); memory_config then plays no part.
    memory_config: the allocated output's memory config (default DRAM interleaved; L1 interleaved accepted).
    max_grid: ttnn.CoreCoord or (x, y), default the whole compute grid; work units (output tile-row, column tile)
        split over the cores of (0, 0) .. max_grid - 1 (one unit, even, uneven, many units a core, idle cores).
    Refusals today (AssertionError, raised before any device work): dtype != float32, layout != TILE, input not
        DRAM interleaved, last dim % 64 != 0 ("hd % 32 == 0"), rows % 64 != 0 ("% 64 != 0"), an output_tensor of
        another dtype, layout or shape, or not interleaved.
    The input is never written. Program cache: one program per (shape, memory configs, grid); buffer addresses and
        data change on a hit.
Reference: float64 torch (conftest ``reference``), the golden's 1e-6 relative-norm tolerance plus a per-element bound.
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

SOURCE_ENTRY = "ttnn.bringup.pair_pool_ttnn:pair_pool"
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
        if name.startswith("ttnn.bringup.pair_pool_ttnn") or name.startswith("ttnn._ttnn.operations.generic"):
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
                raise AssertionError("native pair_pool must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def pair_pool_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"pair_pool acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch  # noqa: E402

# One device for the module: the cache tests use shapes no other test in this module runs, so their first call is a
# known miss whatever ran before.
pytestmark = pytest.mark.use_module_device


def _entries(device):
    return device.num_program_cache_entries()


def _grid(device):
    g = device.compute_with_storage_grid_size()
    return g.x, g.y


def _addr(t):
    return t.buffer_address()


# --------------------------------------------------------------------------- planner branches / API forms
# units = (flattened rows / 64) * (hd / 32); max_grid bounds the cores.
BRANCH_CASES = [
    # one unit on the whole grid: every other core idle
    pytest.param((64, 64), {}, id="min_one_unit_rank2"),
    # 8 units over 2x2 cores: an even split, 2 a core
    pytest.param((1, 1, 128, 256), {"max_grid": (2, 2)}, id="even_8u_2x2"),
    # 9 units over 2x2 cores: uneven, a partial last round; 3 column tiles (hd 96)
    pytest.param((1, 1, 192, 192), {"max_grid": ttnn.CoreCoord(2, 2)}, id="uneven_9u_2x2_corecoord"),
    # every unit on one core
    pytest.param((1, 1, 128, 128), {"max_grid": (1, 1)}, id="single_core_4u"),
    # a one-row and a one-column core rectangle
    pytest.param((1, 1, 256, 128), {"max_grid": (3, 1)}, id="row_grid_3x1"),
    pytest.param((1, 1, 256, 128), {"max_grid": (1, 3)}, id="col_grid_1x3"),
    # leading dims flattened with the rows: rank 3 and rank 4 with batch > 1
    pytest.param((3, 64, 64), {}, id="rank3_b3"),
    pytest.param((2, 3, 128, 192), {"max_grid": (2, 3)}, id="rank4_b2x3"),
    # the model's shape (the compressor's 640 x 1024) on the whole grid: more units than cores
    pytest.param((1, 1, 640, 1024), {}, id="model_640x1024"),
]


@pytest.mark.parametrize(("shape", "kwargs"), BRANCH_CASES)
def test_branches(device, pair_pool_op, pp, shape, kwargs):
    x = pp.make_input(shape, seed=sum(shape))
    xd = pp.to_dev(x, device)
    out = pair_pool_op(xd, **kwargs)
    pp.assert_output_spec(out, shape)
    pp.check(ttnn.to_torch(out), x, label=f"{shape} {kwargs}")
    # the input is never written
    assert torch.equal(ttnn.to_torch(xd), x), "pair_pool modified its input"


@pytest.mark.parametrize(
    ("shape", "kwargs"),
    [
        pytest.param((1, 1, 192, 192), {"max_grid": (2, 2)}, id="uneven_2x2"),
        pytest.param((1, 1, 640, 1024), {}, id="model_full_grid"),
    ],
)
def test_hard_selection_pattern(device, pair_pool_op, pp, shape, kwargs):
    """Gates that pick one row of a pair per column (|gap| 60): the output is (to fp32) a single input element, the
    choice varying per column, column tile and pair, so any index, parity or tile mix-up shows."""
    x = pp.make_selector_input(shape, seed=7)
    out = pair_pool_op(pp.to_dev(x, device), **kwargs)
    got = ttnn.to_torch(out)
    pp.check(got, x, label="selector")
    want = pp.reference(x).float()
    assert torch.allclose(got, want, rtol=1e-6, atol=0), (got - want).abs().max()


def test_extreme_gates(device, pair_pool_op, pp):
    """Gates of large magnitude and both signs (softmax must subtract the max: exp(+-80) unguarded overflows /
    underflows), equal gates, and zero kv."""
    shape = (1, 1, 128, 128)
    x = pp.make_input(shape, seed=11, spread=1.0)
    hd = 64
    x[..., 0:2, hd : hd + 16] = 80.0  # equal, large: 1/2 each
    x[..., 2:4, hd : hd + 16] = -80.0  # equal, very negative: 1/2 each
    x[..., 4, hd : hd + 16] = 80.0
    x[..., 5, hd : hd + 16] = -80.0  # all weight on the even row
    x[..., 6, hd : hd + 16] = -85.0
    x[..., 7, hd : hd + 16] = 85.0  # all weight on the odd row
    x[..., 8:10, :hd] = 0.0  # zero kv: zero out
    out = pair_pool_op(pp.to_dev(x, device))
    got = ttnn.to_torch(out)
    pp.check(got, x, label="extreme gates")
    assert torch.equal(got[..., 4, :], torch.zeros_like(got[..., 4, :])), "zero kv must give exact zeros"


def test_defaults_and_keywords(device, pair_pool_op, pp):
    """The defaults equal their explicit values: memory_config DRAM, max_grid the whole grid (as tuple and CoreCoord);
    bit-identical outputs (one program's arithmetic) and deterministic repeats."""
    shape = (1, 1, 128, 192)
    x = pp.make_input(shape, seed=5)
    xd = pp.to_dev(x, device)
    gx, gy = _grid(device)
    outs = [
        pair_pool_op(xd),
        pair_pool_op(xd),
        pair_pool_op(xd, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        pair_pool_op(xd, max_grid=(gx, gy)),
        pair_pool_op(xd, max_grid=ttnn.CoreCoord(gx, gy)),
        pair_pool_op(xd, output_tensor=None, memory_config=None, max_grid=None),
    ]
    first = ttnn.to_torch(outs[0])
    pp.check(first, x, label="defaults")
    for i, o in enumerate(outs[1:], 1):
        pp.assert_output_spec(o, shape)
        assert torch.equal(ttnn.to_torch(o), first), f"call {i} differs from the default call"
    # every allocated output is its own buffer
    assert len({_addr(o) for o in outs}) == len(outs)


# --------------------------------------------------------------------------- outputs and memory effects


def test_output_tensor_dram(device, pair_pool_op, pp):
    """A caller's DRAM buffer is written and returned (the same buffer); memory_config plays no part then."""
    shape = (1, 1, 192, 128)
    x = pp.make_input(shape, seed=21)
    xd = pp.to_dev(x, device)
    buf = pp.alloc_out(shape, device)
    r = pair_pool_op(xd, output_tensor=buf, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert _addr(r) == _addr(buf), "the result must be the caller's buffer"
    pp.assert_output_spec(r, shape, ttnn.BufferType.DRAM)
    got = ttnn.to_torch(buf)
    pp.check(got, x, label="output_tensor dram")
    assert torch.equal(ttnn.to_torch(xd), x)
    # an op-allocated output of the same input holds the same bytes
    assert torch.equal(ttnn.to_torch(pair_pool_op(xd)), got)


def test_output_l1(device, pair_pool_op, pp):
    """L1 interleaved output, by memory_config and by a caller's L1 buffer."""
    shape = (1, 1, 128, 128)
    x = pp.make_input(shape, seed=22)
    xd = pp.to_dev(x, device)
    a = pair_pool_op(xd, memory_config=ttnn.L1_MEMORY_CONFIG)
    pp.assert_output_spec(a, shape, ttnn.BufferType.L1)
    pp.check(ttnn.to_torch(a), x, label="memory_config L1")
    buf = pp.alloc_out(shape, device, ttnn.L1_MEMORY_CONFIG)
    r = pair_pool_op(xd, output_tensor=buf, max_grid=(2, 1))
    assert _addr(r) == _addr(buf)
    pp.assert_output_spec(r, shape, ttnn.BufferType.L1)
    assert torch.equal(ttnn.to_torch(buf), ttnn.to_torch(a))


def test_output_buffer_reused_overwritten(device, pair_pool_op, pp):
    """One caller buffer across calls with different inputs: every call overwrites all of it."""
    shape = (1, 1, 128, 256)
    buf = pp.alloc_out(shape, device)
    for seed in (31, 32):
        x = pp.make_input(shape, seed=seed)
        r = pair_pool_op(pp.to_dev(x, device), output_tensor=buf, max_grid=(2, 2))
        assert _addr(r) == _addr(buf)
        pp.check(ttnn.to_torch(buf), x, label=f"reused buffer seed {seed}")


# --------------------------------------------------------------------------- program cache sequences
# Device program-cache entries, counted around the op call only (inputs and buffers are made before, readback after).
# A program is defined by the shape (units, tile counts), the memory configs and the grid; the input and output
# addresses and the data are per-call (patched on a hit). Each test uses shapes no other test of this module runs.


def test_cache_fresh_buffers(device, pair_pool_op, pp):
    """A -> new A -> new A: one miss, then hits on fresh live input and output buffers with new data; every output
    uses its own call's input and buffer, and earlier outputs are left untouched."""
    shape = (1, 1, 320, 320)  # 25 units
    kept = []
    for i, seed in enumerate((41, 42, 43)):
        x = pp.make_input(shape, seed=seed)
        xd = pp.to_dev(x, device)
        before = _entries(device)
        out = pair_pool_op(xd, max_grid=(3, 2))
        delta = _entries(device) - before
        assert delta == (1 if i == 0 else 0), f"call {i}: {delta} new program(s)"
        kept.append((x, xd, out))
    assert len({_addr(xd) for _, xd, _ in kept}) == 3, "inputs must be live at distinct addresses"
    assert len({_addr(o) for _, _, o in kept}) == 3, "outputs must be live at distinct addresses"
    for x, _, out in kept:
        pp.check(ttnn.to_torch(out), x, label="fresh buffers")
    # a hit into a caller's buffer (a new output address), then back to an allocated output
    x = pp.make_input(shape, seed=44)
    xd = pp.to_dev(x, device)
    buf = pp.alloc_out(shape, device)
    before = _entries(device)
    r = pair_pool_op(xd, output_tensor=buf, max_grid=(3, 2))
    r2 = pair_pool_op(xd, max_grid=(3, 2))
    assert _entries(device) - before == 0, "an output_tensor of the same spec is the same program"
    assert _addr(r) == _addr(buf)
    pp.check(ttnn.to_torch(buf), x, label="hit into output_tensor")
    assert torch.equal(ttnn.to_torch(r2), ttnn.to_torch(buf))


def test_cache_program_fields(device, pair_pool_op, pp):
    """A -> B -> A over each program-defining field (shape: rows and hd; grid; output memory), every return a hit
    with fresh data, every change a miss; the tuple and CoreCoord forms of one grid are one program."""
    a_shape, a_grid = (1, 1, 384, 192), (2, 2)  # 6 x 3 units

    def run(shape, seed, expect_delta, label, **kwargs):
        x = pp.make_input(shape, seed=seed)
        xd = pp.to_dev(x, device)
        before = _entries(device)
        out = pair_pool_op(xd, **kwargs)
        delta = _entries(device) - before
        assert delta == expect_delta, f"{label}: {delta} new program(s), expected {expect_delta}"
        pp.check(ttnn.to_torch(out), x, label=label)
        return out

    run(a_shape, 51, 1, "A first", max_grid=a_grid)
    run((1, 1, 384, 256), 52, 1, "B: hd 128 -> other column tiles", max_grid=a_grid)
    run(a_shape, 53, 0, "A again", max_grid=a_grid)
    run((1, 1, 448, 192), 54, 1, "C: more rows", max_grid=a_grid)
    run(a_shape, 55, 0, "A again", max_grid=ttnn.CoreCoord(*a_grid))
    run(a_shape, 56, 1, "D: grid 3x2", max_grid=(3, 2))
    run(a_shape, 57, 0, "A again", max_grid=a_grid)
    out = run(a_shape, 58, 1, "E: L1 output", max_grid=a_grid, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    run(a_shape, 59, 0, "E again (L1)", max_grid=a_grid, memory_config=ttnn.L1_MEMORY_CONFIG)
    run(a_shape, 60, 0, "A again (DRAM)", max_grid=a_grid)


def test_cache_valid_invalid_valid(device, pair_pool_op, pp, expect_error):
    """A warmed program does not let an invalid call through; the refusal builds no program; the next valid call hits."""
    shape = (1, 1, 512, 192)
    x = pp.make_input(shape, seed=61)
    before = _entries(device)
    pp.check(ttnn.to_torch(pair_pool_op(pp.to_dev(x, device))), x, label="warm")
    assert _entries(device) - before == 1
    before = _entries(device)
    with expect_error(AssertionError, "BFLOAT16"):
        pair_pool_op(pp.to_dev(x, device, dtype=ttnn.bfloat16))
    good = pp.to_dev(x, device)
    with expect_error(AssertionError, "(?i)shape"):
        pair_pool_op(good, output_tensor=pp.alloc_out((1, 1, 512, 384), device))  # [.., 256, 192], not [.., 256, 96]
    assert _entries(device) - before == 0, "a refused call must not build a program"
    x2 = pp.make_input(shape, seed=62)
    out = pair_pool_op(pp.to_dev(x2, device))
    assert _entries(device) - before == 0, "the valid call after a refusal must hit"
    pp.check(ttnn.to_torch(out), x2, label="after refusal")


# --------------------------------------------------------------------------- refusals (support contract)


def test_refuse_dtype(device, pair_pool_op, pp, expect_error):
    with expect_error(AssertionError, "BFLOAT16"):
        pair_pool_op(pp.to_dev(torch.zeros(1, 1, 64, 64), device, dtype=ttnn.bfloat16))


def test_refuse_layout(device, pair_pool_op, pp, expect_error):
    with expect_error(AssertionError, "(?i)row_major"):
        pair_pool_op(pp.to_dev(torch.zeros(1, 1, 64, 64), device, layout=ttnn.ROW_MAJOR_LAYOUT))


def test_refuse_l1_input(device, pair_pool_op, pp, expect_error):
    with expect_error(AssertionError, "L1"):
        pair_pool_op(pp.to_dev(torch.zeros(1, 1, 64, 64), device, memory_config=ttnn.L1_MEMORY_CONFIG))


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        pytest.param((1, 1, 64, 96), "hd % 32 == 0", id="hd48"),
        pytest.param((1, 1, 64, 32), "hd % 32 == 0", id="hd16"),
        pytest.param((1, 1, 96, 64), "% 64 != 0", id="rows96"),
        pytest.param((1, 1, 32, 64), "% 64 != 0", id="rows32"),
        pytest.param((1, 1, 100, 64), "% 64 != 0", id="rows100_padded"),
    ],
)
def test_refuse_shape(device, pair_pool_op, pp, expect_error, shape, message):
    with expect_error(AssertionError, message):
        pair_pool_op(pp.to_dev(torch.zeros(shape), device))


def test_refuse_output_tensor(device, pair_pool_op, pp, expect_error):
    """The caller's buffer must be exactly the output spec: float32, TILE, the output shape, interleaved."""
    shape = (1, 1, 64, 128)
    x = pp.make_input(shape, seed=71)
    xd = pp.to_dev(x, device)
    with expect_error(AssertionError, "BFLOAT16"):
        pair_pool_op(xd, output_tensor=pp.alloc_out(shape, device, dtype=ttnn.bfloat16))
    with expect_error(AssertionError, "(?i)row_major|layout|shape"):
        pair_pool_op(xd, output_tensor=pp.alloc_out(shape, device, layout=ttnn.ROW_MAJOR_LAYOUT))
    with expect_error(AssertionError, "(?i)shape"):
        pair_pool_op(xd, output_tensor=pp.alloc_out((1, 1, 128, 128), device))  # [.., 64, 64], not [.., 32, 64]
    sharded = ttnn.create_sharded_memory_config(
        [32, 64], ttnn.CoreGrid(x=1, y=1), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )
    with expect_error(AssertionError, "(?i)sharded"):
        pair_pool_op(xd, output_tensor=pp.alloc_out(shape, device, sharded))
    # the input is still usable: a valid call after the refusals
    pp.check(ttnn.to_torch(pair_pool_op(xd)), x, label="after output refusals")

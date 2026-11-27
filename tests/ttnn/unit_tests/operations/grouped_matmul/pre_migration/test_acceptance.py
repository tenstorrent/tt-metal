# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.bringup.grouped_matmul_ttnn:grouped_matmul``.

Every test calls the operation through the ``grouped_matmul_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (README / docstring of ``grouped_matmul``; the planner and descriptor modules are not read):
    grouped_matmul(x, w, *, dtype=None, compute_kernel_config=None, output_tensor=None, memory_config=None,
                   max_grid=None, block_w=None, depth=2, rt=1, w_layout=None, w_inflight=1, x_ahead=1,
                   one_group=False, diag=()) -> ttnn.Tensor
    y[g] = x[g] @ w[g]; x [1, G, M, K], w [1, G, K, N] -> [1, G, M, N]; TILE; x, w (and ``output_tensor``)
    INTERLEAVED (x, w in DRAM); any tile dtypes; output dtype ``dtype`` or x's; a new output in ``memory_config``
    (default DRAM); ``output_tensor`` of exactly that spec is written and returned as is. One grid row a group:
    G <= rows of ``max_grid`` (CoreCoord or (x, y), default the whole grid). ``w_layout`` (a ``bank.BankLayout``):
    w is ``bank.upload``'s WIDTH_SHARDED DRAM tensor, output bytes equal to the interleaved path at the same knobs.
    Refusals today (AssertionError from the op's checks): non-TILE or non-INTERLEAVED-DRAM x / w, an output_tensor
    whose dtype / shape / layout / memory layout differs, an unknown ``diag`` stub, depth < 2, rt x column tiles over
    DEST, a call with no layout (K mismatch, G over the grid's rows), a bank call with a non-bank w, w_inflight /
    x_ahead out of range, a block_w that does not divide K's tiles. Accepted exception types: AssertionError (source)
    or RuntimeError (a C++ TT_FATAL of the same check); the message must name the violated field.
    Cache (device program cache of the mesh, cleared at each cache test's start): a new configuration adds exactly
    one entry; fresh buffers / data / a caller's output buffer hit (+0); block_w, depth, rt, output dtype, x dtype,
    compute config, max_grid and w_layout each select their own program.
    Not covered (not promised or diagnostic only): ``diag`` stubs (wrong output by design), an output aliasing an
    input, batch dims other than [1, G], sharded outputs.
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

SOURCE_ENTRY = "ttnn.bringup.grouped_matmul_ttnn:grouped_matmul"
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
        if name.startswith("ttnn.bringup.grouped_matmul_ttnn") or name.startswith("ttnn._ttnn.operations.generic"):
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
                raise AssertionError("native grouped_matmul must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def grouped_matmul_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"grouped_matmul acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch  # noqa: E402

from tests.ttnn.unit_tests.operations.grouped_matmul.pre_migration.gmm_helpers import (  # noqa: E402
    REL_TOL_HIFI2,
    REL_TOL_HIFI4_EXACT,
    addr,
    assert_close,
    assert_output_spec,
    chip0,
    chips,
    compute_config,
    entries,
    operands,
    rel_err,
    to_dev,
    truth_of,
)
from ttnn.bringup.grouped_matmul_ttnn import bank  # noqa: E402

BF16, BF8, BF4, F32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32
HIFI4 = ttnn.MathFidelity.HiFi4
EXPECTED_REFUSAL = (AssertionError, RuntimeError)  # source: the op's assert; native: a TT_FATAL of the same check


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return g.x, g.y


def _kwargs(mesh, spec):
    """A case's keyword arguments: ``fidelity`` becomes a compute config, ``max_grid`` stays as given (tuple or
    CoreCoord, both accepted)."""
    kw = dict(spec)
    fid = kw.pop("fidelity", None)
    if fid is not None:
        kw["compute_kernel_config"] = compute_config(mesh, fid)
    return kw


def _fresh_cache(mesh):
    """Known cache state for a cache-count test: empty, caching on (asserted by the first call's +1)."""
    ttnn.synchronize_device(mesh)
    mesh.clear_program_cache()
    mesh.enable_program_cache()
    assert entries(mesh) == 0


def _call_counted(mesh, op, *args, **kwargs):
    """The op alone between two cache reads (inputs already on device, readback after)."""
    before = entries(mesh)
    y = op(*args, **kwargs)
    ttnn.synchronize_device(mesh)
    return y, entries(mesh) - before


# --------------------------------------------------------------------------- planner branches, each a miss then a hit

# (id, G, M, K, N, x dtype, w dtype, output dtype (None: the default, x's), kwargs, tolerance)
# Every branch is reached through the public inputs only: the column split (even / uneven / one column / fewer column
# tiles than grid columns), the K blocking (default block_w, block_w 1, a K of 3 tiles, many blocks at depth 4), the
# DEST mode (rt 2: rt x column tiles over half of DEST, full sync), the dtype mix (CB page sizes) and the fidelity.
BRANCHES = [
    ("min_one_tile", 1, 32, 32, 32, BF16, BF16, None, {}, REL_TOL_HIFI2),
    (
        "uneven_cols_4x3_tuple_grid",
        3,
        96,
        512,
        352,
        BF8,
        BF8,
        BF8,
        dict(max_grid=(4, 3), block_w=8, depth=3),
        REL_TOL_HIFI2,
    ),
    ("one_col_1x2", 2, 32, 1024, 64, BF8, BF8, BF16, dict(max_grid=ttnn.CoreCoord(1, 2)), REL_TOL_HIFI2),
    ("two_col_tiles_full_grid", 2, 64, 256, 64, BF8, BF8, BF8, {}, REL_TOL_HIFI2),
    ("k_three_tiles_default_bw", 2, 64, 96, 128, BF16, BF8, BF16, {}, REL_TOL_HIFI2),
    ("block_w1", 2, 32, 256, 128, BF8, BF8, BF8, dict(block_w=1), REL_TOL_HIFI2),
    ("many_blocks_bw4_d4", 4, 64, 2048, 512, BF8, BF8, BF8, dict(block_w=4, depth=4), REL_TOL_HIFI2),
    ("rt2_full_sync_2x2", 2, 128, 256, 256, BF8, BF8, F32, dict(max_grid=ttnn.CoreCoord(2, 2), rt=2), REL_TOL_HIFI2),
    ("bf16_hifi4_fp32_out", 2, 64, 512, 256, BF16, BF16, F32, dict(fidelity=HIFI4), REL_TOL_HIFI4_EXACT),
    ("bfp4_weight", 2, 64, 512, 256, BF16, BF4, BF16, {}, REL_TOL_HIFI2),
    ("fp32_operands", 2, 32, 256, 128, F32, F32, F32, {}, REL_TOL_HIFI2),
]


@pytest.mark.parametrize("G, M, K, N, xdt, wdt, odt, spec, tol", [pytest.param(*c[1:], id=c[0]) for c in BRANCHES])
def test_branch(gmm_mesh, grouped_matmul_op, G, M, K, N, xdt, wdt, odt, spec, tol):
    mesh = gmm_mesh
    kw = _kwargs(mesh, spec)
    if odt is not None:
        kw["dtype"] = odt
    want_dt = odt or xdt
    _fresh_cache(mesh)
    xt, wt = operands(G * 31 + M + K, G, M, K, N)
    x, w = to_dev(mesh, xt, xdt), to_dev(mesh, wt, wdt)
    x_before, w_before = chip0(x), chip0(w)
    y, d = _call_counted(mesh, grouped_matmul_op, x, w, **kw)
    assert d == 1, f"first call of a new configuration: {d} cache entries added, 1 expected"
    assert_output_spec(y, [1, G, M, N], want_dt)
    assert_close(y, x_before @ w_before, tol, "miss")
    assert torch.equal(chip0(x), x_before) and torch.equal(chip0(w), w_before), "an input was modified"

    # a hit on fresh buffers with other data (the first call's tensors kept alive: other addresses)
    xt2, wt2 = operands(G * 31 + M + K + 1000, G, M, K, N)
    x2, w2 = to_dev(mesh, -xt2, xdt), to_dev(mesh, wt2, wdt)
    assert addr(x2) != addr(x) and addr(w2) != addr(w)
    y2, d = _call_counted(mesh, grouped_matmul_op, x2, w2, **kw)
    assert d == 0, f"same configuration on fresh buffers: {d} cache entries added, 0 expected (a hit)"
    assert addr(y2) != addr(y)
    assert_output_spec(y2, [1, G, M, N], want_dt)
    assert_close(y2, truth_of(x2, w2), tol, "hit")
    assert rel_err(chip0(y), x_before @ w_before) <= tol, "the first output changed after the second call"


def test_groups_fill_grid_rows(gmm_mesh, grouped_matmul_op):
    """G equal to the whole grid's rows (the limit, just inside): every row a group."""
    mesh = gmm_mesh
    _, rows = _grid(mesh)
    G, M, K, N = rows, 32, 128, 64
    xt, wt = operands(7, G, M, K, N)
    x, w = to_dev(mesh, xt, BF8), to_dev(mesh, wt, BF8)
    y = grouped_matmul_op(x, w)
    assert_output_spec(y, [1, G, M, N], BF8)
    assert_close(y, truth_of(x, w), REL_TOL_HIFI2, f"G={G}")


def test_defaults_and_conversions(gmm_mesh, grouped_matmul_op):
    """Defaults: no keyword at all gives x's dtype in DRAM; dtype=None, the default compute config spelled out
    (HiFi2, fp32 DEST) and max_grid as a tuple vs a CoreCoord give the same bytes."""
    mesh = gmm_mesh
    G, M, K, N = 2, 64, 256, 128
    xt, wt = operands(11, G, M, K, N)
    x, w = to_dev(mesh, xt, BF16), to_dev(mesh, wt, BF8)
    y = grouped_matmul_op(x, w)
    assert_output_spec(y, [1, G, M, N], BF16)
    ref = assert_close(y, truth_of(x, w), REL_TOL_HIFI2, "defaults")
    rows = _grid(mesh)
    variants = {
        "dtype_none": dict(dtype=None),
        "explicit_default_ckc": dict(compute_kernel_config=compute_config(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG),
        "max_grid_tuple": dict(max_grid=rows),
        "max_grid_corecoord": dict(max_grid=ttnn.CoreCoord(*rows)),
    }
    for name, kw in variants.items():
        yv = grouped_matmul_op(x, w, **kw)
        assert_output_spec(yv, [1, G, M, N], BF16)
        assert torch.equal(chip0(yv), ref), f"{name}: differs from the all-defaults call"


def test_max_grid_tuple_and_corecoord_share_a_program(gmm_mesh, grouped_matmul_op):
    """max_grid (4, 2) and CoreCoord(4, 2) are one configuration: the second is a hit with the same bytes."""
    mesh = gmm_mesh
    G, M, K, N = 2, 64, 256, 224
    xt, wt = operands(13, G, M, K, N)
    x, w = to_dev(mesh, xt, BF8), to_dev(mesh, wt, BF8)
    _fresh_cache(mesh)
    y1, d1 = _call_counted(mesh, grouped_matmul_op, x, w, max_grid=(4, 2))
    y2, d2 = _call_counted(mesh, grouped_matmul_op, x, w, max_grid=ttnn.CoreCoord(4, 2))
    assert (d1, d2) == (1, 0), (d1, d2)
    ref = assert_close(y1, truth_of(x, w), REL_TOL_HIFI2, "tuple")
    assert torch.equal(chip0(y2), ref)


def test_output_tensor_written_and_returned(gmm_mesh, grouped_matmul_op):
    """A caller's output (pre-filled with a sentinel) is the returned object, at the same address, fully overwritten."""
    mesh = gmm_mesh
    G, M, K, N = 3, 96, 512, 352
    xt, wt = operands(17, G, M, K, N)
    x, w = to_dev(mesh, xt, BF8), to_dev(mesh, wt, BF8)
    out = to_dev(mesh, torch.full((1, G, M, N), 1.0e4), BF16)
    a0 = addr(out)
    y = grouped_matmul_op(x, w, dtype=BF16, output_tensor=out, max_grid=(4, 3))
    assert y is out
    assert addr(y) == a0
    assert_output_spec(y, [1, G, M, N], BF16)
    got = assert_close(y, truth_of(x, w), REL_TOL_HIFI2, "output_tensor")
    assert float(got.abs().max()) < 1.0e3, "sentinel left in the caller's output"


def test_l1_output_memory_config(gmm_mesh, grouped_matmul_op):
    """memory_config of a new output: L1 interleaved, the right values there. (Its own shape: no other test's
    configuration, so a cached program of another output buffer type cannot be involved.)"""
    mesh = gmm_mesh
    G, M, K, N = 2, 96, 256, 160
    xt, wt = operands(19, G, M, K, N)
    x, w = to_dev(mesh, xt, BF8), to_dev(mesh, wt, BF8)
    y = grouped_matmul_op(x, w, dtype=BF16, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert_output_spec(y, [1, G, M, N], BF16, ttnn.BufferType.L1)
    assert_close(y, truth_of(x, w), REL_TOL_HIFI2, "L1 output")


def test_per_chip_operands(gmm_mesh, grouped_matmul_op):
    """x different on every chip (sharded over the mesh), w replicated: each chip multiplies its own x."""
    mesh = gmm_mesh
    n = mesh.get_num_devices()
    G, M, K, N = 2, 64, 256, 128
    xt = torch.cat([operands(100 + i, G, M, K, N)[0] for i in range(n)], dim=0)
    _, wt = operands(99, G, M, K, N)
    x = to_dev(mesh, xt, BF8, mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    w = to_dev(mesh, wt, BF8)
    y = grouped_matmul_op(x, w)
    assert_output_spec(y, [1, G, M, N], BF8)
    wq = chip0(w)
    xs, ys = chips(x), chips(y)
    assert len(ys) == n
    for i in range(n):
        rel = rel_err(ys[i], xs[i] @ wq)
        assert rel <= REL_TOL_HIFI2, f"chip {i}: rel {rel:.3e}"
    assert not torch.equal(ys[0], ys[1]), "chips gave the same output for different x"


# --------------------------------------------------------------------------- cache sequences


def test_cache_fresh_buffers_hit(gmm_mesh, grouped_matmul_op):
    """A -> new A -> new A: other data and live addresses each call; one entry, then hits; each output its own."""
    mesh = gmm_mesh
    G, M, K, N = 4, 64, 1024, 512
    kw = dict(dtype=BF8, block_w=4, depth=4, compute_kernel_config=compute_config(mesh))
    _fresh_cache(mesh)
    kept, deltas = [], []
    for i in range(3):
        xt, wt = operands(200 + i, G, M, K, N)
        x, w = to_dev(mesh, xt, BF8), to_dev(mesh, wt, BF8)
        y, d = _call_counted(mesh, grouped_matmul_op, x, w, **kw)
        deltas.append(d)
        kept.append((x, w, y, truth_of(x, w)))
    assert deltas == [1, 0, 0], deltas
    for col in range(3):
        a = [addr(k[col]) for k in kept]
        assert len(set(a)) == 3, f"tensor {col}: addresses not distinct {a}"
    for i, (x, w, y, truth) in enumerate(kept):
        assert_close(y, truth, REL_TOL_HIFI2, f"call {i}")


# Program-defining fields: each gives its own program (CB sizes / formats, compile-time block sizes, DEST mode,
# fidelity, core set), so A -> B -> A is miss, miss, hit.
PROGRAM_FIELDS = [
    ("block_w", dict(block_w=4), dict(block_w=8)),
    ("depth", dict(depth=2), dict(depth=3)),
    ("rt", dict(rt=1), dict(rt=2)),
    ("out_dtype", dict(dtype=BF16), dict(dtype=F32)),
    ("fidelity", dict(fidelity=ttnn.MathFidelity.HiFi2), dict(fidelity=HIFI4)),
    ("max_grid", dict(max_grid=(4, 2)), dict(max_grid=(2, 2))),
]


@pytest.mark.parametrize("a, b", [pytest.param(a, b, id=n) for n, a, b in PROGRAM_FIELDS])
def test_cache_program_field_switch(gmm_mesh, grouped_matmul_op, a, b):
    mesh = gmm_mesh
    G, M, K, N = 2, 128, 512, 256
    xt, wt = operands(300, G, M, K, N)
    x, w = to_dev(mesh, xt, BF8), to_dev(mesh, wt, BF8)
    truth = truth_of(x, w)
    _fresh_cache(mesh)
    deltas = []
    for i, spec in enumerate((a, b, a)):
        y, d = _call_counted(mesh, grouped_matmul_op, x, w, **_kwargs(mesh, spec))
        deltas.append(d)
        assert_close(y, truth, REL_TOL_HIFI2, f"step {i} {spec}")
        if "dtype" in spec:
            assert y.dtype == spec["dtype"]
    assert deltas == [1, 1, 0], deltas


def test_cache_x_dtype_switch(gmm_mesh, grouped_matmul_op):
    """x bfloat8_b -> bfloat16 -> bfloat8_b (output dtype pinned): another input format, another program."""
    mesh = gmm_mesh
    G, M, K, N = 2, 64, 256, 128
    xt, wt = operands(310, G, M, K, N)
    w = to_dev(mesh, wt, BF8)
    _fresh_cache(mesh)
    deltas = []
    for dt in (BF8, BF16, BF8):
        x = to_dev(mesh, xt, dt)
        y, d = _call_counted(mesh, grouped_matmul_op, x, w, dtype=BF16)
        deltas.append(d)
        assert_close(y, truth_of(x, w), REL_TOL_HIFI2, f"x {dt}")
    assert deltas == [1, 1, 0], deltas


def test_cache_output_tensor_absent_present_absent(gmm_mesh, grouped_matmul_op):
    """A new output -> a caller's output -> a new output: one program, the caller's buffer returned and written."""
    mesh = gmm_mesh
    G, M, K, N = 2, 64, 512, 256
    _fresh_cache(mesh)
    deltas, kept = [], []
    for i, preallocated in enumerate((False, True, False)):
        xt, wt = operands(320 + i, G, M, K, N)
        x, w = to_dev(mesh, xt, BF8), to_dev(mesh, wt, BF8)
        kw = dict(dtype=BF16)
        out = None
        if preallocated:
            out = to_dev(mesh, torch.full((1, G, M, N), -7.0e3), BF16)
            kw["output_tensor"] = out
        y, d = _call_counted(mesh, grouped_matmul_op, x, w, **kw)
        deltas.append(d)
        if preallocated:
            assert y is out
        assert_output_spec(y, [1, G, M, N], BF16)
        kept.append((y, truth_of(x, w)))
    assert deltas == [1, 0, 0], deltas
    for i, (y, truth) in enumerate(kept):
        assert_close(y, truth, REL_TOL_HIFI2, f"step {i}")


def test_cache_output_memory_config_l1_dram_l1(gmm_mesh, grouped_matmul_op):
    """L1 output -> DRAM output -> L1 output of one configuration (its own shape, used by no other test): each output
    in the buffer type asked for, with the right values; the return to L1 reuses the first program. The output's
    buffer type is a program-defining field (the writer addresses an L1 or a DRAM buffer). (L1 first: on a route that
    keys the two alike, the DRAM call then writes worker L1, not other live DRAM tensors.)"""
    mesh = gmm_mesh
    G, M, K, N = 2, 32, 512, 192
    _fresh_cache(mesh)
    deltas = []
    steps = (
        (ttnn.L1_MEMORY_CONFIG, ttnn.BufferType.L1),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.BufferType.DRAM),
        (ttnn.L1_MEMORY_CONFIG, ttnn.BufferType.L1),
    )
    for i, (mc, bt) in enumerate(steps):
        xt, wt = operands(330 + i, G, M, K, N)
        x, w = to_dev(mesh, xt, BF8), to_dev(mesh, wt, BF8)
        y, d = _call_counted(mesh, grouped_matmul_op, x, w, dtype=BF16, memory_config=mc)
        deltas.append(d)
        assert_output_spec(y, [1, G, M, N], BF16, bt)
        assert_close(y, truth_of(x, w), REL_TOL_HIFI2, f"step {i} {bt}")
    assert deltas == [1, 1, 0], deltas


def test_cache_valid_invalid_valid(gmm_mesh, grouped_matmul_op, expect_error):
    """A warmed configuration still refuses a wrong caller output (no entry added, the buffer untouched); the next
    valid call hits and is correct."""
    mesh = gmm_mesh
    G, M, K, N = 2, 64, 256, 128
    xt, wt = operands(340, G, M, K, N)
    x, w = to_dev(mesh, xt, BF8), to_dev(mesh, wt, BF8)
    _fresh_cache(mesh)
    y, d = _call_counted(mesh, grouped_matmul_op, x, w, dtype=BF16)
    assert d == 1
    bad = to_dev(mesh, torch.full((1, G, M, N), 5.0), F32)
    before = entries(mesh)
    with expect_error(EXPECTED_REFUSAL, r"(?i)dtype|float32"):
        grouped_matmul_op(x, w, dtype=BF16, output_tensor=bad)
    ttnn.synchronize_device(mesh)
    assert entries(mesh) == before
    assert torch.equal(chip0(bad), torch.full((1, G, M, N), 5.0, dtype=torch.float64)), "refused call wrote its output"
    x2 = to_dev(mesh, operands(341, G, M, K, N)[0], BF8)
    y2, d = _call_counted(mesh, grouped_matmul_op, x2, w, dtype=BF16)
    assert d == 0
    assert_close(y2, truth_of(x2, w), REL_TOL_HIFI2, "after refusal")
    assert_close(y, truth_of(x, w), REL_TOL_HIFI2, "first output")


# --------------------------------------------------------------------------- bank-local weights (w_layout)

BANK_G, BANK_M, BANK_K, BANK_N = 8, 32, 256, 1024


def _bank_layout(mesh, **kw):
    lay = bank.plan(mesh, BANK_G, BANK_K, BANK_N, max_cores=110, **kw)
    assert lay is not None, kw
    return lay


@pytest.mark.parametrize(
    "plan_kw, call_kw",
    [
        pytest.param(dict(cores_per_bank=12), dict(block_w=2, depth=2, w_inflight=1, x_ahead=1), id="band12"),
        pytest.param(
            dict(cores_per_bank=10),
            dict(block_w=4, depth=4, w_inflight=2, x_ahead=1, one_group=True),
            id="band10_one_group",
        ),
        pytest.param(
            dict(cores_per_bank=11, shape="rect"), dict(block_w=2, depth=3, w_inflight=3, x_ahead=2), id="rect11"
        ),
    ],
)
def test_bank_path_matches_interleaved_and_hits(gmm_mesh, grouped_matmul_op, plan_kw, call_kw):
    """The bank path's bytes equal the interleaved path's at the same knobs; then a hit on a fresh x, a fresh bank
    weight (other values) and a caller's output, checked against fp64."""
    mesh = gmm_mesh
    lay = _bank_layout(mesh, **plan_kw)
    common = dict(
        dtype=BF8, compute_kernel_config=compute_config(mesh), block_w=call_kw["block_w"], depth=call_kw["depth"]
    )
    bank_kw = {k: v for k, v in call_kw.items() if k not in ("block_w", "depth")}
    xt, wt = operands(400, BANK_G, BANK_M, BANK_K, BANK_N)
    x, w = to_dev(mesh, xt, BF8), to_dev(mesh, wt, BF8)
    wb = bank.upload(wt, lay, mesh)
    _fresh_cache(mesh)
    today, d0 = _call_counted(mesh, grouped_matmul_op, x, w, **common)
    y, d1 = _call_counted(mesh, grouped_matmul_op, x, wb, w_layout=lay, **common, **bank_kw)
    assert (d0, d1) == (1, 1), (d0, d1)
    assert_output_spec(y, [1, BANK_G, BANK_M, BANK_N], BF8)
    ref = assert_close(today, truth_of(x, w), REL_TOL_HIFI2, "interleaved")
    for i, c in enumerate(chips(y)):
        assert torch.equal(c, ref), f"chip {i}: bank path differs from the interleaved path"

    xt2, wt2 = operands(401, BANK_G, BANK_M, BANK_K, BANK_N)
    x2, w2 = to_dev(mesh, xt2, BF8), to_dev(mesh, wt2, BF8)
    wb2 = bank.upload(wt2, lay, mesh)
    out = to_dev(mesh, torch.full((1, BANK_G, BANK_M, BANK_N), 3.0e3), BF8)
    assert addr(x2) != addr(x) and addr(wb2) != addr(wb)
    y2, d2 = _call_counted(mesh, grouped_matmul_op, x2, wb2, w_layout=lay, output_tensor=out, **common, **bank_kw)
    assert d2 == 0, f"bank path on fresh buffers: {d2} entries added, 0 expected"
    assert y2 is out
    assert_close(y2, truth_of(x2, w2), REL_TOL_HIFI2, "bank hit")
    assert torch.equal(chip0(y), ref), "the first bank output changed"


def test_bank_and_interleaved_alternate(gmm_mesh, grouped_matmul_op):
    """absent -> present -> absent w_layout on the same x: each call correct, the return to interleaved a hit."""
    mesh = gmm_mesh
    lay = _bank_layout(mesh)
    kw = dict(dtype=BF16, block_w=2)
    xt, wt = operands(410, BANK_G, BANK_M, BANK_K, BANK_N)
    x, w = to_dev(mesh, xt, BF8), to_dev(mesh, wt, BF8)
    wb = bank.upload(wt, lay, mesh)
    truth = truth_of(x, w)
    _fresh_cache(mesh)
    y0, d0 = _call_counted(mesh, grouped_matmul_op, x, w, **kw)
    y1, d1 = _call_counted(mesh, grouped_matmul_op, x, wb, w_layout=lay, **kw)
    y2, d2 = _call_counted(mesh, grouped_matmul_op, x, w, **kw)
    assert (d0, d1, d2) == (1, 1, 0), (d0, d1, d2)
    ref = assert_close(y0, truth, REL_TOL_HIFI2, "interleaved")
    assert torch.equal(chip0(y1), ref) and torch.equal(chip0(y2), ref)


# --------------------------------------------------------------------------- refusals


def _refusal_inputs(mesh, name):
    """(args, kwargs, exception types, message regex) of one refused call."""
    G, M, K, N = 2, 64, 256, 128
    xt, wt = operands(500, G, M, K, N)
    x = lambda **k: to_dev(mesh, xt, BF8, **k)  # noqa: E731
    w = lambda **k: to_dev(mesh, wt, BF8, **k)  # noqa: E731
    if name == "x_row_major":
        return (
            (to_dev(mesh, xt, BF16, layout=ttnn.ROW_MAJOR_LAYOUT), w()),
            {},
            EXPECTED_REFUSAL,
            r"(?i)row_major|tile|layout",
        )
    if name == "w_row_major":
        return (
            (x(), to_dev(mesh, wt, BF16, layout=ttnn.ROW_MAJOR_LAYOUT)),
            {},
            EXPECTED_REFUSAL,
            r"(?i)row_major|tile|layout",
        )
    if name == "x_in_l1":
        return (x(memory_config=ttnn.L1_MEMORY_CONFIG), w()), {}, EXPECTED_REFUSAL, r"(?i)L1|DRAM|buffer"
    if name == "w_in_l1":
        return (x(), w(memory_config=ttnn.L1_MEMORY_CONFIG)), {}, EXPECTED_REFUSAL, r"(?i)L1|DRAM|buffer"
    if name == "out_wrong_dtype":
        out = to_dev(mesh, torch.zeros(1, G, M, N), BF16)
        return (x(), w()), dict(output_tensor=out), EXPECTED_REFUSAL, r"(?i)dtype|bfloat16"
    if name == "out_wrong_shape":
        out = to_dev(mesh, torch.zeros(1, G, M, N + 32), BF8)
        return (x(), w()), dict(output_tensor=out), EXPECTED_REFUSAL, r"(?i)shape"
    if name == "out_row_major":
        out = to_dev(mesh, torch.zeros(1, G, M, N), BF16, layout=ttnn.ROW_MAJOR_LAYOUT)
        return (x(), w()), dict(dtype=BF16, output_tensor=out), EXPECTED_REFUSAL, r"(?i)layout|tile|row_major|shape"
    if name == "diag_unknown":
        return (x(), w()), dict(diag=("bogus_stub",)), EXPECTED_REFUSAL, r"(?i)bogus_stub|diag"
    if name == "depth_1":
        return (x(), w()), dict(depth=1), EXPECTED_REFUSAL, r"(?i)depth"
    if name == "rt_over_dest":
        # one column (max_grid 1 x 2): 4 column tiles a core; rt 8 of the 8 row tiles: 32 tiles, over DEST in any mode
        xr = to_dev(mesh, operands(503, G, 256, K, N)[0], BF8)
        return (xr, w()), dict(max_grid=(1, 2), rt=8), EXPECTED_REFUSAL, r"(?i)\brt\b|dest"
    if name == "k_mismatch":
        w_short = to_dev(mesh, wt[:, :, : K // 2, :], BF8)
        return (x(), w_short), {}, EXPECTED_REFUSAL, r"(?i)layout|support|shape|mismatch"
    if name == "groups_over_grid_rows":
        return (x(), w()), dict(max_grid=(4, 1)), EXPECTED_REFUSAL, r"(?i)layout|support|grid|group|rows"
    if name == "positional_dtype":
        return (x(), w(), BF16), {}, TypeError, r"(?i)positional|incompatible|argument"
    # bank path
    lay = bank.plan(mesh, BANK_G, BANK_K, BANK_N, max_cores=110)
    assert lay is not None
    bx = to_dev(mesh, operands(501, BANK_G, BANK_M, BANK_K, BANK_N)[0], BF8)
    bwt = operands(502, BANK_G, BANK_M, BANK_K, BANK_N)[1]
    if name == "bank_interleaved_w":
        return (bx, to_dev(mesh, bwt, BF8)), dict(w_layout=lay), EXPECTED_REFUSAL, r"(?i)bank|shard|w_layout"
    wb = bank.upload(bwt, lay, mesh)
    if name == "bank_w_inflight_over_depth":
        return (bx, wb), dict(w_layout=lay, block_w=2, depth=2, w_inflight=3), EXPECTED_REFUSAL, r"(?i)w_inflight"
    if name == "bank_x_ahead_at_depth":
        return (bx, wb), dict(w_layout=lay, block_w=2, depth=2, x_ahead=2), EXPECTED_REFUSAL, r"(?i)x_ahead"
    if name == "bank_block_w_not_dividing_k":
        return (bx, wb), dict(w_layout=lay, block_w=3), EXPECTED_REFUSAL, r"(?i)block_w"
    raise AssertionError(name)


REFUSALS = [
    "x_row_major",
    "w_row_major",
    "x_in_l1",
    "w_in_l1",
    "out_wrong_dtype",
    "out_wrong_shape",
    "out_row_major",
    "diag_unknown",
    "depth_1",
    "rt_over_dest",
    "k_mismatch",
    "groups_over_grid_rows",
    "positional_dtype",
    "bank_interleaved_w",
    "bank_w_inflight_over_depth",
    "bank_x_ahead_at_depth",
    "bank_block_w_not_dividing_k",
]


@pytest.mark.parametrize("name", REFUSALS)
def test_refusal(gmm_mesh, grouped_matmul_op, expect_error, name):
    args, kwargs, exc, msg = _refusal_inputs(gmm_mesh, name)
    with expect_error(exc, msg):
        grouped_matmul_op(*args, **kwargs)
    ttnn.synchronize_device(gmm_mesh)

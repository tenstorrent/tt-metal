# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.bringup.fabric_reduce_scatter_ttnn:fabric_reduce_scatter``.

Every test calls the operation through the ``fabric_reduce_scatter_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (``fabric_reduce_scatter(input_tensor, *, cluster_axis=0, topology=Linear, num_links=1,
scheme="ring", output=None, placement="auto", dim=2, exact_fp32_add=False, cache_descriptors=False,
program_hash=False, prepared=False)``), on the op's unit-test mesh (FABRIC_2D; BRINGUP_TEST_MESH, e.g. 8x4, else
2x2):
  * input: a per-chip partial [1, 1, R, W], TILE, DRAM interleaved, bf16 or fp32. dim 2 (or -2): R % (32 G) == 0,
    chip gets rows [p Sb, (p + 1) Sb) of the group sum, [1, 1, R / G, W]; dim 3 (or -1): R % 32 == 0 and
    W % (32 G) == 0, chip gets columns, [1, 1, R, W / G]. Groups: cluster_axis 0 (a column, chip owns block r),
    1 (a row, block c), None (the whole mesh, block r * cols + c). Topology Linear; Ring only where every hop of
    the ring is a direct link (a group of 2 is a line), else ValueError. num_links per hop (up to the hop's usable
    links); placement "auto" / "simple".
  * output: a new [1, 1, Sb, W] TILE DRAM-interleaved tensor of the input dtype, or the preallocated ``output`` (its
    buffer is the one returned and every value is overwritten). The input is not modified.
  * exact_fp32_add: FLOAT32 only; every add one IEEE fp32 rounding (a group of 2: torch's a + b bit for bit).
  * cache_descriptors / program_hash / prepared: host-side caching options; same results as the default path.
  * refusals: AssertionError for exact_fp32_add on non-fp32, rows / columns that do not split into G tile-aligned
    blocks, dim outside {2, 3, -1, -2}, a non-TILE or non-[1, 1, ...] input, a preallocated output of the wrong
    shape or dtype, program_hash without cache_descriptors, prepared without program_hash; ValueError for a non-DRAM
    or non-interleaved input, a TILE input that is not tile-aligned, an unknown scheme, dual_cycles without
    cluster_axis=None / Ring / a >= 3x3 torus, more links than a hop has, a ring hop that is not a direct link.
  * program cache (default path): the first call of a configuration compiles; the same configuration with fresh
    input / output buffers and new data, or with / without a preallocated output, reuses it; dim and dtype are
    program-defining; going back to an earlier configuration reuses its program.
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

SOURCE_ENTRY = "ttnn.bringup.fabric_reduce_scatter_ttnn:fabric_reduce_scatter"
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
        if name.startswith("ttnn.bringup.fabric_reduce_scatter_ttnn") or name.startswith(
            "ttnn._ttnn.operations.generic"
        ):
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
                raise AssertionError("native fabric_reduce_scatter must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def fabric_reduce_scatter_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"fabric_reduce_scatter acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch  # noqa: E402

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params  # noqa: E402

# The op's own unit-test mesh (ttnn.bringup._testing.mesh.QB_MESHES): FABRIC_2D, BRINGUP_TEST_MESH=RxC when set (8x4
# on the BH Galaxy), else 2x2. Shapes below are given in tiles per output block, so they fit any mesh.
_MESH = tuple(int(v) for v in (os.environ.get("BRINGUP_TEST_MESH") or "2x2").split("x"))
MESH = pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(_MESH, fabric2d_device_params(), id="x".join(map(str, _MESH)))],
    indirect=["mesh_device", "device_params"],
)

LINEAR, RING = ttnn.Topology.Linear, ttnn.Topology.Ring


def _check_dispatches(op, **kwargs):
    """The source route runs the reduce-scatter through ttnn.generic_op (the first call of a configuration adds a
    setup dispatch), except ``prepared`` (ttnn.bringup.run_prepared, which may make none); the native route makes
    none (its guard fails any)."""
    if op.mode == "native":
        assert op.last_dispatches == 0
    elif not kwargs.get("prepared"):
        assert op.last_dispatches >= 1, op.last_dispatches


def _chips(parts):
    return [parts[r, c] for r in range(parts.shape[0]) for c in range(parts.shape[1])]


def _run_and_check(op, rs, mesh_device, parts, dtype, **kwargs):
    """Upload ``parts``, call the op, check the output's metadata, its values (exactly: integer partials) against
    the torch reference, and that the input is untouched. Returns (input, output)."""
    x = rs.upload(mesh_device, parts, dtype)
    cluster_axis = kwargs.get("cluster_axis", 0)
    dim = kwargs.get("dim", 2)
    g = rs.group_size(mesh_device, cluster_axis)
    out = op(x, **kwargs)
    _check_dispatches(op, **kwargs)
    rs.assert_output_meta(out, rs.out_shape(list(x.shape), g, dim), dtype)
    rs.assert_exact(rs.per_chip(out), rs.expected(parts, cluster_axis, dim), "output")
    rs.assert_exact(rs.per_chip(x), _chips(parts), "input")
    return x, out


# --------------------------------------------------------------------------- planner branches

# (cluster_axis, dim, dtype, exact_fp32_add, num_links, placement, block tiles): one case per group shape, block
# direction, dtype / add path, link count, placement and work size (one tile per block; an odd 15-tile block; a
# block of many chunks per lane).
BRANCHES = [
    pytest.param(0, 2, ttnn.bfloat16, False, 1, "auto", (1, 1), id="axis0_rows_one_tile"),
    pytest.param(0, 2, ttnn.bfloat16, False, 1, "auto", (3, 5), id="axis0_rows_15_tiles"),
    pytest.param(1, 3, ttnn.bfloat16, False, 2, "auto", (3, 4), id="axis1_cols_links2"),
    pytest.param(None, 2, ttnn.bfloat16, False, 1, "auto", (1, 3), id="mesh_rows"),
    pytest.param(None, 3, ttnn.bfloat16, False, 2, "auto", (2, 1), id="mesh_cols_links2"),
    pytest.param(None, 2, ttnn.float32, True, 1, "auto", (2, 2), id="mesh_rows_fp32_exact"),
    pytest.param(1, 2, ttnn.bfloat16, False, 1, "simple", (4, 4), id="axis1_rows_placement_simple"),
    pytest.param(1, 2, ttnn.float32, False, 1, "auto", (2, 2), id="axis1_rows_fp32_fpu"),
    pytest.param(0, -1, ttnn.float32, True, 2, "auto", (2, 4), id="axis0_dim_neg1_fp32_exact"),
    pytest.param(1, -2, ttnn.bfloat16, False, 1, "auto", (2, 3), id="axis1_dim_neg2"),
    pytest.param(1, 2, ttnn.bfloat16, False, 2, "auto", (8, 64), id="axis1_rows_many_chunks"),
]


@MESH
@pytest.mark.parametrize(("cluster_axis", "dim", "dtype", "exact", "num_links", "placement", "tiles"), BRANCHES)
def test_branch(
    mesh_device,
    device_params,
    fabric_reduce_scatter_op,
    rs,
    cluster_axis,
    dim,
    dtype,
    exact,
    num_links,
    placement,
    tiles,
):
    rows, cols = tuple(mesh_device.shape)
    R, W = rs.in_shape(rs.group_size(mesh_device, cluster_axis), dim, tiles)
    kwargs = dict(cluster_axis=cluster_axis, num_links=num_links, dim=dim, placement=placement)
    if exact:
        kwargs["exact_fp32_add"] = True
    _, out = _run_and_check(
        fabric_reduce_scatter_op, rs, mesh_device, rs.int_parts(rows, cols, R, W, 1), dtype, **kwargs
    )
    # a repeat (program cache hit) on other data in fresh buffers stays exact
    _run_and_check(fabric_reduce_scatter_op, rs, mesh_device, rs.int_parts(rows, cols, R, W, 2), dtype, **kwargs)
    ttnn.deallocate(out)


@MESH
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_exact_fp32_bound(mesh_device, device_params, fabric_reduce_scatter_op, rs, cluster_axis):
    """exact_fp32_add on full-mantissa, wide-exponent fp32 partials: each add is one IEEE fp32 rounding, so a group
    of G is within G - 1 roundings of the float64 sum (a group of 2: torch's a + b bit for bit); the default FPU path
    on the same inputs misses that bound (the control proves the input distinguishes the two)."""
    rows, cols = tuple(mesh_device.shape)
    g = rs.group_size(mesh_device, cluster_axis)
    parts = rs.fp32_spread_parts(rows, cols, *rs.in_shape(g, 3, (4, 4)), seed=21 + cluster_axis)
    x = rs.upload(mesh_device, parts, ttnn.float32)
    kw = dict(cluster_axis=cluster_axis, dim=3)
    got = rs.per_chip(fabric_reduce_scatter_op(x, exact_fp32_add=True, **kw))
    off = rs.per_chip(fabric_reduce_scatter_op(x, **kw))
    want64 = rs.expected(parts, cluster_axis, 3)
    bound = [max(g - 1, 1) * 2.0**-24 * b for b in rs.expected(parts.abs(), cluster_axis, 3)]

    def worst(chips):
        return max(float(((t.double() - w).abs() / b).max()) for t, w, b in zip(chips, want64, bound))

    if g == 2:
        rs.assert_exact(got, rs.expected(parts, cluster_axis, 3, acc=torch.float32), "exact pair vs fp32 a + b")
    assert worst(got) <= 1.0, f"exact adds: |err| / ({g - 1}-rounding bound) = {worst(got):.3f}"
    assert worst(off) > 1.0, "control: the FPU add path should miss the exact-add bound"


# --------------------------------------------------------------------------- output buffer


@MESH
def test_preallocated_output(mesh_device, device_params, fabric_reduce_scatter_op, rs):
    """A preallocated ``output`` is the returned buffer, every stale value in it is overwritten, it can be reused
    across calls (other data), and an allocated output after it leaves it alone."""
    rows, cols = tuple(mesh_device.shape)
    R, W = rs.in_shape(rows, 2, (2, 2))
    kw = dict(cluster_axis=0, num_links=1)
    output = rs.upload(mesh_device, torch.full((rows, cols, R // rows, W), 99.0), ttnn.bfloat16)
    addr = output.buffer_address()
    for seed in (1, 2):
        parts = rs.int_parts(rows, cols, R, W, seed=seed)
        x = rs.upload(mesh_device, parts, ttnn.bfloat16)
        out = fabric_reduce_scatter_op(x, output=output, **kw)
        assert out.buffer_address() == addr, "the preallocated output must be the returned buffer"
        rs.assert_output_meta(out, [1, 1, R // rows, W], ttnn.bfloat16)
        rs.assert_exact(rs.per_chip(output), rs.expected(parts, 0, 2), f"preallocated output, call {seed}")
    parts = rs.int_parts(rows, cols, R, W, seed=3)
    out = fabric_reduce_scatter_op(rs.upload(mesh_device, parts, ttnn.bfloat16), **kw)
    assert out.buffer_address() != addr
    rs.assert_exact(rs.per_chip(out), rs.expected(parts, 0, 2), "allocated output")
    last = rs.expected(rs.int_parts(rows, cols, R, W, seed=2), 0, 2)
    rs.assert_exact(rs.per_chip(output), last, "the preallocated output after an allocated call")


# --------------------------------------------------------------------------- program cache


@MESH
def test_program_cache_sequence(mesh_device, device_params, fabric_reduce_scatter_op, rs, expect_error):
    """Default path: A (miss) -> A fresh buffers (hit) -> A again (hit) -> A into a preallocated output (hit) ->
    a refused call -> A (hit) -> B = dim 3 (miss: column blocks need other reader / writer programs) -> A (hit) ->
    C = fp32 (miss: the data format is compiled in) -> A (hit) -> B (hit). Entries are counted around the op call
    only, after upload and before readback; every output is checked against its own fresh data."""
    op = fabric_reduce_scatter_op
    mesh_device.enable_program_cache()
    rows, cols = tuple(mesh_device.shape)
    R, W = 64 * cols, 64 * cols  # two tiles a block either way (dim 2 rows, dim 3 columns)
    kw = dict(cluster_axis=1, num_links=1)
    keep = []  # every earlier input and output stays alive, so fresh buffers really get new addresses
    seed = [0]

    def call(dtype=ttnn.bfloat16, dim=2, output=None):
        seed[0] += 1
        parts = rs.int_parts(rows, cols, R, W, seed=seed[0])
        x = rs.upload(mesh_device, parts, dtype)
        assert all(x.buffer_address() != k.buffer_address() for k in keep), "input reuses a live address"
        before = rs.cache_entries(mesh_device)
        out = op(x, dim=dim, output=output, **kw)
        delta = rs.cache_entries(mesh_device) - before
        rs.assert_output_meta(out, rs.out_shape([1, 1, R, W], cols, dim), dtype)
        rs.assert_exact(rs.per_chip(out), rs.expected(parts, 1, dim), f"call {seed[0]} dim {dim} {dtype}")
        keep.extend([x, out])
        logger.info(f"cache sequence call {seed[0]}: dim {dim} {dtype} output={output is not None}: +{delta} entries")
        return delta

    assert call() > 0, "the first call of a configuration must compile (a cache miss)"
    assert call() == 0, "same configuration, fresh buffers: must hit"
    assert call() == 0, "same configuration, fresh buffers again: must hit"
    pre = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R // cols, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    assert call(output=pre) == 0, "a preallocated output must not change the program"
    bad = rs.upload(mesh_device, rs.int_parts(rows, cols, R, W, seed=99), ttnn.bfloat16)
    with expect_error(AssertionError, "exact_fp32_add needs FLOAT32"):
        op(bad, exact_fp32_add=True, **kw)
    assert call() == 0, "a valid call after a refused one must hit"
    assert call(dim=3) > 0, "dim 3 (column blocks) must be its own program"
    assert call() == 0, "back to dim 2: must hit"
    assert call(dtype=ttnn.float32) > 0, "fp32 must be its own program"
    assert call() == 0, "back to bf16: must hit"
    assert call(dim=3) == 0, "back to dim 3: must hit"


@MESH
def test_host_cache_modes(mesh_device, device_params, fabric_reduce_scatter_op, rs):
    """The default path, cache_descriptors, program_hash and prepared give the same exact result; in every mode a
    call on other buffers with other data hits the program cache and follows the new data, and the first buffers
    used again afterwards (the descriptor / prepared program built for them) are still right."""
    op = fabric_reduce_scatter_op
    mesh_device.enable_program_cache()
    rows, cols = tuple(mesh_device.shape)
    R, W = rs.in_shape(rows, 3, (8, 2))
    base = dict(cluster_axis=0, num_links=1, dim=3)
    modes = [
        ("default", {}),
        ("cache_descriptors", dict(cache_descriptors=True)),
        ("program_hash", dict(cache_descriptors=True, program_hash=True)),
        ("prepared", dict(cache_descriptors=True, program_hash=True, prepared=True)),
        ("default_again", {}),
    ]
    parts_a = rs.int_parts(rows, cols, R, W, seed=5)
    parts_b = rs.int_parts(rows, cols, R, W, seed=6)
    xa = rs.upload(mesh_device, parts_a, ttnn.bfloat16)
    xb = rs.upload(mesh_device, parts_b, ttnn.bfloat16)
    assert xa.buffer_address() != xb.buffer_address()
    want_a, want_b = rs.expected(parts_a, 0, 3), rs.expected(parts_b, 0, 3)
    out_a = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R, W // rows]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    for name, extra in modes:
        kw = dict(base, **extra)
        out = op(xa, output=out_a, **kw)
        _check_dispatches(op, **kw)
        assert out.buffer_address() == out_a.buffer_address()
        rs.assert_exact(rs.per_chip(out), want_a, f"{name}: first buffers")
        before = rs.cache_entries(mesh_device)
        out_b = op(xb, **kw)
        assert rs.cache_entries(mesh_device) - before == 0, f"{name}: other buffers must hit the program cache"
        rs.assert_exact(rs.per_chip(out_b), want_b, f"{name}: other buffers")
        before = rs.cache_entries(mesh_device)
        op(xa, output=out_a, **kw)
        assert rs.cache_entries(mesh_device) - before == 0, f"{name}: the first buffers again must hit"
        rs.assert_exact(rs.per_chip(out_a), want_a, f"{name}: first buffers after other ones")
        ttnn.deallocate(out_b)
    rs.assert_exact(rs.per_chip(xa), _chips(parts_a), "input a")


# --------------------------------------------------------------------------- refusals


def _zeros(rs, m, R, W, dtype=ttnn.bfloat16, **kw):
    rows, cols = tuple(m.shape)
    return rs.upload(m, torch.zeros(rows, cols, R, W), dtype, **kw)


def _valid_rows(m):  # a valid dim-2 row count for cluster_axis 0
    return 32 * tuple(m.shape)[0]


def _bad_dtype_exact(rs, m):
    return _zeros(rs, m, 64, 64), dict(cluster_axis=1, exact_fp32_add=True)


def _bad_cols(rs, m):
    return _zeros(rs, m, 64, 32, ttnn.float32), dict(cluster_axis=1, dim=3, exact_fp32_add=True)


def _bad_rows(rs, m):
    return _zeros(rs, m, 96, 64), dict(cluster_axis=0, dim=2)


def _bad_rows_mesh(rs, m):  # 64 rows never split into G >= 4 tile blocks over the whole mesh
    return _zeros(rs, m, 64, 64), dict(cluster_axis=None, dim=2)


def _bad_dim(rs, m):
    return _zeros(rs, m, _valid_rows(m), 64), dict(cluster_axis=0, dim=1)


def _not_tile_aligned(rs, m):
    return _zeros(rs, m, 48, 64), dict(cluster_axis=0)


def _l1_input(rs, m):
    return _zeros(rs, m, _valid_rows(m), 64, memory_config=ttnn.L1_MEMORY_CONFIG), dict(cluster_axis=0)


def _row_major(rs, m):
    return _zeros(rs, m, 64, 64, layout=ttnn.ROW_MAJOR_LAYOUT), dict(cluster_axis=0)


def _batched(rs, m):
    rows, cols = tuple(m.shape)
    x = ttnn.from_torch(
        torch.zeros(2 * rows, cols, 64, 64),
        device=m,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(m, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    assert list(x.shape) == [2, 1, 64, 64], list(x.shape)
    return x, dict(cluster_axis=0)


def _output_shape(rs, m):  # wants [1, 1, 32, 64]
    return _zeros(rs, m, _valid_rows(m), 64), dict(cluster_axis=0, output=_zeros(rs, m, 64, 64))


def _output_dtype(rs, m):
    return _zeros(rs, m, _valid_rows(m), 64), dict(cluster_axis=0, output=_zeros(rs, m, 32, 64, ttnn.float32))


def _hash_without_cache(rs, m):
    return _zeros(rs, m, 64, 64), dict(cluster_axis=1, program_hash=True)


def _prepared_without_hash(rs, m):
    return _zeros(rs, m, 64, 64), dict(cluster_axis=1, cache_descriptors=True, prepared=True)


def _unknown_scheme(rs, m):
    return _zeros(rs, m, _valid_rows(m), 64), dict(cluster_axis=0, scheme="bogus")


def _dual_cycles_on_axis(rs, m):
    return _zeros(rs, m, _valid_rows(m), 64), dict(cluster_axis=0, topology=RING, scheme="dual_cycles")


def _too_many_links(rs, m):
    return _zeros(rs, m, _valid_rows(m), 64), dict(cluster_axis=0, num_links=64)


def _ring_not_closed(rs, m):
    if tuple(m.shape)[0] <= 2:
        pytest.skip("a 2-chip axis is a line under Ring (no closing hop to refuse)")
    return _zeros(rs, m, _valid_rows(m), 64), dict(cluster_axis=0, topology=RING)


REFUSALS = [
    pytest.param(_bad_dtype_exact, AssertionError, "exact_fp32_add needs FLOAT32", id="exact_needs_fp32"),
    pytest.param(_bad_cols, AssertionError, r"columns 32 must split into \d+", id="cols_not_split"),
    pytest.param(_bad_rows, AssertionError, r"rows 96 must split into \d+", id="rows_not_split"),
    pytest.param(_bad_rows_mesh, AssertionError, r"rows 64 must split into \d+", id="rows_not_split_mesh"),
    pytest.param(_bad_dim, AssertionError, "dim 1", id="dim_1"),
    pytest.param(_not_tile_aligned, ValueError, "must be tile-aligned", id="not_tile_aligned"),
    pytest.param(_l1_input, ValueError, "input must be DRAM interleaved", id="l1_input"),
    pytest.param(_row_major, AssertionError, r"\[1, 1, 64, 64\]", id="row_major_input"),
    pytest.param(_batched, AssertionError, r"\[2, 1, 64, 64\]", id="batched_input"),
    pytest.param(_output_shape, AssertionError, r"\[1, 1, 32, 64\]", id="output_wrong_shape"),
    pytest.param(_output_dtype, AssertionError, r"\[1, 1, 32, 64\]", id="output_wrong_dtype"),
    pytest.param(_hash_without_cache, AssertionError, "program_hash needs cache_descriptors", id="hash_needs_cache"),
    pytest.param(_prepared_without_hash, AssertionError, "prepared needs program_hash", id="prepared_needs_hash"),
    pytest.param(_unknown_scheme, ValueError, "unknown scheme", id="unknown_scheme"),
    pytest.param(_dual_cycles_on_axis, ValueError, "dual_cycles needs cluster_axis=None", id="dual_cycles_on_axis"),
    pytest.param(_too_many_links, ValueError, "64 requested", id="too_many_links"),
    pytest.param(_ring_not_closed, ValueError, "is not a direct link", id="ring_not_direct_link"),
]


@MESH
@pytest.mark.parametrize(("make", "error", "message"), REFUSALS)
def test_refusal(mesh_device, device_params, fabric_reduce_scatter_op, rs, expect_error, make, error, message):
    """The invalid call is refused with the source's exception type and diagnostic; a valid call right after it is
    still exact. (Whether a one-time setup dispatch ran before the refusal is not part of the contract.)"""
    x, kwargs = make(rs, mesh_device)
    with expect_error(error, message):
        fabric_reduce_scatter_op(x, **kwargs)
    rows, cols = tuple(mesh_device.shape)
    parts = rs.int_parts(rows, cols, _valid_rows(mesh_device), 64, seed=3)
    _run_and_check(fabric_reduce_scatter_op, rs, mesh_device, parts, ttnn.bfloat16, cluster_axis=0)

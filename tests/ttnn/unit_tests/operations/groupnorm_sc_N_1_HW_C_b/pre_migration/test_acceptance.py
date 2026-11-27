# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.operations.groupnorm_sc_N_1_HW_C_b:groupnorm_sc_N_1_HW_C_b``.

Every test calls the operation through the ``groupnorm_sc_N_1_HW_C_b_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (the source op, ``groupnorm_sc_N_1_HW_C_b(input_tensor, num_groups, *, gamma=None, beta=None,
eps=1e-5, in_place=False)``):

* GroupNorm over an (N, 1, HW, C) channels-last activation: each image's C channels in ``num_groups`` groups of
  C / num_groups channels, normalized over HW x C/G, then ``* gamma[c] + beta[c]``. One ``ttnn.generic_op`` per call.
* Inputs: bfloat16 / float32 / bfloat8_b (TILE only), TILE or ROW_MAJOR, any HW and C (tile-aligned or not), DRAM
  interleaved; or bfloat16 L1 BLOCK_SHARDED (ROW_MAJOR orientation, even or ragged shards, HW and C tile-aligned).
  gamma / beta: (1, 1, 1, C), DRAM, sharing one dtype and layout (bf16 / fp32 / bf8b TILE; sharded input: bf16 RM).
* Output: one tensor of the input's shape and dtype. DRAM interleaved input -> TILE, DRAM interleaved. BLOCK_SHARDED
  input -> the input's layout and shard spec in L1. ``in_place=True`` (BLOCK_SHARDED only) writes the result over
  the input's own shards and returns that buffer. Out of place, the input, gamma and beta are never written.
* Refusals: ``ValueError`` for structure (rank != 4, shape[1] != 1, num_groups < 1 or not dividing C, gamma / beta
  not (1, 1, 1, C), eps <= 0, in_place on a non-sharded input); ``UnsupportedAxisValue`` (gamma / beta that differ in
  dtype or layout, an axis value outside SUPPORTED: e.g. beta without gamma, an int dtype, height sharding, a shard
  grid that leaves a core without data); ``ExcludedCell`` (sharded fp32 / bf8b input, sharded non-bf16-RM gamma,
  sharded unaligned HW or C); plain ``NotImplementedError`` for plans the op does not build (no feasible column
  slice; a block-sharded geometry with more than 32 group lanes per statistic unit). No refusal dispatches anything.
* Program cache: every distinct program-defining configuration (shape, num_groups, dtype, layout, affine presence and
  format, placement, grid) is one new entry; fresh buffers, new data and new gamma / beta values reuse it. On the
  source, in_place and eps reuse the out-of-place / first-eps program (asserted as a hit-or-miss, recorded).
* The plan depends on the L1 free at call time: a large sharded call can be refused (NotImplementedError) while other
  L1 tensors are alive. The suite frees each test's tensors before the next (conftest).

Cases: this file (API, refusals, one case per planner branch with a proven cache hit on fresh buffers, cache
sequences, memory effects, consumers) and ``test_golden_cut.py`` (every SUPPORTED axis value, the smallest and the
largest golden INPUTS per topology, golden reference and tolerances).
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

SOURCE_ENTRY = "ttnn.operations.groupnorm_sc_N_1_HW_C_b:groupnorm_sc_N_1_HW_C_b"
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
        if name.startswith("ttnn.operations.groupnorm_sc_N_1_HW_C_b") or name.startswith(
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
                raise AssertionError("native groupnorm_sc_N_1_HW_C_b must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def groupnorm_sc_N_1_HW_C_b_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"groupnorm_sc_N_1_HW_C_b acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch  # noqa: E402

from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C_b.pre_migration import gn_support as gs  # noqa: E402
from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C_b.pre_migration.gn_support import (  # noqa: E402
    BF8,
    BF16,
    F32,
    RM,
    TILE,
    Case,
)
from eval.metrics import default_tolerance  # noqa: E402
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue  # noqa: E402

G8x8, G4x8, G4x4 = (8, 8), (4, 8), (4, 4)

# The golden regression's offset-heavy check runs at the shared bf16 default (eval.metrics.default_tolerance).
OFFSET_TOLERANCE = default_tolerance(BF16)


# --------------------------------------------------------------------------- API


def test_defaults_and_keywords(device, groupnorm_sc_N_1_HW_C_b_op):
    """Defaults (no affine, eps=1e-5, out of place), the explicit spelling of the same call and keyword arguments are
    one call: bitwise-equal outputs, each matching the reference."""
    op = groupnorm_sc_N_1_HW_C_b_op
    p = gs.Prepared(Case((1, 1, 64, 128), 4, affine="no_affine"), device, seed=1)
    outs = [
        op(p.tt_x, 4),
        op(p.tt_x, 4, gamma=None, beta=None, eps=1e-5, in_place=False),
        op(p.tt_x, num_groups=4),
        op(input_tensor=p.tt_x, num_groups=4),
    ]
    gs.expect_dispatches(op)
    for out in outs:
        p.check(out)
    first = ttnn.to_torch(outs[0])
    for out in outs[1:]:
        assert torch.equal(ttnn.to_torch(out), first), "the same call must give the same bits"


def test_affine_and_options_are_keyword_only(device, groupnorm_sc_N_1_HW_C_b_op, expect_error):
    op = groupnorm_sc_N_1_HW_C_b_op
    p = gs.Prepared(Case((1, 1, 64, 128), 4), device, seed=2)
    with expect_error(TypeError, r"(?i)argument"):
        op(p.tt_x, 4, p.tt_gamma)
    out = p.call(op)
    p.check(out)


def test_num_groups_must_be_an_int(device, groupnorm_sc_N_1_HW_C_b_op, expect_error):
    """num_groups is an int: a float, even an integral one, is refused before anything runs. The source raises
    ValueError naming num_groups; a typed binding refuses the conversion with a TypeError naming the arguments."""
    op = groupnorm_sc_N_1_HW_C_b_op
    p = gs.Prepared(Case((1, 1, 64, 128), 4), device, seed=3)
    before = gs.entries(device)
    with expect_error((ValueError, TypeError), r"num_groups|argument"):
        op(p.tt_x, 4.0, **p.kwargs())
    gs.expect_dispatches(op, 0)
    assert gs.entries(device) == before, "a refused call must not build a program"
    p.check(p.call(op))


# --------------------------------------------------------------------------- refusals


def _dram(device, shape, dtype=BF16, layout=TILE, torch_dtype=torch.bfloat16):
    return ttnn.from_torch(
        torch.randn(shape).to(torch_dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _affine(device, C, dtype=BF16, layout=RM, torch_dtype=torch.bfloat16, shape=None):
    return _dram(device, shape or (1, 1, 1, C), dtype=dtype, layout=layout, torch_dtype=torch_dtype)


def _sharded(device, shape, grid, shard, *, layout=TILE, dtype=BF16, torch_dtype=torch.bfloat16, mem=None, cores=None):
    gx, gy = grid
    crs = cores or ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    cfg = ttnn.MemoryConfig(
        mem or ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, list(shard), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        torch.randn(shape).to(torch_dtype), dtype=dtype, layout=layout, device=device, memory_config=cfg
    )


def _x(device):
    return _dram(device, (1, 1, 32, 64))


def _height_sharded(device):
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    mem = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    return _sharded(device, (1, 1, 256, 64), (8, 1), (32, 64), mem=mem, cores=cores)


# (build(device) -> (args, kwargs), exception type, message regex). Each input breaks exactly one rule and passes
# every rule checked before it.
REFUSALS = [
    # Structural ValueErrors.
    pytest.param(lambda d: ((_dram(d, (1, 32, 64)), 2), {}), ValueError, r"rank", id="rank_3"),
    pytest.param(lambda d: ((_dram(d, (1, 2, 32, 64)), 2), {}), ValueError, r"shape\[1\]", id="shape1_not_1"),
    pytest.param(lambda d: ((_x(d), 3), {}), ValueError, r"num_groups", id="num_groups_not_dividing_C"),
    pytest.param(lambda d: ((_x(d), 0), {}), ValueError, r"num_groups", id="num_groups_zero"),
    pytest.param(lambda d: ((_x(d), 2), {"gamma": _affine(d, 32)}), ValueError, r"gamma", id="gamma_not_1x1x1xC"),
    pytest.param(
        lambda d: ((_x(d), 2), {"gamma": _affine(d, 64), "beta": _affine(d, 64, shape=(1, 1, 32, 64))}),
        ValueError,
        r"beta",
        id="beta_not_1x1x1xC",
    ),
    pytest.param(lambda d: ((_x(d), 2), {"eps": 0.0}), ValueError, r"eps", id="eps_zero"),
    pytest.param(lambda d: ((_x(d), 2), {"eps": -1e-5}), ValueError, r"eps", id="eps_negative"),
    pytest.param(lambda d: ((_x(d), 2), {"in_place": True}), ValueError, r"in_place", id="in_place_interleaved"),
    # Typed support refusals (UnsupportedAxisValue: outside SUPPORTED).
    pytest.param(
        lambda d: ((_x(d), 2), {"gamma": _affine(d, 64), "beta": _affine(d, 64, F32, torch_dtype=torch.float32)}),
        UnsupportedAxisValue,
        r"gamma and beta",
        id="gamma_beta_dtype_differ",
    ),
    pytest.param(
        lambda d: ((_x(d), 2), {"gamma": _affine(d, 64), "beta": _affine(d, 64, layout=TILE)}),
        UnsupportedAxisValue,
        r"gamma and beta",
        id="gamma_beta_layout_differ",
    ),
    pytest.param(lambda d: ((_x(d), 2), {"beta": _affine(d, 64)}), UnsupportedAxisValue, r"affine", id="beta_only"),
    pytest.param(
        lambda d: ((_dram(d, (1, 1, 32, 64), ttnn.int32, torch_dtype=torch.int32), 2), {}),
        UnsupportedAxisValue,
        r"dtype",
        id="dtype_int32",
    ),
    pytest.param(lambda d: ((_height_sharded(d), 2), {}), UnsupportedAxisValue, r"memory_layout", id="height_sharded"),
    # C=160 on 4 columns: 40 -> one 64-wide TILE shard, 3 shard columns for 4 grid columns (a core without data).
    pytest.param(
        lambda d: ((_sharded(d, (1, 1, 256, 160), (4, 2), (128, 64)), 32), {}),
        UnsupportedAxisValue,
        r"shard_fit",
        id="shard_fit_illegal",
    ),
    # Typed support refusals (ExcludedCell: inside SUPPORTED, excluded on BLOCK_SHARDED).
    pytest.param(
        lambda d: ((_sharded(d, (1, 1, 256, 256), (2, 2), (128, 128), dtype=F32, torch_dtype=torch.float32), 8), {}),
        ExcludedCell,
        r"unsupported combination",
        id="sharded_float32",
    ),
    pytest.param(
        lambda d: ((_sharded(d, (1, 1, 256, 256), (2, 2), (128, 128), dtype=BF8), 8), {}),
        ExcludedCell,
        r"unsupported combination",
        id="sharded_bfloat8_b",
    ),
    pytest.param(
        lambda d: (
            (_sharded(d, (1, 1, 256, 256), (2, 2), (128, 128)), 8),
            {"gamma": _affine(d, 256, layout=TILE), "beta": _affine(d, 256, layout=TILE)},
        ),
        ExcludedCell,
        r"unsupported combination",
        id="sharded_tile_gamma_beta",
    ),
    pytest.param(
        lambda d: ((_sharded(d, (1, 1, 100, 64), (2, 2), (50, 32), layout=RM), 2), {}),
        ExcludedCell,
        r"unsupported combination",
        id="sharded_hw_non_aligned",
    ),
    # Plans the op does not build (plain NotImplementedError).
    # Cg = 2047: a group span is lcm(2047, 32) = all 65504 channels (2047 tiles), so even one column slice holds
    # 2047-tile per-channel vectors: no column-slice count fits L1.
    pytest.param(
        lambda d: ((_dram(d, (1, 1, 32, 65504)), 32), {}),
        NotImplementedError,
        r"column-slice",
        id="interleaved_no_feasible_column_slice",
    ),
    # C=1280 on 7 columns: 192-wide shards, Cg = 20 does not divide them, so the statistic unit is the whole grid
    # width and holds all 64 groups: more than 32 group lanes.
    pytest.param(
        lambda d: ((_sharded(d, (1, 1, 1024, 1280), (7, 8), (128, 192)), 64), {}),
        NotImplementedError,
        r"block-sharded input",
        id="sharded_more_than_32_group_lanes",
    ),
]


@pytest.mark.parametrize("build, error, message", REFUSALS)
def test_refusal(device, groupnorm_sc_N_1_HW_C_b_op, expect_error, build, error, message):
    op = groupnorm_sc_N_1_HW_C_b_op
    args, kwargs = build(device)
    before = device.num_program_cache_entries()
    with expect_error(error, message):
        op(*args, **kwargs)
    gs.expect_dispatches(op, 0)
    assert device.num_program_cache_entries() == before, "a refused call must not build a program"


# --------------------------------------------------------------------------- planner branches

# One case per program branch the plan takes, reached through the public inputs. Every case runs twice from an empty
# cache: a miss (one new program), then a hit on fresh buffers holding other data, both checked against the
# reference (the first output after the second call, so a hit that clobbers it fails too).
PLANNER_CASES = [
    # --- DRAM interleaved: the output is TILE, DRAM interleaved ---
    pytest.param(Case((1, 1, 32, 32), 1), id="dram_tile_single_tile_g1"),
    pytest.param(Case((1, 1, 64, 128), 4, layout=RM), id="dram_rm_cg32"),
    pytest.param(Case((1, 1, 256, 320), 32), id="dram_tile_cg10_two_spans"),
    pytest.param(Case((2, 1, 128, 192), 8, layout=RM, affine="gamma_only"), id="dram_rm_cg24_n2_gamma_only"),
    pytest.param(Case((1, 1, 1024, 32), 8, affine="no_affine"), id="dram_tile_cg4_no_affine"),
    pytest.param(Case((2, 1, 64, 288), 9), id="dram_tile_odd_group_count"),
    # Just inside the num_groups limit: num_groups == C (one channel per group).
    pytest.param(Case((1, 1, 64, 64), 64), id="dram_tile_num_groups_eq_C"),
    # 256 tile rows of one unit: the combine fans in from more than 32 cores (two gather slot rows).
    pytest.param(Case((1, 1, 8192, 32), 1), id="dram_tile_fan_in_gt_32"),
    # 8 images x up to 32 column slices: more units than cores, several blocks per core.
    pytest.param(Case((8, 1, 64, 1024), 32, layout=RM), id="dram_rm_many_units"),
    pytest.param(Case((1, 1, 64, 4096), 32), id="dram_tile_wide_c"),
    # SDXL VAE norm: ~600-1000 tiles per core on a 64-110 core grid, more than L1 holds: the streamed / partially
    # resident regime.
    pytest.param(Case((1, 1, 262144, 256), 32, eps=1e-6), id="dram_tile_deep_streamed"),
    # The same regime with 4 KB fp32 pages and fp32 gamma / beta (SDXL VAE 65536x512 in fp32).
    pytest.param(
        Case((1, 1, 65536, 512), 32, dtype=F32, affine_dtype=F32, eps=1e-6), id="dram_fp32_tile_deep_streamed"
    ),
    # hw_non_aligned: the last tile row of each image is partial.
    pytest.param(Case((1, 1, 33, 64), 2, layout=RM), id="dram_rm_hw_non_aligned_hr1"),
    pytest.param(Case((2, 1, 100, 128), 4, pad_fill=float("nan")), id="dram_tile_hw_non_aligned_nan_padding"),
    # c_non_aligned: the last column slice is partial.
    pytest.param(Case((1, 1, 256, 330), 33), id="dram_tile_c_non_aligned_3_slices"),
    pytest.param(Case((1, 1, 128, 80), 4, pad_fill=float("nan")), id="dram_tile_c_non_aligned_nan_padding"),
    pytest.param(Case((2, 1, 96, 6), 3, layout=RM), id="dram_rm_c_lt_16"),
    pytest.param(Case((2, 1, 50, 100), 4, layout=RM, affine_dtype=F32), id="dram_rm_hw_and_c_non_aligned_fp32_affine"),
    # Formats.
    pytest.param(
        Case((2, 1, 512, 128), 4, dtype=F32, layout=RM, affine_dtype=F32, affine_layout=TILE),
        id="dram_fp32_rm_fp32_tile_affine",
    ),
    pytest.param(
        Case((1, 1, 47, 320), 32, dtype=F32, affine_dtype=F32, pad_fill=1.0e4), id="dram_fp32_tile_hw_pad_1e4"
    ),
    pytest.param(
        Case((1, 1, 64, 96), 32, dtype=BF8, affine="gamma_only", affine_dtype=BF8, affine_layout=TILE),
        id="dram_bf8b_cg3_gamma_only_bf8b",
    ),
    pytest.param(Case((2, 1, 50, 300), 30, dtype=BF8), id="dram_bf8b_hw_and_c_non_aligned"),
    pytest.param(
        Case((2, 1, 100, 128), 4, layout=RM, affine_dtype=BF8, affine_layout=TILE), id="dram_rm_bf16_bf8b_affine"
    ),
    # bfloat8_b on the largest golden DRAM shape (Mochi per device, HW % 32 = 24): 1088-byte pages change how many
    # tiles stay resident per core against the bf16 plan of the same shape (test_golden_cut dram_max).
    pytest.param(
        Case((4, 1, 6360, 768), 32, dtype=BF8, affine_dtype=BF8, affine_layout=TILE), id="dram_bf8b_mochi_4x6360x768"
    ),
    # Offset-heavy activations (mean 10 >> std 1, the golden regression's distribution and its default bf16
    # tolerance): statistics that lose the variance to cancellation fail here.
    pytest.param(
        Case((1, 1, 4096, 640), 32, pattern="offset10", tolerance=OFFSET_TOLERANCE), id="dram_tile_offset_mean10"
    ),
    # --- L1 BLOCK_SHARDED (bf16, gamma / beta bf16 RM): the output keeps the input's layout and shard spec ---
    pytest.param(Case((1, 1, 4096, 320), 32, layout=RM, grid=G8x8, canary=True), id="bs_rm_8x8_narrow_k4"),
    pytest.param(
        Case((1, 1, 4096, 384), 32, layout=RM, grid=G8x8, in_place=True, canary=True), id="bs_rm_8x8_k2_in_place"
    ),
    pytest.param(Case((2, 1, 256, 1280), 32, grid=G8x8, in_place=True, canary=True), id="bs_tile_8x8_n2_in_place"),
    pytest.param(Case((1, 1, 4096, 640), 32, grid=G4x8, canary=True), id="bs_tile_4x8"),
    pytest.param(
        Case((2, 1, 1024, 640), 32, layout=RM, grid=G4x8, in_place=True, canary=True), id="bs_rm_4x8_n2_in_place"
    ),
    pytest.param(
        Case((1, 1, 256, 640), 32, layout=RM, grid=G4x4, affine="gamma_only", canary=True), id="bs_rm_4x4_gamma_only"
    ),
    pytest.param(
        Case((1, 1, 4096, 960), 32, layout=RM, grid=G8x8, affine="no_affine", canary=True),
        id="bs_rm_8x8_120_wide_no_affine",
    ),
    pytest.param(Case((1, 1, 1024, 1280), 64, layout=RM, grid=G8x8, canary=True), id="bs_rm_8x8_g64"),
    pytest.param(
        Case(
            (1, 1, 1024, 1280), 32, layout=RM, grid=G8x8, in_place=True, pattern="offset10", tolerance=OFFSET_TOLERANCE
        ),
        id="bs_rm_8x8_offset_mean10_in_place",
    ),
    # Ragged: padded last row, padded last column, shards straddling images, groups straddling core columns.
    pytest.param(
        Case((1, 1, 1024, 1280), 32, grid=(8, 7), in_place=True, canary=True), id="bs_tile_8x7_pad_row_in_place"
    ),
    pytest.param(Case((1, 1, 4096, 320), 32, layout=RM, grid=(8, 7), canary=True), id="bs_rm_8x7_pad_row_k4"),
    pytest.param(Case((1, 1, 4096, 640), 32, layout=RM, grid=(6, 8), canary=True), id="bs_rm_6x8_pad_col"),
    pytest.param(Case((2, 1, 256, 1280), 32, grid=(8, 3), canary=True), id="bs_tile_8x3_straddle_images"),
    pytest.param(
        Case((1, 1, 1024, 1280), 32, grid=(7, 8), affine="no_affine", canary=True),
        id="bs_tile_7x8_groups_straddle_columns",
    ),
    pytest.param(
        Case((1, 1, 1024, 1280), 32, layout=RM, grid=(7, 8), in_place=True, canary=True),
        id="bs_rm_7x8_groups_straddle_columns_in_place",
    ),
]


@pytest.mark.parametrize("case", PLANNER_CASES)
def test_planner_branch(device, groupnorm_sc_N_1_HW_C_b_op, case):
    op = groupnorm_sc_N_1_HW_C_b_op
    gs.fresh_cache(device)
    first = gs.Prepared(case, device, seed=11)
    n0 = gs.entries(device)
    out_first = first.call(op)
    n1 = gs.entries(device)
    second = gs.Prepared(case, device, seed=12)  # fresh buffers and data; `first` and its output stay alive
    n2 = gs.entries(device)
    out_second = second.call(op)
    n3 = gs.entries(device)
    assert n1 - n0 == 1, f"the first call of a configuration builds one program, got {n1 - n0}"
    assert n3 - n2 == 0, f"a call with fresh buffers must reuse the program, got {n3 - n2} new entries"
    assert second.tt_x.buffer_address() != first.tt_x.buffer_address()
    first.check(out_first)
    second.check(out_second)


def test_bs_rm_in_place_l1_tight(device, groupnorm_sc_N_1_HW_C_b_op):
    """SDXL unet_rm_ip_16384x640: a 320 KB per-core RM shard normalized in place. Only part of its tilized copy fits
    in L1 beside it, so the head chunks are re-tilized from the shard in pass 2 and the output rows are written back
    over rows already consumed. The op plans against the L1 free at call time, so each call runs with no other shard
    of this size alive: the second call's fresh input is made after the first one is freed (it may reuse the freed
    address; its data differ)."""
    op = groupnorm_sc_N_1_HW_C_b_op
    case = Case((1, 1, 16384, 640), 32, layout=RM, grid=G8x8, in_place=True)
    gs.fresh_cache(device)
    first = gs.Prepared(case, device, seed=90)
    n0 = gs.entries(device)
    out = first.call(op)
    n1 = gs.entries(device)
    first.check(out)
    ttnn.deallocate(out, force=True)
    del first, out
    second = gs.Prepared(case, device, seed=91)
    n2 = gs.entries(device)
    out = second.call(op)
    n3 = gs.entries(device)
    assert (n1 - n0, n3 - n2) == (1, 0), f"miss then hit expected, got {(n1 - n0, n3 - n2)}"
    second.check(out)


# --------------------------------------------------------------------------- program cache sequences

BASE = Case((1, 1, 128, 256), 8)
BS_RM = Case((1, 1, 1024, 1280), 32, layout=RM, grid=G8x8)
BS_TILE = Case((1, 1, 1024, 1280), 32, grid=G8x8)

# (A, B): B changes one program-defining field of A. A, B, A, B from an empty cache: each first visit builds one
# program (the field changes the kernels, the circular-buffer set or the tensor list), each return reuses it.
# in_place is not one of them on the source (it only moves the output circular buffers onto the input's shards):
# test_cache_alias_transitions covers it.
FIELD_CHANGES = [
    pytest.param(BASE, BASE.with_(num_groups=16), id="num_groups"),
    pytest.param(BASE.with_(affine="no_affine"), BASE, id="affine_absent_present"),
    pytest.param(BASE.with_(affine="gamma_only"), BASE, id="beta_absent_present"),
    pytest.param(BASE, BASE.with_(affine_dtype=F32, affine_layout=TILE), id="affine_format"),
    pytest.param(BASE, BASE.with_(layout=RM), id="layout"),
    pytest.param(BASE, BASE.with_(dtype=F32), id="dtype"),
    pytest.param(BASE, BASE.with_(shape=(1, 1, 120, 256)), id="shape_hw"),
    pytest.param(BASE, BASE.with_(shape=(2, 1, 128, 256)), id="shape_n"),
    pytest.param(BS_TILE.with_(grid=None), BS_TILE, id="dram_vs_block_sharded"),
    pytest.param(
        Case((1, 1, 4096, 640), 32, layout=RM, grid=G4x8), Case((1, 1, 4096, 640), 32, layout=RM, grid=G8x8), id="grid"
    ),
]


@pytest.mark.parametrize("a, b", FIELD_CHANGES)
def test_cache_program_defining_field(device, groupnorm_sc_N_1_HW_C_b_op, a, b):
    op = groupnorm_sc_N_1_HW_C_b_op
    gs.fresh_cache(device)
    deltas, runs = [], []
    for seed, case in enumerate([a, b, a, b]):
        p = gs.Prepared(case, device, seed=20 + seed)
        before = gs.entries(device)
        out = p.call(op)
        deltas.append(gs.entries(device) - before)
        runs.append((p, out))
    assert deltas == [1, 1, 0, 0], f"A, B, A, B cache deltas {deltas}"
    for p, out in runs:
        p.check(out)


def test_cache_eps_scalar(device, groupnorm_sc_N_1_HW_C_b_op, record_property):
    """eps a -> b -> a -> b on the same input buffers: every output follows the eps of its own call. Returning to an
    eps already seen reuses its program."""
    op = groupnorm_sc_N_1_HW_C_b_op
    case = Case((1, 1, 256, 320), 32)
    gs.fresh_cache(device)
    p = gs.Prepared(case, device, seed=30)
    eps_values = [1e-5, 0.25, 1e-5, 0.25]
    refs = {e: gs.reference(case, p.x_host, p.gamma_host, p.beta_host, eps=e) for e in set(eps_values)}
    gap = (refs[0.25].float() - refs[1e-5].float()).abs().max().item()
    assert gap > 0.1, "the two eps values must give visibly different results"
    deltas, outs = [], []
    for e in eps_values:
        before = gs.entries(device)
        outs.append(p.call(op, eps=e))
        deltas.append(gs.entries(device) - before)
    record_property("eps_change_delta", deltas[1])
    assert deltas[0] == 1 and deltas[2:] == [0, 0], f"eps a, b, a, b cache deltas {deltas}"
    assert deltas[1] in (0, 1)
    for e, out in zip(eps_values, outs):
        p.check(out, expected=refs[e])


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(Case((1, 1, 256, 320), 32, layout=RM), id="dram_rm"),
        pytest.param(BS_TILE.with_(in_place=True, canary=True), id="bs_tile_in_place"),
        pytest.param(Case((2, 1, 256, 1280), 32, layout=RM, grid=(8, 3), canary=True), id="bs_rm_ragged"),
    ],
)
def test_cache_fresh_buffers_and_values(device, groupnorm_sc_N_1_HW_C_b_op, case):
    """A -> new A -> new A: the same metadata on new live buffers (input, gamma, beta, output) with other data; each
    output follows its own call's tensors."""
    op = groupnorm_sc_N_1_HW_C_b_op
    gs.fresh_cache(device)
    runs, deltas = [], []
    for seed in (40, 41, 42):
        p = gs.Prepared(case, device, seed=seed)
        before = gs.entries(device)
        out = p.call(op)
        deltas.append(gs.entries(device) - before)
        runs.append((p, out))
    assert deltas == [1, 0, 0], f"A, new A, new A cache deltas {deltas}"
    for buffers in (lambda p: p.tt_x, lambda p: p.tt_gamma, lambda p: p.tt_beta):
        addresses = {buffers(p).buffer_address() for p, _ in runs}
        assert len(addresses) == len(runs), "every call must run on its own live buffers"
    for p, out in runs:
        p.check(out)


@pytest.mark.parametrize("base", [pytest.param(Case((1, 1, 256, 320), 32), id="dram"), pytest.param(BS_TILE, id="bs")])
def test_cache_optional_absent_present_absent(device, groupnorm_sc_N_1_HW_C_b_op, base):
    op = groupnorm_sc_N_1_HW_C_b_op
    gs.fresh_cache(device)
    sequence = ["no_affine", "gamma_only", "gamma_beta", "no_affine", "gamma_only", "gamma_beta", "no_affine"]
    deltas, runs = [], []
    for seed, affine in enumerate(sequence):
        p = gs.Prepared(base.with_(affine=affine), device, seed=50 + seed)
        before = gs.entries(device)
        out = p.call(op)
        deltas.append(gs.entries(device) - before)
        runs.append((p, out))
    assert deltas == [1, 1, 1, 0, 0, 0, 0], f"affine sequence cache deltas {deltas}"
    for p, out in runs:
        p.check(out)


@pytest.mark.parametrize("layout", [pytest.param(TILE, id="tile"), pytest.param(RM, id="rm")])
@pytest.mark.parametrize(
    "start_in_place", [pytest.param(False, id="separate_first"), pytest.param(True, id="shared_first")]
)
def test_cache_alias_transitions(device, groupnorm_sc_N_1_HW_C_b_op, layout, start_in_place, record_property):
    """separate -> shared -> separate -> shared (and starting shared), on fresh live buffers each call: the in-place
    output is the input's own buffer, the out-of-place one never is, and every output is right.

    The source keeps one program for both (in_place only moves the output circular buffers onto the input's shards;
    the generic program key excludes buffer addresses), so its switch is a hit that must re-point those buffers. A
    port may specialize in_place instead (switch delta 1, recorded); either way every return reuses its program."""
    op = groupnorm_sc_N_1_HW_C_b_op
    gs.fresh_cache(device)
    flags = [start_in_place, not start_in_place, start_in_place, not start_in_place]
    deltas, runs = [], []
    for seed, in_place in enumerate(flags):
        p = gs.Prepared(BS_TILE.with_(layout=layout, in_place=in_place, canary=True), device, seed=60 + seed)
        before = gs.entries(device)
        out = p.call(op)
        deltas.append(gs.entries(device) - before)
        runs.append((p, out))
    record_property("alias_switch_delta", deltas[1])
    assert deltas[0] == 1 and deltas[1] in (0, 1) and deltas[2:] == [0, 0], f"alias transition cache deltas {deltas}"
    for p, out in runs:
        p.check(out)


@pytest.mark.parametrize(
    "case, invalid",
    [
        pytest.param(
            Case((1, 1, 64, 128), 4),
            [
                ({"eps": 0.0}, ValueError, r"eps"),
                ({"num_groups": 3}, ValueError, r"num_groups"),
                ({"gamma": None}, UnsupportedAxisValue, r"affine"),
                ({"in_place": True}, ValueError, r"in_place"),
            ],
            id="dram",
        ),
        pytest.param(
            BS_RM,
            [
                ({"affine": (F32, RM)}, ExcludedCell, r"unsupported combination"),
                ({"num_groups": 0}, ValueError, r"num_groups"),
            ],
            id="bs",
        ),
    ],
)
def test_cache_valid_invalid_valid(device, groupnorm_sc_N_1_HW_C_b_op, expect_error, case, invalid):
    """A warmed program does not let an invalid call through, and the next valid call still hits and is right."""
    op = groupnorm_sc_N_1_HW_C_b_op
    gs.fresh_cache(device)
    warm = gs.Prepared(case, device, seed=70)
    out_warm = warm.call(op)
    assert gs.entries(device) == 1
    for change, error, message in invalid:
        kwargs = warm.kwargs()
        num_groups = change.get("num_groups", case.num_groups)
        if "affine" in change:
            dtype, layout = change["affine"]
            C = case.shape[-1]
            kwargs["gamma"] = _affine(device, C, dtype, layout, torch_dtype=torch.float32)
            kwargs["beta"] = _affine(device, C, dtype, layout, torch_dtype=torch.float32)
        kwargs.update({k: v for k, v in change.items() if k not in ("num_groups", "affine")})
        with expect_error(error, message):
            op(warm.tt_x, num_groups, **kwargs)
        gs.expect_dispatches(op, 0)
        assert gs.entries(device) == 1, "a refused call must not build a program"
    again = gs.Prepared(case, device, seed=71)
    out_again = again.call(op)
    assert gs.entries(device) == 1, "the valid call after the refusals must hit"
    warm.check(out_warm)
    again.check(out_again)


# --------------------------------------------------------------------------- consumers


def test_output_feeds_other_ops(device, groupnorm_sc_N_1_HW_C_b_op):
    """The DRAM output is a regular TILE tensor: ttnn.silu consumes it, and the op takes it back as its own input."""
    op = groupnorm_sc_N_1_HW_C_b_op
    case = Case((1, 1, 256, 320), 32)
    p = gs.Prepared(case, device, seed=80)
    out = p.call(op)
    p.check(out)
    activated = ttnn.to_torch(ttnn.silu(out)).float()
    expected_silu = torch.nn.functional.silu(ttnn.to_torch(out).float())
    assert torch.allclose(activated, expected_silu, rtol=0.02, atol=0.02)
    again = op(out, case.num_groups, gamma=p.tt_gamma, beta=p.tt_beta)
    gs.expect_dispatches(op)
    expected_again = gs.reference(case, ttnn.to_torch(out), p.gamma_host, p.beta_host)
    gs.check_output(
        again, expected_again, shape=list(case.shape), dtype=BF16, expected_layout=TILE, tolerance=gs.TOLERANCES[BF16]
    )


def test_in_place_chain(device, groupnorm_sc_N_1_HW_C_b_op):
    """The SDXL resnet pattern: two in-place norms on one block-sharded RM buffer, then an interleaved copy."""
    op = groupnorm_sc_N_1_HW_C_b_op
    case = BS_RM.with_(in_place=True, canary=True)
    p = gs.Prepared(case, device, seed=81)
    out = p.call(op)
    p.check(out)
    first_result = ttnn.to_torch(out)
    out2 = op(out, case.num_groups, gamma=p.tt_gamma, beta=p.tt_beta, in_place=True)
    gs.expect_dispatches(op)
    assert out2.buffer_address() == p.in_addr
    expected2 = gs.reference(case, first_result, p.gamma_host, p.beta_host)
    gs.check_output(
        out2, expected2, shape=list(case.shape), dtype=BF16, expected_layout=RM, tolerance=gs.TOLERANCES[BF16]
    )
    interleaved = ttnn.to_memory_config(out2, ttnn.DRAM_MEMORY_CONFIG)
    assert torch.equal(ttnn.to_torch(interleaved), ttnn.to_torch(out2))
    assert torch.equal(ttnn.to_torch(p.canary), p.canary_host)

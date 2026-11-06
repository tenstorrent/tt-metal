# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.operations.distributed_rms_norm:distributed_rms_norm``.

Every test calls the operation through the ``distributed_rms_norm_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (the source's public signature and its refusals; nothing about its planner):

    distributed_rms_norm(input_tensor, *, cluster_axis, gamma=None, epsilon=1e-6, topology=Linear, num_links=None,
                         compute_kernel_config=None, memory_config=None, exchange_mode=None) -> ttnn.Tensor

- A multi-device op: the G devices of a group along ``cluster_axis`` hold the hidden dim W = G * W_local and device p
  returns its slice of ``rms_norm(concat_p x[p], (W,), gamma, eps)`` with x's per-device shape, dtype and TILE layout.
  Checked per device against a torch reference (the golden PCC floors, plus a median got/true ratio within 1%,
  since PCC cannot see a per-row scale error such as a wrong eps or G). The input is never written.
- Supported: dtype {float32, bfloat16, bfloat8_b}, TILE layout, rank 2-4, tile-aligned or not; gamma absent or
  {float32, bfloat16} as TILE (1,1,1,W_local) or ROW_MAJOR sticks (1,1,ceil(W_local/32),32); input INTERLEAVED
  (DRAM / L1), HEIGHT_SHARDED or WIDTH_SHARDED; Linear, or Ring (a real ring only for G >= 3 on a wrapping fabric);
  num_links None (= every usable link) or 1..usable; fp32_dest_acc_en True/False; exchange_mode None / "direct" /
  "relay" / "mcast" (1D fabric only), every mode giving the same output.
- Output placement: ``memory_config`` when given (interleaved or sharded with a shard spec, on any grid), else
  DRAM for an interleaved input and the input's own memory config for a sharded one.
- One program per call: the source makes exactly one ``ttnn.generic_op`` dispatch (``last_dispatches == 1``).
- Caller errors raise ValueError (accepted as RuntimeError on the port, as the golden spec does): rank < 2, a gamma
  of the wrong per-device shape, cluster_axis not 0/1 or of size 1, Ring over a fabric that does not wrap that axis,
  num_links outside [1, usable], an unknown exchange_mode.
- Support refusals raise NotImplementedError (UnsupportedAxisValue / ExcludedCell on the source): a dtype, layout,
  rank, gamma dtype or input/output memory layout outside SUPPORTED, float32 input with fp32_dest_acc_en=False, a
  sharded output memory_config without a shard spec, exchange_mode="mcast" on a 2D fabric, a non-ROW_MAJOR shard
  orientation, and a row too wide for L1.
- Program cache: fresh input/gamma/output buffers and new data hit the program of the first call; returning to an
  earlier configuration (gamma presence/layout, dtype, memory layout, exchange mode, epsilon, num_links, shape) hits
  its program again; a refused call between two valid ones leaves the cache usable.

Runs on the fabric mesh of ``conftest.py`` (``drms_mesh``): the system mesh under FABRIC_2D_TORUS_XY, and a 1 x N
line of all devices under FABRIC_2D_TORUS_XY (G = N, and a real ring) and under FABRIC_1D (mcast, non-wrapping).
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

SOURCE_ENTRY = "ttnn.operations.distributed_rms_norm:distributed_rms_norm"
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
        if name.startswith("ttnn.operations.distributed_rms_norm") or name.startswith("ttnn._ttnn.operations.generic"):
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
                raise AssertionError("native distributed_rms_norm must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def distributed_rms_norm_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"distributed_rms_norm acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch

from tests.ttnn.unit_tests.operations.distributed_rms_norm.pre_migration import drms_pre_migration_helpers as H
from tests.ttnn.unit_tests.operations.distributed_rms_norm.pre_migration.drms_pre_migration_helpers import Case

RING = ttnn.Topology.Ring
BF16, FP32, BF8B = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT


def _one_program(op):
    """The source plans one MeshProgramDescriptor and dispatches it once per call (0 on the native route)."""
    if op.mode == "source":
        assert op.last_dispatches == 1, f"expected one ttnn.generic_op dispatch, saw {op.last_dispatches}"


def _entries(mesh):
    return mesh.num_program_cache_entries()


def _empty_cache(mesh):
    ttnn.synchronize_device(mesh)
    mesh.clear_program_cache()
    mesh.enable_program_cache()
    assert _entries(mesh) == 0


def _run(op, call, mesh):
    """One call; returns (output, cache entries it added). Counted around the op only (setup and readback excluded)."""
    before = _entries(mesh)
    out = op(call.tx, **call.kwargs)
    added = _entries(mesh) - before
    _one_program(op)
    return out, added


def _addresses(*tensors):
    return [t.buffer_address() for t in tensors if t is not None]


# --------------------------------------------------------------------------- successful calls, one per branch
# Each is checked against the torch reference on a first call, then on a second call with fresh input / gamma /
# output buffers and new data while the first call's tensors stay alive: a program-cache hit (no new entry) whose
# result follows the current buffers.

SUCCESS_CASES = [
    # smallest call: one tile per device, gamma TILE bf16
    pytest.param(Case((1, 1, 32, 32)), id="single_tile"),
    # the other mesh axis, no gamma, non-default epsilon
    pytest.param(Case((1, 1, 64, 128), cluster_axis=0, gamma=None, epsilon=1e-5), id="axis0_no_gamma_eps"),
    # decode: one tile-row wide enough for the W-split local plan (1 wave)
    pytest.param(Case((1, 1, 32, 1792)), id="decode_wsplit"),
    # W-split, several waves of tile-rows; gamma as ROW_MAJOR sticks
    pytest.param(Case((1, 1, 640, 1792), cluster_axis=0, gamma=(BF16, RM)), id="wsplit_two_waves_sticks"),
    # prefill: many rows, the row-parallel local plan; fp32 gamma
    pytest.param(Case((1, 1, 2048, 1280), gamma=(FP32, TILE)), id="prefill_row_parallel"),
    # more rows than one resident block holds on the grid (fp32)
    pytest.param(Case((1, 1, 4096, 1024), dtype=FP32, gamma=(FP32, TILE)), id="fp32_multi_block"),
    # both dims off the tile grid; fp32 sticks
    pytest.param(Case((1, 1, 17, 50), gamma=(FP32, RM)), id="both_non_aligned_fp32_sticks"),
    # W off the tile grid in bfloat8_b: the pad must stay out of the sum of squares
    pytest.param(Case((1, 1, 64, 1000), cluster_axis=0, dtype=BF8B, gamma=(BF16, RM)), id="w_non_aligned_bf8b"),
    # H off the tile grid, 16-bit DEST
    pytest.param(Case((1, 1, 100, 512), fp32_dest_acc_en=False), id="h_non_aligned_dest16"),
    pytest.param(Case((128, 256), dtype=FP32, gamma=None), id="rank2_fp32"),
    pytest.param(Case((1, 32, 1792), cluster_axis=0), id="rank3"),
    pytest.param(Case((2, 4, 128, 512), dtype=BF8B, fp32_dest_acc_en=False), id="rank4_batched_bf8b_dest16"),
    # a long batched sequence: many waves of exchange records per call
    pytest.param(Case((4, 1, 4096, 1792)), id="long_sequence_batched"),
    # interleaved L1 input: the output still defaults to DRAM
    pytest.param(Case((1, 1, 64, 512), in_mem="l1"), id="l1_interleaved_in"),
    # sharded inputs: the output defaults to the input's own shard
    pytest.param(Case((1, 1, 640, 1792), in_mem="height"), id="height_sharded"),
    pytest.param(Case((1, 1, 32, 1792), in_mem="width"), id="width_sharded"),
    # one tile-row: the whole height shard on a single core (kept small: the source plans from the L1 left free, so
    # a large shard whose first call's tensors stay live re-plans the second call and compiles again)
    pytest.param(Case((1, 1, 32, 1024), in_mem="height"), id="height_sharded_one_core"),
    # output placement independent of the input's
    pytest.param(Case((1, 1, 32, 1792), in_mem="width", out_mem="dram"), id="width_to_dram"),
    pytest.param(Case((1, 1, 640, 1792), in_mem="height", out_mem="dram"), id="height_to_dram"),
    pytest.param(Case((1, 1, 32, 1792), out_mem="width_8x3"), id="dram_to_width_8x3"),
    pytest.param(Case((1, 1, 64, 1000), out_mem="l1"), id="dram_to_l1"),
    # explicit options on the system mesh
    pytest.param(Case((1, 1, 640, 1792), num_links=1), id="num_links_1"),
    pytest.param(Case((1, 1, 64, 128), topology=RING), id="ring_of_two_is_a_line"),
    pytest.param(Case((1, 1, 32, 1792), exchange_mode="direct"), id="exchange_direct"),
    pytest.param(Case((1, 1, 640, 1792), exchange_mode="relay"), id="exchange_relay"),
    # a group of all N devices: Linear, and a real ring (FABRIC_2D_TORUS_XY wraps the 1 x N line)
    pytest.param(Case((1, 1, 64, 128), mesh="line"), id="line_linear"),
    pytest.param(Case((1, 1, 640, 1792), topology=RING, mesh="line"), id="line_ring"),
    pytest.param(
        Case((1, 1, 32, 1792), topology=RING, num_links=1, exchange_mode="relay", mesh="line"),
        id="line_ring_relay_one_link",
    ),
    # the 1D-fabric-only multicast schedule, and the default schedule on a 1D fabric
    pytest.param(Case((1, 1, 640, 1792), exchange_mode="mcast", mesh="line_1d"), id="line_1d_mcast"),
    pytest.param(Case((1, 1, 32, 32), gamma=None, exchange_mode="mcast", mesh="line_1d"), id="line_1d_mcast_tiny"),
    pytest.param(Case((1, 1, 32, 1792), mesh="line_1d"), id="line_1d_default"),
]


@pytest.mark.parametrize("case", SUCCESS_CASES)
def test_call_matches_reference(case, drms_mesh, distributed_rms_norm_op):
    op = distributed_rms_norm_op
    first = H.build(case, drms_mesh, seed=1)
    out1, _ = _run(op, first, drms_mesh)
    H.check_output(out1, first, drms_mesh)
    H.check_input_untouched(first, drms_mesh)

    second = H.build(case, drms_mesh, seed=2)
    assert not torch.equal(second.x, first.x)
    out2, added = _run(op, second, drms_mesh)
    assert added == 0, f"a call on fresh buffers of the same configuration added {added} program(s)"
    assert out2.buffer_address() != out1.buffer_address(), "the second call must return a fresh output buffer"
    assert second.tx.buffer_address() != first.tx.buffer_address()
    H.check_output(out2, second, drms_mesh)
    H.check_input_untouched(second, drms_mesh)
    # the first output is the first call's alone
    H.check_output(out1, first, drms_mesh)


# --------------------------------------------------------------------------- program cache sequences


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(Case((1, 1, 32, 1792)), id="interleaved"),
        # three calls stay live: a shard small enough that they leave the planner the same L1
        pytest.param(Case((1, 1, 320, 1792), in_mem="height"), id="height_sharded"),
        pytest.param(Case((1, 1, 32, 1792), in_mem="width", out_mem="width_8x3"), id="width_to_width_regrid"),
        pytest.param(Case((1, 1, 64, 1000), gamma=(FP32, RM), out_mem="l1"), id="sticks_to_l1"),
    ],
)
def test_cache_miss_then_hits_on_fresh_buffers(case, drms_mesh, distributed_rms_norm_op):
    """From an empty cache: a miss, then two hits on fresh live buffers with new data; every result is current."""
    op = distributed_rms_norm_op
    _empty_cache(drms_mesh)
    calls, outs = [], []
    for i, seed in enumerate((11, 12, 13)):
        call = H.build(case, drms_mesh, seed=seed)
        out, added = _run(op, call, drms_mesh)
        if i == 0:
            assert added >= 1, "the first call on an empty cache must compile its program (is caching enabled?)"
        else:
            assert added == 0, f"call {i} on fresh buffers added {added} program(s)"
        calls.append(call)
        outs.append(out)
    for name, tensors in (
        ("input", [c.tx for c in calls]),
        ("gamma", [c.tg for c in calls]),
        ("output", outs),
    ):
        addrs = _addresses(*tensors)
        assert len(set(addrs)) == len(addrs), f"{name} buffers of live calls must differ: {addrs}"
    for call, out in zip(calls, outs):
        H.check_output(out, call, drms_mesh)


# (A, B, B must compile its own program). A -> B -> A -> B, each on fresh buffers: returning to a configuration
# reuses its program. B is asserted to miss only where it plainly changes the kernels or their data formats; for
# epsilon, num_links, shape and output buffer type it is recorded, since a port may pass them at runtime.
VARIANTS = [
    pytest.param(Case((1, 1, 64, 512), gamma=None), Case((1, 1, 64, 512)), True, id="gamma_absent_present"),
    pytest.param(Case((1, 1, 64, 512)), Case((1, 1, 64, 512), gamma=(BF16, RM)), True, id="gamma_tile_sticks"),
    pytest.param(Case((1, 1, 64, 512)), Case((1, 1, 64, 512), gamma=(FP32, TILE)), True, id="gamma_bf16_fp32"),
    pytest.param(Case((1, 1, 64, 512)), Case((1, 1, 64, 512), dtype=FP32), True, id="dtype_bf16_fp32"),
    pytest.param(Case((1, 1, 64, 512)), Case((1, 1, 64, 512), fp32_dest_acc_en=False), True, id="dest_fp32_16"),
    pytest.param(Case((1, 1, 32, 1792)), Case((1, 1, 32, 1792), in_mem="width"), True, id="interleaved_width"),
    pytest.param(
        Case((1, 1, 640, 1792), exchange_mode="direct"),
        Case((1, 1, 640, 1792), exchange_mode="relay"),
        True,
        id="exchange_direct_relay",
    ),
    pytest.param(Case((1, 1, 64, 512)), Case((1, 1, 64, 512), epsilon=0.5), False, id="epsilon"),
    pytest.param(Case((1, 1, 640, 1792)), Case((1, 1, 640, 1792), num_links=1), False, id="num_links_all_1"),
    pytest.param(Case((1, 1, 32, 1792)), Case((1, 1, 640, 1792)), False, id="shape"),
    pytest.param(Case((1, 1, 64, 512)), Case((1, 1, 64, 512), out_mem="l1"), False, id="output_dram_l1"),
]


@pytest.mark.parametrize(("a", "b", "b_misses"), VARIANTS)
def test_cache_returns_to_each_variant(a, b, b_misses, drms_mesh, distributed_rms_norm_op, record_property):
    op = distributed_rms_norm_op
    _empty_cache(drms_mesh)
    live = []
    for step, (case, seed) in enumerate(((a, 21), (b, 22), (a, 23), (b, 24))):
        call = H.build(case, drms_mesh, seed=seed)
        out, added = _run(op, call, drms_mesh)
        if step == 0:
            assert added >= 1, "the first call on an empty cache must compile its program"
        elif step == 1:
            record_property("b_added", added)
            if b_misses:
                assert added >= 1, "B changes the program, so it must not reuse A's"
        else:
            assert added == 0, f"step {step} returns to a cached configuration but added {added} program(s)"
        H.check_output(out, call, drms_mesh)
        live.append((call, out))
    # the older outputs still hold their own call's result
    for call, out in live:
        H.check_output(out, call, drms_mesh)


def test_cache_survives_refusals(drms_mesh, distributed_rms_norm_op, expect_error):
    """valid -> invalid -> valid: a warmed configuration still refuses invalid calls, which add no program, and the
    next valid call hits and is correct."""
    op = distributed_rms_norm_op
    case = Case((1, 1, 64, 512))
    _empty_cache(drms_mesh)
    warm = H.build(case, drms_mesh, seed=31)
    out, added = _run(op, warm, drms_mesh)
    assert added >= 1
    H.check_output(out, warm, drms_mesh)

    before = _entries(drms_mesh)
    with expect_error((ValueError, RuntimeError), "num_links"):
        op(warm.tx, **{**warm.kwargs, "num_links": 99})
    with expect_error((ValueError, RuntimeError), "exchange_mode"):
        op(warm.tx, **{**warm.kwargs, "exchange_mode": "bogus"})
    with expect_error(NotImplementedError, "mcast"):
        op(warm.tx, **{**warm.kwargs, "exchange_mode": "mcast"})
    assert _entries(drms_mesh) == before, "a refused call must not add a program"

    again = H.build(case, drms_mesh, seed=32)
    out2, added = _run(op, again, drms_mesh)
    assert added == 0
    H.check_output(out2, again, drms_mesh)


@pytest.mark.parametrize(
    ("mesh_cfg", "topology", "modes"),
    [
        pytest.param("system", None, ("direct", "relay"), id="system_linear"),
        pytest.param("line", RING, ("direct", "relay"), id="line_torus_ring"),
        pytest.param("line_1d", None, ("direct", "relay", "mcast"), id="line_1d_linear"),
        pytest.param("ring_1d", RING, ("direct", "relay", "mcast"), id="ring_1d_ring"),
    ],
)
def test_exchange_modes_are_bit_identical(mesh_cfg, topology, modes, drms_mesh, distributed_rms_norm_op):
    """Every exchange schedule computes the same result: each mode matches the reference and gives the same bits
    as "direct", on the same input."""
    op = distributed_rms_norm_op
    base = Case((1, 1, 640, 1792), topology=topology, mesh=mesh_cfg)
    call = H.build(base, drms_mesh, seed=41)
    outs = {}
    for mode in modes:
        moded = H.Call(
            base.but(exchange_mode=mode), call.x, call.gamma, call.tx, call.tg, {**call.kwargs, "exchange_mode": mode}
        )
        out, _ = _run(op, moded, drms_mesh)
        outs[mode] = H.check_output(out, moded, drms_mesh)
    for mode, got in outs.items():
        for r, row in enumerate(got):
            for c, actual in enumerate(row):
                assert torch.equal(actual, outs["direct"][r][c]), f"{mode} differs from direct on device ({r},{c})"


# --------------------------------------------------------------------------- refusals


def _mesh_input(mesh, shape, *, dtype=BF16, layout=TILE, mem=None, seed=0):
    torch.manual_seed(seed)
    rows, cols = tuple(mesh.shape)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32)
    return H.to_mesh(x, mesh, dtype=dtype, layout=layout, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG)


def _gamma_of_shape(mesh, shape, *, dtype=BF16, layout=TILE):
    rows, cols = tuple(mesh.shape)
    return H.to_mesh(torch.randn((rows, cols, *shape)), mesh, dtype=dtype, layout=layout)


def _rank1(mesh):
    return ttnn.from_torch(
        torch.randn(64),
        dtype=BF16,
        layout=TILE,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _block_sharded(shape):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shape[-2] // 2, shape[-1] // 2], ttnn.ShardOrientation.ROW_MAJOR),
    )


S = (1, 1, 64, 256)

# (id, mesh_cfg, error, message pattern, build(mesh) -> (input, kwargs))
REFUSALS = [
    # caller errors (ValueError on the source)
    ("rank_1", "system", (ValueError, RuntimeError), "rank", lambda m: (_rank1(m), {"cluster_axis": 1})),
    (
        "gamma_tile_wrong_width",
        "system",
        (ValueError, RuntimeError),
        "gamma",
        lambda m: (_mesh_input(m, S), {"cluster_axis": 1, "gamma": _gamma_of_shape(m, (1, 1, 1, 128))}),
    ),
    (
        "gamma_row_major_not_sticks",
        "system",
        (ValueError, RuntimeError),
        "gamma",
        lambda m: (
            _mesh_input(m, S),
            {"cluster_axis": 1, "gamma": _gamma_of_shape(m, (1, 1, 1, 256), layout=RM)},
        ),
    ),
    (
        "cluster_axis_2",
        "system",
        (ValueError, RuntimeError),
        "cluster_axis",
        lambda m: (_mesh_input(m, S), {"cluster_axis": 2}),
    ),
    (
        "cluster_axis_of_size_1",
        "line",
        (ValueError, RuntimeError),
        "cluster_axis",
        lambda m: (_mesh_input(m, S), {"cluster_axis": 0}),
    ),
    (
        "num_links_0",
        "system",
        (ValueError, RuntimeError),
        "num_links",
        lambda m: (_mesh_input(m, S), {"cluster_axis": 1, "num_links": 0}),
    ),
    (
        "num_links_above_usable",
        "system",
        (ValueError, RuntimeError),
        "num_links",
        lambda m: (_mesh_input(m, S), {"cluster_axis": 1, "num_links": 99}),
    ),
    (
        "unknown_exchange_mode",
        "system",
        (ValueError, RuntimeError),
        "exchange_mode",
        lambda m: (_mesh_input(m, S), {"cluster_axis": 1, "exchange_mode": "broadcast"}),
    ),
    (
        "ring_on_non_wrapping_fabric",
        "line_1d",
        (ValueError, RuntimeError),
        "(?i)ring",
        lambda m: (_mesh_input(m, S), {"cluster_axis": 1, "topology": RING}),
    ),
    # support refusals (NotImplementedError: UnsupportedAxisValue / ExcludedCell on the source)
    (
        "dtype_bfloat4_b",
        "system",
        NotImplementedError,
        "dtype",
        lambda m: (_mesh_input(m, S, dtype=ttnn.bfloat4_b), {"cluster_axis": 1}),
    ),
    (
        "row_major_input",
        "system",
        NotImplementedError,
        "layout",
        lambda m: (_mesh_input(m, S, layout=RM), {"cluster_axis": 1}),
    ),
    (
        "rank_5",
        "system",
        NotImplementedError,
        "rank",
        lambda m: (_mesh_input(m, (1, 1, 1, 64, 256)), {"cluster_axis": 1}),
    ),
    (
        "gamma_bfloat8_b",
        "system",
        NotImplementedError,
        "gamma",
        lambda m: (_mesh_input(m, S), {"cluster_axis": 1, "gamma": _gamma_of_shape(m, (1, 1, 1, 256), dtype=BF8B)}),
    ),
    (
        "block_sharded_input",
        "system",
        NotImplementedError,
        "memory_layout",
        lambda m: (_mesh_input(m, S, mem=_block_sharded(S)), {"cluster_axis": 1}),
    ),
    (
        "block_sharded_output",
        "system",
        NotImplementedError,
        "memory_layout",
        lambda m: (_mesh_input(m, S), {"cluster_axis": 1, "memory_config": _block_sharded(S)}),
    ),
    (
        "sharded_output_without_shard_spec",
        "system",
        NotImplementedError,
        "shard spec",
        lambda m: (
            _mesh_input(m, S),
            {
                "cluster_axis": 1,
                "memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
            },
        ),
    ),
    (
        "fp32_with_dest16",
        "system",
        NotImplementedError,
        "",
        lambda m: (
            _mesh_input(m, S, dtype=FP32),
            {"cluster_axis": 1, "compute_kernel_config": H.compute_config(False)},
        ),
    ),
    (
        "mcast_on_2d_fabric",
        "system",
        NotImplementedError,
        "mcast",
        lambda m: (_mesh_input(m, S), {"cluster_axis": 1, "exchange_mode": "mcast"}),
    ),
    (
        "col_major_shard_orientation",
        "system",
        NotImplementedError,
        "(?i)orientation",
        lambda m: (
            _mesh_input(
                m, (1, 1, 32, 1792), mem=H.width_sharded((1, 1, 32, 1792), orientation=ttnn.ShardOrientation.COL_MAJOR)
            ),
            {"cluster_axis": 1},
        ),
    ),
]


@pytest.mark.parametrize(
    ("mesh_cfg", "error", "pattern", "make"),
    [pytest.param(cfg, err, pat, make, id=name) for name, cfg, err, pat, make in REFUSALS],
)
def test_refusal(mesh_cfg, error, pattern, make, drms_mesh, distributed_rms_norm_op, expect_error):
    tx, kwargs = make(drms_mesh)
    before = _entries(drms_mesh)
    with expect_error(error, pattern):
        distributed_rms_norm_op(tx, **kwargs)
    assert _entries(drms_mesh) == before, "a refused call must not add a program"
    if distributed_rms_norm_op.mode == "source":
        assert distributed_rms_norm_op.last_dispatches == 0, "a refused call must not dispatch"

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib

import pytest
import torch
import ttnn
import numpy as np
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal

TILE_HEIGHT = 32


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([1, 4, 4, 2], [1, 4, 128, 2], 2),
        ([8, 8, 8, 8], [8, 8, 8, 8], -1),
        ([32, 64, 128], [32, 64, 128], -1),
        ([64, 128, 256], [64, 128, 128], -1),
        ([1, 2048, 1, 64], [1, 2048, 1, 32], -1),
        ([1, 1, 1, 1], [1, 1, 1, 1], -1),
        ([4, 4], [4, 4], 1),
        ([128, 64], [128, 32], 1),
        ([16, 16, 16], [16, 16, 16], 0),
        ([1, 1, 1, 1], [1, 1, 1, 1], 1),
        ([64, 128, 256], [64, 128, 128], 1),
        ([256, 2, 32], [160, 2, 32], 1),
        ([2, 256, 2, 32], [2, 128, 2, 32], 1),
        ([2, 32, 96], [2, 32, 32], 1),
        ([128, 128], [128, 64], 1),
        ([1, 2, 128, 1, 768], [1, 2, 8, 1, 768], 2),
        ([1, 2, 8, 1, 768], [1, 2, 8, 1, 128], -1),
        ([1, 2, 8, 2, 768], [1, 2, 8, 2, 128], -1),
        ([1, 1, 2, 8, 2, 768], [1, 1, 2, 8, 2, 128], -2),
    ],
)
def test_gather_general(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([8, 8, 8, 8], [8, 8, 8, 8], -1),
        ([32, 64, 128], [32, 64, 128], -1),
        ([64, 128, 256], [64, 128, 128], -1),
        ([1, 2048, 1, 64], [1, 2048, 1, 32], -1),
        ([1, 1, 1, 1], [1, 1, 1, 1], -1),
    ],
)
def test_gather_preallocated_output(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, input_shape[dim], index_shape, dtype=torch.int64)
    output = torch.zeros_like(index, dtype=torch_dtype)

    torch_gather = torch.gather(input, dim, index, out=output)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)
    ttnn_output = ttnn.from_torch(output, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    ttnn.gather(ttnn_input, dim, index=ttnn_index, out=ttnn_output)

    assert ttnn_output.shape == index.shape

    assert_allclose(torch_gather, ttnn.to_torch(ttnn_output))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([1, 1, 512, 64], [1, 1, 512, 32], -1),  # 16 cores
        ([1, 1, 2048, 64], [1, 1, 2048, 32], -1),  # 64 cores
        ([1, 1, 2240, 64], [1, 1, 2240, 32], -1),  # 70 cores
    ],
)
def test_gather_multicore_cases(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim, torch_input_datatype, ttnn_input_datatype, ttnn_index_datatype",
    [
        ([1, 1, 512, 64], [1, 1, 512, 32], -1, torch.float32, ttnn.float32, ttnn.uint16),
        ([128, 64], [128, 32], 1, torch.bfloat16, ttnn.bfloat16, ttnn.uint16),
        ([2, 32, 96], [2, 32, 32], -1, torch.float32, ttnn.float32, ttnn.uint32),
    ],
)
def test_gather_datatype_cases(
    input_shape, index_shape, dim, torch_input_datatype, ttnn_input_datatype, ttnn_index_datatype, device
):
    torch.manual_seed(0)

    input = torch.randn(input_shape, dtype=torch_input_datatype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn_input_datatype, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn_index_datatype, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([32, 256 * TILE_HEIGHT], [32, 64 * TILE_HEIGHT], -1),
        ([1, 1, 32, 256 * TILE_HEIGHT], [1, 1, 32, 128 * TILE_HEIGHT], -1),
        ([1, 1, 32, 63 * TILE_HEIGHT], [1, 1, 32, 63 * TILE_HEIGHT], -1),
        ([1, 1, 32, 20 * TILE_HEIGHT], [1, 1, 32, 20 * TILE_HEIGHT], -1),
        ([1, 1, 32, 96 * TILE_HEIGHT], [1, 1, 32, 96 * TILE_HEIGHT], -1),
        ([1, 1, 32, 256 * TILE_HEIGHT], [1, 1, 32, 256 * TILE_HEIGHT], -1),
        ([1, 151936], [1, 151936], -1),
        ([1, 128256], [1, 128256], -1),
        # Wt_input=8193 -> bitmap_words=257 > BITMAP_WORDS_MAX=256; exercises use_bitmap=false fallback.
        ([1, 262176], [1, 32], -1),
    ],
)
def test_gather_long_tensor(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    max_uint32 = np.iinfo(np.uint32).max
    max_idx_val = min(input_shape[dim], max_uint32)
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, max_idx_val, index_shape, dtype=torch.int64)  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim, runs",
    [
        ([64, 64], [64, 32], -1, 10),
        ([1, 1, 32, 2048 * TILE_HEIGHT], [1, 1, 32, 2048 * TILE_HEIGHT], -1, 2),
        ([32, 128], [32, 128], -1, 5),
    ],
)
def test_gather_cache_run(input_shape, index_shape, dim, runs, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16

    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, input_shape[dim], index_shape, dtype=torch.int64)

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    for _ in range(runs):
        ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)
        assert ttnn_gather.shape == index.shape
        assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([32, 64, 128], [32, 64, 128], -1),
        ([32, 8192], [32, 2048], -1),
    ],
)
def test_gather_sub_core_grids(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
        ]
    )

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index, sub_core_grids=sub_core_grids)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([32, 32, 64 * TILE_HEIGHT], [32, 32, 64 * TILE_HEIGHT], -1),
        ([64, 64, 128 * TILE_HEIGHT], [64, 64, 128 * TILE_HEIGHT], -1),
    ],
)
def test_gather_multirow(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    max_uint32 = np.iinfo(np.uint32).max
    max_idx_val = min(input_shape[dim], max_uint32)
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, max_idx_val, index_shape, dtype=torch.int64)

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


# --- Codegen-path coverage ---
#
# ttnn.gather routes gate-supported cases to codegen and the rest to native, and offers no way to
# ask for one: the verification-only entries below live in the private module for that reason (see
# gather_force.hpp). The nightly routing suite only asserts the *rejected* cases fall back to
# native, so nothing there fails if a codegen kernel itself breaks -- these pin codegen and compare
# it against native on the same input. That comparison is exact because gather only moves values,
# so any mismatch is a real kernel bug rather than tolerance.
#
# One case per factory select_program_factory() can pick:
#   Wt_index == 1                    -> interleaved, one buffered index row per core
#   Wt_index >= 2 and Ht < the grid  -> tiled, index tile columns split across the cores
# The streaming factory is reached by L1 pressure rather than by shape, so its witness row has to be
# computed from the device; it gets its own test below.
_CODEGEN_CASES = [
    # (input_shape, index_shape, dim)
    ([1, 1, 32, 64], [1, 1, 32, 32], -1),
    ([1, 1, 128, 256], [1, 1, 128, 192], -1),
]
_CODEGEN_CASE_IDS = ["interleaved", "tiled"]

_force_native = ttnn._ttnn.operations.data_movement.gather_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.gather_force_codegen

_TILE_BYTES = 32 * 32 * 2


def _codegen_tensors(device, input_shape, index_shape, dim):
    """The gathered axis fixes the index dtype: uint16 cannot name a position past 65535."""
    axis_len = input_shape[dim]
    xt = ttnn.from_torch(
        torch.rand(input_shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    index = torch.randint(0, axis_len, index_shape, dtype=torch.int32)
    index_dtype = ttnn.uint16 if axis_len <= 65535 else ttnn.uint32
    return xt, ttnn.from_torch(index, dtype=index_dtype, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("input_shape, index_shape, dim", _CODEGEN_CASES, ids=_CODEGEN_CASE_IDS)
def test_gather_codegen(device, input_shape, index_shape, dim):
    torch.manual_seed(42)
    xt, it = _codegen_tensors(device, input_shape, index_shape, dim)

    golden = _force_native(xt, dim, it)
    output = _force_codegen(xt, dim, it)

    assert output.shape == golden.shape, f"Output shape {output.shape} does not match native shape {golden.shape}"
    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(output))


def test_gather_codegen_streaming(device):
    # An input row the per-core L1 budget can only just hold streams as ONE block whose depth is the
    # row itself; a wider row splits into two half-depth blocks and fits under any budget. So the
    # witness row is the budget in tile pages net of the index and output pages sharing it, and it
    # has to come from the device -- Wormhole's L1 is 36 pages smaller than Blackhole's, and a row
    # hardcoded for one is not brimful on the other. Wt_index then sizes the interleaved plan's
    # max(4, Wt_index)-deep output CB past the same budget, so selection reaches streaming.
    torch.manual_seed(42)
    budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
    wt_input = (budget - 2 * _TILE_BYTES) // _TILE_BYTES
    xt, it = _codegen_tensors(device, [1, 1, 64, 32 * wt_input], [1, 1, 64, 32 * 16], -1)

    golden = _force_native(xt, -1, it)
    output = _force_codegen(xt, -1, it)

    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(output))


@pytest.mark.parametrize("input_shape, index_shape, dim", _CODEGEN_CASES, ids=_CODEGEN_CASE_IDS)
def test_pc_gather_codegen(device, input_shape, index_shape, dim):
    torch.manual_seed(42)
    num_iters = 3
    # A distinct allocation per iteration: the cached program has to rebind its Buffer*s rather than
    # replay the first dispatch's addresses.
    tensors = [_codegen_tensors(device, input_shape, index_shape, dim) for _ in range(num_iters)]
    goldens = [ttnn.to_torch(_force_native(xt, dim, it)) for xt, it in tensors]

    for i in range(num_iters):
        xt, it = tensors[i]
        with device.cache_entries_counter.measure():
            output = _force_codegen(xt, dim, it)

        assert_equal(goldens[i], ttnn.to_torch(output))
        if i == 0:
            base_count = device.cache_entries_counter.total
        else:
            assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"


# The codegen path picks its factory, and the streaming factory its block depth, off the LIVE L1
# frontier (gather_usable_l1), and compute_program_hash keys on the resulting plan. In eager mode a
# frontier that drops below what a cached plan was built against is caught on enqueue by
# Program::validate_circular_buffer_region. A capture can re-decide nothing: it refuses to compile a
# program that is not already cached. So what a capture does depends entirely on whether the frontier
# it captures under is the one the warm-up cached a plan for, which is what this test pins down. It
# does not assert which factory served the call.
#
# The mirror-image question -- a replay writing over L1 allocated after the capture -- is a framework
# property, not a gather one: it reproduces on the native prim too, and belongs to the documented
# "Unsafe allocations" contract, with TT_METAL_TRACE_ALLOC_TRACKING as its detector and coverage in
# tests/ttnn/unit_tests/base_functionality/test_single_device_trace.py. Testing it here would
# attribute a framework bug to this op. See
# tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrestness.md.
_TRACE_DEVICE_PARAMS = {"trace_region_size": 4 * 1024 * 1024}

# Mirrors kGatherWriteBatchTiles in gather_common.hpp, which fixes the row-buffered output CB's floor
# depth (gather_output_cb_tiles). Hardcoded because the kernels take it as a compile-time constant and
# nothing exposes it to Python.
_GATHER_WRITE_BATCH_TILES = 4


def _best_effort(action):
    """Cleanup that must not mask the failure that made it necessary: once a capture has gone wrong
    these calls fail too, and the original error is the one worth surfacing."""
    try:
        action()
    except RuntimeError:
        pass


@contextlib.contextmanager
def _capturing(device):
    """Trace capture that closes even when the captured op raises.

    A queue left mid-capture fails every later call against the device, the teardown synchronize
    included, so without this one failed assertion wedges the card for the rest of the session.
    """
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        yield trace_id
    finally:
        _best_effort(lambda: ttnn.end_trace_capture(device, trace_id, cq_id=0))


def _interleaved_plan_bytes(wt_input, wt_index):
    """The row-buffered footprint gather_interleaved_fits_l1() weighs against the L1 budget: Wt_input
    input pages, one index page and gather_output_cb_tiles(Wt_index) output pages."""
    return (wt_input + 1 + max(_GATHER_WRITE_BATCH_TILES, wt_index)) * _TILE_BYTES


def _trace_row_width(device):
    """An input row whose row-buffered CB plan needs about half a clear L1 frontier.

    Halving the frontier then has to push select_program_factory() off that plan onto streaming, with
    both sides of the flip far enough from the boundary that per-bank alignment padding cannot decide
    it.
    """
    info = ttnn._ttnn.reports.get_device_info(device)
    wt_input = (info.cb_limit // 2) // _TILE_BYTES - (1 + _GATHER_WRITE_BATCH_TILES)
    if wt_input < 8:
        pytest.skip(f"device CB budget of {info.cb_limit} B is too small to size a row either side of the flip")
    return wt_input


def _pin_l1_headroom(device, plan_bytes, headroom_divisor):
    """Drop the live L1 frontier to plan_bytes/headroom_divisor above the CB base, inside the region
    the row-buffered plan needs.

    Interleaved L1 spreads pages round-robin over every bank, so a tensor of N tiles per bank lowers
    the lowest occupied address by the same amount on all of them; the contents are irrelevant, only
    the occupancy is. Where it lands is asserted rather than assumed, on both sides: above the region
    and the plan still fits, below the CB base and the tensor is not in worker L1 at all. Either way
    there is nothing left to collide with and every assertion downstream would hold vacuously.
    """
    info = ttnn._ttnn.reports.get_device_info(device)
    tiles_per_bank = (info.cb_limit - plan_bytes // headroom_divisor) // _TILE_BYTES
    if tiles_per_bank <= 0:
        pytest.skip("device L1 is too small to leave a meaningful headroom window")
    resident = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32 * tiles_per_bank, 32 * info.l1_num_banks]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    )
    headroom = resident.buffer_address() - info.address_at_first_l1_cb_buffer
    print(
        f"\n[l1] cb_base={info.address_at_first_l1_cb_buffer} cb_limit={info.cb_limit} "
        f"plan={plan_bytes} pinned_at={resident.buffer_address()} headroom={headroom}"
    )
    if not 0 < headroom < plan_bytes:
        ttnn.deallocate(resident)
        pytest.fail(f"pinned L1 tensor left {headroom} B above the CB base, outside (0, {plan_bytes})")
    return resident


def _trace_case(device):
    """Device tensors, the CB footprint their row-buffered plan needs, and a second host input to swap
    in after capture. Wt_index of 1 keeps selection on the row-buffered factory rather than the tiled
    one, so that footprint is the whole CB region.

    The index is all zeros, so the expected result is column 0 of whatever input is resident. That
    keeps both goldens on the host: running a device op between the capture and the replay would
    allocate against the very frontier the test has pinned.
    """
    wt_input = _trace_row_width(device)
    shape = [1, 1, 64, 32 * wt_input]
    before_swap, after_swap = torch.randn(shape), torch.randn(shape)
    index = torch.zeros([1, 1, 64, 32], dtype=torch.int64)
    xt = ttnn.from_torch(before_swap, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    it = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    # Gathering then rounding equals rounding then gathering -- gather only selects elements -- so a
    # host golden taken this way is bit-exact against the bfloat16 device result.
    goldens = tuple(torch.gather(host, -1, index).bfloat16().float() for host in (before_swap, after_swap))
    return xt, it, _interleaved_plan_bytes(wt_input, 1), after_swap, goldens


@pytest.mark.parametrize(
    "pin_before_warmup",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="#46533: a live-L1-derived plan misses the cache when the frontier moves "
                "between warm-up and capture, and a capture cannot compile the plan it then needs",
            ),
        ),
    ],
    ids=["warm_under_pressure", "warm_then_pressure"],
)
@pytest.mark.parametrize("device_params", [_TRACE_DEVICE_PARAMS], indirect=True)
def test_gather_codegen_trace_capture_under_pressure(device, pin_before_warmup):
    """Capturing with L1 constrained must record the plan that frontier admits.

    A capture compiles nothing: the program has to be cached already. warm_under_pressure warms up
    with the pin in place, so the constrained plan is the one cached and the capture finds it -- this
    is the case that shows the op IS trace-safe when a model warms up under the conditions it will
    capture under, which is what the framework asks of a caller
    (tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrestness.md).

    warm_then_pressure warms up with L1 clear and pins afterwards -- the shape a model takes when it
    warms up before it traces. The frontier moved, so the plan the capture needs is a different one,
    and whether it is cached is not something the caller can arrange. That is #46533, a property of
    every live-L1-derived plan in the tree rather than of this op, so it is recorded here as xfail
    rather than worked around.
    """
    torch.manual_seed(42)
    xt, it, plan_bytes, after_swap, (golden_before, golden_after) = _trace_case(device)

    trace_id = None
    resident = None
    try:
        if pin_before_warmup:
            resident = _pin_l1_headroom(device, plan_bytes, 2)
        warm = _force_codegen(xt, -1, it)
        assert_equal(golden_before, ttnn.to_torch(warm))
        ttnn.deallocate(warm)
        entries_after_warmup = device.num_program_cache_entries()
        if not pin_before_warmup:
            resident = _pin_l1_headroom(device, plan_bytes, 2)

        with _capturing(device) as trace_id:
            output = _force_codegen(xt, -1, it)
        ttnn.synchronize_device(device)
        # The whole premise of the capture: the plan it needs was already cached, so it compiled
        # nothing. A capture that had to build a program would raise instead, so this is the check
        # that the warm-up cached the plan the pinned frontier actually admits.
        entries_after_capture = device.num_program_cache_entries()
        assert entries_after_capture == entries_after_warmup, (
            f"trace capture added {entries_after_capture - entries_after_warmup} program cache "
            f"entr(y/ies): the plan it needed was not the one warm-up cached"
        )

        # Swap the input, keeping the buffer address the trace baked. The assertion below needs it:
        # the capture already left a correct result in `output`, so against an unchanged input a
        # replay that never reached the device would compare equal too.
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(after_swap, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), xt)
        ttnn.synchronize_device(device)

        ttnn.execute_trace(device, trace_id, blocking=True)
        ttnn.synchronize_device(device)
        assert_equal(golden_after, ttnn.to_torch(output))
    finally:
        if trace_id is not None:
            _best_effort(lambda: ttnn.release_trace(device, trace_id))
        if resident is not None:
            ttnn.deallocate(resident)

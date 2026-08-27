# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import threading
from itertools import chain

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import ttnn

from loguru import logger
from tests.ttnn.utils_for_testing import assert_equal

TEST_PADDING_VALUE = -42

RM = ttnn.ROW_MAJOR_LAYOUT
TL = ttnn.TILE_LAYOUT


def _case(shape, layout, dim, keepdim, dtype):
    """Single argmax test case tuple (readable shorthand for parametrization)."""
    return (shape, layout, dim, keepdim, dtype)


def _argmax_misc_and_rank_special():
    return [
        _case([], RM, None, True, torch.bfloat16),
        _case([32], RM, -1, False, torch.float32),
        _case([32, 0], RM, 1, True, torch.bfloat16),
        _case([64], RM, -1, True, torch.bfloat16),
        _case([1, 512], RM, -1, True, torch.float32),
        _case([1, 1024], RM, -1, True, torch.int32),
        _case([1, 65], RM, -1, True, torch.uint8),
        _case([8, 10, 129], RM, 2, True, torch.bfloat16),
        _case([1, 8, 160], RM, -1, False, torch.bfloat16),
        _case([1, 256, 1024 * 8], RM, -1, False, torch.float32),
        _case([32, 32, 32, 1], RM, -1, True, torch.float32),
        _case([128], RM, -1, True, torch.float32),
        _case([256], RM, -1, False, torch.bfloat16),
        _case([128], TL, -1, True, torch.float32),
        _case([256], TL, -1, False, torch.bfloat16),
    ]


def _argmax_row_major_wide_reduce_last_dim():
    return [
        _case([64, 128], RM, -1, True, torch.float32),
        _case([64, 128], RM, -1, False, torch.int32),
        _case([32, 64, 128], RM, -1, True, torch.float32),
        _case([32, 64, 128], RM, -1, True, torch.bfloat16),
        _case([32, 64, 128], TL, -1, False, torch.bfloat16),
        _case([16, 32, 64, 128], RM, -1, True, torch.bfloat16),
        _case([16, 32, 64, 128], RM, -1, True, torch.float32),
        _case([16, 32, 64, 128], RM, -1, True, torch.int32),
        _case([16, 32, 64, 128], TL, -1, True, torch.bfloat16),
        _case([16, 32, 64, 128], TL, -1, False, torch.float32),
        _case([16, 32, 70, 130], TL, -1, True, torch.bfloat16),
        _case([16, 32, 70, 130], TL, -1, False, torch.bfloat16),
        _case([8, 16, 32, 64], RM, -1, True, torch.float32),
        _case([8, 16, 32, 64], RM, -1, False, torch.bfloat16),
        _case([4, 8, 16, 32], RM, -1, False, torch.float32),
        _case([100, 200], RM, -1, True, torch.bfloat16),
        _case([100, 200], RM, -1, False, torch.float32),
        _case([50, 100, 200], RM, -1, True, torch.int32),
        _case([25, 50, 100], RM, -1, False, torch.uint8),
        _case([12, 24, 48, 96], RM, -1, True, torch.bfloat16),
        _case([1, 8, 20, 18], TL, -1, True, torch.bfloat16),
    ]


def _argmax_nc_hw_mixed_shapes():
    """Non-last dims on TILE and ROW_MAJOR (padding / rank coverage)."""
    return [
        _case([4, 32, 32], TL, 0, True, torch.bfloat16),
        _case([2, 64, 64], TL, -2, True, torch.bfloat16),
        _case([2, 64, 64], TL, 1, False, torch.bfloat16),
        _case([1, 70, 130], TL, -2, True, torch.bfloat16),
        _case([1, 2, 32, 32], TL, 2, True, torch.float32),
        _case([2, 64, 64], RM, 1, True, torch.bfloat16),
        _case([2, 64, 64], RM, -2, False, torch.bfloat16),
        _case([1, 48, 96], RM, -2, True, torch.float32),
        _case([4, 32, 32], TL, 0, False, torch.bfloat16),
        _case([4, 32, 32], RM, 0, True, torch.bfloat16),
    ]


def _argmax_nc_nd_rank4():
    return [
        _case([2, 3, 64, 64], TL, 0, True, torch.float32),
        _case([2, 3, 64, 64], TL, 1, True, torch.float32),
        _case([2, 3, 64, 64], TL, 1, False, torch.float32),
        _case([2, 3, 64, 64], TL, -2 - 1, False, torch.bfloat16),
        _case([2, 3, 64, 64], RM, 1, True, torch.bfloat16),
        _case([2, 5, 70, 130], TL, 0, True, torch.float32),
        _case([2, 5, 70, 130], TL, 0, True, torch.bfloat16),
        _case([2, 5, 70, 130], TL, 1, False, torch.bfloat16),
        _case([1, 5, 32, 32], TL, 1, False, torch.float32),
        _case([5, 1, 64, 64], TL, 0, True, torch.bfloat16),
        _case([3, 5, 256, 256], TL, 0, True, torch.bfloat16),
        _case([2, 3, 64, 64], TL, 2, True, torch.bfloat16),
        _case([2, 3, 70, 130], TL, 2, False, torch.bfloat16),
    ]


def _argmax_nc_nd_rank5():
    return [
        _case([2, 3, 4, 32, 32], TL, 0, True, torch.bfloat16),
        _case([2, 3, 4, 32, 32], TL, 1, False, torch.bfloat16),
        _case([2, 3, 4, 32, 32], TL, 2, True, torch.float32),
        _case([2, 3, 4, 64, 64], TL, 3, True, torch.bfloat16),
    ]


def argmax_torch_ttnn_cases():
    yield from chain(
        _argmax_misc_and_rank_special(),
        _argmax_row_major_wide_reduce_last_dim(),
        _argmax_nc_hw_mixed_shapes(),
        _argmax_nc_nd_rank4(),
        _argmax_nc_nd_rank5(),
    )


@pytest.mark.parametrize(
    argnames="tensor_shape, tensor_layout, dim, keepdim, dtype",
    argvalues=list(argmax_torch_ttnn_cases()),
)
def test_argmax(device, tensor_shape, tensor_layout, dim, keepdim, dtype):
    """
    Test the compatibility of the torch and ttnn output for argmax of different
    tensor shapes, dim values, and data types.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    torch.manual_seed(0)
    rank = len(tensor_shape)

    # Create tensor based on data type
    if dtype in [torch.int32, torch.uint8]:
        # Use randint for integer types
        torch_tensor = (
            torch.randint(0, 100, tensor_shape, dtype=dtype) if rank > 0 else torch.randint(0, 100, (), dtype=dtype)
        )
    else:
        # Use randn for floating point types
        torch_tensor = torch.randn(*tensor_shape, dtype=dtype) if rank > 0 else torch.randn((), dtype=dtype)

    # Convert torch uint8 to appropriate ttnn type
    if dtype == torch.uint8:  # PyTorch does not have uint32/uint16, so we use uint8
        ttnn_dtype = ttnn.uint32
        ttnn_tensor = ttnn.from_torch(torch_tensor, device=device, dtype=ttnn_dtype, layout=tensor_layout)
        if tensor_layout == ttnn.TILE_LAYOUT:
            ttnn_tensor = ttnn.fill_implicit_tile_padding(ttnn_tensor, TEST_PADDING_VALUE)

    else:
        ttnn_tensor = ttnn.from_torch(torch_tensor, device=device, layout=tensor_layout)
        if tensor_layout == ttnn.TILE_LAYOUT:
            ttnn_tensor = ttnn.fill_implicit_tile_padding(ttnn_tensor, TEST_PADDING_VALUE)

    torch_op, ttnn_op = getattr(torch, "argmax"), getattr(ttnn, "argmax")

    # Run on both and flag exceptions
    torch_errored = False
    torch_error_msg = ""
    try:
        torch_result = torch_op(torch_tensor, dim=dim, keepdim=keepdim) if dim is not None else torch_op(torch_tensor)
    except (IndexError, RuntimeError) as e:
        torch_errored = True
        torch_error_msg = str(e)

    ttnn_errored = False
    ttnn_error_msg = ""
    try:
        if dim is not None:
            ttnn_result = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim)
        else:
            ttnn_result = ttnn_op(ttnn_tensor)
    except RuntimeError as e:
        ttnn_errored = True
        ttnn_error_msg = str(e)

    assert (
        torch_errored == ttnn_errored
    ), f"mismatch in errors raised: torch: {torch_errored} ({torch_error_msg}), ttnn: {ttnn_errored} ({ttnn_error_msg})"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        logger.warning(f"both torch and ttnn raised errors: torch: {torch_error_msg}, ttnn: {ttnn_error_msg}")
        return

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result)).to(torch.int32)

    # test for equivalance
    assert_equal(torch_result, ttnn_result)


def test_argmax_nc_ties_first_index_wins(device):
    """Constant tensor: argmax tie-break must match PyTorch (smallest index wins)."""
    t = torch.full([4, 3, 64, 64], 1.0, dtype=torch.bfloat16)
    for dim in (0, 1):
        ref = torch.argmax(t, dim=dim, keepdim=True)
        ttnn_t = ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn_t = ttnn.fill_implicit_tile_padding(ttnn_t, TEST_PADDING_VALUE)
        out = ttnn.argmax(ttnn_t, dim=dim, keepdim=True)
        assert_equal(ref, ttnn.to_torch(ttnn.from_device(out)).to(torch.int32))


def test_argmax_nc_preallocated_output(device):
    torch.manual_seed(0)
    t = torch.randn(2, 3, 64, 64, dtype=torch.float32)
    ttnn_in = ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_in = ttnn.fill_implicit_tile_padding(ttnn_in, TEST_PADDING_VALUE)
    ref = torch.argmax(t, dim=1, keepdim=True)
    out_shape = ref.shape
    ttnn_out = ttnn.zeros(list(out_shape), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    result = ttnn.argmax(ttnn_in, dim=1, keepdim=True, output_tensor=ttnn_out)
    assert_equal(ref, ttnn.to_torch(ttnn.from_device(result)).to(torch.int32))


@pytest.mark.timeout(120, method="thread")
def test_argmax_reduce_all_multicore_no_deadlock(device):
    """
    Guard for the multi-core reduce_all (whole-tensor) argmax path.

    The reduce core signals each iteration via start_sem (set + multicast), which increases
    monotonically.  In the reduce_all path there is no per-iteration done_sem back-pressure (it is
    lifted out of the k-loop), so the reduce core free-runs and can advance start_sem past a given
    value before a lagging worker samples it.  Workers therefore wait with wait_min(k+1) (>=) rather
    than an exact match, so a skipped-over value cannot strand a worker at noc_semaphore_wait and
    deadlock the op.

    A ROW_MAJOR tensor with dim=None routes to the multi-core reduce_all reader; the shape is large
    enough to split across many cores and run several k-iterations.  The check guards correctness of
    the reduce_all multi-core path and, via the worker-thread watchdog + pytest-timeout, converts a
    deadlock into an assertion failure instead of a hung session.
    """
    torch.manual_seed(0)
    t = torch.randn((64, 4096), dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(t, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    result = {}

    def _run():
        try:
            out = ttnn.argmax(ttnn_in)  # dim=None -> reduce_all multi-core path
            ttnn.synchronize_device(device)
            result["out"] = out
        except Exception as exc:  # surface device/compile errors to the main thread
            result["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=60.0)

    assert not worker.is_alive(), (
        "ttnn.argmax(dim=None) did not complete on the multi-core reduce_all path: a worker's "
        "start_sem wait was starved -- workers must wait_min(>=) on the monotonic start_sem, since "
        "an exact-match wait can be lapped by the free-running reduce core."
    )
    if "error" in result:
        raise result["error"]

    ref = int(torch.argmax(t.reshape(-1)))
    got = int(ttnn.to_torch(ttnn.from_device(result["out"])).item())
    assert got == ref, f"argmax reduce_all mismatch: got {got}, expected {ref}"


def _run_argmax_in_worker(fn, timeout_s=60.0):
    """Run `fn` on a worker thread so a device hang surfaces as an assertion, not a hung session."""
    result = {}

    def _run():
        try:
            result["out"] = fn()
        except Exception as exc:  # surface device/compile errors to the main thread
            result["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    return worker, result


@pytest.mark.timeout(120, method="thread")
@pytest.mark.parametrize(
    "tensor_shape, core_ranges",
    [
        # More sub-grid cores than blocks of work
        ([8, 32], [((0, 0), (1, 1))]),  # 4 cores, 2 blocks
        ([1, 64], [((0, 0), (7, 0))]),  # 8 cores, 4 blocks
        # Fewer cores than blocks
        ([1, 80], [((0, 0), (3, 0))]),  # 4 cores, 5 blocks -> 2 blocks each = 128 units for 80
        ([1, 144], [((0, 0), (7, 0))]),  # 8 cores, 9 blocks -> 2 blocks each = 256 units for 144
        # Two ranges exercise the second core group (cores1 / red_dim_units1).
        ([4, 80], [((0, 0), (1, 0)), ((0, 1), (1, 1))]),
        # Widths that already split cleanly
        ([1, 128], [((0, 0), (7, 0))]),
        ([8, 1024], [((0, 0), (7, 0))]),
    ],
)
def test_argmax_sub_core_grids_work_split(device, tensor_shape, core_ranges):
    """
    Guard the sub_core_grids work-split: rounded-up per-core shares can overshoot the
    reduction dim by more than one core's share, and trimming only the last core then underflowed
    uint32 into a large read size and a hang.  Slices are now clamped per core instead.
    """
    torch.manual_seed(0)
    grids = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(*end)) for start, end in core_ranges}
    )

    torch_tensor = torch.randn(*tensor_shape, dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def _run():
        out = ttnn.argmax(ttnn_tensor, dim=-1, keepdim=False, sub_core_grids=grids)
        ttnn.synchronize_device(device)
        return out

    worker, result = _run_argmax_in_worker(_run)

    assert not worker.is_alive(), f"ttnn.argmax timed out for shape={tensor_shape}, " f"sub_core_grids={core_ranges}"
    if "error" in result:
        raise result["error"]

    ttnn_result = ttnn.to_torch(ttnn.from_device(result["out"])).to(torch.int32)
    assert_equal(torch.argmax(torch_tensor, dim=-1, keepdim=False), ttnn_result)


@pytest.mark.timeout(120, method="thread")
def test_argmax_reduce_all_sub_core_grids_idle_cores(device):
    """
    dim=None uses the kernel's separate reduce_all block (its own partial-write and semaphore
    sequence); drive it with a sub_core_grids wide enough to leave cores with no work.
    """
    torch.manual_seed(0)
    # 4 cores for a 32-wide reduction dim = 2 blocks of work, so two cores are left idle.
    grids = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    t = torch.randn(8, 32, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(t, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def _run():
        out = ttnn.argmax(ttnn_in, sub_core_grids=grids)  # dim=None -> reduce_all
        ttnn.synchronize_device(device)
        return out

    worker, result = _run_argmax_in_worker(_run)

    assert not worker.is_alive(), "ttnn.argmax(dim=None) timed out with sub_core_grids leaving idle cores"
    if "error" in result:
        raise result["error"]

    got = int(ttnn.to_torch(ttnn.from_device(result["out"])).item())
    assert got == int(torch.argmax(t.reshape(-1))), f"reduce_all mismatch: got {got}"

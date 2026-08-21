# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""sub_core_grids fence coverage for ttnn.experimental.topk_large_indices.

The fence confines the op to one rectangular core block (e.g. an 8x10 = 80-core
rectangle) so a concurrent workload (e.g. a CCL) can own the remaining cores.
Correctness must hold both for a fence at the grid origin and for an OFFSET
fence (origin x=5 leaves columns 0-4 untouched — the 80/40 partition case).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="topk_large_indices is Blackhole-only"),
]


def _fence(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


# 80-core rectangles on the 13x10 P150 worker grid: one at the origin, one
# offset to columns 5..12 (columns 0..4 free for a concurrent CCL).
FENCES = {
    "origin_8x10": (0, 0, 7, 9),
    "offset_x5_8x10": (5, 0, 12, 9),
}


def _assert_topk_values_match(torch_input, tt_indices, k, valid_length=None):
    """Tie-safe check: indices are in range and distinct per row, and the
    gathered value multiset equals torch.topk's over the searched prefix."""
    searched = torch_input if valid_length is None else torch_input[..., :valid_length]
    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)
    assert list(indices.shape) == list(torch_input.shape[:-1]) + [k]
    assert indices.min() >= 0
    limit = searched.shape[-1]
    assert indices.max() < limit
    flat = indices.reshape(-1, k)
    for row_indices in flat:
        assert row_indices.unique().numel() == k
    actual_values = torch.gather(searched.float(), dim=-1, index=indices)
    ref_values, _ = torch.topk(searched.float(), k, dim=-1, largest=True, sorted=True)
    assert_equal(actual_values.sort(dim=-1).values, ref_values.sort(dim=-1).values)
    # Sorted-descending output contract: gathered values must already be sorted.
    assert_equal(actual_values, actual_values.sort(dim=-1, descending=True).values)


@pytest.mark.parametrize("fence_name", FENCES.keys())
@pytest.mark.parametrize(
    "rows,n,k,valid_length",
    [
        # 160 rows exceed an 80-core fence. Explicit sub_core_grids intentionally
        # disables the two-dispatch hybrid, covering the restricted single-launch
        # path on the production 1M/512k prefill cell.
        (160, 1048576, 2048, 524288),
        # Multi-row rectangle (tree) engine inside the fence.
        (32, 65536, 2048, None),
        # Ragged width (non-power-of-two, tail chunk) on the fenced grid.
        (3, 77777, 512, None),
    ],
    ids=["prefill_160x1M_valid512k", "rect_32x65536", "ragged_3x77777_k512"],
)
def test_topk_large_indices_fenced_matches_torch(device, fence_name, rows, n, k, valid_length):
    x0, y0, x1, y1 = FENCES[fence_name]
    grid = device.compute_with_storage_grid_size()
    if x1 >= grid.x or y1 >= grid.y:
        pytest.skip(f"fence ({x1},{y1}) does not fit the {grid.x}x{grid.y} worker grid")

    torch.manual_seed(7)
    torch_input = torch.randn(1, 1, rows, n, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_indices = ttnn.experimental.topk_large_indices(
        tt_input, k=k, valid_length=valid_length, sub_core_grids=_fence(x0, y0, x1, y1)
    )

    _assert_topk_values_match(torch_input, tt_indices, k, valid_length)


def test_topk_large_indices_fence_matches_unfenced_values(device):
    """The fenced run selects the same value multiset as the unfenced run
    (tie identity may differ — the engine split changes merge order)."""
    torch.manual_seed(11)
    rows, n, k = 8, 131072, 1024
    torch_input = torch.randn(1, 1, rows, n, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    unfenced = ttnn.experimental.topk_large_indices(tt_input, k=k)
    fenced = ttnn.experimental.topk_large_indices(tt_input, k=k, sub_core_grids=_fence(0, 0, 7, 9))

    unfenced_idx = ttnn.to_torch(unfenced, dtype=torch.uint32).to(torch.int64)
    fenced_idx = ttnn.to_torch(fenced, dtype=torch.uint32).to(torch.int64)
    unfenced_vals = torch.gather(torch_input.float(), dim=-1, index=unfenced_idx)
    fenced_vals = torch.gather(torch_input.float(), dim=-1, index=fenced_idx)
    assert_equal(fenced_vals.sort(dim=-1).values, unfenced_vals.sort(dim=-1).values)


def test_topk_large_indices_non_rectangular_grid_falls_back_row_parallel(device):
    """A non-rectangular core set is legal: the op runs the row-parallel
    engine over the enumerated cores (trees need a dense rectangle)."""
    torch.manual_seed(13)
    torch_input = torch.randn(1, 1, 37, 65536, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    two_ranges = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(3, 3), ttnn.CoreCoord(4, 4)),
        ]
    )
    tt_indices = ttnn.experimental.topk_large_indices(tt_input, k=256, sub_core_grids=two_ranges)
    _assert_topk_values_match(torch_input, tt_indices, 256)


def test_topk_large_indices_fence_rejects_out_of_grid(device, expect_error):
    torch_input = torch.randn(1, 1, 2, 4096, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid = device.compute_with_storage_grid_size()
    with expect_error(RuntimeError, "must be fully contained"):
        ttnn.experimental.topk_large_indices(tt_input, k=256, sub_core_grids=_fence(0, 0, grid.x, grid.y))

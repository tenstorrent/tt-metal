# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal


def run_get_rows(device, torch_input, torch_indices, input_dtype, index_dtype):
    """Gathers rows on device and returns (ttnn output as torch, torch reference)."""
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices, dtype=index_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    output = ttnn.get_rows(ttnn_input, ttnn_indices)
    torch_output = ttnn.to_torch(output)
    reference = torch.index_select(torch_input, 0, torch_indices.to(torch.int64))
    return torch_output, reference


@pytest.mark.parametrize(
    "H, W",
    [
        (128, 576),  # KV-cache shaped
        (4, 3),
        (33, 65),  # non-power-of-two, non-tile-aligned
        (1, 64),
        (2048, 128),
    ],
)
@pytest.mark.parametrize("index_dtype", [ttnn.uint32, ttnn.int32], ids=["u32", "i32"])
def test_get_rows_basic(device, H, W, index_dtype):
    torch.manual_seed(0)
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    torch_indices = torch.arange(H, dtype=torch.int64) % H
    torch_output, reference = run_get_rows(device, torch_input, torch_indices, ttnn.bfloat16, index_dtype)
    assert_equal(torch_output, reference)


def test_get_rows_ggml_spec_example(device):
    """The worked example from the GGML metalium spec (issue #15942): rows 0 and 2 of a [3, 4] table."""
    torch_input = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        dtype=torch.bfloat16,
    )
    torch_indices = torch.tensor([0, 2], dtype=torch.int64)
    torch_output, reference = run_get_rows(device, torch_input, torch_indices, ttnn.bfloat16, ttnn.int32)
    expected = torch.tensor(
        [
            [1, 2, 3, 4],
            [9, 10, 11, 12],
        ],
        dtype=torch.bfloat16,
    )
    assert_equal(reference, expected)  # sanity: the reference itself
    assert_equal(torch_output, expected)


def test_get_rows_docstring_example(device):
    """The ordering example from the operation documentation: indices [3, 0, 2]."""
    torch_input = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
            [40, 41, 42],
        ],
        dtype=torch.bfloat16,
    )
    torch_indices = torch.tensor([3, 0, 2], dtype=torch.int64)
    torch_output, reference = run_get_rows(device, torch_input, torch_indices, ttnn.bfloat16, ttnn.uint32)
    expected = torch.tensor(
        [
            [40, 41, 42],
            [10, 11, 12],
            [30, 31, 32],
        ],
        dtype=torch.bfloat16,
    )
    assert_equal(reference, expected)  # sanity: the reference itself
    assert_equal(torch_output, expected)


@pytest.mark.parametrize("N", [1, 7, 32, 511, 512])
def test_get_rows_random_indices(device, N):
    torch.manual_seed(1)
    H, W = 1024, 256
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    torch_indices = torch.randint(0, H, (N,), dtype=torch.int64)
    torch_output, reference = run_get_rows(device, torch_input, torch_indices, ttnn.bfloat16, ttnn.uint32)
    assert_equal(torch_output, reference)


def test_get_rows_reordered_and_duplicate_indices(device):
    torch.manual_seed(2)
    H, W = 64, 128
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    # Reordered with duplicates, exercising both first and last row.
    torch_indices = torch.tensor([H - 1, 0, 0, 5, H - 1, 5], dtype=torch.int64)
    torch_output, reference = run_get_rows(device, torch_input, torch_indices, ttnn.bfloat16, ttnn.uint32)
    assert_equal(torch_output, reference)


@pytest.mark.parametrize("W", [2, 32, 4096, 32768], ids=["w2", "w32", "w4k", "w32k"])
def test_get_rows_row_widths(device, W):
    """Single-row gathers at several widths, including rows much wider than a tile."""
    torch.manual_seed(3)
    H = 8
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    torch_indices = torch.tensor([3], dtype=torch.int64)
    torch_output, reference = run_get_rows(device, torch_input, torch_indices, ttnn.bfloat16, ttnn.uint32)
    assert_equal(torch_output, reference)


@pytest.mark.parametrize(
    "input_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.int32], ids=["bf16", "fp32", "u32", "i32"]
)
def test_get_rows_dtypes(device, input_dtype):
    torch_dtype = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.uint32: torch.int32,
        ttnn.int32: torch.int32,
    }[
        input_dtype
    ]
    torch.manual_seed(4)
    H, W = 96, 160
    if input_dtype == ttnn.uint32:
        torch_input = torch.randint(0, 2**31 - 1, (H, W), dtype=torch.int32)
    else:
        torch_input = torch.randn(H, W, dtype=torch_dtype)
    torch_indices = torch.randperm(H, generator=torch.Generator().manual_seed(4))[:40].to(torch.int64)
    torch_output, reference = run_get_rows(device, torch_input, torch_indices, input_dtype, ttnn.uint32)
    assert_equal(torch_output, reference)


def test_get_rows_first_and_last_row(device):
    torch.manual_seed(5)
    H, W = 37, 96
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    torch_indices = torch.tensor([0, H - 1, 0, H - 1], dtype=torch.int64)
    torch_output, reference = run_get_rows(device, torch_input, torch_indices, ttnn.bfloat16, ttnn.uint32)
    assert_equal(torch_output, reference)


def test_get_rows_rejects_negative_index(device, expect_error):
    torch_input = torch.randn(16, 64, dtype=torch.bfloat16)
    torch_indices = torch.tensor([0, -1], dtype=torch.int64)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "out of range"):
        ttnn.get_rows(ttnn_input, ttnn_indices)


def test_get_rows_rejects_out_of_range_index(device, expect_error):
    torch_input = torch.randn(16, 64, dtype=torch.bfloat16)
    torch_indices = torch.tensor([15, 16], dtype=torch.int64)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "out of range"):
        ttnn.get_rows(ttnn_input, ttnn_indices)


def test_get_rows_rejects_wrong_index_dtype(device, expect_error):
    torch_input = torch.randn(16, 64, dtype=torch.bfloat16)
    torch_indices = torch.zeros(4, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "indices dtype"):
        ttnn.get_rows(ttnn_input, ttnn_indices)


def test_get_rows_rejects_wrong_index_rank(device, expect_error):
    torch_input = torch.randn(16, 64, dtype=torch.bfloat16)
    torch_indices = torch.zeros(2, 2, dtype=torch.int64)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "rank 1"):
        ttnn.get_rows(ttnn_input, ttnn_indices)


def test_get_rows_rejects_wrong_input_rank(device, expect_error):
    torch_input = torch.randn(4, 16, 64, dtype=torch.bfloat16)
    torch_indices = torch.zeros(4, dtype=torch.int64)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "rank 2"):
        ttnn.get_rows(ttnn_input, ttnn_indices)


def test_get_rows_rejects_tile_layout(device, expect_error):
    torch_input = torch.randn(64, 64, dtype=torch.bfloat16)
    torch_indices = torch.tensor([1, 2], dtype=torch.int64)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "ROW_MAJOR"):
        ttnn.get_rows(ttnn_input, ttnn_indices)


def test_get_rows_empty_indices_rejected(device, expect_error):
    """Empty index vectors are explicitly out of scope for this milestone (#15942)."""
    torch_input = torch.randn(16, 64, dtype=torch.bfloat16)
    torch_indices = torch.empty(0, dtype=torch.int64)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "empty"):
        ttnn.get_rows(ttnn_input, ttnn_indices)


def test_get_rows_preallocated_output(device):
    torch.manual_seed(6)
    H, W, N = 128, 96, 33
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    torch_indices = torch.randint(0, H, (N,), dtype=torch.int64)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.empty(
        [N, W], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.get_rows(ttnn_input, ttnn_indices, out=output)
    reference = torch.index_select(torch_input, 0, torch_indices)
    assert_equal(ttnn.to_torch(output), reference)

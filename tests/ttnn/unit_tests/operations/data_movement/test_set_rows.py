# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal


def run_set_rows(device, torch_input, torch_indices, torch_updates, input_dtype, index_dtype):
    """Writes rows on device and returns (ttnn result as torch, torch reference)."""
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=index_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_updates = ttnn.from_torch(torch_updates, dtype=input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.set_rows(ttnn_input, ttnn_indices, ttnn_updates)
    torch_output = ttnn.to_torch(output)
    reference = torch_input.clone()
    reference.index_put_((torch_indices.to(torch.int64),), torch_updates)
    return torch_output, reference


@pytest.mark.parametrize(
    "H, W",
    [
        (128, 576),  # KV-cache shaped
        (4, 3),
        (33, 65),  # non-power-of-two, non-tile-aligned
        (2048, 128),
    ],
)
@pytest.mark.parametrize("index_dtype", [ttnn.uint32, ttnn.int32], ids=["u32", "i32"])
def test_set_rows_basic(device, H, W, index_dtype):
    torch.manual_seed(0)
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    torch_indices = torch.arange(0, H, 7, dtype=torch.int64)
    torch_updates = torch.randn(torch_indices.numel(), W, dtype=torch.bfloat16)
    torch_output, reference = run_set_rows(device, torch_input, torch_indices, torch_updates, ttnn.bfloat16, index_dtype)
    assert_equal(torch_output, reference)


def test_set_rows_untouched_rows_preserved(device):
    """The whole output must match, not just the written rows."""
    torch.manual_seed(1)
    H, W = 64, 96
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    torch_indices = torch.tensor([5, 30, 63], dtype=torch.int64)
    torch_updates = torch.randn(3, W, dtype=torch.bfloat16)
    torch_output, reference = run_set_rows(device, torch_input, torch_indices, torch_updates, ttnn.bfloat16, ttnn.uint32)
    assert_equal(torch_output, reference)
    # And the untouched rows are bit-identical to the input.
    mask = torch.ones(H, dtype=torch.bool)
    mask[torch_indices] = False
    assert_equal(torch_output[mask], torch_input[mask])


def test_set_rows_first_and_last_row(device):
    torch.manual_seed(2)
    H, W = 37, 64
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    torch_indices = torch.tensor([0, H - 1, 0, H - 1], dtype=torch.int64)
    torch_updates = torch.randn(4, W, dtype=torch.bfloat16)
    torch_output, reference = run_set_rows(device, torch_input, torch_indices, torch_updates, ttnn.bfloat16, ttnn.uint32)
    assert_equal(torch_output, reference)


@pytest.mark.parametrize("N", [1, 7, 32, 511, 512])
def test_set_rows_random_indices(device, N):
    torch.manual_seed(3)
    H, W = 1024, 256
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    torch_indices = torch.randint(0, H, (N,), dtype=torch.int64)
    torch_updates = torch.randn(N, W, dtype=torch.bfloat16)
    torch_output, reference = run_set_rows(device, torch_input, torch_indices, torch_updates, ttnn.bfloat16, ttnn.uint32)
    assert_equal(torch_output, reference)


@pytest.mark.parametrize("W", [2, 32, 4096], ids=["w2", "w32", "w4k"])
def test_set_rows_row_widths(device, W):
    torch.manual_seed(4)
    H = 8
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    torch_indices = torch.tensor([3], dtype=torch.int64)
    torch_updates = torch.randn(1, W, dtype=torch.bfloat16)
    torch_output, reference = run_set_rows(device, torch_input, torch_indices, torch_updates, ttnn.bfloat16, ttnn.uint32)
    assert_equal(torch_output, reference)


def test_set_rows_bf16(device):
    torch.manual_seed(5)
    H, W, N = 96, 160, 40
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    perm = torch.randperm(H, generator=torch.Generator().manual_seed(5))
    torch_indices = perm[:N].to(torch.int64)
    torch_updates = torch.randn(N, W, dtype=torch.bfloat16)
    torch_output, reference = run_set_rows(device, torch_input, torch_indices, torch_updates, ttnn.bfloat16, ttnn.uint32)
    assert_equal(torch_output, reference)


def test_set_rows_rejects_negative_index(device, expect_error):
    torch_input = torch.randn(16, 64, dtype=torch.bfloat16)
    torch_indices = torch.tensor([0, -1], dtype=torch.int64)
    torch_updates = torch.randn(2, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_updates = ttnn.from_torch(torch_updates, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "out of range"):
        ttnn.set_rows(ttnn_input, ttnn_indices, ttnn_updates)


def test_set_rows_rejects_out_of_range_index(device, expect_error):
    torch_input = torch.randn(16, 64, dtype=torch.bfloat16)
    torch_indices = torch.tensor([15, 16], dtype=torch.int64)
    torch_updates = torch.randn(2, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_updates = ttnn.from_torch(torch_updates, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "out of range"):
        ttnn.set_rows(ttnn_input, ttnn_indices, ttnn_updates)


def test_set_rows_rejects_wrong_index_dtype(device, expect_error):
    torch_input = torch.randn(16, 64, dtype=torch.bfloat16)
    torch_indices = torch.zeros(4, dtype=torch.float32)
    torch_updates = torch.randn(4, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_updates = ttnn.from_torch(torch_updates, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError):
        ttnn.set_rows(ttnn_input, ttnn_indices, ttnn_updates)


def test_set_rows_rejects_wrong_index_rank(device, expect_error):
    torch_input = torch.randn(16, 64, dtype=torch.bfloat16)
    torch_indices = torch.zeros(2, 2, dtype=torch.int64)
    torch_updates = torch.randn(4, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_updates = ttnn.from_torch(torch_updates, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "rank 1"):
        ttnn.set_rows(ttnn_input, ttnn_indices, ttnn_updates)


def test_set_rows_rejects_wrong_input_rank(device, expect_error):
    torch_input = torch.randn(4, 16, 64, dtype=torch.bfloat16)
    torch_indices = torch.zeros(4, dtype=torch.int64)
    torch_updates = torch.randn(4, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_updates = ttnn.from_torch(torch_updates, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "rank 2"):
        ttnn.set_rows(ttnn_input, ttnn_indices, ttnn_updates)


def test_set_rows_rejects_update_count_mismatch(device, expect_error):
    torch_input = torch.randn(16, 64, dtype=torch.bfloat16)
    torch_indices = torch.tensor([1, 2, 3], dtype=torch.int64)
    torch_updates = torch.randn(2, 64, dtype=torch.bfloat16)  # 2 updates for 3 indices
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_updates = ttnn.from_torch(torch_updates, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "one row per index"):
        ttnn.set_rows(ttnn_input, ttnn_indices, ttnn_updates)


def test_set_rows_rejects_empty_indices(device, expect_error):
    """Empty index vectors are explicitly out of scope for this milestone (#15942)."""
    torch_input = torch.randn(16, 64, dtype=torch.bfloat16)
    torch_indices = torch.empty(0, dtype=torch.int64)
    torch_updates = torch.empty(0, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_updates = ttnn.from_torch(torch_updates, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "empty"):
        ttnn.set_rows(ttnn_input, ttnn_indices, ttnn_updates)


def test_set_rows_preallocated_output(device):
    torch.manual_seed(6)
    H, W, N = 128, 96, 33
    torch_input = torch.randn(H, W, dtype=torch.bfloat16)
    torch_indices = torch.randint(0, H, (N,), dtype=torch.int64)
    torch_updates = torch.randn(N, W, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_indices = ttnn.from_torch(torch_indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_updates = ttnn.from_torch(torch_updates, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.empty(
        [H, W], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.set_rows(ttnn_input, ttnn_indices, ttnn_updates, out=output)
    reference = torch_input.clone()
    reference.index_put_((torch_indices,), torch_updates)
    assert_equal(ttnn.to_torch(output), reference)

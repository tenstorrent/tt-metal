# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_fold import run_fold_test

pytestmark = pytest.mark.use_module_device


@pytest.mark.merge_gate
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float], ids=["bfloat16", "float32"])
@pytest.mark.parametrize(
    "input_shape, output_size, kernel_size, dilation, padding, stride",
    [
        # 32-value input rows are DRAM-aligned, so the reader reads them directly.
        ((1, 9, 32), (6, 10), (3, 3), (1, 1), (0, 0), (1, 1)),
        # 36-value input rows are not DRAM-aligned, so the reader goes through the scratch buffer (always on
        # Blackhole). 240 output rows give each core several; dilation, padding and stride hit every skip branch.
        ((5, 64, 36), (12, 12), (4, 4), (2, 2), (3, 3), (2, 2)),
    ],
    ids=["aligned_row", "unaligned_row"],
)
def test_moreh_fold(input_shape, output_size, kernel_size, dilation, padding, stride, dtype, device):
    torch.manual_seed(0)
    run_fold_test(device, input_shape, output_size, kernel_size, dilation, padding, stride, dtype)


@pytest.mark.merge_gate
def test_moreh_fold_rank_2_input(device):
    torch.manual_seed(0)
    # A rank-2 input has no batch dim; the factory then sets N = 1.
    run_fold_test(device, (128, 324), (32, 32), (4, 4), (2, 2), (5, 5), (2, 2), torch.bfloat16)


@pytest.mark.merge_gate
def test_moreh_fold_l1_short_row(device):
    torch.manual_seed(0)
    # Not the nightly helper: it has no memory_config argument.
    # Regression for #57475: an 8-value (16-byte) output row is shorter than its 32-byte-rounded page, and an
    # L1 output exposed the writer spilling that rounding padding past the row.
    torch_input = torch.randn((1, 4, 21), dtype=torch.bfloat16) + 1
    expected = torch.nn.functional.fold(torch_input, (4, 8), (2, 2), (1, 1), (0, 0), (1, 1))

    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output = ttnn.moreh_fold(
        tt_input, None, (4, 8), (2, 2), (1, 1), (0, 0), (1, 1), memory_config=ttnn.L1_MEMORY_CONFIG
    )

    passing, output_pcc = comp_allclose_and_pcc(expected, ttnn.to_torch(tt_output), rtol=0.05, atol=0.05)
    assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_fold_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_fold_test(device, (1, 9, 32), (6, 10), (3, 3), (1, 1), (0, 0), (1, 1), torch.bfloat16)
    num_program_cache_entries = device.num_program_cache_entries()
    # Without this, the equality below would also pass for an op that never caches a program.
    assert num_program_cache_entries > 0
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = ttnn.from_torch(torch.zeros([1, 9, 32]), device=device)
    run_fold_test(device, (1, 9, 32), (6, 10), (3, 3), (1, 1), (0, 0), (1, 1), torch.bfloat16)
    assert device.num_program_cache_entries() == num_program_cache_entries

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.skipif(not is_wormhole_b0(), reason="Wormhole-only program")

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "shape, perm, dtype, in_mem, out_mem",
    [
        ((64, 49, 3, 3, 32), (2, 0, 3, 1, 4), ttnn.bfloat16, L1, L1),
        ((25, 49, 3, 12, 32), (2, 0, 3, 1, 4), ttnn.bfloat16, L1, L1),
        ((16, 64, 3, 6, 32), (2, 0, 3, 1, 4), ttnn.bfloat16, L1, L1),
        ((25, 49, 3, 12, 32), (2, 0, 3, 1, 4), ttnn.float32, L1, L1),
        ((25, 49, 3, 12, 32), (2, 0, 3, 1, 4), ttnn.bfloat16, DRAM, DRAM),
        ((9, 49, 3, 24, 32), (2, 0, 3, 1, 4), ttnn.bfloat16, L1, DRAM),
        ((9, 49, 3, 24, 32), (2, 0, 3, 1, 4), ttnn.bfloat16, DRAM, L1),
        ((3, 5, 7, 2, 24), (2, 0, 3, 1, 4), ttnn.bfloat16, L1, L1),
        ((3, 5, 7, 2, 24), (2, 0, 3, 1, 4), ttnn.bfloat16, DRAM, L1),
        ((1, 8, 64, 32), (0, 2, 1, 3), ttnn.bfloat16, L1, L1),
        ((2, 3, 4, 5, 6, 16), (0, 3, 1, 4, 2, 5), ttnn.bfloat16, L1, L1),
        ((7, 11, 13, 40), (2, 1, 0, 3), ttnn.bfloat16, L1, L1),
        ((5, 9, 512), (1, 0, 2), ttnn.bfloat16, L1, L1),
        ((5, 9, 1024), (1, 0, 2), ttnn.bfloat16, DRAM, DRAM),
        ((3, 7, 2048), (1, 0, 2), ttnn.bfloat16, L1, L1),
        ((2, 64, 256), (1, 0, 2), ttnn.bfloat16, L1, L1),
        ((4, 64, 256), (1, 0, 2), ttnn.bfloat16, DRAM, L1),
        ((2, 64, 272), (1, 0, 2), ttnn.bfloat16, L1, L1),
        ((2, 63, 256), (1, 0, 2), ttnn.bfloat16, DRAM, DRAM),
        ((5, 25, 1, 32), (1, 0, 2, 3), ttnn.bfloat16, L1, L1),
        ((9, 49, 3, 6, 32), (2, 0, 3, 1, 4), ttnn.int32, L1, L1),
        ((2, 1000, 3, 16), (1, 2, 0, 3), ttnn.float32, DRAM, DRAM),
    ],
)
def test_permute_row_invariant_exact(device, shape, perm, dtype, in_mem, out_mem):
    torch.manual_seed(0)
    for _ in range(2):
        torch_input = torch.randn(shape, dtype=torch.float32)
        ttnn_input = ttnn.from_torch(
            torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
        )
        output = ttnn.permute(ttnn_input, perm, memory_config=out_mem)
        assert_equal(ttnn.to_torch(ttnn_input).permute(perm), ttnn.to_torch(output))


_HOG_PAGE_BYTES = 16


def _l1_hog_leaving(device, free_bytes_per_bank):
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    pages_per_bank = (view.largest_contiguous_bytes_free_per_bank - free_bytes_per_bank) // _HOG_PAGE_BYTES
    if pages_per_bank <= 0:
        pytest.skip("L1 already has less free space than the test needs")
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape((pages_per_bank * view.num_banks, _HOG_PAGE_BYTES // 2)),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )


def test_permute_row_invariant_with_less_free_l1(device):
    torch.manual_seed(0)
    perm = (2, 0, 3, 1, 4)
    torch_input = torch.randn((9, 49, 3, 6, 32), dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
    )
    expected = ttnn.to_torch(ttnn_input).permute(perm)
    assert_equal(expected, ttnn.to_torch(ttnn.permute(ttnn_input, perm, memory_config=DRAM)))
    entries = device.num_program_cache_entries()
    hog = _l1_hog_leaving(device, 24 * 1024)
    assert_equal(expected, ttnn.to_torch(ttnn.permute(ttnn_input, perm, memory_config=DRAM)))
    assert device.num_program_cache_entries() == entries + 1
    ttnn.deallocate(hog)
    assert_equal(expected, ttnn.to_torch(ttnn.permute(ttnn_input, perm, memory_config=DRAM)))
    assert device.num_program_cache_entries() == entries + 1

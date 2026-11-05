# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.skipif(not is_wormhole_b0(), reason="Wormhole-only program")


@pytest.mark.parametrize(
    "shape, dtype, mem_config, repeats",
    [
        ((1, 8, 384, 1), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, (1, 1, 1, 768)),
        ((1, 8, 384, 1), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (1, 1, 1, 768)),
        ((1, 32, 384, 1), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, (1, 1, 1, 384)),
        ((1, 1, 1024, 2), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, (1, 1, 1, 512)),
        ((1, 1, 512, 4), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, (1, 1, 1, 256)),
        ((1, 1, 256, 1), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, (1, 1, 1, 100)),
        ((1, 1, 256, 1), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, (1, 1, 1, 33)),
        ((1, 1, 77, 1), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, (1, 1, 1, 6)),
        ((1, 1, 77, 2), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (1, 1, 1, 24)),
        ((1, 8, 384, 1), ttnn.float32, ttnn.L1_MEMORY_CONFIG, (1, 1, 1, 768)),
        ((1, 1, 300, 2), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG, (1, 1, 1, 37)),
        ((1, 1, 300, 1), ttnn.float32, ttnn.L1_MEMORY_CONFIG, (1, 1, 1, 3)),
        ((1, 1, 600, 1), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (1, 1, 1, 96)),
        ((1, 1, 64, 2), ttnn.float32, ttnn.L1_MEMORY_CONFIG, (1, 1, 1, 2100)),
    ],
)
def test_repeat_last_dim_row_major_exact(device, shape, dtype, mem_config, repeats):
    torch.manual_seed(0)
    for _ in range(2):
        torch_input = torch.randn(shape, dtype=torch.float32)
        ttnn_input = ttnn.from_torch(
            torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
        )
        output = ttnn.repeat(ttnn_input, ttnn.Shape(repeats))
        assert_equal(ttnn.to_torch(ttnn_input).repeat(repeats), ttnn.to_torch(output))


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


def test_repeat_last_dim_row_major_with_less_free_l1(device):
    torch.manual_seed(0)
    repeats = (1, 1, 1, 768)
    torch_input = torch.randn((1, 8, 384, 1), dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    expected = ttnn.to_torch(ttnn_input).repeat(repeats)
    assert_equal(expected, ttnn.to_torch(ttnn.repeat(ttnn_input, ttnn.Shape(repeats))))
    hog = _l1_hog_leaving(device, 18 * 1024)
    assert_equal(expected, ttnn.to_torch(ttnn.repeat(ttnn_input, ttnn.Shape(repeats))))
    ttnn.deallocate(hog)
    assert_equal(expected, ttnn.to_torch(ttnn.repeat(ttnn_input, ttnn.Shape(repeats))))

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_equal


def test_fmod_nan(device):
    torch_input_a = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)
    torch_input_b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.fmod(tt_in_a, tt_in_b)
    output_tensor = ttnn.to_torch(tt_result)

    assert torch.equal(torch.isnan(golden), torch.isnan(output_tensor))

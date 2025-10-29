# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp


def test_sigmoid_accurate_arange(device):
    # Generate all possible bit patterns for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)

    # Mask NaN
    mask = torch.isnan(input_tensor)
    input_tensor[mask] = 1.0

    # Exp Working range - Overflow from 88.5(inf), Underflow till -87(<0)
    low = -87.0
    high = 88.5
    # masking to working range
    mask = (input_tensor >= low) & (input_tensor <= high)
    input_tensor = input_tensor[mask]

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.sigmoid_accurate)
    golden = golden_function(input_tensor, device=device)
    tt_result = ttnn.sigmoid_accurate(tt_in, fast_and_approximate_mode=False)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 3, allow_nonfinite=True)

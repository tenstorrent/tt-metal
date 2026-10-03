# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("value", [-0.0, 0.0])
def test_log1p_signed_zero(dtype, value, device):
    # IEEE 754 requires log1p(+-0) = +-0; torch.log1p agrees. calculate_log1p_fp32's first
    # step (u = a + 1.0f) destroys the sign of an exact zero input before it can propagate
    # through the rest of the range-reduction, so the accurate path previously always
    # returned +0.0 for log1p(-0.0). Regression test for that sign loss.
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    input_tensor = torch.tensor([value], dtype=torch_dtype)

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.log1p(tt_in)
    result = ttnn.to_torch(tt_result)

    golden_function = ttnn.get_golden_function(ttnn.log1p)
    golden = golden_function(input_tensor)

    # assert_with_ulp / plain equality both treat +0.0 == -0.0, so the sign has to be
    # checked explicitly via copysign rather than via a numeric-difference assertion.
    assert torch.copysign(torch.tensor(1.0), result[0]) == torch.copysign(torch.tensor(1.0), golden[0]), (
        f"log1p({value}) sign mismatch: got {result[0]} (torch reference {golden[0]})"
    )

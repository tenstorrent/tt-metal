# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_equal

pytestmark = pytest.mark.use_module_device


TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def _unary_with_preallocated_output(device, torch_input, in_dtype, out_dtype, ttnn_op):
    input_tensor = ttnn.from_torch(torch_input, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.from_torch(
        torch.zeros(torch_input.shape, dtype=TORCH_DTYPE[out_dtype]),
        dtype=out_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn_op(input_tensor, output_tensor=output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn_op)
    golden = golden_fn(torch_input).to(TORCH_DTYPE[out_dtype])
    result = ttnn.to_torch(output_tensor)
    assert output_tensor.dtype == out_dtype
    assert_with_ulp(golden, result, 0)
    return output_tensor


def test_neg_mixed_dtype(device):
    """Preallocated output dtype can differ from the input. Every in/out pair is
    invoked back-to-back to check for program-cache collision.
    ttsim [5082] ERROR: UndefinedBehavior: tensix_unpacr: unpack_to_dst=0 in_data_format=0 out_data_format=0"""
    torch.manual_seed(0)
    fixed_input = torch.ones((64, 64), dtype=torch.float32) * 1.22
    ttnn_op = ttnn.neg

    for in_dt in (ttnn.bfloat16, ttnn.float32):
        for out_dt in (ttnn.bfloat16, ttnn.float32):
            torch_input = fixed_input.to(TORCH_DTYPE[in_dt])
            _unary_with_preallocated_output(device, torch_input, in_dt, out_dt, ttnn_op)

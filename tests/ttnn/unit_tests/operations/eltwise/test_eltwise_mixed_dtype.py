# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device

# bfloat8_b has no torch equivalent; host tensors use bfloat16 (same mantissa width)
# so from_torch packing and ULP comparison share one resolution.
TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}

FLOAT_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32)
# ULP is measured in the output dtype. bfloat8_b is compared at bfloat16 resolution
# (see assert_with_ulp). Same-format bf16/fp32 neg is exact; block-float packing
# and host vs device bfp8 packers can differ.
ULP_THRESHOLD = {
    (ttnn.bfloat16, ttnn.bfloat16): 0,
    (ttnn.bfloat16, ttnn.float32): 0,
    (ttnn.float32, ttnn.bfloat16): 0,
    (ttnn.float32, ttnn.float32): 0,
    (ttnn.bfloat8_b, ttnn.bfloat8_b): 1,
    (ttnn.bfloat8_b, ttnn.bfloat16): 1,
    (ttnn.bfloat8_b, ttnn.float32): 1,
    (ttnn.bfloat16, ttnn.bfloat8_b): 1,
    (ttnn.float32, ttnn.bfloat8_b): 1,
}


def _unary_with_preallocated_output(device, torch_input, in_dtype, out_dtype, ttnn_op, ulp_threshold=0):
    input_tensor = ttnn.from_torch(torch_input, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.from_torch(
        torch.zeros(torch_input.shape, dtype=TORCH_DTYPE[out_dtype]),
        dtype=out_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn_op(input_tensor, output_tensor=output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn_op)
    # Device-visible input captures bfloat8_b packing so golden matches the kernel.
    # Pass the ttnn tensor into assert_with_ulp so bfloat8_b is compared at bfloat16
    # resolution; ttnn.to_torch() would upcast it to float32 and make ULP too strict.
    golden = golden_fn(ttnn.to_torch(input_tensor)).to(TORCH_DTYPE[out_dtype])
    assert output_tensor.dtype == out_dtype
    assert_with_ulp(golden, output_tensor, ulp_threshold)


def test_neg_mixed_dtype(device):
    """Preallocated output dtype can differ from the input. Every in/out pair is
    invoked back-to-back to check for program-cache collision."""
    torch.manual_seed(0)
    fixed_input = torch.ones((64, 64), dtype=torch.float32) * 1.22
    ttnn_op = ttnn.neg

    for in_dt in FLOAT_DTYPES:
        for out_dt in FLOAT_DTYPES:
            torch_input = fixed_input.to(TORCH_DTYPE[in_dt])
            _unary_with_preallocated_output(
                device, torch_input, in_dt, out_dt, ttnn_op, ulp_threshold=ULP_THRESHOLD[(in_dt, out_dt)]
            )

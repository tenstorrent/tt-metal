# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from tests.ttnn.utils_for_testing import assert_with_ulp, generate_all_bfloat16_bitpatterns


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_sigmoid(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)

    tt_output_tensor_on_device = ttnn.sigmoid_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.sigmoid_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor, 0.90)
    assert comp_pass


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bw_sigmoid_all_bitpatterns(device, dtype):
    """Exhaustive: every bfloat16 bit pattern as the input, with an all-ones gradient.

    d/dx sigmoid(x) is sigmoid(x) * sigmoid(-x), which is a normal float out to |x| = 87.34,
    so no finite input in the swept set has a zero gradient.  Building it as s * (1 - s) lost
    the whole x >= 6.25 (bfloat16) / x >= 16.75 (float32) tail: s rounds to exactly 1 there
    and the subtraction cancels to 0.

    The reference is evaluated in float64 rather than through the registered golden so that
    the comparison is not limited by the reference's own float32 rounding.

    Excluded: subnormal inputs (hardware flushes them to zero), NaN and +/-inf (covered by
    the special-value tests), and inputs whose exact gradient is itself subnormal.
    """
    x2d = generate_all_bfloat16_bitpatterns(dtype)
    x = x2d.flatten()
    tt_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32

    tt_in = ttnn.from_torch(
        x2d, dtype=tt_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_grad = ttnn.from_torch(
        torch.ones_like(x2d),
        dtype=tt_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.to_torch(ttnn.sigmoid_bw(tt_grad, tt_in)[0]).flatten().to(dtype)
    x64 = x.double()
    golden64 = torch.sigmoid(x64) * torch.sigmoid(-x64)
    golden = golden64.to(dtype)

    tiny = torch.finfo(torch.float32).tiny
    finite_in = torch.isfinite(x) & ((x == 0) | (x.abs() >= tiny))
    normal_out = (golden64.abs() >= tiny) & torch.isfinite(golden64)
    checked = finite_in & normal_out

    lost = checked & (result == 0)
    assert lost.sum() == 0, f"{int(lost.sum())} inputs returned a zero gradient, first at x={float(x[lost][0])}"
    assert_with_ulp(expected_result=golden[checked], actual_result=result[checked], ulp_threshold=8)

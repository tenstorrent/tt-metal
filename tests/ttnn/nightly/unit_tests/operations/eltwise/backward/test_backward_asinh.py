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
def test_bw_asinh(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.asinh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.asinh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bw_asinh_all_bitpatterns(device, dtype):
    """Exhaustive: every bfloat16 bit pattern as the input, with an all-ones gradient.

    d/dx asinh(x) is 1/sqrt(1 + x^2), a normal float for every |x| up to 2^126, but squaring
    overflows above 2^64 and rsqrt(inf) is 0, so the gradient used to vanish over the whole
    2^64 < |x| <= FLT_MAX band - 15,874 of the 65,536 bit patterns.

    The reference is evaluated in float64 rather than through the registered golden so that
    the comparison is not limited by the reference's own float32 rounding.

    Excluded: subnormal inputs (hardware flushes them to zero), NaN and +/-inf (covered by
    the special-value tests), and gradients whose exact value is below 2^-100, where the
    accuracy is that of ttnn.reciprocal at the bottom of the normal range rather than
    anything this derivative does (ttnn.reciprocal(2^126) is itself 0, not FLT_MIN).
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

    result = ttnn.to_torch(ttnn.asinh_bw(tt_grad, tt_in)[0]).flatten().to(dtype)
    x64 = x.double()
    golden64 = 1.0 / torch.sqrt(1.0 + x64 * x64)
    golden = golden64.to(dtype)

    tiny = torch.finfo(torch.float32).tiny
    finite_in = torch.isfinite(x) & ((x == 0) | (x.abs() >= tiny))
    checked = finite_in & (golden64 >= 2.0**-100)

    lost = checked & (result == 0)
    assert lost.sum() == 0, f"{int(lost.sum())} inputs returned a zero gradient, first at x={float(x[lost][0])}"
    assert_with_ulp(expected_result=golden[checked], actual_result=result[checked], ulp_threshold=4)

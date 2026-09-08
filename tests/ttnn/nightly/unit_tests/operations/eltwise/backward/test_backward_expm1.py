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
def test_bw_expm1(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.expm1_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.expm1_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bw_expm1_all_bitpatterns(device, dtype):
    """Exhaustive: every bfloat16 bit pattern as the input, with an all-ones gradient.

    d/dx expm1(x) is exp(x), which is a normal float down to x = -87.34, so no finite input
    in the swept set has a zero gradient.  Building it as expm1(x) + 1 lost the whole
    x <= -6.25 (bfloat16) / x <= -17 (float32) tail to cancellation and returned exactly 0.

    The reference is evaluated in float64 rather than through the registered golden so that
    the comparison is not limited by the reference's own float32 rounding.

    Excluded: subnormal inputs (hardware flushes them to zero), NaN and +/-inf (covered by
    the special-value tests), and inputs whose exact gradient is outside the float32 normal
    range (exp overflows above x = 88.7 and is subnormal below x = -87.34).
    """
    x2d = generate_all_bfloat16_bitpatterns(dtype)
    x = x2d.flatten()

    tt_in = ttnn.from_torch(
        x2d,
        dtype=ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_grad = ttnn.from_torch(
        torch.ones_like(x2d),
        dtype=ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.to_torch(ttnn.expm1_bw(tt_grad, tt_in)[0]).flatten().to(dtype)
    golden64 = torch.exp(x.double())
    golden = golden64.to(dtype)

    tiny = torch.finfo(torch.float32).tiny
    finite_in = torch.isfinite(x) & ((x == 0) | (x.abs() >= tiny))
    normal_out = (golden64.abs() >= tiny) & (golden64.abs() <= torch.finfo(torch.float32).max)
    checked = finite_in & normal_out

    lost = checked & (result == 0)
    assert lost.sum() == 0, f"{int(lost.sum())} inputs returned a zero gradient, first at x={float(x[lost][0])}"
    assert_with_ulp(golden[checked], result[checked], ulp_threshold=2)

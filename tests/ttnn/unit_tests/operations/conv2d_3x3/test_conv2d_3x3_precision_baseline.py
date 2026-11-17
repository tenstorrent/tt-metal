# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for conv2d_3x3 on the supported interleaved ROW_MAJOR bf16 path.

Reference: torch fp32 conv on the same bf16 activation and the same (bf16) torch weight, so a bf8_b
weight run carries its quantization error in the numbers. Printed per case (run with -s):
PCC, max / mean abs error, relative RMS (rms(err) / std(ref)), bf16 ULP vs the bf16-rounded
reference, and the spread of actual / expected (a tight cluster away from 1.0 is a scale bug,
not rounding).
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import calculate_detailed_ulp_stats, comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights

# Golden-suite gates (eval/golden_tests/conv2d_3x3/helpers.py TOLERANCES), keyed by the coarser dtype.
GATES = {ttnn.bfloat16: (0.997, 0.04), ttnn.bfloat8_b: (0.99, 0.12)}

# ((H, W, C_in), C_out, padding, dilation)
SHAPES = [
    ((8, 8, 32), 32, 1, 1),  # single core, K = 288
    ((16, 16, 64), 128, 1, 1),  # 8 cores, multi-tile N
    ((32, 64, 128), 128, 6, 6),  # 64 cores, dilated (ASPP-like reach)
    ((8, 16, 2048), 256, 1, 1),  # 4 cores, K = 18432, K-blocks below one kernel row
]


def _shape_id(case):
    (H, W, c_in), c_out, padding, dilation = case
    return f"H{H}_W{W}_Ci{c_in}_Co{c_out}_p{padding}_d{dilation}"


def _ratio_spread(actual, expected):
    mask = torch.isfinite(actual) & (expected.abs() > 1e-3 * expected.abs().max())
    r = (actual[mask] / expected[mask]).double()
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return q[1].item(), q[0].item(), q[2].item()


@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["w_bf16", "w_bf8b"])
@pytest.mark.parametrize("case", SHAPES, ids=[_shape_id(c) for c in SHAPES])
def test_conv2d_3x3_precision_baseline(case, weight_dtype, device):
    torch.manual_seed(1234)
    (H, W, c_in), c_out, padding, dilation = case
    prepare_kwargs = dict(
        stride=1,
        padding=padding,
        dilation=dilation,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weight_dtype=weight_dtype,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_weight = prepare_conv2d_weights(torch_weight, **prepare_kwargs)
    tt_bias = prepare_conv2d_bias(torch_bias, **prepare_kwargs)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = conv2d_3x3(
        tt_input, tt_weight, input_height=H, input_width=W, bias=tt_bias, padding=padding, dilation=dilation
    )
    actual = ttnn.to_torch(tt_output).float()

    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), padding=padding, dilation=dilation)
    expected = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)

    err = actual - expected
    rel_rms = (err.pow(2).mean().sqrt() / expected.std()).item()
    _, allclose_msg = comp_allclose(expected, actual)
    ulp = calculate_detailed_ulp_stats(expected.to(torch.bfloat16), actual.to(torch.bfloat16))
    r_med, r_p5, r_p95 = _ratio_spread(actual, expected)
    pcc_gate, rms_gate = GATES[weight_dtype]
    _, pcc_msg = assert_with_pcc(expected, actual, pcc_gate)
    print(
        f"\nBASELINE {_shape_id(case)} w={weight_dtype}: {pcc_msg} | max_abs={err.abs().max().item():.4g} "
        f"mean_abs={err.abs().mean().item():.4g} rel_rms={rel_rms:.4g} | {allclose_msg} | "
        f"ulp median={ulp['median_ulp']:.1f} p99={ulp['p99_ulp']:.1f} max={ulp['max_ulp']:.0f} "
        f"exact={ulp['perfect_matches']:.3f} | ratio median={r_med:.5f} p5={r_p5:.5f} p95={r_p95:.5f}"
    )
    assert rel_rms <= rms_gate, f"relative RMS {rel_rms:.4g} above the golden gate {rms_gate}"

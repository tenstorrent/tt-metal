# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics

import os

TEST_PADDING_VALUE = -42
pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("h", [24, 32, 384])
@pytest.mark.parametrize("w", [42, 64, 1024])
def test_rms_norm(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.fill_implicit_tile_padding(weight, TEST_PADDING_VALUE)
    output_tensor = ttnn.rms_norm(input_tensor, weight=weight)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.035,
        atol=0.043,
        frobenius_threshold=0.008,
        ulp_threshold=9,
        check_ulp=True,
    )


# The weight_dtype axis exercises the ROW_MAJOR weight reader for both a 2-byte (bf16) and a 4-byte
# (fp32) element, so its element-size handling is covered rather than only the bf16 case.
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [24, 128])
@pytest.mark.parametrize("w", [32, 4096])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2])
@pytest.mark.parametrize("math_approx_mode", [True, False])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
@pytest.mark.parametrize("packer_l1_acc", [True, False])
def test_rms_norm_row_major(
    device, batch_size, h, w, weight_dtype, math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc
):
    if weight_dtype == torch.float32 and w == 4096:
        # This path buffers the full width of the weight in SRAM; at w=4096 the fp32 weight CB
        # alone is 4096 tiles-worth of Float32 and, together with the other per-width CBs, exceeds
        # the SRAM budget on a single core. The fp32 ROW_MAJOR weight is still exercised at w=32.
        pytest.skip("fp32 weight at w=4096 exceeds the single-core SRAM budget for this CB layout")

    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=weight_dtype)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    # For ROW_MAJOR layout, weight's last padded dim needs to equal tile width,
    # additionally, weight's volume needs to be equal to the last padded dim of the input.
    tile_width = 32
    assert w % tile_width == 0
    torch_weight_reshaped = torch_weight.reshape(w // tile_width, tile_width)
    # from_torch infers the ttnn dtype from the torch tensor, so the weight's ttnn dtype follows weight_dtype.
    weight = ttnn.from_torch(torch_weight_reshaped, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )

    output_tensor = ttnn.rms_norm(input_tensor, weight=weight, compute_kernel_config=compute_config)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.091,
        atol=0.129,
        frobenius_threshold=0.09,
    )


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [24, 2048])
@pytest.mark.parametrize("w", [42, 4022])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_with_weight_and_residual(device, batch_size, h, w, dtype):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=dtype)
    torch_residual_input_tensor = torch.rand((batch_size, h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor + torch_residual_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    residual_input_tensor = ttnn.fill_implicit_tile_padding(residual_input_tensor, TEST_PADDING_VALUE)
    weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.fill_implicit_tile_padding(weight, TEST_PADDING_VALUE)
    # Data is unpacked as Tf32, fp32 dest accumulation is required
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    output_tensor = ttnn.rms_norm(
        input_tensor, residual_input_tensor=residual_input_tensor, weight=weight, compute_kernel_config=compute_config
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if dtype == torch.bfloat16:
        rtol = 0.055
        atol = 0.069
        frobenius_threshold = 0.012
    else:
        rtol = 0.052
        atol = 0.064
        frobenius_threshold = 0.012

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )

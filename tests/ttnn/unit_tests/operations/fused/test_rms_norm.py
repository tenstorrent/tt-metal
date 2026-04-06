# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("h", [32, 384])
@pytest.mark.parametrize("w", [64, 1024])
def test_rms_norm(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.rms_norm(input_tensor, weight=weight)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [32, 4096])
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2])
@pytest.mark.parametrize("math_approx_mode", [True, False])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
@pytest.mark.parametrize("packer_l1_acc", [True, False])
def test_rms_norm_row_major(device, batch_size, h, w, math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc):
    torch.manual_seed(0)

    if fp32_dest_acc_en and device.arch() == ttnn.device.Arch.BLACKHOLE:
        pytest.skip("Skipping test on Blackhole with fp32_dest_acc_en=True, see issue #38561")

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)

    # For ROW_MAJOR layout, weight's last padded dim needs to equal tile width,
    # additionally, weight's volume needs to be equal to the last padded dim of the input.
    tile_width = 32
    assert w % tile_width == 0
    torch_weight_reshaped = torch_weight.reshape(w // tile_width, tile_width)
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

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [4022])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_with_weight_and_residual(device, batch_size, h, w, dtype):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=dtype)
    torch_residual_input_tensor = torch.rand((batch_size, h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor + torch_residual_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.rms_norm(input_tensor, residual_input_tensor=residual_input_tensor, weight=weight)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)

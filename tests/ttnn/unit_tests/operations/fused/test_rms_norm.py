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


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [24, 128])
@pytest.mark.parametrize("w", [32, 4096])
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2])
@pytest.mark.parametrize("math_approx_mode", [True, False])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
@pytest.mark.parametrize("packer_l1_acc", [True, False])
def test_rms_norm_row_major(device, batch_size, h, w, math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

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


@pytest.mark.parametrize("h, w", [(32, 2048), (2048, 2048), (64, 96), (24, 42)])
@pytest.mark.parametrize("weight_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
def test_rms_norm_residual_output(device, h, w, weight_layout, fp32_dest_acc_en, buffer_type):
    """residual_output_tensor: one op writes h = x + y and n = rms_norm(h) * gamma, bit-identical to
    ttnn.add followed by ttnn.rms_norm with the same compute config."""
    if weight_layout == ttnn.ROW_MAJOR_LAYOUT and w % 32 != 0:
        pytest.skip("ROW_MAJOR gamma is built as [1, 1, w // 32, 32]")
    torch.manual_seed(0)
    torch_x = torch.randn((1, 1, h, w), dtype=torch.bfloat16)
    torch_y = torch.randn((1, 1, h, w), dtype=torch.bfloat16) * 0.3
    torch_weight = torch.randn((w,), dtype=torch.bfloat16) * 0.1 + 1
    mem_config = ttnn.L1_MEMORY_CONFIG if buffer_type == ttnn.BufferType.L1 else ttnn.DRAM_MEMORY_CONFIG

    x = ttnn.from_torch(torch_x, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    y = ttnn.from_torch(torch_y, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    if weight_layout == ttnn.ROW_MAJOR_LAYOUT:
        weight = ttnn.from_torch(
            torch_weight.reshape(1, 1, w // 32, 32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )
    else:
        weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )
    eps = 1e-6

    h_ref = ttnn.add(x, y, memory_config=mem_config)
    n_ref = ttnn.rms_norm(
        h_ref, epsilon=eps, weight=weight, memory_config=mem_config, compute_kernel_config=compute_config
    )

    for _ in range(2):  # the second call is a program cache hit with a new residual output
        h_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, h, w]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem_config
        )
        n_out = ttnn.rms_norm(
            x,
            epsilon=eps,
            weight=weight,
            residual_input_tensor=y,
            memory_config=mem_config,
            compute_kernel_config=compute_config,
            residual_output_tensor=h_out,
        )
        assert torch.equal(ttnn.to_torch(h_out), ttnn.to_torch(h_ref))
        assert torch.equal(ttnn.to_torch(n_out), ttnn.to_torch(n_ref))


def test_rms_norm_residual_output_validation(device, expect_error):
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    y = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    h_bad_shape = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 64, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    h_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    with expect_error(RuntimeError, "residual_output_tensor requires residual_input_tensor"):
        ttnn.rms_norm(x, residual_output_tensor=h_out)
    with expect_error(RuntimeError, "shapes must match the input"):
        ttnn.rms_norm(x, residual_input_tensor=y, residual_output_tensor=h_bad_shape)
    with expect_error(RuntimeError, "must not alias"):
        ttnn.rms_norm(x, residual_input_tensor=y, residual_output_tensor=y)

# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics

import os

TEST_PADDING_VALUE = -42
pytestmark = pytest.mark.use_module_device


@pytest.mark.merge_gate
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


@pytest.mark.parametrize(
    "x, eps, w, b",
    [
        (0.0, 1e-5, None, None),
        (0.001, 0.25, None, None),
        (-2.0, 4.0, None, None),
        (3.0, 1e-5, None, None),
        (0.001, 0.25, 2.0, 0.5),
        (0.0, 1e-5, 1.5, -0.25),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_rms_norm_0d(device, x, eps, w, b, dtype):
    # RMSNorm over a single element is x / sqrt(x^2 + eps); epsilon must not be dropped.
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.tensor(x, dtype=torch_dtype)
    torch_weight = None if w is None else torch.tensor(w, dtype=torch_dtype)
    torch_bias = None if b is None else torch.tensor(b, dtype=torch_dtype)

    expected = torch.nn.functional.rms_norm(
        torch_input.reshape(1).double(),
        (1,),
        None if torch_weight is None else torch_weight.reshape(1).double(),
        eps,
    )
    if torch_bias is not None:
        expected = expected + torch_bias.double()

    def to_device(t):
        return None if t is None else ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.rms_norm(
        to_device(torch_input), epsilon=eps, weight=to_device(torch_weight), bias=to_device(torch_bias)
    )
    output = ttnn.to_torch(output).double()

    assert output.shape == torch_input.shape
    rtol, atol = (1e-3, 1e-6) if dtype == ttnn.float32 else (2e-2, 1e-4)
    assert torch.allclose(
        output.reshape(1), expected, rtol=rtol, atol=atol
    ), f"x={x} eps={eps} w={w} b={b}: ttnn={output.item()!r} torch={expected.item()!r}"

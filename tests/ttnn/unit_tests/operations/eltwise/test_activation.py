# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp, assert_allclose


def run_activation_unary_test(device, h, w, ttnn_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hardtanh(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.hardtanh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sigmoid_accurate(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sigmoid_accurate)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hardswish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.hardswish)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log_sigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.log_sigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_mish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.mish)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_relu6(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.relu6)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gelu(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.gelu)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([1, 3, 320, 320])),
    ),
)
@pytest.mark.parametrize(
    "low, high, atol, rtol",
    [
        (-6, -3, 1e-2, 1e-2),  # Strong negative saturation region
        (-3, 0, 1e-3, 1e-3),  # Negative transition region
        (0, 3, 1e-2, 1e-2),  # Positive transition region
        (3, 6, 1e-3, 1e-3),  # Positive saturation region
    ],
)
def test_gelu_accurate_allclose(input_shapes, low, high, atol, rtol, device):
    """Test GELU accuracy using allclose for different input regions matching analysis ranges"""
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.gelu)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.gelu(tt_in)
    result = ttnn.to_torch(tt_result)
    # Use allclose with range-specific tolerances
    assert_allclose(result, golden, atol=atol, rtol=rtol)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hardsigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.hardsigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sign(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sign)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_softsign(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.softsign)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_swish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.swish)


def run_activation_softplus_test(device, h, w, beta, threshold, ttnn_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, beta=beta, threshold=threshold)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output_tensor = ttnn_function(input_tensor_a, beta=beta, threshold=threshold, queue_id=0)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("beta", [-1, 0.5, 1, 2])
@pytest.mark.parametrize("threshold", [-20, 5, 10, 20, 40])
def test_softplus(device, h, w, beta, threshold):
    run_activation_softplus_test(device, h, w, beta, threshold, ttnn.softplus)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tanhshrink(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.tanhshrink)


def run_activation_unary_test_glu(device, batch_size, h, w, dim, ttnn_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16).unsqueeze(0)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dim", [-1, 3])
def test_glu(device, batch_size, h, w, dim):
    run_activation_unary_test_glu(device, batch_size, h, w, dim, ttnn.glu)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dim", [-1, 3])
def test_reglu(device, batch_size, h, w, dim):
    run_activation_unary_test_glu(device, batch_size, h, w, dim, ttnn.reglu)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dim", [-1, 3])
def test_swiglu(device, batch_size, h, w, dim):
    run_activation_unary_test_glu(device, batch_size, h, w, dim, ttnn.swiglu)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dim", [-1, 3])
def test_geglu(device, batch_size, h, w, dim):
    run_activation_unary_test_glu(device, batch_size, h, w, dim, ttnn.geglu)


def torch_prelu(x, *args, weight, **kwargs):
    result = torch.nn.functional.prelu(x, torch.tensor(weight, dtype=x.dtype))
    return result


def run_activation_test_elu(device, h, w, scalar, ttnn_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, alpha=scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, alpha=scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_activation_test_leaky_relu(device, h, w, scalar, ttnn_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, negative_slope=scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, negative_slope=scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_activation_test_scalarB(device, h, w, scalar, ttnn_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_activation_test_scalarB_key(device, h, w, value, ttnn_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, value=value)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, value)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("scalar", [-0.5, 0, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_elu(device, h, w, scalar):
    run_activation_test_elu(device, h, w, scalar, ttnn.elu)


@pytest.mark.parametrize("alpha", [1, 2.5, 5.0, -1, -5, 0])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype",
    [(torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16), (torch.bfloat16, ttnn.bfloat4_b)],
)
def test_scalarB_celu(device, h, w, alpha, torch_dtype, ttnn_dtype):
    if alpha == 0:
        pytest.skip("alpha=0 is not supported")

    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.celu)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat4_b:
        torch_input_tensor_a = ttnn.to_torch(input_tensor_a)

    torch_output_tensor = golden_function(torch_input_tensor_a, alpha=alpha)

    output_tensor = ttnn.celu(input_tensor_a, alpha=alpha)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(torch_output_tensor, output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("scalar", [0.5, 1.0])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_hardshrink(device, h, w, scalar):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.hardshrink)
    torch_output_tensor = golden_function(torch_input_tensor_a, lambd=scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.hardshrink(input_tensor_a, lambd=scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("value", [0.88])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_heaviside(device, h, w, value):
    run_activation_test_scalarB_key(device, h, w, value, ttnn.heaviside)


@pytest.mark.parametrize("scalar", [-0.5, 0, 0.1, 0.01, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_leaky_relu(device, h, w, scalar):
    run_activation_test_leaky_relu(device, h, w, scalar, ttnn.leaky_relu)


@pytest.mark.parametrize("weight", [-0.5, 1.0, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_prelu(device, h, w, weight):
    torch.manual_seed(0)
    ttnn_function = ttnn.prelu
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_prelu(torch_input_tensor_a, weight=weight)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, weight)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("scalar", [0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_softshrink(device, h, w, scalar):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.softshrink)
    torch_output_tensor = golden_function(torch_input_tensor_a, lambd=scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.softshrink(input_tensor_a, lambd=scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


def run_activation_test_scalarBC_key(device, h, w, scalar1, scalar2, ttnn_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)

    torch_output_tensor = golden_function(torch_input_tensor_a, scalar1, scalar2)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("min", [-0.5, -0.1, -5.5])
@pytest.mark.parametrize("max", [0.5, 1.5, 27.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarBC_clip(device, h, w, min, max):
    run_activation_test_scalarBC_key(device, h, w, min, max, ttnn.clip)


def run_activation_test_threshold(device, h, w, scalar1, scalar2, ttnn_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)

    torch_output_tensor = golden_function(torch_input_tensor_a, value=scalar1, threshold=scalar2)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("value", [-0.5, -0.1, -5.5])
@pytest.mark.parametrize("threshold", [-0.5, 1.5, 27.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_threshold(device, h, w, value, threshold):
    run_activation_test_threshold(device, h, w, value, threshold, ttnn.threshold)

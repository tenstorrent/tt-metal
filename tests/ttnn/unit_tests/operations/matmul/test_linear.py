# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from ttnn.operations.activations import get_golden_function_for_activation
from loguru import logger

from models.common.utility_functions import (
    torch_random,
    is_blackhole,
    skip_for_blackhole,
    is_llk_assert_enabled,
    skip_for_slow_dispatch,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_numeric_metrics
from tests.ttnn.unit_tests.operations.matmul.test_matmul import is_tiny_tile_combo_supported


pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("batch_sizes", [(1,)])
@pytest.mark.parametrize("m_size", [384])
@pytest.mark.parametrize("k_size", [1024])
@pytest.mark.parametrize("n_size", [1024])
@pytest.mark.parametrize("use_bias", [True, False])
def test_linear(
    batch_sizes,
    m_size,
    k_size,
    n_size,
    use_bias,
    *,
    device,
):
    torch.manual_seed(0)
    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    if use_bias:
        torch_bias = torch_random((n_size,), -0.1, 0.1, dtype=torch.float32)
    else:
        torch_bias = None
    torch_output_tensor = torch.nn.functional.linear(
        torch_input_tensor_a, torch_input_tensor_b.T.contiguous(), bias=torch_bias
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    if use_bias:
        bias = ttnn.from_torch(
            torch_bias.reshape((1, n_size)),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        bias = None

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        atol=0.001 * k_size,
        rtol=0.016 * k_size,
        frobenius_threshold=0.001 * k_size,
        pcc_threshold=0.999,
        check_ulp=False,
    )


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [384])
@pytest.mark.parametrize("k_size", [1024])
@pytest.mark.parametrize("n_size", [1024])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("core_grid", [False])
def test_linear_with_core_grid(
    batch_size,
    m_size,
    k_size,
    n_size,
    use_bias,
    core_grid,
    *,
    device,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")
    torch.manual_seed(0)
    input_shape_a = (batch_size, 1, m_size, k_size)
    input_shape_b = (k_size, n_size)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    if use_bias:
        torch_bias = torch_random((n_size,), -0.1, 0.1, dtype=torch.float32)
    else:
        torch_bias = None
    torch_output_tensor = torch.nn.functional.linear(
        torch_input_tensor_a, torch_input_tensor_b.T.contiguous(), bias=torch_bias
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    if use_bias:
        bias = ttnn.from_torch(
            torch_bias.reshape((1, n_size)),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        bias = None

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        core_grid=ttnn.CoreGrid(y=batch_size, x=6),
    )

    output_tensor = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        atol=0.001 * k_size,
        rtol=0.055 * k_size,
        frobenius_threshold=0.001 * k_size,
        pcc_threshold=0.999,
        check_ulp=False,
    )


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [32, 64])
@pytest.mark.parametrize("k_size", [1024])
@pytest.mark.parametrize("n_size", [1024])
@pytest.mark.parametrize("activation", [None, "relu", "silu", "gelu", "gelu_approx", "gelu_tanh", "relu6"])
def test_wide_linear_with_argument_for_core_grid_set_to_device_grid(
    device, batch_size, m_size, k_size, n_size, activation
):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    if activation is not None:
        torch_output_tensor = get_golden_function_for_activation(activation)(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.linear(input_tensor_a, input_tensor_b, core_grid=device.core_grid, activation=activation)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        atol=0.005 * k_size,
        rtol=3.125 * k_size,
        frobenius_threshold=0.001 * k_size,
        pcc_threshold=0.997,
        check_ulp=False,
    )


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [32, 64])
@pytest.mark.parametrize("k_size", [1024])
@pytest.mark.parametrize("n_size", [1024])
@pytest.mark.parametrize(
    "activation",
    [
        None,
        "relu",
        "silu",
        "gelu",
        "sigmoid",
        "hardsigmoid",
        "mish",
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH),
    ],
)
def test_linear_with_compound_activation(device, batch_size, m_size, k_size, n_size, activation):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)

    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b
    if activation is not None:
        torch_output_tensor = get_golden_function_for_activation(activation)(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    # We supply no program config or core grid, so this uses the unfused path.
    output_tensor = ttnn.linear(input_tensor_a, input_tensor_b, activation=activation)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        atol=0.003 * k_size,
        rtol=1.321 * k_size,
        frobenius_threshold=0.001 * k_size,
        pcc_threshold=0.997,
        check_ulp=False,
    )


@pytest.mark.parametrize(
    "activation",
    [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDSIGMOID),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, -1.0, 1.0),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SELU, 1.6732632, 1.0507009),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0),
    ],
)
def test_linear_fused_activation_numerical_stability(device, activation):
    """Each supported fused activation must produce finite outputs (no NaN, no Inf)."""
    torch.manual_seed(0)
    m, k, n = 64, 128, 64
    in0 = torch.randn(1, 1, m, k, dtype=torch.bfloat16)
    in1 = torch.randn(1, 1, k, n, dtype=torch.bfloat16)

    in0_t = ttnn.from_torch(in0, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.linear(in0_t, in1_t, bias=None, activation=activation)
    out_torch = ttnn.to_torch(out).to(torch.float32)

    assert not torch.isnan(out_torch).any(), f"{activation.op_type} produced NaN"
    assert not torch.isinf(out_torch).any(), f"{activation.op_type} produced Inf"


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [32, 64])
@pytest.mark.parametrize("k_size", [1024])
@pytest.mark.parametrize("n_size", [1024])
@pytest.mark.parametrize("activation", [None, "relu"])
def test_linear_by_passing_in_1D_systolic_array_program_config(device, batch_size, m_size, k_size, n_size, activation):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b
    if activation == "relu":
        torch_output_tensor = torch.relu(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        activation=activation,
        core_grid=device.core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        atol=0.005 * k_size,
        rtol=2.266 * k_size,
        frobenius_threshold=0.001 * k_size,
        pcc_threshold=0.997,
        check_ulp=False,
    )


@pytest.mark.parametrize("m_size", [32, 512])
@pytest.mark.parametrize("k_size", [1024, 2048])
@pytest.mark.parametrize("n_size", [1024, 2048])
def test_linear_fp32_acc(device, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((1, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        core_grid=device.core_grid,
        compute_kernel_config=compute_kernel_config,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        atol=0.063 * k_size,
        rtol=0.115 * k_size,
        frobenius_threshold=0.001 * k_size,
        pcc_threshold=0.997,
        check_ulp=False,
    )


def test_bloom_ff2_linear(device):
    torch.manual_seed(0)
    torch_input_tensor = torch_random((8, 384, 4096), -0.1, 0.1, dtype=torch.float32)
    torch_weight = torch_random((4096, 1024), -0.1, 0.1, dtype=torch.float32)
    torch_bias = torch_random((1024,), -0.01, 0.01, dtype=torch.float32)

    torch_output = torch_input_tensor @ torch_weight + torch_bias

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    weights = ttnn.from_torch(
        torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    bias = ttnn.from_torch(
        torch_bias.reshape((1, -1)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    output = ttnn.linear(
        input_tensor,
        weights,
        bias=bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    output_torch = ttnn.to_torch(output)
    assert_numeric_metrics(
        torch_output,
        output_torch,
        atol=0.001 * 4096,
        rtol=0.02 * 4096,
        frobenius_threshold=0.001 * 4096,
        pcc_threshold=0.9992,
        check_ulp=False,
    )


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [32, 64])
@pytest.mark.parametrize("k_size", [1024, 2048])
@pytest.mark.parametrize("n_size", [1024, 2048])
@pytest.mark.parametrize("activation", [None, "relu"])
def test_linear_by_passing_in_1D_systolic_array_program_config_and_optional_outout_tensor(
    device, batch_size, m_size, k_size, n_size, activation
):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b
    if activation == "relu":
        torch_output_tensor = torch.relu(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    torch_opt_output_tensor = torch.zeros_like(torch_output_tensor)
    optional_output_tensor = ttnn.from_torch(torch_opt_output_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        activation=activation,
        core_grid=device.core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)

    ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        activation=activation,
        optional_output_tensor=optional_output_tensor,
        core_grid=device.core_grid,
    )

    optional_output_tensor = ttnn.to_torch(optional_output_tensor)

    assert len(output_tensor.shape) == len(torch_output_tensor.shape) == len(optional_output_tensor.shape)
    assert output_tensor.shape == torch_output_tensor.shape == optional_output_tensor.shape
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        atol=0.0059 * k_size,
        rtol=7.9688 * k_size,
        frobenius_threshold=0.0001 * k_size,
        pcc_threshold=0.997,
    )
    assert_with_pcc(torch_output_tensor, optional_output_tensor, 0.997)
    assert_with_pcc(optional_output_tensor, output_tensor, 0.997)


def test_linear_with_fp32_dest_acc_and_bias(device):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.rand([64, 1, 256, 384])
    torch_input_tensor_b = torch.rand([1, 1, 1152, 384])
    torch_input_tensor_c = torch.rand([1, 1, 1, 1152])
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch.transpose(torch_input_tensor_b, -1, -2))
    torch_output_tensor += torch_input_tensor_c

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    input_tensor_c = ttnn.from_torch(torch_input_tensor_c, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    output1 = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=input_tensor_c,
        compute_kernel_config=compute_kernel_config,
        core_grid=ttnn.CoreGrid(y=8, x=7),
        transpose_b=True,
    )
    output_tensor = ttnn.to_torch(output1)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        atol=0.002 * 384,
        rtol=0.001 * 384,
        frobenius_threshold=0.001 * 384,
        pcc_threshold=0.99,
        check_ulp=False,
    )


def test_resnet50_linear(device):
    torch.manual_seed(0)
    batch_size = 16
    input_channels = 2048
    output_channels = 1000
    input_shape = [1, 1, batch_size, input_channels]
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_weight_tensor = torch.randn([1, 1, output_channels, input_channels], dtype=torch.bfloat16)
    torch_bias_tensor = torch.randn([1, 1, 1, output_channels], dtype=torch.bfloat16)
    torch_out_golden_tensor = torch.nn.functional.linear(
        torch_input_tensor[0, 0, :, :], torch_weight_tensor[0, 0, :, :], bias=torch_bias_tensor[0, 0, :, :]
    )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight_tensor = ttnn.from_torch(
        torch.permute(torch_weight_tensor, (0, 1, 3, 2)), ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )
    tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    grid_size = (8, 4)
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
            )
        }
    )
    x = tt_input_tensor
    shard_shape = [
        x.volume() // x.padded_shape[-1],
        x.padded_shape[-1] // (grid_size[0] * grid_size[1]),
    ]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    width_sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    x = ttnn.to_memory_config(x, width_sharded_mem_config)

    tt_output_tensor_on_device = ttnn.linear(
        x,
        tt_weight_tensor,
        bias=tt_bias_tensor,
        program_config=matmul_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=compute_config,
    )
    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_numeric_metrics(
        torch_out_golden_tensor,
        torch_output_tensor[0, 0, :, :],
        atol=0.003 * 2048,
        rtol=0.258 * 2048,
        frobenius_threshold=0.001 * 2048,
        pcc_threshold=0.99,
        check_ulp=False,
    )


@pytest.mark.parametrize(
    "shape_a,shape_b,shape_bias",
    [
        # Vector-vector: (k) x (k) -> scalar
        ((32,), (32,), None),  # No bias for scalar output
        ((1,), (1,), tuple()),
        # Vector-matrix: (k) x (k, n) -> (n)
        ((32,), (32, 32), (32,)),  # Standard bias shape
        ((32,), (32, 32), (1,)),  # Broadcast bias
        ((1,), (1, 32), None),
        ((1,), (1, 32), (1,)),
        # Matrix-vector: (m, k) x (k) -> (m)
        ((32, 32), (32,), (32,)),  # Standard bias shape
        ((32, 32), (32,), (1,)),  # Broadcast bias
        # GH Issue #16599
        pytest.param((32, 1, 32), (32, 32), (1, 32), marks=pytest.mark.xfail),  # 4D tensors with no bias
        ((32, 2, 32), (32, 32), (1, 32)),  # 4D tensors with no bias
    ],
)
def test_vector_linear(device, shape_a, shape_b, shape_bias) -> tuple:
    """
    Test the compatibility of the torch and ttnn linear for the given operation and different
    tensor shapes.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    """
    torch.manual_seed(0)
    # Create random tensors with appropriate dimensions
    torch_a = torch.randn(*shape_a, dtype=torch.bfloat16)
    torch_b = torch.randn(*shape_b, dtype=torch.bfloat16)

    # For torch.linear, weight matrix is expected to be (out_features, in_features)
    # but internally it's transposed during the operation
    torch_weight = torch_b
    if len(shape_b) >= 2:
        torch_weight = torch.transpose(torch_weight, -1, -2)

    # Create bias tensor if shape_bias is not empty
    torch_bias = None
    ttnn_bias = None
    if shape_bias is not None:
        torch_bias = torch.randn(*shape_bias, dtype=torch.bfloat16) if shape_bias != tuple() else torch.randn(())
        ttnn_bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    # Create ttnn tensors
    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Handle exceptions in torch
    torch_errored = False
    torch_error_msg = ""
    try:
        torch_result = torch.nn.functional.linear(torch_a, torch_weight, torch_bias)
    except Exception as e:
        torch_errored = True
        torch_error_msg = str(e)

    # Run ttnn linear with the same operations
    ttnn_errored = False
    ttnn_error_msg = ""
    try:
        ttnn_result = ttnn.linear(ttnn_a, ttnn_b, bias=ttnn_bias)
    except Exception as e:
        ttnn_errored = True
        ttnn_error_msg = str(e)

    # Compare error behavior
    if torch_errored != ttnn_errored:
        return (
            False,
            f"mismatch in errors raised: torch: {torch_errored} ({torch_error_msg}), ttnn: {ttnn_errored} ({ttnn_error_msg})",
        )

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        logger.warning(f"both torch and ttnn raised errors: torch: {torch_error_msg}, ttnn: {ttnn_error_msg}")
        return (True, "")

    # Convert ttnn result to torch for comparison
    ttnn_result_torch = ttnn.to_torch(ttnn.from_device(ttnn_result))

    # Check shape compatibility
    if ttnn_result_torch.shape != torch_result.shape:
        assert False, f"mismatch in shape: torch: {torch_result.shape}, ttnn: {ttnn_result_torch.shape}"

    # Check values with numeric metrics
    k_value = shape_a[-1] if len(shape_a) > 0 else 1
    assert_numeric_metrics(
        torch_result,
        ttnn_result_torch,
        atol=0.0157 * k_value,
        rtol=0.1954 * k_value,
        frobenius_threshold=0.0047 * k_value,
        pcc_threshold=0.99,
    )

    # Allow some tolerance for numeric differences
    atol = rtol = 0.1

    assert torch.allclose(torch_result, ttnn_result_torch, atol=atol, rtol=rtol, equal_nan=True), (
        f"mismatch in allclose: torch: {torch_result}, ttnn: {ttnn_result_torch}",
    )


@pytest.mark.parametrize("in0_block_w", [1, 2, 4, 8])
@pytest.mark.parametrize("out_subblock", [[1, 4]])
@pytest.mark.parametrize("out_block", [[1, 4]])
@pytest.mark.parametrize("num_cores", [62, 64])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_bias_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("compute_config_params", [[ttnn.MathFidelity.LoFi, True, False, False, False]])
def test_linear_yolov7(
    device,
    in0_block_w,
    out_subblock,
    out_block,
    num_cores,
    input_dtype,
    weights_bias_dtype,
    output_dtype,
    compute_config_params,
):
    torch.manual_seed(0)
    nhw = 6400
    input_channels = 512
    output_channels = 512

    input_shape = [1, 1, nhw, input_channels]
    torch_input_tensor = torch.randn([1, 1, nhw, input_channels], dtype=torch.bfloat16)  # Original size
    torch_weight_tensor = torch.randn([1, 1, output_channels, input_channels], dtype=torch.bfloat16)
    torch_bias_tensor = torch.randn([1, 1, 1, output_channels], dtype=torch.bfloat16)
    torch_out_golden_tensor = torch.nn.functional.linear(
        torch_input_tensor[0, 0, :, :], torch_weight_tensor[0, 0, :, :], bias=torch_bias_tensor[0, 0, :, :]
    )
    torch_out_golden_tensor = torch.nn.functional.silu(torch_out_golden_tensor)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    tt_weight_tensor = ttnn.from_torch(
        torch.permute(torch_weight_tensor, (0, 1, 3, 2)),
        weights_bias_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor,
        weights_bias_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=compute_config_params[0],
        math_approx_mode=compute_config_params[1],
        fp32_dest_acc_en=compute_config_params[2],
        packer_l1_acc=compute_config_params[3],
        dst_full_sync_en=compute_config_params[4],
    )
    grid_size = (8, 8)
    per_core_M = (nhw + 32 * num_cores - 1) // (32 * num_cores)
    per_core_N = output_channels // 32

    matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock[0],
        out_subblock_w=out_subblock[1],
        out_block_h=out_block[0],
        out_block_w=out_block[1],
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        mcast_in0=False,
    )

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 7),
            ),
        }
    )

    shard_height = (nhw + num_cores * 32 - 1) // (num_cores * 32) * 32
    x = tt_input_tensor
    in0_shard_shape = [shard_height, input_channels]
    in0_shard_spec = ttnn.ShardSpec(shard_grid, in0_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    height_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )
    x = ttnn.to_memory_config(x, height_sharded_mem_config)

    out_shard_shape = [shard_height, output_channels]
    out_shard_spec = ttnn.ShardSpec(shard_grid, out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard_spec)

    tt_output_tensor_on_device = ttnn.linear(
        x,
        tt_weight_tensor,
        bias=tt_bias_tensor,
        program_config=matmul_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
        compute_kernel_config=compute_config,
    )
    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_numeric_metrics(
        torch_out_golden_tensor,
        torch_output_tensor[0, 0, :, :],
        atol=0.014 * 512,
        rtol=24.25 * 512,
        frobenius_threshold=0.001 * 512,
        pcc_threshold=0.99,
        check_ulp=False,
    )


# ============================================================================
# Sub-device tests: verify ttnn.linear works on a sub-device whose start core
# is NOT (0, 0).  This exercises the sub_device_start_core offset that was
# added to the 2-D and 1-D multicast program factories.
# ============================================================================


def _setup_subdevice(device, skip_rows=1):
    """Create two sub-devices: row(s) 0..skip_rows-1 as a 'dummy' sub-device
    and the remaining rows as the 'worker' sub-device.  Returns a tuple of
    (sub_device_manager, worker_sub_device_id, worker_core_grid) that the
    caller must tear down via _teardown_subdevice().
    """
    grid = device.compute_with_storage_grid_size()
    cols, rows = grid.x, grid.y
    assert rows > skip_rows, f"Device grid has only {rows} rows; need >{skip_rows} for this sub-device test"

    dummy_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, skip_rows - 1))})
    worker_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, skip_rows), ttnn.CoreCoord(cols - 1, rows - 1))})

    dummy_sub_device = ttnn.SubDevice([dummy_crs])
    worker_sub_device = ttnn.SubDevice([worker_crs])

    dummy_sub_device_id = ttnn.SubDeviceId(0)
    worker_sub_device_id = ttnn.SubDeviceId(1)

    sub_device_manager = device.create_sub_device_manager([dummy_sub_device, worker_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)
    device.set_sub_device_stall_group([dummy_sub_device_id, worker_sub_device_id])

    worker_core_grid = ttnn.CoreGrid(x=cols, y=rows - skip_rows)
    return sub_device_manager, worker_sub_device_id, worker_core_grid, worker_crs


def _teardown_subdevice(device, sub_device_manager):
    """Clean up the sub-device manager."""
    device.reset_sub_device_stall_group()
    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("m_size", [128, 384])
@pytest.mark.parametrize("k_size", [512])
@pytest.mark.parametrize("n_size", [512])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_linear_on_subdevice(device, m_size, k_size, n_size, use_bias, transpose_b):
    """Run ttnn.linear on a sub-device that starts at row 1 (not row 0).

    This is the pattern used when overlapping CCL operations on row 0
    with matmul compute on the remaining rows.
    """
    grid = device.compute_with_storage_grid_size()
    if grid.y < 2:
        pytest.skip("Need at least 2 rows for sub-device test")

    sub_device_manager, worker_sub_device_id, worker_core_grid, worker_crs = _setup_subdevice(device)
    try:
        torch.manual_seed(0)
        torch_input_a = torch.randn((1, 1, m_size, k_size), dtype=torch.bfloat16)
        torch_input_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
        torch_bias = torch.randn((1, n_size), dtype=torch.bfloat16) if use_bias else None
        if transpose_b:
            torch_output = torch_input_a @ torch_input_b.T
        else:
            torch_output = torch_input_a @ torch_input_b
        if torch_bias is not None:
            torch_output = torch_output + torch_bias

        input_a = ttnn.from_torch(torch_input_a, dtype=ttnn.bfloat16, device=device)
        # Use to_layout outside of ttnn.from_torch to avoid introducing new arguments (subdevice_id / sub_core_grids) for ttnn.from_torch.
        # There is ongoing activity to unify the interface for subdevices; once complete, to_layout can be removed from here.
        input_a = ttnn.to_layout(input_a, ttnn.TILE_LAYOUT, sub_core_grids=worker_crs)
        input_b = ttnn.from_torch(torch_input_b, dtype=ttnn.bfloat16, device=device)
        input_b = ttnn.to_layout(input_b, ttnn.TILE_LAYOUT, sub_core_grids=worker_crs)
        bias = None
        if use_bias:
            bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, device=device)
            bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT, sub_core_grids=worker_crs)

        output = ttnn.linear(
            input_a,
            input_b,
            transpose_b=transpose_b,
            bias=bias,
            core_grid=worker_core_grid,
            sub_device_id=worker_sub_device_id,
        )
        output = ttnn.to_torch(output)
        assert_numeric_metrics(
            torch_output,
            output,
            atol=0.007 * k_size,
            rtol=7.313 * k_size,
            frobenius_threshold=0.001 * k_size,
            pcc_threshold=0.999,
            check_ulp=False,
        )
    finally:
        _teardown_subdevice(device, sub_device_manager)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("m_size", [128])
@pytest.mark.parametrize("k_size", [512])
@pytest.mark.parametrize("n_size", [512])
@pytest.mark.parametrize("skip_rows", [1, 2, 5])
def test_linear_on_subdevice_variable_start_row(device, m_size, k_size, n_size, skip_rows):
    """Verify that the sub-device start core offset works for different
    starting rows, not just row 1.
    """
    grid = device.compute_with_storage_grid_size()
    if grid.y <= skip_rows:
        pytest.skip(f"Need at least {skip_rows + 1} rows for this sub-device test")

    sub_device_manager, worker_sub_device_id, worker_core_grid, worker_crs = _setup_subdevice(
        device, skip_rows=skip_rows
    )
    try:
        torch.manual_seed(0)
        torch_input_a = torch.randn((1, 1, m_size, k_size), dtype=torch.bfloat16)
        torch_input_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
        torch_output = torch_input_a @ torch_input_b

        input_a = ttnn.from_torch(torch_input_a, dtype=ttnn.bfloat16, device=device)
        input_a = ttnn.to_layout(input_a, ttnn.TILE_LAYOUT, sub_core_grids=worker_crs)
        input_b = ttnn.from_torch(torch_input_b, dtype=ttnn.bfloat16, device=device)
        input_b = ttnn.to_layout(input_b, ttnn.TILE_LAYOUT, sub_core_grids=worker_crs)
        output = ttnn.linear(
            input_a,
            input_b,
            core_grid=worker_core_grid,
            sub_device_id=worker_sub_device_id,
        )
        output = ttnn.to_torch(output)
        assert_numeric_metrics(
            torch_output,
            output,
            atol=0.005 * k_size,
            rtol=4.188 * k_size,
            frobenius_threshold=0.001 * k_size,
            pcc_threshold=0.999,
            check_ulp=False,
        )
    finally:
        _teardown_subdevice(device, sub_device_manager)


@pytest.mark.parametrize(
    "batch_size, seq_len, k_size, n_size, fp32_dest_acc",
    [
        (64, 256, 384, 1536, True),
        (64, 256, 384, 1536, False),
    ],
)
def test_linear_bias_cb_estimation_with_large_n_small_k(device, batch_size, seq_len, k_size, n_size, fp32_dest_acc):
    """Regression test for issue #36316.

    When N >> K and bias is present, the bias circular buffer was being
    underestimated (sized by in0_block_w instead of per_core_N/out_block_w),
    causing L1 overflow. fp32_dest_acc_en=True doubles the intermediate tile
    size from 2048 to 4096 bytes, pushing CB totals closer to L1 limits on BH
    devices. The fp32_dest_acc=False variant exercises the same code path with
    smaller intermediates, which can hit L1 limits on WH devices where
    available L1 is smaller.
    """
    torch.manual_seed(0)
    torch_input_a = torch.randn((batch_size, 1, seq_len, k_size), dtype=torch.bfloat16)
    torch_input_b = torch.randn((n_size, k_size), dtype=torch.bfloat16)
    torch_bias = torch.randn((1, 1, 1, n_size), dtype=torch.bfloat16)

    torch_output = torch_input_a @ torch_input_b.T + torch_bias

    input_a = ttnn.from_torch(torch_input_a, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    input_b = ttnn.from_torch(torch_input_b, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=True,
    )

    output = ttnn.linear(
        input_a,
        input_b,
        bias=bias,
        transpose_b=True,
        core_grid=device.core_grid,
        compute_kernel_config=compute_kernel_config,
    )
    output = ttnn.to_torch(output)
    assert_numeric_metrics(
        torch_output,
        output,
        atol=0.004 * k_size,
        rtol=4.334 * k_size,
        frobenius_threshold=0.001 * k_size,
        pcc_threshold=0.99,
        check_ulp=False,
    )


def run_linear_bias_broadcast(device, a, b, bias=None, optional_output=None):
    a_tt = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    bias_tt = None
    if bias is not None:
        bias_tt = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    optional_tt = None
    if optional_output is not None:
        optional_tt = ttnn.from_torch(optional_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.linear(a_tt, b_tt, bias=bias_tt, optional_output_tensor=optional_tt)
    return ttnn.to_torch(result)


@pytest.mark.parametrize(
    "a_shape, b_shape, bias_shape",
    [
        ((256, 1024), (1024, 512), (1, 1, 512)),
        ((1, 1024), (1024, 512), (1, 512, 512)),
        ((2, 16, 32), (2, 32, 8), None),
        ((32, 64), (64, 16), (1, 17, 16)),  # Broadcast error: Invalid dimension"
    ],
)
def test_linear_bias_broadcast(device, a_shape, b_shape, bias_shape, expect_error):
    torch.manual_seed(0)

    a = torch.randn(*a_shape, dtype=torch.bfloat16)
    b = torch.randn(*b_shape, dtype=torch.bfloat16)

    if bias_shape is not None:
        bias = torch.randn(*bias_shape, dtype=torch.bfloat16)
    else:
        bias = None

    torch_failed = False
    try:
        if bias_shape is None:
            expected = torch.matmul(a, b)
        else:
            expected = torch.matmul(a, b) + bias

    except Exception:
        torch_failed = True

    if torch_failed:
        with expect_error(Exception, "."):
            run_linear_bias_broadcast(device, a, b, bias)
    else:
        result = run_linear_bias_broadcast(device, a, b, bias)
        assert result.shape == expected.shape
        assert_numeric_metrics(
            expected, result, pcc_threshold=0.999, check_ulp=False, check_frobenius=False, check_allclose=False
        )


@pytest.mark.parametrize(
    "a_shape, b_shape, bias_shape, optional_shape",
    [
        ((8, 64), (64, 4), (1, 1, 4), (1, 1, 1, 8, 4)),
        ((8, 64), (64, 4), (1, 1, 4), (1, 3, 8)),  # Invalid optional output tensor
    ],
)
def test_linear_bias_broadcast_with_optional_shape(device, a_shape, b_shape, bias_shape, optional_shape, expect_error):
    torch.manual_seed(0)

    a = torch.randn(*a_shape, dtype=torch.bfloat16)
    b = torch.randn(*b_shape, dtype=torch.bfloat16)
    bias = torch.randn(*bias_shape, dtype=torch.bfloat16)

    optional = None
    if optional_shape is not None:
        optional = torch.empty(*optional_shape, dtype=torch.bfloat16)

    torch_failed = False
    try:
        expected = torch.matmul(a, b) + bias
    except Exception:
        torch_failed = True

    if torch_failed:
        with expect_error(Exception, "."):
            run_linear_bias_broadcast(device, a, b, bias, optional)
    else:
        if expected.numel() != optional.numel():
            with expect_error(Exception, "."):
                run_linear_bias_broadcast(device, a, b, bias, optional)
        else:
            result = run_linear_bias_broadcast(device, a, b, bias, optional)

            if optional_shape is not None:
                assert result.shape == optional_shape
            else:
                assert result.shape == expected.shape


def test_linear_bias_wrong_height_rejected_on_multicore_reuse_program_config(device, expect_error):
    """Fused bias on MatmulMultiCoreReuseProgramConfig must be the full [M, N] block.

    The factory reads M*N bias tiles, so a shorter bias would read past the buffer. This shape is
    valid for the matmul (N == per_core_N) but not for the bias, so it must be rejected at bias
    validation.
    """
    torch.manual_seed(0)
    m, k, n = 64, 64, 64  # M == 2 tiles; a single-tile-row bias does not cover it
    in0 = ttnn.from_torch(torch.randn(1, 1, m, k, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    in1 = ttnn.from_torch(torch.randn(1, 1, k, n, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch.randn(1, 1, 32, n, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=k // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m // 32,
        per_core_N=n // 32,
    )

    with expect_error(RuntimeError, r"padded second last dimension of bias, 32, not equal to expected bias height, 64"):
        ttnn.linear(in0, in1, bias=bias, program_config=program_config)


@pytest.mark.parametrize("bias_rank", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("m,k,n", [(32, 32, 32)])
def test_linear_broadcast_bias_ranks(device, m, k, n, bias_rank):
    """
    ``ttnn.linear`` with broadcastable bias shapes (logical rank 0-4) vs torch ``matmul + bias``.
    """

    if bias_rank == 0:
        pytest.skip(f"Rank-0 bias linear not supported")

    torch.manual_seed(0)
    torch_input = torch.randn((m, k), dtype=torch.bfloat16)
    torch_weight = torch.randn((n, k), dtype=torch.bfloat16)
    if bias_rank == 1:
        torch_bias = torch.randn((n,), dtype=torch.bfloat16)
    elif bias_rank == 2:
        torch_bias = torch.randn((1, n), dtype=torch.bfloat16)
    elif bias_rank == 3:
        torch_bias = torch.randn((1, 1, n), dtype=torch.bfloat16)
    else:
        torch_bias = torch.randn((1, 1, 1, n), dtype=torch.bfloat16)

    torch_mat = torch.matmul(torch_input, torch_weight.T)
    torch_output = torch_mat + torch_bias

    input_tensor = ttnn.from_torch(
        torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    weight_tensor = ttnn.from_torch(
        torch_weight, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    bias_tensor = ttnn.from_torch(
        torch_bias, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_tensor = ttnn.linear(input_tensor, weight_tensor, bias=bias_tensor, transpose_b=True)

    output = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output, output, pcc=0.99)


def _skip_unless_fused_full_mn_tiny_tile_supported(transpose_tile, tile_w, tile_h):
    if not is_tiny_tile_combo_supported(transpose_tile, tile_w, tile_h, True) and is_llk_assert_enabled():
        pytest.skip("Unsupported tiny-tile combination (see _TINY_TILE_SUPPORTED_COMBOS).")


def pad_to_dram_banks(num, tile_w, lcm=32 * 12):
    remainder = num % lcm
    if remainder == 0:
        return num
    padding_needed = lcm - remainder
    padded_number = num + padding_needed
    return padded_number


@skip_for_blackhole("TinyTile Matmul needs to be fixed on BH. Issue #31385")
@pytest.mark.parametrize("k_dram", [128, 256])
@pytest.mark.parametrize(
    "m,n,tile_h,tile_w,transpose_tile",
    [
        (32, 32, 32, 32, False),
        (16, 32, 16, 32, False),
    ],
)
def test_linear_fused_non_broadcast_bias_dram_sharded_in1(device, k_dram, m, n, tile_h, tile_w, transpose_tile):
    """Fused bias [1,1,M,N] with MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig (grid 1x1)."""
    _skip_unless_fused_full_mn_tiny_tile_supported(transpose_tile, tile_w, tile_h)
    torch.manual_seed(0)
    in1_dtype = ttnn.bfloat16
    num_banks = device.dram_grid_size().x if is_blackhole() else 12
    n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_banks)
    in0_shape = [1, 1, m, k_dram]
    in1_shape = [1, 1, k_dram, n]
    in1_shard_shape = [k_dram, n_padded // num_banks]
    num_cores = 1
    in0_block_w = k_dram // num_cores // 32
    out_block_h = m // tile_h
    out_block_w = n // num_cores // tile_w
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    in0_memory_config = ttnn.create_sharded_memory_config(
        (1, 1, m, k_dram),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, 32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, tile_w), transpose_tile=transpose_tile),
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )
    bias_dram = torch.randn([1, 1, m, n]).bfloat16().float()
    bias_shard_shape = [tile_h, n_padded // num_banks]
    bias_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    bias_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), bias_shard_grid)})
    bias_shard_spec = ttnn.ShardSpec(bias_shard_grid, bias_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    bias_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, bias_shard_spec)
    bias_t = ttnn.from_torch(
        bias_dram,
        tile=ttnn.Tile((tile_h, tile_w)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bias_mem_config,
    )
    dram_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // 4,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fused_activation=None,
    )
    dram_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    output_dram = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=dram_program_config,
        memory_config=sharded_mem_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=dram_compute_kernel_config,
        output_tile=ttnn.Tile([tile_h, tile_w]),
    )
    pt_dram = in0 @ in1 + bias_dram
    for o in ttnn.get_device_tensors(output_dram):
        assert_numeric_metrics(
            pt_dram,
            ttnn.to_torch(o),
            atol=0.004 * k_dram,
            rtol=0.227 * k_dram,
            frobenius_threshold=0.001 * k_dram,
            pcc_threshold=0.999,
            check_ulp=False,
        )


@skip_for_blackhole("TinyTile Matmul needs to be fixed on BH. Issue #31385")
@pytest.mark.parametrize("m,k,n", [(32, 32, 32), (32, 64, 32)])
def test_linear_fused_non_broadcast_bias_width_sharded_in0_in1(device, m, k, n):
    """Fused bias [1,1,M,N] with width-sharded activations/weights/bias and 1D mcast program config."""
    _skip_unless_fused_full_mn_tiny_tile_supported(False, 32, 32)
    torch.manual_seed(0)
    num_act = num_mm = 1
    core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    mem_config_weights = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range, [k, n // num_mm], ttnn.ShardOrientation.ROW_MAJOR),
    )
    mem_config_bias = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range, [32, n // num_mm], ttnn.ShardOrientation.ROW_MAJOR),
    )
    mem_config_input = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range, [m, k // num_act], ttnn.ShardOrientation.ROW_MAJOR),
    )
    sharded_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    input_tensor = torch.randn([1, 1, m, k], dtype=torch.bfloat16)
    tt_input = ttnn.as_tensor(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config_input,
    )
    weights_tensor = torch.randn([1, 1, k, n], dtype=torch.bfloat16)
    weight_tt = ttnn.as_tensor(
        weights_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config_weights,
    )
    bias_tensor = torch.randn([1, 1, m, n], dtype=torch.bfloat16) * 2.0
    bias_tt = ttnn.as_tensor(
        bias_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config_bias,
    )
    sharded_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=k // num_mm // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m // 32,
        per_core_N=n // 32,
        mcast_in0=True,
        fused_activation=None,
        fuse_batch=True,
    )
    tt_out = ttnn.linear(
        tt_input,
        weight_tt,
        bias=bias_tt,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=sharded_program_config,
        compute_kernel_config=sharded_compute_kernel_config,
    )
    matmul_ref = torch.matmul(input_tensor, weights_tensor) + bias_tensor
    tt_mm_out = ttnn.to_torch(ttnn.from_device(tt_out))
    assert_numeric_metrics(
        matmul_ref,
        tt_mm_out,
        atol=0.018 * k,
        rtol=2.57 * k,
        frobenius_threshold=0.001 * k,
        pcc_threshold=0.993,
    )


@pytest.mark.parametrize(
    "a_shape, b_shape, bias_shape",
    [
        ((1, 3, 128, 32), (1, 3, 32, 64), (1, 1, 1, 64)),
        ((1, 3, 128, 32), (1, 3, 32, 32), (1, 1, 1, 32)),
        ((1, 3, 128, 32), (1, 3, 32, 64), (1, 1, 1, 64)),
        ((1, 3, 32, 32), (1, 3, 32, 64), (1, 1, 32, 64)),
        ((1, 3, 32, 32), (1, 3, 32, 64), (1, 3, 32, 64)),
        ((1, 2, 16, 64), (1, 2, 64, 64), (1, 2, 16, 64)),
    ],
)
def test_linear_with_batched(device, a_shape, b_shape, bias_shape):
    torch.manual_seed(0)

    # Create inputs
    tensor_a_torch = torch.randn(a_shape)
    tensor_b_torch = torch.randn(b_shape)
    bias_torch = torch.randn(bias_shape)

    # PyTorch reference
    result_torch = torch.matmul(tensor_a_torch, tensor_b_torch) + bias_torch

    # TTNN tensors
    tensor_a_ttnn = ttnn.from_torch(tensor_a_torch, device=device)
    tensor_b_ttnn = ttnn.from_torch(tensor_b_torch, device=device)
    bias_ttnn = ttnn.from_torch(bias_torch, device=device)

    # Convert to TILE layout
    tensor_a_ttnn = ttnn.to_layout(tensor_a_ttnn, ttnn.Layout.TILE)
    tensor_b_ttnn = ttnn.to_layout(tensor_b_ttnn, ttnn.Layout.TILE)
    bias_ttnn = ttnn.to_layout(bias_ttnn, ttnn.Layout.TILE)

    # Run TTNN
    result_ttnn = ttnn.linear(
        tensor_a_ttnn,
        tensor_b_ttnn,
        bias=bias_ttnn,
    )

    result_ttnn_torch = ttnn.to_torch(ttnn.from_device(result_ttnn))

    # test for equivalance
    assert_numeric_metrics(
        result_torch,
        result_ttnn_torch,
        atol=0.1,
        rtol=0.1,
        frobenius_threshold=0.002,
        pcc_threshold=0.99,
    )


def _reuse_program_config(device, m, k, n):
    """MatmulMultiCoreReuseProgramConfig

    Each batch element's matrix is a single program block
    (per_core_M/per_core_N is the whole [M, N]).
    It's the requirement for the fused-bias path: the full [M, N] bias is loaded once and reused across the batch.
    """
    tile = 32
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        in0_block_w=k // tile,
        out_subblock_h=1,
        out_subblock_w=n // tile,
        per_core_M=m // tile,
        per_core_N=n // tile,
    )


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("batch", [8, 24])
@pytest.mark.parametrize(
    "m, k, n",
    [
        (32, 32, 32),  # GDN shape: square single tile, per_core_M == per_core_N == 1
        (32, 128, 32),  # larger K
        (32, 32, 64),  # per_core_N == 2 within one N block, single M tile-row
        (64, 64, 64),  # M == 2 tiles: genuine full [M, N] bias, varies per M-row
        (128, 128, 128),  # M == N == 4 tiles: multi-tile block in both M and N
    ],
)
def test_linear_batched_reuse_fused_elementwise_bias(device, dtype, batch, m, k, n):
    """Fused full-tile (elementwise) bias on the batched MatmulMultiCoreReuseProgramConfig path."""
    torch.manual_seed(0)
    torch_a = torch_random((batch, 1, m, k), -0.1, 0.1, dtype=torch.float32)
    torch_b = torch_random((batch, 1, k, n), -0.1, 0.1, dtype=torch.float32)
    torch_bias = torch_random((1, 1, m, n), -0.1, 0.1, dtype=torch.float32)  # full-tile, broadcast over batch
    torch_output = torch.matmul(torch_a, torch_b) + torch_bias

    a = ttnn.from_torch(torch_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.linear(a, b, bias=bias, program_config=_reuse_program_config(device, m, k, n))
    output = ttnn.to_torch(output)

    pcc = 0.9997 if dtype == ttnn.float32 else 0.99
    assert_with_pcc(torch_output, output, pcc)


def _captured_device_op_names(fn):
    """Device-op names recorded while running `fn` under graph capture (no device dispatch)."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
    try:
        fn()
    finally:
        captured = ttnn.graph.end_graph_capture()
    return [
        node["params"]["name"]
        for node in captured
        if node.get("node_type") == "function_start" and "name" in node.get("params", {})
    ]


@pytest.mark.parametrize(
    "m, k, n, bias_shape, expect_fused",
    [
        (64, 64, 64, (1, 1, 64, 64), True),  # full [M, N] bias -> fused into the matmul
        (128, 128, 128, (1, 1, 128, 128), True),  # full [M, N], M = N = 4 tiles -> fused
        (64, 64, 64, (1, 64), False),  # row-vector bias -> post-processed via add()
    ],
)
def test_linear_batched_reuse_bias_routing(device, m, k, n, bias_shape, expect_fused):
    """A full [M, N] bias fuses into the matmul; other biases fall back to post-processed add."""
    torch.manual_seed(0)
    a = ttnn.from_torch(torch.randn(8, 1, m, k) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(8, 1, k, n) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch.randn(*bias_shape) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    pc = _reuse_program_config(device, m, k, n)

    names = _captured_device_op_names(lambda: ttnn.linear(a, b, bias=bias, program_config=pc))

    assert any("Matmul" in nm for nm in names), f"no matmul op captured: {names}"
    has_binary = any("Binary" in nm for nm in names)
    if expect_fused:
        assert not has_binary, f"expected fused bias (no post-process add), captured: {names}"
    else:
        assert has_binary, f"expected a post-process add, captured: {names}"


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_linear_batched_reuse_row_vector_bias_post_processed_correct(device, dtype):
    """A row-vector bias on the batched reuse config is post-processed (not fused) and stays correct."""
    batch, m, k, n = 8, 64, 64, 64  # M = 2 tiles; row-vector bias does not cover it
    torch.manual_seed(0)
    torch_a = torch_random((batch, 1, m, k), -0.1, 0.1, dtype=torch.float32)
    torch_b = torch_random((batch, 1, k, n), -0.1, 0.1, dtype=torch.float32)
    torch_bias = torch_random((n,), -0.1, 0.1, dtype=torch.float32)
    torch_output = torch.matmul(torch_a, torch_b) + torch_bias

    a = ttnn.from_torch(torch_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias.reshape(1, n), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn.linear(a, b, bias=bias, program_config=_reuse_program_config(device, m, k, n)))
    pcc = 0.9997 if dtype == ttnn.float32 else 0.99
    assert_with_pcc(torch_output, output, pcc)


def test_linear_batched_reuse_fused_bias_program_cache(device):
    """Fused bias stays correct across a program-cache hit with a different bias buffer.

    The bias buffer address is a runtime-arg binding. A second launch with the same program key
    but a distinct bias tensor must re-patch that binding; Launch twice with different bias buffers,
    assert the second is a cache hit, and verify both outputs (a stale binding fails the second PCC).
    """
    batch, m, k, n = 8, 64, 64, 64  # M = 2 tiles: full [M, N] fused bias
    torch.manual_seed(0)
    torch_a = torch_random((batch, 1, m, k), -0.1, 0.1, dtype=torch.float32)
    torch_b = torch_random((batch, 1, k, n), -0.1, 0.1, dtype=torch.float32)
    # Distinct value ranges -> distinct buffers; a retained first binding corrupts the second result.
    torch_biases = [
        torch_random((1, 1, m, n), -0.1, 0.1, dtype=torch.float32),
        torch_random((1, 1, m, n), 0.5, 1.0, dtype=torch.float32),
    ]
    pc = _reuse_program_config(device, m, k, n)
    a = ttnn.from_torch(torch_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    biases = [ttnn.from_torch(tb, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device) for tb in torch_biases]

    device.enable_program_cache()
    device.clear_program_cache()

    before = device.num_program_cache_entries()
    out0 = ttnn.linear(a, b, bias=biases[0], program_config=pc)
    after_first = device.num_program_cache_entries()
    out1 = ttnn.linear(a, b, bias=biases[1], program_config=pc)
    after_second = device.num_program_cache_entries()

    assert after_first == before + 1, (
        f"first launch should add exactly one program-cache entry, added {after_first - before} "
        f"(is the program cache enabled?)"
    )
    assert after_second == after_first, (
        f"second launch recompiled instead of hitting the program cache "
        f"({after_first - before} then {after_second - after_first} new entries)"
    )
    assert_with_pcc(torch.matmul(torch_a, torch_b) + torch_biases[0], ttnn.to_torch(out0), 0.9997)
    assert_with_pcc(torch.matmul(torch_a, torch_b) + torch_biases[1], ttnn.to_torch(out1), 0.9997)

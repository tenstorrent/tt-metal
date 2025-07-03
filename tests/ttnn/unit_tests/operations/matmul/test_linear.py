# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.utility_functions import torch_random, is_wormhole_b0, skip_for_grayskull


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

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


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

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [32, 64])
@pytest.mark.parametrize("k_size", [1024])
@pytest.mark.parametrize("n_size", [1024])
@pytest.mark.parametrize("activation", [None, "relu", "silu"])
def test_wide_linear_with_argument_for_core_grid_set_to_device_grid(
    device, batch_size, m_size, k_size, n_size, activation
):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b
    if activation == "relu":
        torch_output_tensor = torch.relu(torch_output_tensor)
    elif activation == "silu":
        torch_output_tensor = torch.nn.functional.silu(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.linear(input_tensor_a, input_tensor_b, core_grid=device.core_grid, activation=activation)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


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
    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


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

    if is_wormhole_b0():
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    else:
        # Grayskull doesn't support fp32 but test passing a GS config is ok
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
        )

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        core_grid=device.core_grid,
        compute_kernel_config=compute_kernel_config,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


def test_bloom_ff2_linear(device):
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

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.9992


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
    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)
    assert_with_pcc(torch_output_tensor, optional_output_tensor, 0.997)
    assert_with_pcc(optional_output_tensor, output_tensor, 0.997)


@skip_for_grayskull()
def test_linear_with_fp32_dest_acc_and_bias(device):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.rand([64, 1, 256, 384])
    torch_input_tensor_b = torch.rand([1, 1, 1152, 384])
    torch_input_tensor_c = torch.rand([1, 1, 1, 1152])
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
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
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


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
    assert_with_pcc(torch_out_golden_tensor, torch_output_tensor[0, 0, :, :], pcc=0.99)


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

    # Check values with PCC
    assert_with_pcc(torch_result, ttnn_result_torch, 0.99)

    # Allow some tolerance for numeric differences
    atol = rtol = 0.1

    assert torch.allclose(torch_result, ttnn_result_torch, atol=atol, rtol=rtol, equal_nan=True), (
        f"mismatch in allclose: torch: {torch_result}, ttnn: {ttnn_result_torch}",
    )

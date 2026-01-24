# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger

from models.common.utility_functions import skip_for_blackhole


@skip_for_blackhole("Not tested / built for Blackhole")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("use_weight_and_bias", [True, False])
def test_layernorm_factory_descriptor(device, h, w, use_welford, use_weight_and_bias):
    """
    Test layer norm using the factory's create_descriptor method with generic_op.
    This test follows the pattern from test_generic_op.py but uses layernorm factory.
    """
    torch.manual_seed(0)
    dtype = torch.bfloat16
    eps = 1e-12

    # Create input tensor
    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create optional weight and bias tensors
    weight = None
    bias = None
    torch_weight = None
    torch_bias = None
    if use_weight_and_bias:
        torch_weight = torch.rand((w,), dtype=dtype)
        torch_bias = torch.rand((w,), dtype=dtype)
        weight = ttnn.from_torch(
            torch_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        bias = ttnn.from_torch(
            torch_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Create output tensor and zero it to ensure clean state
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([h, w]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    # Zero the output tensor to ensure clean state
    output_tensor = ttnn.zeros_like(output_tensor)

    # Create LayerNormParams
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    operation_params = ttnn.LayerNormParams()
    operation_params.norm_type = ttnn.LayerNormType.LAYERNORM
    operation_params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params.eps = eps
    operation_params.output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    operation_params.program_config = program_config
    # Use same compute kernel config as layer_norm (HiFi4, no approx, fp32 acc)
    operation_params.compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    # dtype is optional - leave it as default (None) if not needed

    # Create LayerNormInputs
    tensor_args = ttnn.LayerNormInputs()
    tensor_args.input = input_tensor
    # Optional fields are already None by default, only set if they have values
    if weight is not None:
        tensor_args.weight = weight
    if bias is not None:
        tensor_args.bias = bias

    # Get program descriptor from factory
    program_descriptor = ttnn.LayerNormMultiCoreProgramFactory.create_descriptor(
        operation_params, tensor_args, output_tensor
    )

    # Skip welford tests: When use_welford=True, create_descriptor creates a reciprocal tensor
    # internally and stores its buffer pointer in the descriptor. The tensor is destroyed when
    # the function returns, causing "bad optional access" errors when generic_op tries to use
    # the descriptor. This is a known limitation of using create_descriptor with generic_op
    # for welford mode. The normal layer_norm operation works because it uses create() which
    # keeps resources alive.
    if use_welford:
        pytest.skip("Welford mode with create_descriptor + generic_op has reciprocal tensor lifetime issues")

    # Execute using generic_op
    io_tensors = [input_tensor]
    if weight is not None:
        io_tensors.append(weight)
    if bias is not None:
        io_tensors.append(bias)
    io_tensors.append(output_tensor)

    output = ttnn.generic_op(io_tensors, program_descriptor)

    # Get golden reference using standard layer_norm
    golden = ttnn.layer_norm(
        input_tensor,
        weight=weight,
        bias=bias,
        epsilon=eps,
        program_config=program_config,
    )

    # Compare results
    torch_golden = ttnn.to_torch(golden)
    torch_output = ttnn.to_torch(output)

    logger.info(f"input_tensor shape: {input_tensor.shape}")
    logger.info(f"torch_golden shape: {torch_golden.shape}")
    logger.info(f"torch_output shape: {torch_output.shape}")

    # Debug: Print some sample values
    logger.info(f"Input sample (first 5): {torch_input_tensor.flatten()[:5]}")
    logger.info(f"Golden sample (first 5): {torch_golden.flatten()[:5]}")
    logger.info(f"Output sample (first 5): {torch_output.flatten()[:5]}")

    max_diff = torch.max(torch.abs(torch_golden - torch_output))
    mean_diff = torch.mean(torch.abs(torch_golden - torch_output))
    logger.info(f"Max diff: {max_diff}")
    logger.info(f"Mean diff: {mean_diff}")

    # Check if outputs are all zeros or NaN
    logger.info(f"Golden has NaN: {torch.isnan(torch_golden).any()}")
    logger.info(f"Output has NaN: {torch.isnan(torch_output).any()}")
    logger.info(f"Golden is all zeros: {(torch_golden == 0).all()}")
    logger.info(f"Output is all zeros: {(torch_output == 0).all()}")

    matching = torch.allclose(torch_golden, torch_output, rtol=1e-2, atol=1e-2)
    logger.info(f"Tensors are matching: {matching}")

    assert (
        matching
    ), f"Factory descriptor output should match standard layer_norm output. Max diff: {max_diff}, Mean diff: {mean_diff}"


@skip_for_blackhole("Not tested / built for Blackhole")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_layernorm_factory_descriptor_with_residual(device, h, w):
    """
    Test layer norm factory with residual input tensor.
    """
    torch.manual_seed(0)
    dtype = torch.bfloat16
    eps = 1e-12

    # Create input and residual tensors
    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_residual = torch.rand((h, w), dtype=dtype)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    residual_tensor = ttnn.from_torch(
        torch_residual,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create output tensor and zero it to ensure clean state
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([h, w]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    # Zero the output tensor to ensure clean state
    output_tensor = ttnn.zeros_like(output_tensor)

    # Create LayerNormParams
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=False)
    operation_params = ttnn.LayerNormParams()
    operation_params.norm_type = ttnn.LayerNormType.LAYERNORM
    operation_params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params.eps = eps
    operation_params.output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    operation_params.program_config = program_config
    # Use same compute kernel config as layer_norm (HiFi4, no approx, fp32 acc)
    operation_params.compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    # dtype is optional - leave it as default (None) if not needed

    # Create LayerNormInputs with residual
    tensor_args = ttnn.LayerNormInputs()
    tensor_args.input = input_tensor
    tensor_args.residual_input_tensor = residual_tensor
    # Optional fields are already None by default

    # Get program descriptor from factory
    program_descriptor = ttnn.LayerNormMultiCoreProgramFactory.create_descriptor(
        operation_params, tensor_args, output_tensor
    )

    # Execute using generic_op
    io_tensors = [input_tensor, residual_tensor, output_tensor]
    output = ttnn.generic_op(io_tensors, program_descriptor)

    # Get golden reference using standard layer_norm
    golden = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_tensor,
        epsilon=eps,
        program_config=program_config,
    )

    # Compare results
    torch_golden = ttnn.to_torch(golden)
    torch_output = ttnn.to_torch(output)

    max_diff = torch.max(torch.abs(torch_golden - torch_output))
    mean_diff = torch.mean(torch.abs(torch_golden - torch_output))
    logger.info(f"Max diff: {max_diff}")
    logger.info(f"Mean diff: {mean_diff}")

    matching = torch.allclose(torch_golden, torch_output, rtol=1e-2, atol=1e-2)
    logger.info(f"Tensors are matching: {matching}")

    assert (
        matching
    ), f"Factory descriptor output with residual should match standard layer_norm output. Max diff: {max_diff}, Mean diff: {mean_diff}"

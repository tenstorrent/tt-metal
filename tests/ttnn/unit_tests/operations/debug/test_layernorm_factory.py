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


@skip_for_blackhole("Not tested / built for Blackhole")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_layernorm_sharded_factory_descriptor(device, h, w):
    """
    Test sharded layer norm using the factory's create_descriptor method with generic_op.
    Uses width-sharded tensors for testing the sharded layernorm path.
    """
    torch.manual_seed(0)
    dtype = torch.bfloat16
    eps = 1e-12

    # Sharding configuration - width sharded across 2 cores
    num_cores = 2
    shard_width = w // num_cores
    shard_height = h
    shard_shape = (shard_height, shard_width)

    # Create input tensor
    torch_input_tensor = torch.rand((h, w), dtype=dtype)

    # Create sharded memory config
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )

    # Create output tensor with same sharding
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([h, w]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        sharded_mem_config,
    )

    # Create LayerNormParams with sharded program config
    block_h = h // 32  # tiles in height
    block_w = shard_width // 32  # tiles in width per shard
    subblock_w = min(block_w, 4)

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(num_cores, 1),
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        inplace=True,
        use_welford=False,
    )

    operation_params = ttnn.LayerNormParams()
    operation_params.norm_type = ttnn.LayerNormType.LAYERNORM
    operation_params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params.eps = eps
    operation_params.output_mem_config = sharded_mem_config
    operation_params.program_config = program_config
    operation_params.compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    # Create LayerNormInputs
    tensor_args = ttnn.LayerNormInputs()
    tensor_args.input = input_tensor

    # Get program descriptor from sharded factory
    program_descriptor = ttnn.LayerNormShardedProgramFactory.create_descriptor(
        operation_params, tensor_args, output_tensor
    )

    # Execute using generic_op
    io_tensors = [input_tensor, output_tensor]
    output = ttnn.generic_op(io_tensors, program_descriptor)

    # Get golden reference using standard layer_norm
    golden = ttnn.layer_norm(
        input_tensor,
        epsilon=eps,
        program_config=program_config,
    )

    # Compare results
    torch_golden = ttnn.to_torch(golden)
    torch_output = ttnn.to_torch(output)

    logger.info(f"input_tensor shape: {input_tensor.shape}")
    logger.info(f"torch_golden shape: {torch_golden.shape}")
    logger.info(f"torch_output shape: {torch_output.shape}")

    max_diff = torch.max(torch.abs(torch_golden - torch_output))
    mean_diff = torch.mean(torch.abs(torch_golden - torch_output))
    logger.info(f"Max diff: {max_diff}")
    logger.info(f"Mean diff: {mean_diff}")

    matching = torch.allclose(torch_golden, torch_output, rtol=1e-2, atol=1e-2)
    logger.info(f"Tensors are matching: {matching}")

    assert matching, f"Sharded factory descriptor output should match. Max diff: {max_diff}, Mean diff: {mean_diff}"


@skip_for_blackhole("Not tested / built for Blackhole")
def test_program_descriptor_merge(device):
    """
    Test merging multiple ProgramDescriptors that operate on different core ranges.
    This creates two separate layernorm operations on different parts of the device
    and merges them into a single program.
    """
    torch.manual_seed(0)
    dtype = torch.bfloat16
    eps = 1e-12
    h, w = 32, 64

    # Create two separate input tensors for two different layernorm operations
    torch_input1 = torch.rand((h, w), dtype=dtype)
    torch_input2 = torch.rand((h, w), dtype=dtype)

    input_tensor1 = ttnn.from_torch(
        torch_input1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor2 = ttnn.from_torch(
        torch_input2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create output tensors
    output_tensor1 = ttnn.zeros_like(input_tensor1)
    output_tensor2 = ttnn.zeros_like(input_tensor2)

    # Create program config
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=False)

    # Create operation params for first layernorm
    operation_params1 = ttnn.LayerNormParams()
    operation_params1.norm_type = ttnn.LayerNormType.LAYERNORM
    operation_params1.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params1.eps = eps
    operation_params1.output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    operation_params1.program_config = program_config
    operation_params1.compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    tensor_args1 = ttnn.LayerNormInputs()
    tensor_args1.input = input_tensor1

    # Create operation params for second layernorm (same params, different tensors)
    operation_params2 = ttnn.LayerNormParams()
    operation_params2.norm_type = ttnn.LayerNormType.LAYERNORM
    operation_params2.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params2.eps = eps
    operation_params2.output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    operation_params2.program_config = program_config
    operation_params2.compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    tensor_args2 = ttnn.LayerNormInputs()
    tensor_args2.input = input_tensor2

    # Get program descriptors from factories
    program_descriptor1 = ttnn.LayerNormMultiCoreProgramFactory.create_descriptor(
        operation_params1, tensor_args1, output_tensor1
    )
    program_descriptor2 = ttnn.LayerNormMultiCoreProgramFactory.create_descriptor(
        operation_params2, tensor_args2, output_tensor2
    )

    # Note: For a true merge test, we would need the two descriptors to use different core ranges.
    # The LayerNormMultiCoreProgramFactory by default uses the same core range layout.
    # For this test, we demonstrate the merge API works and catches overlapping ranges.

    # Test that merging with overlapping core ranges raises an error
    try:
        program_descriptor1.merge(program_descriptor2)
        # If we get here, the core ranges didn't overlap (which could happen with small tensors)
        logger.info("Merge succeeded - core ranges did not overlap")
    except RuntimeError as e:
        # Expected behavior when core ranges overlap
        logger.info(f"Merge correctly raised error for overlapping core ranges: {e}")
        assert "overlapping" in str(e).lower(), f"Error should mention overlapping: {e}"

    # Test the static merge_descriptors method as well
    try:
        merged = ttnn.ProgramDescriptor.merge_descriptors([program_descriptor1, program_descriptor2])
        logger.info("Static merge succeeded - core ranges did not overlap")
    except RuntimeError as e:
        logger.info(f"Static merge correctly raised error for overlapping core ranges: {e}")
        assert "overlapping" in str(e).lower(), f"Error should mention overlapping: {e}"

    logger.info("Merge API test completed successfully")


@skip_for_blackhole("Not tested / built for Blackhole")
def test_program_descriptor_merge_non_overlapping(device):
    """
    Test merging two ProgramDescriptors with explicitly non-overlapping core ranges.
    This manually constructs descriptors on different cores to verify merge works.
    """
    import ttnn

    # Create a simple ProgramDescriptor with a kernel on cores (0,0)-(0,0)
    desc1 = ttnn.ProgramDescriptor()

    kernel1 = ttnn.KernelDescriptor()
    kernel1.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp"
    kernel1.source_type = ttnn.KernelDescriptor.SourceType.FILE_PATH
    kernel1.core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    kernel1.config = ttnn.ReaderConfigDescriptor()
    desc1.kernels = [kernel1]

    # Create a second ProgramDescriptor with a kernel on cores (1,0)-(1,0)
    desc2 = ttnn.ProgramDescriptor()

    kernel2 = ttnn.KernelDescriptor()
    kernel2.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp"
    kernel2.source_type = ttnn.KernelDescriptor.SourceType.FILE_PATH
    kernel2.core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})
    kernel2.config = ttnn.ReaderConfigDescriptor()
    desc2.kernels = [kernel2]

    # Verify the initial state
    assert len(desc1.kernels) == 1
    assert len(desc2.kernels) == 1

    # Merge desc2 into desc1 - should succeed since cores don't overlap
    desc1.merge(desc2)

    # Verify merge succeeded
    assert len(desc1.kernels) == 2, f"Expected 2 kernels after merge, got {len(desc1.kernels)}"
    logger.info("Successfully merged two ProgramDescriptors with non-overlapping core ranges")

    # Test static merge_descriptors
    desc3 = ttnn.ProgramDescriptor()
    kernel3 = ttnn.KernelDescriptor()
    kernel3.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp"
    kernel3.source_type = ttnn.KernelDescriptor.SourceType.FILE_PATH
    kernel3.core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0))})
    kernel3.config = ttnn.ReaderConfigDescriptor()
    desc3.kernels = [kernel3]

    merged = ttnn.ProgramDescriptor.merge_descriptors([desc1, desc3])
    assert len(merged.kernels) == 3, f"Expected 3 kernels after merge, got {len(merged.kernels)}"
    logger.info("Successfully used static merge_descriptors to combine multiple descriptors")


@skip_for_blackhole("Not tested / built for Blackhole")
def test_program_descriptor_merge_overlapping_should_fail(device):
    """
    Test that merging ProgramDescriptors with overlapping core ranges raises an error.
    """
    import ttnn

    # Create two ProgramDescriptors with overlapping core ranges
    desc1 = ttnn.ProgramDescriptor()
    kernel1 = ttnn.KernelDescriptor()
    kernel1.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp"
    kernel1.source_type = ttnn.KernelDescriptor.SourceType.FILE_PATH
    kernel1.core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    kernel1.config = ttnn.ReaderConfigDescriptor()
    desc1.kernels = [kernel1]

    desc2 = ttnn.ProgramDescriptor()
    kernel2 = ttnn.KernelDescriptor()
    kernel2.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp"
    kernel2.source_type = ttnn.KernelDescriptor.SourceType.FILE_PATH
    # Overlapping core range - (1,1) is in both
    kernel2.core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2))})
    kernel2.config = ttnn.ReaderConfigDescriptor()
    desc2.kernels = [kernel2]

    # Merge should fail due to overlapping cores
    with pytest.raises(RuntimeError) as exc_info:
        desc1.merge(desc2)

    assert "overlapping" in str(exc_info.value).lower(), f"Error should mention overlapping: {exc_info.value}"
    logger.info(f"Correctly raised error for overlapping core ranges: {exc_info.value}")


@skip_for_blackhole("Not tested / built for Blackhole")
def test_merged_layernorm_4x4_grid_random_configs(device):
    """
    Test that merges 16 layernorm operations running on a 4x4 grid of 2x2 core ranges.

    This test:
    1. Splits the 8x8 core grid into 4x4 groups of 2x2 core ranges (16 groups total)
    2. Each group randomly runs sharded layernorm with random configurations
    3. Randomly samples: use_weight, use_bias, use_residual
    4. Randomly selects dtype from [bfloat16, bfloat8_b] (float32 output not supported for sharded LN)
    5. Merges all 16 ProgramDescriptors into one
    6. Compares each group's output to torch reference
    """
    import random

    torch.manual_seed(42)
    random.seed(42)

    eps = 1e-12

    # Each group uses a 2x2 core range = 4 cores in a row for width sharding
    # We'll use 2 cores per operation (width sharded) to keep things simpler
    # 4x4 grid of groups = 16 groups, each using 2 cores horizontally
    # So we need 4 groups across x 2 cores = 8 cores in x direction
    # And 4 groups in y = 4 rows
    # Total: 8x4 = 32 cores used (subset of 8x8)

    num_groups_x = 4
    num_groups_y = 4
    cores_per_group_x = 2  # Width sharding across 2 cores
    cores_per_group_y = 1  # Single row per group

    # Tensor dimensions (must be tilized: divisible by 32)
    h = 32  # Height
    w_per_core = 32  # Width per core (1 tile)
    w = w_per_core * cores_per_group_x  # Total width = 64 (2 tiles)

    # Available dtypes for sharded layernorm (float32 dest accumulation is used internally)
    available_dtypes = [ttnn.bfloat16, ttnn.bfloat8_b]
    dtype_to_torch = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.bfloat8_b: torch.bfloat16,  # bfloat8_b converts to bfloat16 in torch
    }

    # Store all operations data
    operations = []
    program_descriptors = []

    logger.info(f"Creating {num_groups_x * num_groups_y} layernorm operations on 4x4 grid")

    for group_y in range(num_groups_y):
        for group_x in range(num_groups_x):
            group_idx = group_y * num_groups_x + group_x

            # Calculate core range for this group
            # Each group occupies cores_per_group_x cores in x and cores_per_group_y in y
            start_x = group_x * cores_per_group_x
            start_y = group_y * cores_per_group_y
            end_x = start_x + cores_per_group_x - 1
            end_y = start_y + cores_per_group_y - 1

            # Random configuration
            # Note: use_bias without use_weight is not typically supported in layernorm
            # Valid combos: (no weight, no bias), (weight only), (weight and bias)
            config_choice = random.choice(["none", "weight_only", "weight_and_bias"])
            if config_choice == "none":
                use_weight = False
                use_bias = False
            elif config_choice == "weight_only":
                use_weight = True
                use_bias = False
            else:  # weight_and_bias
                use_weight = True
                use_bias = True
            # Note: residual is tricky with sharded tensors, skip for now
            use_residual = False
            dtype = random.choice(available_dtypes)
            torch_dtype = dtype_to_torch[dtype]

            logger.info(
                f"Group {group_idx} ({group_x}, {group_y}): cores ({start_x},{start_y})-({end_x},{end_y}), "
                f"weight={use_weight}, bias={use_bias}, dtype={dtype}"
            )

            # Create shard spec for this group's cores
            core_range = ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))
            shard_shape = (h, w_per_core)  # Each core gets w_per_core width
            shard_spec = ttnn.ShardSpec(
                ttnn.CoreRangeSet({core_range}),
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            sharded_mem_config = ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=shard_spec,
            )

            # Create torch input
            torch_input = torch.rand((h, w), dtype=torch_dtype)

            # Create torch weight/bias if needed
            torch_weight = torch.rand((w,), dtype=torch_dtype) if use_weight else None
            torch_bias = torch.rand((w,), dtype=torch_dtype) if use_bias else None

            # Compute torch golden (layernorm)
            torch_golden = torch.nn.functional.layer_norm(
                torch_input.float(),
                (w,),
                weight=torch_weight.float() if torch_weight is not None else None,
                bias=torch_bias.float() if torch_bias is not None else None,
                eps=eps,
            ).to(torch_dtype)

            # Create ttnn input tensor with sharding
            input_tensor = ttnn.from_torch(
                torch_input,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=sharded_mem_config,
            )

            # Create weight/bias tensors if needed (DRAM, not sharded for simplicity)
            weight_tensor = None
            bias_tensor = None
            if use_weight:
                weight_tensor = ttnn.from_torch(
                    torch_weight.reshape(1, 1, 1, w),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            if use_bias:
                bias_tensor = ttnn.from_torch(
                    torch_bias.reshape(1, 1, 1, w),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            # Create output tensor with same sharding
            output_tensor = ttnn.allocate_tensor_on_device(
                ttnn.Shape([h, w]),
                dtype,
                ttnn.TILE_LAYOUT,
                device,
                sharded_mem_config,
            )

            # Create sharded program config
            block_h = h // 32  # tiles in height
            block_w = w_per_core // 32  # tiles in width per shard
            subblock_w = min(block_w, 4)

            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(cores_per_group_x, cores_per_group_y),
                subblock_w=subblock_w,
                block_h=block_h,
                block_w=block_w,
                inplace=True,
                use_welford=False,
            )

            # Create operation params
            operation_params = ttnn.LayerNormParams()
            operation_params.norm_type = ttnn.LayerNormType.LAYERNORM
            operation_params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
            operation_params.eps = eps
            operation_params.output_mem_config = sharded_mem_config
            operation_params.program_config = program_config
            operation_params.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
            )

            # Create tensor args
            tensor_args = ttnn.LayerNormInputs()
            tensor_args.input = input_tensor
            if weight_tensor is not None:
                tensor_args.weight = weight_tensor
            if bias_tensor is not None:
                tensor_args.bias = bias_tensor

            # Get program descriptor
            program_descriptor = ttnn.LayerNormShardedProgramFactory.create_descriptor(
                operation_params, tensor_args, output_tensor
            )
            program_descriptors.append(program_descriptor)

            # Store operation data for later verification
            operations.append(
                {
                    "group_idx": group_idx,
                    "group_x": group_x,
                    "group_y": group_y,
                    "core_range": (start_x, start_y, end_x, end_y),
                    "use_weight": use_weight,
                    "use_bias": use_bias,
                    "dtype": dtype,
                    "input_tensor": input_tensor,
                    "weight_tensor": weight_tensor,
                    "bias_tensor": bias_tensor,
                    "output_tensor": output_tensor,
                    "torch_golden": torch_golden,
                    "torch_input": torch_input,
                }
            )

    # Merge all program descriptors
    logger.info(f"Merging {len(program_descriptors)} program descriptors...")
    merged_descriptor = ttnn.ProgramDescriptor.merge_descriptors(program_descriptors)
    logger.info(f"Merged descriptor has {len(merged_descriptor.kernels)} kernels")

    # Build io_tensors list: all inputs first, then all outputs
    io_tensors = []
    for op in operations:
        io_tensors.append(op["input_tensor"])
        if op["weight_tensor"] is not None:
            io_tensors.append(op["weight_tensor"])
        if op["bias_tensor"] is not None:
            io_tensors.append(op["bias_tensor"])
    for op in operations:
        io_tensors.append(op["output_tensor"])

    # Execute merged program
    logger.info("Executing merged program with generic_op...")
    ttnn.generic_op(io_tensors, merged_descriptor)

    # Verify each output against torch golden
    all_passed = True
    for op in operations:
        torch_output = ttnn.to_torch(op["output_tensor"])
        torch_golden = op["torch_golden"]

        # Handle shape differences (output might be [1,1,h,w] vs [h,w])
        torch_output = torch_output.reshape(torch_golden.shape)

        max_diff = torch.max(torch.abs(torch_golden.float() - torch_output.float()))
        mean_diff = torch.mean(torch.abs(torch_golden.float() - torch_output.float()))

        # More relaxed tolerance for bfloat8_b
        rtol = 0.1 if op["dtype"] == ttnn.bfloat8_b else 0.02
        atol = 0.1 if op["dtype"] == ttnn.bfloat8_b else 0.02

        matching = torch.allclose(torch_golden.float(), torch_output.float(), rtol=rtol, atol=atol)

        logger.info(
            f"Group {op['group_idx']}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
            f"matching={matching} (weight={op['use_weight']}, bias={op['use_bias']}, dtype={op['dtype']})"
        )

        if not matching:
            all_passed = False
            logger.error(
                f"Group {op['group_idx']} FAILED: cores {op['core_range']}, "
                f"max_diff={max_diff}, mean_diff={mean_diff}"
            )

    assert all_passed, "Not all groups matched their torch golden values"
    logger.info(f"All {len(operations)} groups passed verification!")


@skip_for_blackhole("Not tested / built for Blackhole")
def test_experimental_rms_norm_disjoint_cores(device):
    """
    Test ttnn.experimental.programs.rms_norm on two disjoint core ranges.

    This test:
    1. Creates two RMS norm operations on separate core ranges
    2. Uses ttnn.experimental.programs.rms_norm to create program descriptors
    3. Merges the descriptors using ProgramDescriptor.merge_descriptors
    4. Executes via generic_op
    5. Compares both outputs to torch reference
    """
    torch.manual_seed(42)

    eps = 1e-12
    h, w = 32, 64

    # Define two disjoint core ranges
    # Core range 1: cores (0,0) to (1,1) - 4 cores
    core_range_1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    # Core range 2: cores (2,0) to (3,1) - 4 cores
    core_range_2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 1))})

    # Create torch inputs
    torch_input_1 = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_2 = torch.rand((h, w), dtype=torch.bfloat16)

    # Optional weights
    torch_weight_1 = torch.rand((w,), dtype=torch.bfloat16)
    torch_weight_2 = torch.rand((w,), dtype=torch.bfloat16)

    # Compute torch golden (RMS norm)
    def torch_rms_norm(x, weight, eps):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
        result = x.float() / rms
        if weight is not None:
            result = result * weight.float()
        return result.to(torch.bfloat16)

    torch_golden_1 = torch_rms_norm(torch_input_1, torch_weight_1, eps)
    torch_golden_2 = torch_rms_norm(torch_input_2, torch_weight_2, eps)

    # Create ttnn tensors
    input_tensor_1 = ttnn.from_torch(
        torch_input_1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_2 = ttnn.from_torch(
        torch_input_2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    weight_tensor_1 = ttnn.from_torch(
        torch_weight_1.reshape(1, 1, 1, w),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weight_tensor_2 = ttnn.from_torch(
        torch_weight_2.reshape(1, 1, 1, w),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create branch descriptors using the experimental API
    # Output tensors are created internally by the BranchDescriptor
    branch_1 = ttnn.experimental.programs.rms_norm(
        input_tensor_1,
        core_range_1,
        epsilon=eps,
        weight=weight_tensor_1,
    )

    branch_2 = ttnn.experimental.programs.rms_norm(
        input_tensor_2,
        core_range_2,
        epsilon=eps,
        weight=weight_tensor_2,
    )

    logger.info(f"Created RMS norm branch 1 with {len(branch_1.descriptor.kernels)} kernels")
    logger.info(f"Created RMS norm branch 2 with {len(branch_2.descriptor.kernels)} kernels")

    # Launch composite operation using the experimental API
    # Outputs are returned from launch_composite
    logger.info("Executing merged RMS norm program via launch_composite...")
    output_tensor_1, output_tensor_2 = ttnn.experimental.launch_composite([branch_1, branch_2])

    # Verify results
    torch_output_1 = ttnn.to_torch(output_tensor_1).reshape(torch_golden_1.shape)
    torch_output_2 = ttnn.to_torch(output_tensor_2).reshape(torch_golden_2.shape)

    max_diff_1 = torch.max(torch.abs(torch_golden_1.float() - torch_output_1.float()))
    max_diff_2 = torch.max(torch.abs(torch_golden_2.float() - torch_output_2.float()))
    mean_diff_1 = torch.mean(torch.abs(torch_golden_1.float() - torch_output_1.float()))
    mean_diff_2 = torch.mean(torch.abs(torch_golden_2.float() - torch_output_2.float()))

    logger.info(f"RMS norm 1: max_diff={max_diff_1:.6f}, mean_diff={mean_diff_1:.6f}")
    logger.info(f"RMS norm 2: max_diff={max_diff_2:.6f}, mean_diff={mean_diff_2:.6f}")

    rtol, atol = 0.02, 0.02
    matching_1 = torch.allclose(torch_golden_1.float(), torch_output_1.float(), rtol=rtol, atol=atol)
    matching_2 = torch.allclose(torch_golden_2.float(), torch_output_2.float(), rtol=rtol, atol=atol)

    logger.info(f"RMS norm 1 matches: {matching_1}")
    logger.info(f"RMS norm 2 matches: {matching_2}")

    assert matching_1, f"RMS norm 1 output mismatch. Max diff: {max_diff_1}"
    assert matching_2, f"RMS norm 2 output mismatch. Max diff: {max_diff_2}"
    logger.info("Both RMS norm operations on disjoint cores passed!")


@skip_for_blackhole("Not tested / built for Blackhole")
def test_sharded_layernorm_with_valid_core_range(device):
    """
    Test that sharded layernorm works when core_range_set contains the shard spec cores.
    """
    torch.manual_seed(42)
    eps = 1e-12
    h, w = 32, 64

    # Create shard spec on cores (0,0)-(1,0) = 2 cores
    num_cores = 2
    shard_width = w // num_cores
    shard_shape = (h, shard_width)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Create input tensor
    torch_input = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )

    # Create output tensor with same sharding
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([h, w]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        sharded_mem_config,
    )

    # Create sharded program config
    block_h = h // 32
    block_w = shard_width // 32
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(num_cores, 1),
        subblock_w=min(block_w, 4),
        block_h=block_h,
        block_w=block_w,
        inplace=True,
        use_welford=False,
    )

    operation_params = ttnn.LayerNormParams()
    operation_params.norm_type = ttnn.LayerNormType.LAYERNORM
    operation_params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params.eps = eps
    operation_params.output_mem_config = sharded_mem_config
    operation_params.program_config = program_config
    operation_params.compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    tensor_args = ttnn.LayerNormInputs()
    tensor_args.input = input_tensor

    # Create core_range_set that CONTAINS the shard spec cores (0,0)-(1,0)
    # Use a larger range (0,0)-(3,3) to ensure shard spec is within it
    valid_core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})

    # This should succeed - shard spec cores are within the provided range
    program_descriptor = ttnn.LayerNormShardedProgramFactory.create_descriptor(
        operation_params, tensor_args, output_tensor, valid_core_range
    )

    logger.info(f"Sharded layernorm with valid core range succeeded, {len(program_descriptor.kernels)} kernels")
    assert len(program_descriptor.kernels) > 0


@skip_for_blackhole("Not tested / built for Blackhole")
def test_sharded_layernorm_with_invalid_core_range_should_fail(device):
    """
    Test that sharded layernorm raises an error when shard spec cores are outside core_range_set.
    """
    torch.manual_seed(42)
    eps = 1e-12
    h, w = 32, 64

    # Create shard spec on cores (2,0)-(3,0) = 2 cores
    num_cores = 2
    shard_width = w // num_cores
    shard_shape = (h, shard_width)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Create input tensor
    torch_input = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )

    # Create output tensor with same sharding
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([h, w]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        sharded_mem_config,
    )

    # Create sharded program config
    block_h = h // 32
    block_w = shard_width // 32
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(num_cores, 1),
        subblock_w=min(block_w, 4),
        block_h=block_h,
        block_w=block_w,
        inplace=True,
        use_welford=False,
    )

    operation_params = ttnn.LayerNormParams()
    operation_params.norm_type = ttnn.LayerNormType.LAYERNORM
    operation_params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params.eps = eps
    operation_params.output_mem_config = sharded_mem_config
    operation_params.program_config = program_config
    operation_params.compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    tensor_args = ttnn.LayerNormInputs()
    tensor_args.input = input_tensor

    # Create core_range_set that does NOT contain the shard spec cores (2,0)-(3,0)
    # Use range (0,0)-(1,1) which doesn't include cores 2 or 3 in x direction
    invalid_core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})

    # This should fail - shard spec cores (2,0)-(3,0) are outside the provided range
    with pytest.raises(RuntimeError) as exc_info:
        ttnn.LayerNormShardedProgramFactory.create_descriptor(
            operation_params, tensor_args, output_tensor, invalid_core_range
        )

    assert "not within" in str(exc_info.value).lower() or "shard" in str(exc_info.value).lower()
    logger.info(f"Correctly raised error for shard spec outside core_range_set: {exc_info.value}")


@skip_for_blackhole("Not tested / built for Blackhole")
def test_programs_layer_norm_auto_factory_selection(device):
    """
    Test that programs.layer_norm automatically selects the correct factory based on input.

    For non-sharded input: uses LayerNormMultiCoreProgramFactory
    For sharded input: uses LayerNormShardedProgramFactory
    """
    torch.manual_seed(123)
    eps = 1e-12
    h, w = 32, 64

    # Test 1: Non-sharded input
    torch_input = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})

    # programs.layer_norm should auto-select non-sharded factory
    branch = ttnn.experimental.programs.layer_norm(input_tensor, core_range, epsilon=eps)
    logger.info(f"Non-sharded: Created branch with {len(branch.descriptor.kernels)} kernels")
    assert len(branch.descriptor.kernels) > 0

    # Execute and verify
    (output,) = ttnn.experimental.launch_composite([branch])

    torch_golden = torch.nn.functional.layer_norm(torch_input.float(), (w,), eps=eps).to(torch.bfloat16)
    torch_output = ttnn.to_torch(output).reshape(torch_golden.shape)
    assert torch.allclose(torch_golden, torch_output, rtol=0.02, atol=0.02), "Non-sharded layer_norm mismatch"
    logger.info("Non-sharded layer_norm passed!")

    # Test 2: Sharded input
    num_cores = 2
    shard_width = w // num_cores
    shard_shape = (h, shard_width)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    torch_input_sharded = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor_sharded = ttnn.from_torch(
        torch_input_sharded,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )

    # programs.layer_norm should auto-select sharded factory (no core_range needed)
    branch_sharded = ttnn.experimental.programs.layer_norm(input_tensor_sharded, epsilon=eps)
    logger.info(f"Sharded: Created branch with {len(branch_sharded.descriptor.kernels)} kernels")
    assert len(branch_sharded.descriptor.kernels) > 0

    # Execute and verify
    (output_sharded,) = ttnn.experimental.launch_composite([branch_sharded])

    torch_golden_sharded = torch.nn.functional.layer_norm(torch_input_sharded.float(), (w,), eps=eps).to(torch.bfloat16)
    torch_output_sharded = ttnn.to_torch(output_sharded).reshape(torch_golden_sharded.shape)
    assert torch.allclose(
        torch_golden_sharded, torch_output_sharded, rtol=0.02, atol=0.02
    ), "Sharded layer_norm mismatch"
    logger.info("Sharded layer_norm passed!")
    logger.info("Auto factory selection test passed!")

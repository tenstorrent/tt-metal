# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


def torch_rms_norm(x, gamma=None, eps=1e-5):
    """Reference RMS norm implementation in PyTorch."""
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    x_normed = x / rms
    if gamma is not None:
        x_normed = x_normed * gamma
    return x_normed


def torch_layer_norm(x, gamma=None, beta=None, residual=None, eps=1e-5):
    """Reference Layer norm implementation in PyTorch with optional residual."""
    # Add residual if provided
    if residual is not None:
        x = x + residual

    # Compute mean and variance over the last dimension
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalize
    x_normed = (x - mean) / torch.sqrt(var + eps)

    # Apply gamma (scale) and beta (shift)
    if gamma is not None:
        x_normed = x_normed * gamma
    if beta is not None:
        x_normed = x_normed + beta

    return x_normed


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [128])  # Larger height for more tile rows
@pytest.mark.parametrize("w", [64])
def test_parallel_rms_norm(device, batch_size, h, w):
    """Test parallel execution using ttnn.parallel.branch() API."""
    torch.manual_seed(0)

    # Create two input tensors
    torch_input_a = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input_b = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    # Create weights (gamma)
    torch_weight_a = torch.rand((w,), dtype=torch.bfloat16)
    torch_weight_b = torch.rand((w,), dtype=torch.bfloat16)

    # Compute reference outputs
    torch_output_a = torch_rms_norm(torch_input_a, torch_weight_a)
    torch_output_b = torch_rms_norm(torch_input_b, torch_weight_b)

    # Move tensors to device
    input_a = ttnn.from_torch(torch_input_a, device=device, layout=ttnn.TILE_LAYOUT)
    input_b = ttnn.from_torch(torch_input_b, device=device, layout=ttnn.TILE_LAYOUT)
    weight_a = ttnn.from_torch(torch_weight_a, device=device, layout=ttnn.TILE_LAYOUT)
    weight_b = ttnn.from_torch(torch_weight_b, device=device, layout=ttnn.TILE_LAYOUT)

    # Define disjoint core ranges for each branch
    # Use single core per branch on different rows (y-axis) which works correctly
    cores_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    cores_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))])

    # Create branches using ttnn.parallel.branch(operation, *args, cores=..., **kwargs)
    branch_a = ttnn.parallel.branch(
        ttnn.rms_norm,
        input_a,
        cores=cores_a,
        epsilon=1e-5,
        weight=weight_a,
    )
    branch_b = ttnn.parallel.branch(
        ttnn.rms_norm,
        input_b,
        cores=cores_b,
        epsilon=1e-5,
        weight=weight_b,
    )

    # Execute in parallel using ttnn.parallel([...])
    results = ttnn.parallel([branch_a, branch_b])

    # Extract outputs
    output_a = ttnn.from_device(results[0][0])
    output_b = ttnn.from_device(results[1][0])

    # Convert back to torch
    output_a = ttnn.to_torch(output_a)
    output_b = ttnn.to_torch(output_b)

    # Verify results
    assert_with_pcc(torch_output_a, output_a, 0.999)
    assert_with_pcc(torch_output_b, output_b, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [128])  # Same as first test
@pytest.mark.parametrize("w", [64])
def test_parallel_rms_norm_different_seed(device, batch_size, h, w):
    """Test parallel RMS norm with different random seed."""
    torch.manual_seed(42)

    torch_input_a = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input_b = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    # Create weights (gamma) - RMS norm works better with weights
    torch_weight_a = torch.rand((w,), dtype=torch.bfloat16)
    torch_weight_b = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_a = torch_rms_norm(torch_input_a, torch_weight_a)
    torch_output_b = torch_rms_norm(torch_input_b, torch_weight_b)

    input_a = ttnn.from_torch(torch_input_a, device=device, layout=ttnn.TILE_LAYOUT)
    input_b = ttnn.from_torch(torch_input_b, device=device, layout=ttnn.TILE_LAYOUT)
    weight_a = ttnn.from_torch(torch_weight_a, device=device, layout=ttnn.TILE_LAYOUT)
    weight_b = ttnn.from_torch(torch_weight_b, device=device, layout=ttnn.TILE_LAYOUT)

    # Use single core per branch on different rows (y-axis) which works correctly
    cores_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))])
    cores_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(0, 3))])

    # Using ttnn.parallel.branch() API
    branch_a = ttnn.parallel.branch(ttnn.rms_norm, input_a, cores=cores_a, epsilon=1e-5, weight=weight_a)
    branch_b = ttnn.parallel.branch(ttnn.rms_norm, input_b, cores=cores_b, epsilon=1e-5, weight=weight_b)

    results = ttnn.parallel([branch_a, branch_b])

    output_a = ttnn.to_torch(ttnn.from_device(results[0][0]))
    output_b = ttnn.to_torch(ttnn.from_device(results[1][0]))

    assert_with_pcc(torch_output_a, output_a, 0.999)
    assert_with_pcc(torch_output_b, output_b, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [128])  # 4 tile rows (128/32)
@pytest.mark.parametrize("w", [64])  # 2 tile columns (64/32)
def test_rms_full_grid(device, batch_size, h, w):
    """
    Test parallel execution using full 8x8 compute grid split into 4x4 blocks of 2x2 cores.
    Each 2x2 block runs a sharded RMS norm calculation.
    Total: 16 parallel branches.
    """
    torch.manual_seed(123)

    # Grid layout: 8x8 cores split into 4x4 blocks of 2x2 cores
    # Block (i,j) covers cores [(2*i, 2*j), (2*i+1, 2*j+1)]
    num_blocks_x = 4  # 8 cores / 2 cores per block
    num_blocks_y = 4
    num_branches = num_blocks_x * num_blocks_y  # 16 branches

    # Create input and weight tensors for each branch
    torch_inputs = []
    torch_weights = []
    torch_outputs = []

    for i in range(num_branches):
        torch_input = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
        torch_weight = torch.rand((w,), dtype=torch.bfloat16)
        torch_output = torch_rms_norm(torch_input, torch_weight)

        torch_inputs.append(torch_input)
        torch_weights.append(torch_weight)
        torch_outputs.append(torch_output)

    # Move tensors to device
    device_inputs = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in torch_inputs]
    device_weights = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in torch_weights]

    # Create branches for each 2x2 block
    branches = []
    for block_y in range(num_blocks_y):
        for block_x in range(num_blocks_x):
            branch_idx = block_y * num_blocks_x + block_x

            # Calculate core range for this 2x2 block
            start_x = block_x * 2
            start_y = block_y * 2
            end_x = start_x + 1  # 2 cores wide (0,1)
            end_y = start_y + 1  # 2 cores tall (0,1)

            core_range = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))]
            )

            # Create branch for this block
            branch = ttnn.parallel.branch(
                ttnn.rms_norm,
                device_inputs[branch_idx],
                cores=core_range,
                epsilon=1e-5,
                weight=device_weights[branch_idx],
            )
            branches.append(branch)

    # Execute all 16 branches in parallel
    results = ttnn.parallel(branches)

    # Verify each branch output against torch reference
    for i in range(num_branches):
        output = ttnn.to_torch(ttnn.from_device(results[i][0]))
        assert_with_pcc(torch_outputs[i], output, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("num_branches", [2])
def test_layernorm_parallel_interleaved(device, batch_size, h, w, num_branches):
    """
    Test parallel LayerNorm with interleaved (non-sharded) tensors.
    Uses the simpler multi-core program factory to test parallel infrastructure.
    """
    torch.manual_seed(456)

    # Create input, weight, bias, and residual tensors for each branch
    torch_inputs = []
    torch_weights = []
    torch_biases = []
    torch_residuals = []
    torch_outputs = []

    for i in range(num_branches):
        torch_input = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
        torch_weight = torch.rand((w,), dtype=torch.bfloat16)
        torch_bias = torch.rand((w,), dtype=torch.bfloat16)
        torch_residual = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

        # Compute reference: LayerNorm with residual
        torch_output = torch_layer_norm(torch_input, torch_weight, torch_bias, torch_residual)

        torch_inputs.append(torch_input)
        torch_weights.append(torch_weight)
        torch_biases.append(torch_bias)
        torch_residuals.append(torch_residual)
        torch_outputs.append(torch_output)

    # Create branches with different core ranges (non-overlapping single cores)
    branches = []

    for i in range(num_branches):
        # Each branch gets its own core on different rows (y-axis) which works correctly
        core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, i), ttnn.CoreCoord(0, i))})

        # Move tensors to device (interleaved memory)
        input_tensor = ttnn.from_torch(
            torch_inputs[i],
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        residual_tensor = ttnn.from_torch(
            torch_residuals[i],
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        weight_tensor = ttnn.from_torch(
            torch_weights[i],
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        bias_tensor = ttnn.from_torch(
            torch_biases[i],
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

        # Create branch for this core using layer_norm
        branch = ttnn.parallel.branch(
            ttnn.layer_norm,
            input_tensor,
            cores=core_range_set,
            epsilon=1e-5,
            weight=weight_tensor,
            bias=bias_tensor,
            residual_input_tensor=residual_tensor,
        )
        branches.append(branch)

    # Execute all branches in parallel
    results = ttnn.parallel(branches)

    # Verify each branch output against torch reference
    for i in range(num_branches):
        output = ttnn.to_torch(ttnn.from_device(results[i][0]))
        assert_with_pcc(torch_outputs[i], output, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("block_h", [2])  # Height of each core block
@pytest.mark.parametrize("block_w", [2])  # Width of each core block
@pytest.mark.parametrize("num_blocks_h", [4])  # Number of blocks in height (total 4*2=8 cores)
@pytest.mark.parametrize("num_blocks_w", [4])  # Number of blocks in width (total 4*2=8 cores)
def test_layernorm_full_grid(device, batch_size, block_h, block_w, num_blocks_h, num_blocks_w):
    """
    Test LayerNorm across the full 8x8 grid using ttnn::parallel.

    This test divides the 8x8 compute grid into 4x4 blocks of 2x2 cores each.
    Each 2x2 block is a parallel branch running LayerNorm (non-sharded/interleaved) with:
    - residual input
    - weight (gamma)
    - bias (beta)

    This demonstrates 16 parallel LayerNorm operations running concurrently.
    """
    torch.manual_seed(42)

    num_branches = num_blocks_h * num_blocks_w  # 16 branches

    # Tensor dimensions per branch
    h = 128  # 4 tile rows
    w = 64  # 2 tile cols

    # Create input data for each branch
    torch_inputs = []
    torch_residuals = []
    torch_weights = []
    torch_biases = []
    torch_outputs = []

    for _ in range(num_branches):
        torch_input = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
        torch_residual = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
        torch_weight = torch.rand((w,), dtype=torch.bfloat16)
        torch_bias = torch.rand((w,), dtype=torch.bfloat16)

        # Compute reference output
        torch_output = torch_layer_norm(torch_input, torch_weight, torch_bias, torch_residual)

        torch_inputs.append(torch_input)
        torch_residuals.append(torch_residual)
        torch_weights.append(torch_weight)
        torch_biases.append(torch_bias)
        torch_outputs.append(torch_output)

    # Create branches for each 2x2 block
    branches = []

    for block_row in range(num_blocks_h):
        for block_col in range(num_blocks_w):
            branch_idx = block_row * num_blocks_w + block_col

            # Calculate the core range for this 2x2 block
            start_x = block_col * block_w
            start_y = block_row * block_h
            end_x = start_x + block_w - 1
            end_y = start_y + block_h - 1

            core_range_set = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))]
            )

            # Move tensors to device with interleaved (non-sharded) memory
            input_tensor = ttnn.from_torch(
                torch_inputs[branch_idx],
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
            residual_tensor = ttnn.from_torch(
                torch_residuals[branch_idx],
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
            weight_tensor = ttnn.from_torch(
                torch_weights[branch_idx],
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
            bias_tensor = ttnn.from_torch(
                torch_biases[branch_idx],
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )

            # Create branch for this block using layer_norm (interleaved)
            branch = ttnn.parallel.branch(
                ttnn.layer_norm,
                input_tensor,
                cores=core_range_set,
                epsilon=1e-5,
                weight=weight_tensor,
                bias=bias_tensor,
                residual_input_tensor=residual_tensor,
            )
            branches.append(branch)

    # Execute all branches in parallel
    results = ttnn.parallel(branches)

    # Verify each branch output against torch reference
    for i in range(num_branches):
        output = ttnn.to_torch(ttnn.from_device(results[i][0]))
        assert_with_pcc(torch_outputs[i], output, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("block_h", [2])  # Height of each core block
@pytest.mark.parametrize("block_w", [2])  # Width of each core block
@pytest.mark.parametrize("num_blocks_h", [4])  # Number of blocks in height (4*2=8 cores)
@pytest.mark.parametrize("num_blocks_w", [4])  # Number of blocks in width (4*2=8 cores)
@pytest.mark.parametrize("tiles_per_core_h", [1])  # Tiles per core in height
@pytest.mark.parametrize("tiles_per_core_w", [2])  # Tiles per core in width
def test_layernorm_full_grid_sharded(
    device, batch_size, block_h, block_w, num_blocks_h, num_blocks_w, tiles_per_core_h, tiles_per_core_w
):
    """
    Test sharded LayerNorm across the full 8x8 grid using ttnn::parallel.

    This test divides the 8x8 compute grid into 4x4 blocks of 2x2 cores each.
    Each 2x2 block is a parallel branch running sharded LayerNorm with:
    - Sharded input tensor (BLOCK_SHARDED on the 2x2 core block)
    - Sharded residual tensor (same shard spec as input)
    - Weight (gamma) - interleaved
    - Bias (beta) - interleaved

    This demonstrates 16 parallel sharded LayerNorm operations running concurrently.
    """
    torch.manual_seed(42)

    num_branches = num_blocks_h * num_blocks_w  # 16 branches

    # Per-branch tensor dimensions (sharded across 2x2 = 4 cores)
    h = tiles_per_core_h * block_h * 32  # 1 * 2 * 32 = 64 elements per branch
    w = tiles_per_core_w * block_w * 32  # 2 * 2 * 32 = 128 elements per branch

    # Per-core shard dimensions
    shard_h = tiles_per_core_h * 32  # 32
    shard_w = tiles_per_core_w * 32  # 64

    # Create input data for each branch
    torch_inputs = []
    torch_residuals = []
    torch_weights = []
    torch_biases = []
    torch_outputs = []

    for _ in range(num_branches):
        torch_input = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
        torch_residual = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
        torch_weight = torch.rand((w,), dtype=torch.bfloat16)
        torch_bias = torch.rand((w,), dtype=torch.bfloat16)

        # Compute reference output
        torch_output = torch_layer_norm(torch_input, torch_weight, torch_bias, torch_residual)

        torch_inputs.append(torch_input)
        torch_residuals.append(torch_residual)
        torch_weights.append(torch_weight)
        torch_biases.append(torch_bias)
        torch_outputs.append(torch_output)

    # Create branches for each 2x2 block
    branches = []

    for block_row in range(num_blocks_h):
        for block_col in range(num_blocks_w):
            branch_idx = block_row * num_blocks_w + block_col

            # Calculate the core range for this 2x2 block
            start_x = block_col * block_w
            start_y = block_row * block_h
            end_x = start_x + block_w - 1
            end_y = start_y + block_h - 1

            core_range_set = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))]
            )

            # Create sharded memory config for this block
            shard_spec = ttnn.ShardSpec(core_range_set, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
            mem_config = ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=shard_spec,
            )

            # Move tensors to device with sharded memory config
            input_tensor = ttnn.from_torch(
                torch_inputs[branch_idx],
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem_config,
            )
            residual_tensor = ttnn.from_torch(
                torch_residuals[branch_idx],
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem_config,
            )
            weight_tensor = ttnn.from_torch(
                torch_weights[branch_idx],
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
            bias_tensor = ttnn.from_torch(
                torch_biases[branch_idx],
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )

            # Create sharded layernorm program config
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(block_w, block_h),
                subblock_w=tiles_per_core_w,
                block_h=tiles_per_core_h,
                block_w=tiles_per_core_w,
                inplace=False,
            )

            # Create branch for this block using sharded layer_norm
            branch = ttnn.parallel.branch(
                ttnn.layer_norm,
                input_tensor,
                cores=core_range_set,
                epsilon=1e-5,
                weight=weight_tensor,
                bias=bias_tensor,
                residual_input_tensor=residual_tensor,
                memory_config=mem_config,
                program_config=program_config,
            )
            branches.append(branch)

    # Execute all branches in parallel
    results = ttnn.parallel(branches)

    # Verify each branch output against torch reference
    for i in range(num_branches):
        output = ttnn.to_torch(ttnn.from_device(results[i][0]))
        assert_with_pcc(torch_outputs[i], output, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("block_h", [2])  # Height of each core block
@pytest.mark.parametrize("block_w", [2])  # Width of each core block
@pytest.mark.parametrize("num_blocks_h", [4])  # Number of blocks in height (4*2=8 cores)
@pytest.mark.parametrize("num_blocks_w", [4])  # Number of blocks in width (4*2=8 cores)
@pytest.mark.parametrize("tiles_per_core_h", [1])  # Tiles per core in height
@pytest.mark.parametrize("tiles_per_core_w", [2])  # Tiles per core in width
def test_layernorm_heterogeneous_branches(
    device, batch_size, block_h, block_w, num_blocks_h, num_blocks_w, tiles_per_core_h, tiles_per_core_w
):
    """
    Test sharded LayerNorm with heterogeneous branch configurations.

    This test runs 16 parallel LayerNorm operations where each branch has:
    - Different combinations of weight, bias, and residual
    - Different data types (float32, bfloat16, bfloat8_b)
    - Alternating use of welford algorithm

    Branch configurations (16 branches) with varied dtypes and algorithms:
        Branch | Weight | Bias | Residual | DataType  | Welford | CBs
        -------|--------|------|----------|-----------|---------|----
        0      | ✗      | ✗    | ✗        | bfloat16  | ✗       | 14
        1      | ✓      | ✗    | ✗        | bfloat16  | ✓       | 12
        2      | ✓      | ✓    | ✗        | bfloat16  | ✗       | 16
        3      | ✗      | ✗    | ✓        | bfloat16  | ✓       | 11
        4      | ✓      | ✗    | ✓        | bfloat16  | ✗       | 16
        5      | ✓      | ✓    | ✓        | bfloat16  | ✓       | 14
        6      | ✗      | ✗    | ✗        | float32   | ✓       | 10
        7      | ✓      | ✗    | ✗        | float32   | ✗       | 15
        8      | ✓      | ✓    | ✗        | float32   | ✓       | 13
        9      | ✗      | ✗    | ✓        | float32   | ✗       | 15
        10     | ✓      | ✗    | ✓        | float32   | ✓       | 13
        11     | ✓      | ✓    | ✓        | float32   | ✗       | 17
        12     | ✗      | ✗    | ✗        | bfloat8_b | ✗       | 14
        13     | ✓      | ✗    | ✗        | bfloat8_b | ✓       | 12
        14     | ✓      | ✓    | ✗        | bfloat8_b | ✗       | 16
        15     | ✓      | ✓    | ✓        | bfloat8_b | ✓       | 14

    Note: "bias only" (weight=None, bias=set) is excluded due to kernel limitations
    with the sharded LayerNorm in parallel execution.
    """
    # Clear program cache to ensure fresh compilation for heterogeneous configurations
    device.disable_and_clear_program_cache()

    torch.manual_seed(42)

    num_branches = num_blocks_h * num_blocks_w  # 16 branches

    # Per-branch tensor dimensions (sharded across 2x2 = 4 cores)
    h = tiles_per_core_h * block_h * 32  # 1 * 2 * 32 = 64 elements per branch
    w = tiles_per_core_w * block_w * 32  # 2 * 2 * 32 = 128 elements per branch

    # Per-core shard dimensions
    shard_h = tiles_per_core_h * 32  # 32
    shard_w = tiles_per_core_w * 32  # 64

    # Define branch configurations: (has_weight, has_bias, has_residual, torch_dtype, ttnn_dtype, use_welford)
    # Test various combinations with varying dtypes and algorithms:
    # - Weight only, weight+bias, or neither (bias-only has kernel issues)
    # - Residual independently varied
    # - Different dtypes are tested
    # - Welford algorithm alternated
    #
    # NOTE: "bias only" (weight=None, bias=set) is excluded due to kernel limitations
    branch_configs = [
        # bfloat16 variants - alternating welford
        (False, False, False, torch.bfloat16, ttnn.bfloat16, False),  # 0: none, standard
        (True, False, False, torch.bfloat16, ttnn.bfloat16, True),  # 1: weight only, welford
        (True, True, False, torch.bfloat16, ttnn.bfloat16, False),  # 2: weight + bias, standard
        (False, False, True, torch.bfloat16, ttnn.bfloat16, True),  # 3: residual only, welford
        (True, False, True, torch.bfloat16, ttnn.bfloat16, False),  # 4: weight + residual, standard
        (True, True, True, torch.bfloat16, ttnn.bfloat16, True),  # 5: weight + bias + residual, welford
        # float32 variants - alternating welford
        (False, False, False, torch.float32, ttnn.float32, True),  # 6: none, welford
        (True, False, False, torch.float32, ttnn.float32, False),  # 7: weight only, standard
        (True, True, False, torch.float32, ttnn.float32, True),  # 8: weight + bias, welford
        (False, False, True, torch.float32, ttnn.float32, False),  # 9: residual only, standard
        (True, False, True, torch.float32, ttnn.float32, True),  # 10: weight + residual, welford
        (True, True, True, torch.float32, ttnn.float32, False),  # 11: weight + bias + residual, standard
        # bfloat8_b variants - alternating welford
        (False, False, False, torch.bfloat16, ttnn.bfloat8_b, False),  # 12: none, standard
        (True, False, False, torch.bfloat16, ttnn.bfloat8_b, True),  # 13: weight only, welford
        (True, True, False, torch.bfloat16, ttnn.bfloat8_b, False),  # 14: weight + bias, standard
        (True, True, True, torch.bfloat16, ttnn.bfloat8_b, True),  # 15: weight + bias + residual, welford
    ]

    # Create input data for each branch
    torch_inputs = []
    torch_residuals = []
    torch_weights = []
    torch_biases = []
    torch_outputs = []
    configs = []

    for branch_idx in range(num_branches):
        has_weight, has_bias, has_residual, torch_dtype, ttnn_dtype, use_welford = branch_configs[branch_idx]
        configs.append(branch_configs[branch_idx])

        # Create tensors in the appropriate dtype
        torch_input = torch.rand((batch_size, h, w), dtype=torch_dtype)
        torch_weight = torch.rand((w,), dtype=torch_dtype) if has_weight else None
        torch_bias = torch.rand((w,), dtype=torch_dtype) if has_bias else None
        torch_residual = torch.rand((batch_size, h, w), dtype=torch_dtype) if has_residual else None

        # Compute reference output with appropriate parameters
        torch_output = torch_layer_norm(torch_input, torch_weight, torch_bias, torch_residual)

        torch_inputs.append(torch_input)
        torch_weights.append(torch_weight)
        torch_biases.append(torch_bias)
        torch_residuals.append(torch_residual)
        torch_outputs.append(torch_output)

    # Create branches for each 2x2 block
    branches = []

    for block_row in range(num_blocks_h):
        for block_col in range(num_blocks_w):
            branch_idx = block_row * num_blocks_w + block_col
            has_weight, has_bias, has_residual, torch_dtype, ttnn_dtype, use_welford = configs[branch_idx]

            # Calculate the core range for this 2x2 block
            start_x = block_col * block_w
            start_y = block_row * block_h
            end_x = start_x + block_w - 1
            end_y = start_y + block_h - 1

            core_range_set = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))]
            )

            # Create sharded memory config for this block
            shard_spec = ttnn.ShardSpec(core_range_set, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
            mem_config = ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=shard_spec,
            )

            # Move input tensor to device with sharded memory config and appropriate dtype
            input_tensor = ttnn.from_torch(
                torch_inputs[branch_idx],
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn_dtype,
                memory_config=mem_config,
            )

            # Optional weight
            weight_tensor = None
            if has_weight:
                weight_tensor = ttnn.from_torch(
                    torch_weights[branch_idx],
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16 if torch_dtype == torch.bfloat16 else ttnn.float32,
                )

            # Optional bias
            bias_tensor = None
            if has_bias:
                bias_tensor = ttnn.from_torch(
                    torch_biases[branch_idx],
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16 if torch_dtype == torch.bfloat16 else ttnn.float32,
                )

            # Optional residual
            residual_tensor = None
            if has_residual:
                residual_tensor = ttnn.from_torch(
                    torch_residuals[branch_idx],
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn_dtype,
                    memory_config=mem_config,
                )

            # Create sharded layernorm program config with optional welford
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(block_w, block_h),
                subblock_w=tiles_per_core_w,
                block_h=tiles_per_core_h,
                block_w=tiles_per_core_w,
                inplace=False,
                use_welford=use_welford,
            )

            # Create branch for this block using sharded layer_norm
            branch = ttnn.parallel.branch(
                ttnn.layer_norm,
                input_tensor,
                cores=core_range_set,
                epsilon=1e-5,
                weight=weight_tensor,
                bias=bias_tensor,
                residual_input_tensor=residual_tensor,
                memory_config=mem_config,
                program_config=program_config,
            )
            branches.append(branch)

    # Execute all branches in parallel
    results = ttnn.parallel(branches)

    # Verify each branch output against torch reference
    # Use slightly lower PCC for bfloat8_b due to lower precision
    for i in range(num_branches):
        has_weight, has_bias, has_residual, torch_dtype, ttnn_dtype, use_welford = configs[i]
        output = ttnn.to_torch(ttnn.from_device(results[i][0]))

        # bfloat8_b has lower precision, so use a lower PCC threshold
        pcc_threshold = 0.98 if ttnn_dtype == ttnn.bfloat8_b else 0.999
        try:
            assert_with_pcc(torch_outputs[i], output, pcc_threshold)
        except AssertionError as e:
            print(
                f"Branch {i} FAILED: weight={has_weight}, bias={has_bias}, residual={has_residual}, "
                f"dtype={ttnn_dtype}, welford={use_welford}"
            )
            raise


def test_layernorm_heterogeneous_grid_sizes(device):
    """
    Test sharded LayerNorm with heterogeneous core grid block sizes.

    This test runs multiple parallel LayerNorm operations where each branch uses
    a different core grid block size. The branches are placed on disjoint core ranges
    and test various combinations of:
    - Different block sizes (1x1, 1x2, 2x1, 2x2, 4x2, etc.)
    - Different data types
    - With/without weight, bias, residual
    - With/without Welford algorithm

    Grid layout (8x8 grid, partial coverage):
        Branch 0: 2x2 block at (0,0)-(1,1)
        Branch 1: 1x4 block at (2,0)-(2,3)
        Branch 2: 4x1 block at (3,0)-(6,0)
        Branch 3: 2x2 block at (4,2)-(5,3)
        Branch 4: 1x1 block at (7,0)
        Branch 5: 3x2 block at (0,4)-(2,5)
    """
    # Clear program cache to ensure fresh compilation
    device.disable_and_clear_program_cache()

    torch.manual_seed(42)

    # Define branch configurations:
    # (block_w, block_h, start_x, start_y, has_weight, has_bias, has_residual, torch_dtype, ttnn_dtype, use_welford)
    branch_configs = [
        # Branch 0: 2x2 block at (0,0), bfloat16, weight+bias, no welford
        (2, 2, 0, 0, True, True, False, torch.bfloat16, ttnn.bfloat16, False),
        # Branch 1: 1x4 block at (2,0), float32, no weight/bias, welford
        (1, 4, 2, 0, False, False, False, torch.float32, ttnn.float32, True),
        # Branch 2: 4x1 block at (3,0), bfloat16, weight only, no welford
        (4, 1, 3, 0, True, False, False, torch.bfloat16, ttnn.bfloat16, False),
        # Branch 3: 2x2 block at (4,2), float32, weight+bias+residual, welford
        (2, 2, 4, 2, True, True, True, torch.float32, ttnn.float32, True),
        # Branch 4: 1x1 block at (7,0), bfloat8_b, weight+bias, no welford
        (1, 1, 7, 0, True, True, False, torch.bfloat16, ttnn.bfloat8_b, False),
        # Branch 5: 3x2 block at (0,4), bfloat16, residual only, welford
        (3, 2, 0, 4, False, False, True, torch.bfloat16, ttnn.bfloat16, True),
    ]

    num_branches = len(branch_configs)

    # Tiles per core (fixed for simplicity)
    tiles_per_core_h = 1
    tiles_per_core_w = 2

    # Per-core shard dimensions in elements
    shard_h = tiles_per_core_h * 32  # 32
    shard_w = tiles_per_core_w * 32  # 64

    # Create input data and branches
    torch_inputs = []
    torch_weights = []
    torch_biases = []
    torch_residuals = []
    torch_outputs = []
    branches = []

    print("\nBranch configurations:")
    print("-" * 100)
    print(
        f"{'Branch':^7} | {'Grid':^7} | {'Cores':^20} | {'Weight':^6} | {'Bias':^6} | {'Resid':^6} | {'Dtype':^10} | {'Welford':^7}"
    )
    print("-" * 100)

    for i, (
        block_w,
        block_h,
        start_x,
        start_y,
        has_weight,
        has_bias,
        has_residual,
        torch_dtype,
        ttnn_dtype,
        use_welford,
    ) in enumerate(branch_configs):
        # Calculate tensor dimensions for this branch
        h = tiles_per_core_h * block_h * 32  # height depends on block_h
        w = tiles_per_core_w * block_w * 32  # width depends on block_w

        # Calculate core range
        end_x = start_x + block_w - 1
        end_y = start_y + block_h - 1
        core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))]
        )

        print(
            f"{i:^7} | {block_w}x{block_h:^5} | ({start_x},{start_y})-({end_x},{end_y}){' '*(12-len(f'({start_x},{start_y})-({end_x},{end_y})'))} | {'✓' if has_weight else '✗':^6} | {'✓' if has_bias else '✗':^6} | {'✓' if has_residual else '✗':^6} | {str(ttnn_dtype).split('.')[-1]:^10} | {'✓' if use_welford else '✗':^7}"
        )

        # Create torch tensors
        torch_input = torch.rand((1, h, w), dtype=torch_dtype)
        torch_weight = torch.rand((w,), dtype=torch_dtype) if has_weight else None
        torch_bias = torch.rand((w,), dtype=torch_dtype) if has_bias else None
        torch_residual = torch.rand((1, h, w), dtype=torch_dtype) if has_residual else None

        # Compute reference output
        torch_output = torch_layer_norm(torch_input, torch_weight, torch_bias, torch_residual)

        torch_inputs.append(torch_input)
        torch_weights.append(torch_weight)
        torch_biases.append(torch_bias)
        torch_residuals.append(torch_residual)
        torch_outputs.append(torch_output)

        # Create sharded memory config for this block
        shard_spec = ttnn.ShardSpec(core_range_set, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=shard_spec,
        )

        # Move input tensor to device
        input_tensor = ttnn.from_torch(
            torch_input,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn_dtype,
            memory_config=mem_config,
        )

        # Optional weight
        weight_tensor = None
        if has_weight:
            weight_tensor = ttnn.from_torch(
                torch_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16 if torch_dtype == torch.bfloat16 else ttnn.float32,
            )

        # Optional bias
        bias_tensor = None
        if has_bias:
            bias_tensor = ttnn.from_torch(
                torch_bias,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16 if torch_dtype == torch.bfloat16 else ttnn.float32,
            )

        # Optional residual
        residual_tensor = None
        if has_residual:
            residual_tensor = ttnn.from_torch(
                torch_residual,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn_dtype,
                memory_config=mem_config,
            )

        # Create sharded layernorm program config
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(block_w, block_h),
            subblock_w=tiles_per_core_w,
            block_h=tiles_per_core_h,
            block_w=tiles_per_core_w,
            inplace=False,
            use_welford=use_welford,
        )

        # Create branch
        branch = ttnn.parallel.branch(
            ttnn.layer_norm,
            input_tensor,
            cores=core_range_set,
            epsilon=1e-5,
            weight=weight_tensor,
            bias=bias_tensor,
            residual_input_tensor=residual_tensor,
            memory_config=mem_config,
            program_config=program_config,
        )
        branches.append(branch)

    print("-" * 100)

    # Execute all branches in parallel
    print(f"\nExecuting {num_branches} branches in parallel...")
    results = ttnn.parallel(branches)

    # Verify each branch output against torch reference
    print("\nResults:")
    all_passed = True
    for i in range(num_branches):
        (
            block_w,
            block_h,
            start_x,
            start_y,
            has_weight,
            has_bias,
            has_residual,
            torch_dtype,
            ttnn_dtype,
            use_welford,
        ) = branch_configs[i]
        output = ttnn.to_torch(ttnn.from_device(results[i][0]))

        # bfloat8_b has lower precision
        pcc_threshold = 0.98 if ttnn_dtype == ttnn.bfloat8_b else 0.999

        from tests.ttnn.utils_for_testing import comp_pcc

        pcc_pass, pcc_msg = comp_pcc(torch_outputs[i], output, pcc_threshold)

        status = "✓ PASS" if pcc_pass else "✗ FAIL"
        print(f"  Branch {i} ({block_w}x{block_h}): {status} - PCC={pcc_msg:.6f}")

        if not pcc_pass:
            all_passed = False

    assert all_passed, "One or more branches failed PCC check"

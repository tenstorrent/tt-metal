# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp

# Define sharding configurations
height_sharded_memory_config = ttnn.create_sharded_memory_config(
    [128, 160],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6)), ttnn.CoreRange((1, 0), (1, 6))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

width_sharded_memory_config = ttnn.create_sharded_memory_config(
    [1792, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3)), ttnn.CoreRange((1, 0), (1, 3))}),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

block_sharded_memory_config = ttnn.create_sharded_memory_config(
    [256, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([4, 7, 64, 128]),
    ],
)
@pytest.mark.parametrize(
    "sharded_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
    ],
)
@pytest.mark.parametrize(
    "ttnn_op, dtype, atol_threshold, ulp_threshold",
    [
        (ttnn.log_sigmoid, ttnn.bfloat16, 1e-1, 7.0),
    ],
)
def test_unary_sharded_ops(input_shape, sharded_config, ttnn_op, dtype, atol_threshold, ulp_threshold, device):
    """Test unary operations with different sharding strategies and configurable thresholds"""
    torch.manual_seed(2024)

    # Map ttnn dtype to torch dtype
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    # Create input tensor with range suitable for the operation
    torch_input = torch.randn(input_shape, dtype=torch_dtype)

    # Get golden result from torch
    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output = golden_function(torch_input, device=device)

    # Convert to ttnn with sharded memory config
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    # Run operation with sharded input
    ttnn_output_sharded = ttnn_op(ttnn_input, memory_config=sharded_config)

    # Convert output back to torch
    ttnn_output = ttnn.to_torch(ttnn_output_sharded)

    # Compare with golden using specified thresholds
    assert torch.allclose(ttnn_output, torch_output, atol=atol_threshold)
    assert_with_ulp(torch_output, ttnn_output, ulp_threshold)


@pytest.mark.parametrize(
    "ttnn_op, dtype, low, high, atol_threshold, ulp_threshold",
    [
        (ttnn.log_sigmoid, ttnn.bfloat16, -87.0, 10.0, 1e-1, 7.0),
    ],
)
def test_unary_exhaustive_bitpatterns(ttnn_op, dtype, low, high, atol_threshold, ulp_threshold, device):
    """Test unary operations with exhaustive bf16 bit patterns within valid range"""
    torch.manual_seed(2024)

    # Map ttnn dtype to torch dtype
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    # Generate all possible bit patterns for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch_dtype)
    input_tensor_f32 = input_tensor.to(torch.float32)

    # Mask to working range to avoid overflow/underflow
    mask = (input_tensor_f32 >= low) & (input_tensor_f32 <= high)
    input_tensor = input_tensor[mask]

    # Convert to ttnn tensor
    ttnn_input = ttnn.from_torch(
        input_tensor,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get golden result from torch
    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output = golden_function(input_tensor, device=device)

    # Run operation
    ttnn_output = ttnn_op(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    # Compare with golden using specified threshold
    assert_with_ulp(ttnn_output, torch_output, ulp_threshold)
    assert torch.allclose(ttnn_output, torch_output, atol=atol_threshold)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([4, 7, 64, 128]),
    ],
)
@pytest.mark.parametrize(
    "sharded_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
    ],
)
@pytest.mark.parametrize(
    "torch_input_dtype, torch_output_dtype, ttnn_input_dtype, ttnn_output_dtype",
    [
        # uint16 -> bfloat16 conversions
        (torch.uint16, torch.bfloat16, ttnn.uint16, ttnn.bfloat16),
        # bfloat16 -> uint16 conversions
        (torch.bfloat16, torch.uint16, ttnn.bfloat16, ttnn.uint16),
        # int32 -> uint32 conversions
        (torch.int32, torch.uint32, ttnn.int32, ttnn.uint32),
        # uint32 -> float32 conversions
        (torch.uint32, torch.float32, ttnn.uint32, ttnn.float32),
        # float32 -> uint32 conversions
        (torch.float32, torch.uint32, ttnn.float32, ttnn.uint32),
    ],
)
def test_bitcast_sharded(
    input_shape,
    sharded_config,
    torch_input_dtype,
    torch_output_dtype,
    ttnn_input_dtype,
    ttnn_output_dtype,
    device,
):
    """Test bitcast operation with sharded tensors - reinterprets bit pattern without conversion"""
    torch.manual_seed(2024)

    # Generate test values based on dtype
    if torch_input_dtype == torch.uint16:
        torch_input = torch.randint(0, 65535, input_shape, dtype=torch_input_dtype)
    elif torch_input_dtype == torch.bfloat16:
        torch_input = torch.randn(input_shape, dtype=torch_input_dtype)
    elif torch_input_dtype == torch.int32:
        torch_input = torch.randint(-2147483648, 2147483647, input_shape, dtype=torch_input_dtype)
    elif torch_input_dtype == torch.uint32:
        torch_input = torch.randint(0, 4294967295, input_shape, dtype=torch_input_dtype)
    elif torch_input_dtype == torch.float32:
        torch_input = torch.randn(input_shape, dtype=torch_input_dtype)
    else:
        torch_input = torch.randn(input_shape, dtype=torch_input_dtype)

    # Create PyTorch reference using view (bitcast)
    torch_output = torch_input.view(torch_output_dtype)

    # Convert to ttnn with sharded memory config
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn_input_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    # Perform bitcast with sharded input
    ttnn_output_sharded = ttnn.bitcast(ttnn_input, ttnn_output_dtype, memory_config=sharded_config)

    # Convert output back to torch
    ttnn_output = ttnn.to_torch(ttnn_output_sharded, dtype=torch_output_dtype)

    # Compare values - bitcast should preserve exact bit patterns
    # Note: NaN values may convert to inf due to hardware packer limitation
    # For non-NaN, non-inf values, we expect exact match
    torch_output_flat = torch_output.flatten()
    ttnn_output_flat = ttnn_output.flatten()

    for i in range(len(torch_output_flat)):
        expected = torch_output_flat[i].item()
        actual = ttnn_output_flat[i].item()

        if torch.isnan(torch.tensor(expected)):
            # NaN values may convert to inf in bfloat16 due to packer hardware limitation
            assert torch.isinf(torch.tensor(actual)) or torch.isnan(
                torch.tensor(actual)
            ), f"Value {i}: Expected NaN, got {actual}"
        elif torch.isinf(torch.tensor(expected)):
            # Inf values should match
            assert torch.isinf(torch.tensor(actual)), f"Value {i}: Expected Inf, got {actual}"
        else:
            # Normal values should match exactly for bitcast
            if torch_output_dtype == torch.float32:
                # Allow tolerance for precision issues with large numbers
                # Use relative tolerance for large numbers, absolute for small
                abs_diff = abs(expected - actual)
                if abs(expected) > 1.0:
                    # Relative tolerance for large numbers (allow up to 0.1% relative error)
                    rel_tol = abs_diff / abs(expected)
                    assert (
                        rel_tol < 0.001 or abs_diff < 0.002 or expected == actual
                    ), f"Value {i}: Expected {expected}, got {actual}, difference: {abs_diff}, rel_diff: {rel_tol}"
                else:
                    # Absolute tolerance for small numbers
                    assert (
                        abs_diff < 0.002 or expected == actual
                    ), f"Value {i}: Expected {expected}, got {actual}, difference: {abs_diff}"
            elif torch_output_dtype == torch.bfloat16:
                # bfloat16 denormals may be flushed to zero due to hardware limitations
                expected_tensor = torch.tensor(expected)
                is_denormal = (
                    not torch.isnan(expected_tensor)
                    and not torch.isinf(expected_tensor)
                    and expected != 0.0
                    and abs(expected) < torch.finfo(torch.bfloat16).tiny
                )
                if is_denormal:
                    # Allow zero for denormal numbers
                    assert (
                        actual == 0.0 or abs(expected - actual) < 1e-6
                    ), f"Value {i}: Expected denormal {expected}, got {actual}"
                else:
                    # Exact match for normal values
                    assert (
                        expected == actual
                    ), f"Value {i}: Expected {expected}, got {actual}, difference: {abs(expected - actual)}"
            else:
                # For integer types, expect exact match
                assert (
                    expected == actual
                ), f"Value {i}: Expected {expected}, got {actual}, difference: {abs(expected - actual)}"

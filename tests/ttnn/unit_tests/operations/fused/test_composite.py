# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

import models.experimental.ops.descriptors as descriptors
import models.experimental.ops.descriptors.composite as composite
from tests.ttnn.utils_for_testing import assert_allclose


# ============================================================================
# Common Utilities
# ============================================================================


def torch_rms_norm(x, gamma, eps=1e-5):
    """Reference RMS norm implementation in PyTorch."""
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    x_normed = x.float() / rms
    if gamma is not None:
        x_normed = x_normed * gamma.float()
    return x_normed.to(x.dtype)


def torch_layer_norm(x, gamma, eps=1e-5):
    """Reference layer norm implementation in PyTorch."""
    mean = torch.mean(x.float(), dim=-1, keepdim=True)
    var = torch.var(x.float(), dim=-1, keepdim=True, unbiased=False)
    x_normed = (x.float() - mean) / torch.sqrt(var + eps)
    if gamma is not None:
        x_normed = x_normed * gamma.float()
    return x_normed.to(x.dtype)


def assert_outputs_are_close(torch_inputs, torch_weights, ttnn_outputs, rtol=1e-2, atol=2.5e-2):
    for torch_input, torch_weight, ttnn_output in zip(torch_inputs, torch_weights, ttnn_outputs):
        # Compute expected output
        expected = torch_rms_norm(torch_input, torch_weight)

        # Convert actual output to torch
        actual = ttnn.to_torch(ttnn.from_device(ttnn_output))

        assert_allclose(expected, actual, rtol=rtol, atol=atol)


# ============================================================================
# Tests
# ============================================================================


def test_deepseek_v3_q_kv_rms_norm(device):
    """
    Tests the parallel Q/KV RMS norms in
    the DeepSeek V3 MLA block
    """

    def create_q_norm_tensors(device):
        """
        Create q_norm tensors:
        - L1 width sharded on cores (0,0) to (3,3) = 16 cores
        - Shard shape: [32, 96]
        - Total tensor shape: [1, 1, 32, 1536] (16 cores × 96 = 1536)
        - Weights are DRAM interleaved
        """
        torch.manual_seed(42)

        # Core range: (0,0) to (3,3) = 4x4 = 16 cores
        q_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
        num_cores = 16
        shard_width = 96
        total_width = num_cores * shard_width  # 1536

        # Tensor shapes
        input_shape = (1, 1, 32, total_width)
        weight_shape = (1, 1, 1, total_width)

        # Create torch tensors
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

        # Create sharded memory config for input (L1 width sharded)
        shard_spec = ttnn.ShardSpec(q_cores, [32, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=shard_spec,
        )

        # Move input to device with sharded memory config
        input_tensor = ttnn.from_torch(
            torch_input,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=sharded_mem_config,
        )

        # Weights are DRAM interleaved
        weight_tensor = ttnn.from_torch(
            torch_weight,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Program config for sharded RMS norm
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(4, 4),  # 4x4 grid
            subblock_w=shard_width // 32,  # tiles per shard width = 96/32 = 3
            block_h=1,  # 32/32 = 1 tile high
            block_w=shard_width // 32,  # 3 tiles wide per core
            inplace=False,
        )

        return {
            "input": input_tensor,
            "weight": weight_tensor,
            "torch_input": torch_input,
            "torch_weight": torch_weight,
            "cores": q_cores,
            "memory_config": sharded_mem_config,
            "program_config": program_config,
            "name": "q_norm",
        }

    def create_kv_norm_tensors(device):
        """
        Create kv_norm tensors:
        - L1 width sharded on cores (5,0) to (6,7) = 2x8 = 16 cores
        - Shard shape: [32, 32]
        - Total tensor shape: [1, 1, 32, 512] (16 cores × 32 = 512)
        - Weights are DRAM interleaved
        """
        torch.manual_seed(123)

        # Core range: (5,0) to (6,7) = 2 cols × 8 rows = 16 cores
        kv_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 7))])
        num_cores = 16
        shard_width = 32
        total_width = num_cores * shard_width  # 512

        # Tensor shapes
        input_shape = (1, 1, 32, total_width)
        weight_shape = (1, 1, 1, total_width)

        # Create torch tensors
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

        # Create sharded memory config for input (L1 width sharded)
        shard_spec = ttnn.ShardSpec(kv_cores, [32, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=shard_spec,
        )

        # Move input to device with sharded memory config
        input_tensor = ttnn.from_torch(
            torch_input,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=sharded_mem_config,
        )

        # Weights are DRAM interleaved
        weight_tensor = ttnn.from_torch(
            torch_weight,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Program config for sharded RMS norm
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(2, 8),  # 2 cols × 8 rows
            subblock_w=shard_width // 32,  # tiles per shard width = 32/32 = 1
            block_h=1,  # 32/32 = 1 tile high
            block_w=shard_width // 32,  # 1 tile wide per core
            inplace=False,
        )

        return {
            "input": input_tensor,
            "weight": weight_tensor,
            "torch_input": torch_input,
            "torch_weight": torch_weight,
            "cores": kv_cores,
            "memory_config": sharded_mem_config,
            "program_config": program_config,
            "name": "kv_norm",
        }

    q_tensors = create_q_norm_tensors(device)
    kv_tensors = create_kv_norm_tensors(device)

    # Create branches
    q_branch = descriptors.rms_norm(
        q_tensors["input"],
        epsilon=1e-5,
        weight=q_tensors["weight"],
        memory_config=q_tensors["memory_config"],
        core_range_set=q_tensors["cores"],
        program_config=q_tensors["program_config"],
    )
    kv_branch = descriptors.rms_norm(
        kv_tensors["input"],
        epsilon=1e-5,
        weight=kv_tensors["weight"],
        memory_config=kv_tensors["memory_config"],
        core_range_set=kv_tensors["cores"],
        program_config=kv_tensors["program_config"],
    )

    # Run composite (returns list of output lists, one per op descriptor)
    outputs = composite.launch([q_branch, kv_branch])

    # Verify outputs (extract first output tensor from each op's output list)
    assert_outputs_are_close(
        torch_inputs=[q_tensors["torch_input"], kv_tensors["torch_input"]],
        torch_weights=[q_tensors["torch_weight"], kv_tensors["torch_weight"]],
        ttnn_outputs=[outputs[0][0], outputs[1][0]],
    )


def _create_compute_heavy_tensors(device):
    """
    Create two sets of compute-heavy tensors for left_half and right_half norm operations.

    Each half uses 32 cores (4x8 grid) with width sharding.
    Tensor dimensions are chosen to be compute-heavy:
    - Large width (many tiles per core for reduction)
    - Multiple rows (more work per core)

    Left half: cores (0,0)-(3,7) = 32 cores
    Right half: cores (4,0)-(7,7) = 32 cores
    """
    torch.manual_seed(42)

    # Left half: cores (0,0) to (3,7) = 4 columns x 8 rows = 32 cores
    left_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))])
    num_left_cores = 32

    # Right half: cores (4,0) to (7,7) = 4 columns x 8 rows = 32 cores
    right_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 7))])

    # Make it compute-heavy: larger shard width and multiple tile rows
    # Each core processes a shard of [num_rows, shard_width]
    # Total width = num_cores * shard_width
    num_tile_rows = 4  # 4 tile rows = 128 rows
    shard_width_tiles = 8  # 8 tiles per core = 256 elements per row per core
    shard_width = shard_width_tiles * 32  # 256 elements
    shard_height = num_tile_rows * 32  # 128 rows

    total_width = num_left_cores * shard_width  # 32 * 256 = 8192 elements
    total_height = shard_height  # 128 rows

    # Create tensors with shape [1, 1, height, width]
    shape = (1, 1, total_height, total_width)
    weight_shape = (1, 1, 1, total_width)

    torch_left_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_left_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    torch_right_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_right_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    # Create sharded memory configs
    left_shard_spec = ttnn.ShardSpec(left_cores, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    left_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=left_shard_spec,
    )

    right_shard_spec = ttnn.ShardSpec(right_cores, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    right_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=right_shard_spec,
    )

    # Convert to TTNN tensors
    left_input = ttnn.from_torch(
        torch_left_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=left_mem_config,
    )
    left_weight = ttnn.from_torch(
        torch_left_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    right_input = ttnn.from_torch(
        torch_right_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=right_mem_config,
    )
    right_weight = ttnn.from_torch(
        torch_right_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    return {
        "left": {
            "input": left_input,
            "weight": left_weight,
            "torch_input": torch_left_input,
            "torch_weight": torch_left_weight,
            "cores": left_cores,
            "mem_config": left_mem_config,
        },
        "right": {
            "input": right_input,
            "weight": right_weight,
            "torch_input": torch_right_input,
            "torch_weight": torch_right_weight,
            "cores": right_cores,
            "mem_config": right_mem_config,
        },
        "config": {
            "shape": shape,
            "shard_shape": [shard_height, shard_width],
        },
    }


def _run_heavy_composite_norm_test(device, norm_fn, torch_norm_fn):
    """
    Common test logic for heavy composite norm operations.

    Args:
        device: The TTNN device
        norm_fn: The descriptor norm function (descriptors.rms_norm or descriptors.layer_norm)
        torch_norm_fn: The PyTorch reference implementation (torch_rms_norm or torch_layer_norm)
    """
    # Create tensors
    tensors = _create_compute_heavy_tensors(device)

    # Create programs
    left_branch = norm_fn(
        tensors["left"]["input"],
        core_range_set=tensors["left"]["cores"],
        epsilon=1e-5,
        weight=tensors["left"]["weight"],
    )
    right_branch = norm_fn(
        tensors["right"]["input"],
        core_range_set=tensors["right"]["cores"],
        epsilon=1e-5,
        weight=tensors["right"]["weight"],
    )

    # Run composite (returns list of output lists, one per op descriptor)
    outputs = composite.launch([left_branch, right_branch])

    # Verify outputs (extract first output tensor from each op's output list)
    for torch_input, torch_weight, ttnn_output in zip(
        [tensors["left"]["torch_input"], tensors["right"]["torch_input"]],
        [tensors["left"]["torch_weight"], tensors["right"]["torch_weight"]],
        [outputs[0][0], outputs[1][0]],
    ):
        expected = torch_norm_fn(torch_input, torch_weight)
        actual = ttnn.to_torch(ttnn.from_device(ttnn_output))
        assert_allclose(expected, actual, rtol=1e-2, atol=2.5e-2)


def test_composite_rms_heavy(device):
    """
    Test a heavy compute load with RMS norm, where each core processes
    multiple tiles

    Uses large tensors split across the 8x8 grid:
    - Left half: (0,0)-(3,7) = 32 cores
    - Right half: (4,0)-(7,7) = 32 cores

    Each core processes [128, 256] elements = 32,768 elements
    Total per half: 32 cores * 32,768 = 1,048,576 elements
    """
    _run_heavy_composite_norm_test(device, descriptors.rms_norm, torch_rms_norm)


def test_composite_layer_norm_heavy(device):
    """
    Test a heavy compute load with layer norm, where each core processes
    multiple tiles

    Uses large tensors split across the 8x8 grid:
    - Left half: (0,0)-(3,7) = 32 cores
    - Right half: (4,0)-(7,7) = 32 cores

    Each core processes [128, 256] elements = 32,768 elements
    Total per half: 32 cores * 32,768 = 1,048,576 elements
    """
    _run_heavy_composite_norm_test(device, descriptors.layer_norm, torch_layer_norm)


def test_composite_mixed_norm(device):
    """
    Test composite with mixed normalization types.

    Left half uses RMS norm, right half uses layer norm.
    This tests that different operation types can be composed together.
    """
    # Create tensors
    tensors = _create_compute_heavy_tensors(device)

    # Create branches with different norm types
    left_branch = descriptors.rms_norm(
        tensors["left"]["input"],
        core_range_set=tensors["left"]["cores"],
        epsilon=1e-5,
        weight=tensors["left"]["weight"],
    )
    right_branch = descriptors.layer_norm(
        tensors["right"]["input"],
        core_range_set=tensors["right"]["cores"],
        epsilon=1e-5,
        weight=tensors["right"]["weight"],
    )

    # Run composite
    outputs = composite.launch([left_branch, right_branch])

    # Verify left output (RMS norm)
    expected_left = torch_rms_norm(tensors["left"]["torch_input"], tensors["left"]["torch_weight"])
    actual_left = ttnn.to_torch(ttnn.from_device(outputs[0][0]))
    assert_allclose(expected_left, actual_left, rtol=1e-2, atol=2.5e-2)

    # Verify right output (layer norm)
    expected_right = torch_layer_norm(tensors["right"]["torch_input"], tensors["right"]["torch_weight"])
    actual_right = ttnn.to_torch(ttnn.from_device(outputs[1][0]))
    assert_allclose(expected_right, actual_right, rtol=1e-2, atol=2.5e-2)


def test_composite_non_sharded(device):
    """
    Test composite operations with non-sharded (DRAM interleaved) inputs.

    Uses two operations on separate core ranges with DRAM interleaved tensors.
    """
    torch.manual_seed(42)

    # Define non-overlapping core ranges
    left_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    right_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 3))])

    # Create tensor shapes
    shape = (1, 1, 32, 1024)
    weight_shape = (1, 1, 1, 1024)

    # Create torch tensors
    torch_left_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_left_weight = torch.rand(weight_shape, dtype=torch.bfloat16)
    torch_right_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_right_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    # Convert to TTNN with DRAM interleaved memory
    left_input = ttnn.from_torch(
        torch_left_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    left_weight = ttnn.from_torch(
        torch_left_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    right_input = ttnn.from_torch(
        torch_right_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    right_weight = ttnn.from_torch(
        torch_right_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create branches (core_range_set is required for non-sharded inputs)
    left_branch = descriptors.rms_norm(
        left_input,
        core_range_set=left_cores,
        epsilon=1e-5,
        weight=left_weight,
    )
    right_branch = descriptors.layer_norm(
        right_input,
        core_range_set=right_cores,
        epsilon=1e-5,
        weight=right_weight,
    )

    # Run composite
    outputs = composite.launch([left_branch, right_branch])

    # Verify outputs
    expected_left = torch_rms_norm(torch_left_input, torch_left_weight)
    actual_left = ttnn.to_torch(ttnn.from_device(outputs[0][0]))
    assert_allclose(expected_left, actual_left, rtol=1e-2, atol=2.5e-2)

    expected_right = torch_layer_norm(torch_right_input, torch_right_weight)
    actual_right = ttnn.to_torch(ttnn.from_device(outputs[1][0]))
    assert_allclose(expected_right, actual_right, rtol=1e-2, atol=2.5e-2)


def test_composite_8_ops_random_cores(device):
    """
    Test composite with 8 operations on random non-overlapping core ranges.

    Uses a mix of RMS norm and layer norm operations to test:
    - Many operations in a single composite
    - Random core placement
    - Mixed operation types
    """
    torch.manual_seed(123)

    # Define 8 non-overlapping core ranges across the grid
    # Use 2x2 blocks
    core_ranges = [
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),  # 4 cores
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 1))]),  # 4 cores
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 1))]),  # 4 cores
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(7, 1))]),  # 4 cores
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(1, 3))]),  # 4 cores
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(3, 3))]),  # 4 cores
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 2), ttnn.CoreCoord(5, 3))]),  # 4 cores
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(6, 2), ttnn.CoreCoord(7, 3))]),  # 4 cores
    ]

    # Use width sharding - 4 cores per operation, each core gets a shard
    num_cores_per_op = 4
    shard_width = 64  # 2 tiles per core
    total_width = num_cores_per_op * shard_width  # 256

    # Create tensor shapes (same for all ops)
    shape = (1, 1, 32, total_width)  # 1 tile row, distributed across width
    weight_shape = (1, 1, 1, total_width)

    # Create tensors and branches for all 8 operations
    torch_inputs = []
    torch_weights = []
    branches = []
    norm_fns = []  # Track which norm function to use for verification

    for i, cores in enumerate(core_ranges):
        # Create torch tensors
        torch_input = torch.rand(shape, dtype=torch.bfloat16)
        torch_weight = torch.rand(weight_shape, dtype=torch.bfloat16)
        torch_inputs.append(torch_input)
        torch_weights.append(torch_weight)

        # Create sharded memory config
        shard_spec = ttnn.ShardSpec(cores, [32, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=shard_spec,
        )

        # Convert to TTNN with sharded memory
        ttnn_input = ttnn.from_torch(
            torch_input,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=sharded_mem_config,
        )
        ttnn_weight = ttnn.from_torch(
            torch_weight,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Alternate between RMS norm and layer norm
        if i % 2 == 0:
            branch = descriptors.rms_norm(
                ttnn_input,
                core_range_set=cores,
                epsilon=1e-5,
                weight=ttnn_weight,
            )
            norm_fns.append(torch_rms_norm)
        else:
            branch = descriptors.layer_norm(
                ttnn_input,
                core_range_set=cores,
                epsilon=1e-5,
                weight=ttnn_weight,
            )
            norm_fns.append(torch_layer_norm)

        branches.append(branch)

    # Run composite with all 8 operations
    outputs = composite.launch(branches)

    # Verify all outputs
    for i, (torch_input, torch_weight, output, norm_fn) in enumerate(
        zip(torch_inputs, torch_weights, outputs, norm_fns)
    ):
        expected = norm_fn(torch_input, torch_weight)
        actual = ttnn.to_torch(ttnn.from_device(output[0]))
        assert_allclose(expected, actual, rtol=1e-2, atol=2.5e-2)


def test_composite_program_cache(device):
    """Test that composite.launch() properly caches the merged program."""
    # Setup: sharded tensors on non-overlapping cores
    batch_size, seq_len, hidden_dim = 1, 128, 1024
    torch.manual_seed(0)

    # Core ranges for two operations (non-overlapping)
    cores_left = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    cores_right = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 1))})

    # Create compute kernel configs
    arch = device.arch()
    compute_config = ttnn.rmsnorm_default_compute_config(arch)

    # Helper to create inputs and run composite
    def run_composite_ops():
        # Create inputs
        torch_input_left = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
        torch_input_right = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
        torch_weight_left = torch.ones(hidden_dim, dtype=torch.bfloat16)
        torch_weight_right = torch.ones(hidden_dim, dtype=torch.bfloat16)

        # Convert to ttnn tensors (sharded)
        ttnn_input_left = ttnn.from_torch(torch_input_left, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_input_right = ttnn.from_torch(
            torch_input_right, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn_weight_left = ttnn.from_torch(
            torch_weight_left, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn_weight_right = ttnn.from_torch(
            torch_weight_right, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # Create operation descriptors
        left = descriptors.rms_norm(
            ttnn_input_left,
            core_range_set=cores_left,
            weight=ttnn_weight_left,
            compute_kernel_config=compute_config,
        )
        right = descriptors.rms_norm(
            ttnn_input_right,
            core_range_set=cores_right,
            weight=ttnn_weight_right,
            compute_kernel_config=compute_config,
        )

        # Launch composite operation
        outputs = composite.launch([left, right])
        return outputs

    # Get initial cache count
    initial_cache_entries = device.num_program_cache_entries()

    # Run the same composite operation 3 times
    for _ in range(3):
        outputs = run_composite_ops()
        # Verify outputs are valid tensors
        assert len(outputs) == 2
        assert all(len(out) == 1 for out in outputs)

    # Check that only 1 new program was cached (the merged program)
    # Each iteration should reuse the same cached merged program
    final_cache_entries = device.num_program_cache_entries()
    new_entries = final_cache_entries - initial_cache_entries

    assert new_entries == 1, (
        f"Expected 1 new program cache entry for the merged program, "
        f"but got {new_entries} entries (initial: {initial_cache_entries}, final: {final_cache_entries})"
    )


def test_composite_program_cache_different_configs(device):
    """Test that different composite configurations create separate cache entries."""
    # Setup
    batch_size, seq_len, hidden_dim = 1, 128, 1024
    torch.manual_seed(0)

    # Core ranges
    cores_left = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    cores_right = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 1))})

    arch = device.arch()
    compute_config = ttnn.rmsnorm_default_compute_config(arch)

    # Helper to run composite with specific configuration
    def run_composite_with_cores(core_set_left, core_set_right):
        torch_input_left = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
        torch_input_right = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
        torch_weight = torch.ones(hidden_dim, dtype=torch.bfloat16)

        ttnn_input_left = ttnn.from_torch(torch_input_left, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_input_right = ttnn.from_torch(
            torch_input_right, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        left = descriptors.rms_norm(
            ttnn_input_left,
            core_range_set=core_set_left,
            weight=ttnn_weight,
            compute_kernel_config=compute_config,
        )
        right = descriptors.rms_norm(
            ttnn_input_right,
            core_range_set=core_set_right,
            weight=ttnn_weight,
            compute_kernel_config=compute_config,
        )

        return composite.launch([left, right])

    # Get initial cache count
    initial_cache_entries = device.num_program_cache_entries()

    # Run with first configuration twice (should cache 1 program)
    for _ in range(2):
        run_composite_with_cores(cores_left, cores_right)

    cache_after_first = device.num_program_cache_entries()
    assert cache_after_first - initial_cache_entries == 1, "First configuration should add 1 cache entry"

    # Run with different core configuration (should add new cache entry)
    cores_left_new = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 1))})
    cores_right_new = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(7, 1))})

    run_composite_with_cores(cores_left_new, cores_right_new)

    cache_after_second = device.num_program_cache_entries()
    assert cache_after_second - cache_after_first == 1, "Different configuration should add 1 new cache entry"

    # Run first configuration again (should reuse cached program, no new entry)
    run_composite_with_cores(cores_left, cores_right)

    cache_final = device.num_program_cache_entries()
    assert cache_final == cache_after_second, "Rerunning first configuration should not add new cache entry"


def test_composite_layer_norm_welford_non_sharded(device):
    """
    Test composite layer norm with Welford algorithm on non-sharded (DRAM interleaved) inputs.

    Uses two LayerNorm operations on separate core ranges with DRAM interleaved tensors
    and Welford algorithm for numerically stable variance computation.
    """
    torch.manual_seed(42)

    # Define non-overlapping core ranges
    left_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    right_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 3))])

    # Create tensor shapes
    shape = (1, 1, 32, 1024)
    weight_shape = (1, 1, 1, 1024)

    # Create torch tensors
    torch_left_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_left_weight = torch.rand(weight_shape, dtype=torch.bfloat16)
    torch_right_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_right_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    # Convert to TTNN with DRAM interleaved memory
    left_input = ttnn.from_torch(
        torch_left_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    left_weight = ttnn.from_torch(
        torch_left_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    right_input = ttnn.from_torch(
        torch_right_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    right_weight = ttnn.from_torch(
        torch_right_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create program config with Welford enabled
    welford_config = ttnn.LayerNormDefaultProgramConfig(use_welford=True)

    # Create branches with Welford LayerNorm
    left_branch = descriptors.layer_norm(
        left_input,
        core_range_set=left_cores,
        epsilon=1e-5,
        weight=left_weight,
        program_config=welford_config,
    )
    right_branch = descriptors.layer_norm(
        right_input,
        core_range_set=right_cores,
        epsilon=1e-5,
        weight=right_weight,
        program_config=welford_config,
    )

    # Run composite
    outputs = composite.launch([left_branch, right_branch])

    # Verify outputs
    expected_left = torch_layer_norm(torch_left_input, torch_left_weight)
    actual_left = ttnn.to_torch(ttnn.from_device(outputs[0][0]))
    assert_allclose(expected_left, actual_left, rtol=1e-2, atol=2.5e-2)

    expected_right = torch_layer_norm(torch_right_input, torch_right_weight)
    actual_right = ttnn.to_torch(ttnn.from_device(outputs[1][0]))
    assert_allclose(expected_right, actual_right, rtol=1e-2, atol=2.5e-2)


def test_composite_layer_norm_welford_sharded(device):
    """
    Test composite layer norm with Welford algorithm on sharded (L1) inputs.

    Uses two LayerNorm operations on separate core ranges with L1 width-sharded tensors
    and Welford algorithm for numerically stable variance computation.
    """
    torch.manual_seed(42)

    # Define non-overlapping core ranges for sharding
    # Left half: cores (0,0)-(3,7) = 32 cores
    left_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))])
    # Right half: cores (4,0)-(7,7) = 32 cores
    right_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 7))])

    num_cores = 32
    shard_width = 64  # 2 tiles per core
    shard_height = 32  # 1 tile row
    total_width = num_cores * shard_width  # 2048

    # Create tensor shapes
    shape = (1, 1, shard_height, total_width)
    weight_shape = (1, 1, 1, total_width)

    # Create torch tensors
    torch_left_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_left_weight = torch.rand(weight_shape, dtype=torch.bfloat16)
    torch_right_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_right_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    # Create sharded memory configs
    left_shard_spec = ttnn.ShardSpec(left_cores, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    left_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=left_shard_spec,
    )

    right_shard_spec = ttnn.ShardSpec(right_cores, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    right_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=right_shard_spec,
    )

    # Convert to TTNN with sharded memory
    left_input = ttnn.from_torch(
        torch_left_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=left_mem_config,
    )
    left_weight = ttnn.from_torch(
        torch_left_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    right_input = ttnn.from_torch(
        torch_right_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=right_mem_config,
    )
    right_weight = ttnn.from_torch(
        torch_right_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create program config with Welford enabled for sharded
    left_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 8),
        subblock_w=shard_width // 32,
        block_h=shard_height // 32,
        block_w=shard_width // 32,
        inplace=False,
        use_welford=True,
    )
    right_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 8),
        subblock_w=shard_width // 32,
        block_h=shard_height // 32,
        block_w=shard_width // 32,
        inplace=False,
        use_welford=True,
    )

    # Create branches with Welford LayerNorm
    left_branch = descriptors.layer_norm(
        left_input,
        core_range_set=left_cores,
        epsilon=1e-5,
        weight=left_weight,
        memory_config=left_mem_config,
        program_config=left_program_config,
    )
    right_branch = descriptors.layer_norm(
        right_input,
        core_range_set=right_cores,
        epsilon=1e-5,
        weight=right_weight,
        memory_config=right_mem_config,
        program_config=right_program_config,
    )

    # Run composite
    outputs = composite.launch([left_branch, right_branch])

    # Verify outputs
    expected_left = torch_layer_norm(torch_left_input, torch_left_weight)
    actual_left = ttnn.to_torch(ttnn.from_device(outputs[0][0]))
    assert_allclose(expected_left, actual_left, rtol=1e-2, atol=2.5e-2)

    expected_right = torch_layer_norm(torch_right_input, torch_right_weight)
    actual_right = ttnn.to_torch(ttnn.from_device(outputs[1][0]))
    assert_allclose(expected_right, actual_right, rtol=1e-2, atol=2.5e-2)


def test_composite_overlapping_cores_error(device):
    """
    Test that composite.launch() raises an error when core ranges overlap.

    Overlapping core ranges between different operations in a composite
    would cause undefined behavior, so this should be caught and rejected.
    """
    torch.manual_seed(42)

    # Define OVERLAPPING core ranges - both include cores (2,0) to (3,3)
    left_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    right_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(5, 3))])
    # left_cores covers (0,0)-(3,3), right_cores covers (2,0)-(5,3)
    # Overlap: (2,0)-(3,3)

    # Create tensor shapes - use enough tile rows (16) to ensure all 16 cores are used
    # This guarantees the overlapping cores will actually be used
    shape = (1, 1, 512, 512)  # 512/32 = 16 tile rows, uses all 16 cores per branch
    weight_shape = (1, 1, 1, 512)

    # Create torch tensors
    torch_left_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_left_weight = torch.rand(weight_shape, dtype=torch.bfloat16)
    torch_right_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_right_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    # Convert to TTNN with DRAM interleaved memory
    left_input = ttnn.from_torch(
        torch_left_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    left_weight = ttnn.from_torch(
        torch_left_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    right_input = ttnn.from_torch(
        torch_right_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    right_weight = ttnn.from_torch(
        torch_right_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create branches with overlapping core ranges
    left_branch = descriptors.rms_norm(
        left_input,
        core_range_set=left_cores,
        epsilon=1e-5,
        weight=left_weight,
    )
    right_branch = descriptors.rms_norm(
        right_input,
        core_range_set=right_cores,
        epsilon=1e-5,
        weight=right_weight,
    )

    # Attempting to launch with overlapping cores should raise an error
    with pytest.raises(RuntimeError, match="overlapping"):
        composite.launch([left_branch, right_branch])

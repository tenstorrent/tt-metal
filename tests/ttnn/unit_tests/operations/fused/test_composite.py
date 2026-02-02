# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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
    )
    kv_branch = descriptors.rms_norm(
        kv_tensors["input"],
        epsilon=1e-5,
        weight=kv_tensors["weight"],
        memory_config=kv_tensors["memory_config"],
        core_range_set=kv_tensors["cores"],
    )

    # Run composite (returns list of output lists, one per op descriptor)
    outputs = composite.launch([q_branch, kv_branch])

    # Verify outputs (extract first output tensor from each op's output list)
    assert_outputs_are_close(
        torch_inputs=[q_tensors["torch_input"], kv_tensors["torch_input"]],
        torch_weights=[q_tensors["torch_weight"], kv_tensors["torch_weight"]],
        ttnn_outputs=[outputs[0][0], outputs[1][0]],
    )


def test_composite_rms_heavy(device):
    """
    Test a heavy compute load, where each core processes
    multiple tiles

    Uses large tensors split across the 8x8 grid:
    - Left half: (0,0)-(3,7) = 32 cores
    - Right half: (4,0)-(7,7) = 32 cores

    Each core processes [128, 256] elements = 32,768 elements
    Total per half: 32 cores * 32,768 = 1,048,576 elements
    """

    def create_compute_heavy_tensors(device):
        """
        Create two sets of compute-heavy tensors for left_half and right_half RMS norm.

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

    # Create tensors
    tensors = create_compute_heavy_tensors(device)

    # Create programs
    left_branch = descriptors.rms_norm(
        tensors["left"]["input"],
        core_range_set=tensors["left"]["cores"],
        epsilon=1e-5,
        weight=tensors["left"]["weight"],
    )
    right_branch = descriptors.rms_norm(
        tensors["right"]["input"],
        core_range_set=tensors["right"]["cores"],
        epsilon=1e-5,
        weight=tensors["right"]["weight"],
    )

    # Run composite (returns list of output lists, one per op descriptor)
    outputs = composite.launch([left_branch, right_branch])

    # Verify outputs (extract first output tensor from each op's output list)
    assert_outputs_are_close(
        torch_inputs=[tensors["left"]["torch_input"], tensors["right"]["torch_input"]],
        torch_weights=[tensors["left"]["torch_weight"], tensors["right"]["torch_weight"]],
        ttnn_outputs=[outputs[0][0], outputs[1][0]],
    )

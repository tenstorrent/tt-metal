# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn


def get_high_perf_compute_config():
    """
    Returns a high-performance compute kernel configuration.

    Optimizations:
    - MathFidelity.HiFi2: Approx 2x faster than HiFi4, sufficient precision for inference.
    - fp32_dest_acc_en=True: Ensures accumulation happens in FP32 for stability.
    - packer_l1_acc=True: Keeps accumulation buffers in L1 memory, reducing DRAM read/writes.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def l1_width_sharded_memory(hidden_states):
    """
    Applies L1 Width Sharding to a tensor.

    Why:
    - Sharding splits the tensor across multiple cores (e.g., 8x8 grid).
    - Keeps data local to the core processing it, maximizing parallel efficiency.
    - Signficantly reduces NoC (Network-on-Chip) traffic compared to Interleaved DRAM.
    """
    # Handle different rank inputs
    if len(hidden_states.shape) == 4:
        batch_size, __, seq_len, hidden_size = hidden_states.padded_shape
    elif len(hidden_states.shape) == 3:
        batch_size, seq_len, hidden_size = hidden_states.padded_shape
    else:
        # Fallback to standard L1 if shape is unusual
        return ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

    # Calculate optimal grid size
    num_cores = hidden_size // ttnn.TILE_SIZE

    # Try to fit into a standard grid (e.g. 8 wide)
    # This logic might need tuning based on the specific chip topology (Wormhole/Grayskull)
    if num_cores % 8 == 0:
        core_grid = ttnn.CoreGrid(y=num_cores // 8, x=8)
    else:
        # If not divisible by 8, use a flat 1xN grid or similar
        # For safety in this initial pass, we might fallback if it's weird
        # core_grid = ttnn.CoreGrid(y=1, x=num_cores)
        return ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

    shard_width = hidden_size // (core_grid.x * core_grid.y)

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=[batch_size * seq_len, shard_width],
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    return ttnn.to_memory_config(hidden_states, width_sharded_mem_config)

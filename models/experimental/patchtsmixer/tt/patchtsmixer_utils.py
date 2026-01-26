"""
Utility functions for PatchTSMixer TTNN implementation.

Provides HEIGHT sharding helpers optimized for Wormhole (8x7 = 56 cores).
"""

import ttnn


# Constants
TILE_SIZE = 32
WORMHOLE_GRID = [8, 7]  # 56 cores
MAX_CORES = 56


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division: ⌈a/b⌉"""
    return (a + b - 1) // b


def _to_tiles(x: int) -> int:
    """Convert element count to tile count (32 elements per tile)."""
    return _ceil_div(x, TILE_SIZE)


def _pick_divisor_at_most(value: int, at_most: int) -> int:
    """
    Find largest divisor d where d <= at_most and value % d == 0.

    This ensures even distribution of tiles across cores.
    Returns 1 if no valid divisor found.
    """
    for d in range(at_most, 1, -1):
        if value % d == 0:
            return d
    return 1


def choose_height_sharding(
    M: int, K: int, *, max_cores: int = MAX_CORES, min_tiles_per_core: int = 2, min_K_tiles: int = 2
):
    """
    Decide whether to use HEIGHT sharding based on tile dimensions.

    HEIGHT sharding distributes rows (M dimension) across cores while keeping
    the K dimension intact on each core. This is optimal for matmuls where:
    - M (batch * sequence) is large enough to parallelize
    - K (feature dimension) is not tiny (to avoid padding overhead)

    Args:
        M: Number of rows (effective batch size: B * C * Np)
        K: Number of columns (feature dimension: D or patch_length)
        max_cores: Maximum number of cores to use (default: 56 for Wormhole)
        min_tiles_per_core: Minimum tiles per core to justify sharding (default: 2)
        min_K_tiles: Minimum K tiles required to enable sharding (default: 2)
                     Avoids sharding when K is tiny (e.g., channel mixing with C=2-16)

    Returns:
        Tuple of (do_shard, num_cores, shard_shape):
        - do_shard: bool, whether sharding is recommended
        - num_cores: int, number of cores to use (0 if not sharding)
        - shard_shape: [rows, cols] in elements (None if not sharding)

    Example:
        For ETTh2 with batch_size=4:
        - M = 4 * 7 * 64 = 1,792 rows
        - K = 64 columns
        - Mt = 56 tiles, Kt = 2 tiles
        - Recommends: 56 cores, each processing 32 rows
    """
    Mt = _to_tiles(M)
    Kt = _to_tiles(K)

    # Gate 1: K dimension must not be tiny (avoid padding overhead)
    # Example: channel mixing where C=7, Kt=1 would waste 25/32 elements per tile
    if Kt < min_K_tiles:
        return (False, 0, None)

    # Gate 2: Need enough tiles to distribute across at least 2 cores
    max_nc = min(max_cores, Mt // min_tiles_per_core)
    if max_nc < 2:
        return (False, 0, None)

    # Select core count that divides Mt evenly for balanced load
    # Example: Mt=56 → try [56,48,32,28,16,8,7,4,2,1] → pick 56 (perfect!)
    nc = _pick_divisor_at_most(Mt, max_nc)
    if nc < 2:
        return (False, 0, None)

    # Each core processes this many complete tiles
    tiles_per_core = Mt // nc
    shard_rows = tiles_per_core * TILE_SIZE
    shard_cols = K  # Full width per core (HEIGHT sharding)

    return (True, nc, [shard_rows, shard_cols])


def make_height_sharded_mem_config(num_cores: int, shard_shape):
    """
    Create HEIGHT_SHARDED memory configuration for L1.

    Args:
        num_cores: Number of cores to distribute across
        shard_shape: [rows, cols] in elements that each core will process

    Returns:
        MemoryConfig with HEIGHT_SHARDED layout
    """
    core_range_set = ttnn.num_cores_to_corerangeset(
        target_num_cores=num_cores,
        grid_size=WORMHOLE_GRID,
        row_wise=True,
    )

    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            grid=core_range_set,
            shard_shape=shard_shape,
            shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def apply_linear_height_sharded(
    x,
    weight,
    bias,
    M: int,
    K: int,
    out_shape: tuple,
    *,
    use_sharding: bool = True,
    compute_config=None,
):
    """
    Apply linear layer with optional HEIGHT sharding optimization.

    This is a convenience wrapper that:
    1. Decides whether to shard based on dimensions
    2. Reshapes to 2D if sharding
    3. Applies HEIGHT_SHARDED linear
    4. De-shards back to interleaved for compatibility
    5. Reshapes to output shape

    Args:
        x: Input tensor (any rank)
        weight: Weight tensor
        bias: Bias tensor
        M: Effective batch size (product of all dims except last)
        K: Input feature dimension (last dim)
        out_shape: Target output shape (e.g., (B, C, Np, D))
        use_sharding: Enable HEIGHT sharding optimization (default: True)
        compute_config: Compute kernel configuration

    Returns:
        Output tensor in out_shape with interleaved memory layout

    Example:
        x: (4, 7, 64, 64) → M=1792, K=64
        out = apply_linear_height_sharded(
            x, W, b, M=1792, K=64, out_shape=(4, 7, 64, 128)
        )
    """
    if not use_sharding:
        return ttnn.linear(x, weight, bias=bias, compute_kernel_config=compute_config)

    do_shard, nc, shard_shape = choose_height_sharding(M, K)

    if not do_shard:
        return ttnn.linear(x, weight, bias=bias, compute_kernel_config=compute_config)

    # Flatten to 2D matmul view: (1, 1, M, K)
    x_2d = ttnn.reshape(x, (1, 1, M, K))

    # Convert to HEIGHT_SHARDED layout
    shard_mem_config = make_height_sharded_mem_config(nc, shard_shape)
    x_sharded = ttnn.to_memory_config(x_2d, shard_mem_config)

    # Compute with sharded layout (output stays sharded in L1)
    out_sharded = ttnn.linear(
        x_sharded,
        weight,
        bias=bias,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(x_sharded)

    # CRITICAL: Convert back to interleaved for compatibility with
    # downstream ops (permute, softmax, residual add, etc.)
    out_interleaved = ttnn.sharded_to_interleaved(out_sharded, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out_sharded)

    # Reshape to target output shape
    out = ttnn.reshape(out_interleaved, out_shape)

    return out

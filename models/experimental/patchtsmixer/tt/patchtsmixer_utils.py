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
    the K dimension intact on each core. This is optimal when:
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

    # IMPORTANT: shard shape should be tile-aligned sizes, not raw sizes.
    shard_rows = tiles_per_core * TILE_SIZE  # multiple of 32

    shard_cols = Kt * TILE_SIZE  # padded to tiles K

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
    program_config=None,
    fused_activation=None,
    activation: str | None = None,
    program_config_factory=None,
    min_tiles_per_core: int = 2,
    min_K_tiles: int = 1,
    return_sharded: bool = False,
    out_memory_config=None,
):
    """Apply linear with optional HEIGHT sharding and reshape to `out_shape`."""
    # Normalize to matmul view (1,1,M,K) only if needed.
    if (
        len(x.shape) == 4
        and int(x.shape[0]) == 1
        and int(x.shape[1]) == 1
        and int(x.shape[2]) == M
        and int(x.shape[3]) == K
    ):
        x_2d = x
        x_was_reshaped = False
    else:
        x_2d = ttnn.reshape(x, (1, 1, M, K))
        x_was_reshaped = True

    if not use_sharding:
        out_2d = ttnn.linear(
            x_2d,
            weight,
            bias=bias,
            activation=activation,
            compute_kernel_config=compute_config,
        )
        if x_was_reshaped:
            ttnn.deallocate(x_2d)
        out = ttnn.reshape(out_2d, out_shape)
        return out

    do_shard, nc, shard_shape = choose_height_sharding(
        M,
        K,
        min_tiles_per_core=min_tiles_per_core,
        min_K_tiles=min_K_tiles,
    )

    if not do_shard:
        out_2d = ttnn.linear(
            x_2d,
            weight,
            bias=bias,
            activation=activation,
            compute_kernel_config=compute_config,
        )
        if x_was_reshaped:
            ttnn.deallocate(x_2d)
        out = ttnn.reshape(out_2d, out_shape)
        return out

    # shard_shape uses padded K
    K_pad = shard_shape[1]

    # Pad to match the padded shard width when needed
    if K_pad != K:
        x_2d = ttnn.pad(
            x_2d,
            padding=((0, 0), (0, 0), (0, 0), (0, K_pad - K)),
            value=0.0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # if x_2d was a reshape-view, it's now replaced by a new tensor anyway
        x_was_reshaped = True
        # Pad weight along its input-feature dim to match K_pad
        if len(weight.shape) == 4:
            weight = ttnn.pad(
                weight,
                padding=((0, 0), (0, 0), (0, K_pad - K), (0, 0)),
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        elif len(weight.shape) == 2:
            weight = ttnn.pad(
                weight,
                padding=((0, K_pad - K), (0, 0)),
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    shard_mem_config = make_height_sharded_mem_config(nc, shard_shape)
    if x_2d.is_sharded():
        x_sharded = x_2d
    else:
        x_sharded = ttnn.to_memory_config(x_2d, shard_mem_config)
        ttnn.deallocate(x_2d)

    out_cfg = out_memory_config if (return_sharded and out_memory_config is not None) else shard_mem_config
    if out_memory_config is None:
        out_k = int(out_shape[-1])
        out_k_pad = _to_tiles(out_k) * TILE_SIZE
        if out_k_pad != shard_shape[1]:
            out_shard_shape = [shard_shape[0], out_k_pad]
            out_cfg = make_height_sharded_mem_config(nc, out_shard_shape)

    # Build a program_config using the callback.
    if program_config_factory is not None:
        out_k = int(out_shape[-1])
        program_config = program_config_factory(
            nc=nc, M=M, K=K, out_k=out_k, shard_shape=shard_shape, out_memory_config=out_cfg
        )

    if fused_activation is not None:
        if program_config is None:
            raise RuntimeError(
                "fused_activation requires an explicit program_config  (e.g. MatmulMultiCoreReuseMultiCast1DProgramConfig)"
            )
        program_config.fused_activation = fused_activation

    out_sharded = ttnn.linear(
        x_sharded,
        weight,
        bias=bias,
        memory_config=out_cfg,
        compute_kernel_config=compute_config,
        program_config=program_config,
    )
    ttnn.deallocate(x_sharded)

    if return_sharded:
        # Avoid reshaping sharded outputs to prevent view-only tensors.
        return out_sharded

    out_interleaved = ttnn.sharded_to_interleaved(out_sharded, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out_sharded)

    out = ttnn.reshape(out_interleaved, out_shape)
    return out


def make_1d_mcast_prog_config_for_height_sharded(
    *,
    nc: int,
    shard_shape,
    out_k: int,
    fuse_batch: bool = True,
    # fused_activation = None,
    mcast_in0: bool = False,
    in0_block_w_tiles: int | None = None,
    out_subblock_h: int = 1,
    out_subblock_w: int | None = None,
):
    # shard_shape is [rows, cols] in elements, already tile-aligned
    shard_rows_elems, shard_cols_elems = shard_shape

    per_core_M = shard_rows_elems // TILE_SIZE  # tiles along M per core
    K_tiles = shard_cols_elems // TILE_SIZE  # tiles along K per core
    out_k_tiles = _to_tiles(out_k)
    per_core_N = out_k_tiles
    # Wormhole grid is [x,y] in the constants, but TTNN CoreGrid often takes x,y separately.
    # We already use num_cores_to_corerangeset(row_wise=True), so compute grid is essentially 1D.
    # We'll use a 1D-ish compute_with_storage_grid_size = (1, nc) or (nc, 1) depending on row_wise.
    # Since we used row_wise=True, cores are laid out row-wise (x changes first).
    # The safest is to set grid_size to (WORMHOLE_GRID[0], WORMHOLE_GRID[1]) or (nc,1) if supported.
    # Start with full chip grid and let corerangeset handle selection.

    compute_grid = ttnn.CoreCoord(WORMHOLE_GRID[0], WORMHOLE_GRID[1])

    if in0_block_w_tiles is None:
        in0_block_w_tiles = _pick_divisor_at_most(K_tiles, min(4, K_tiles))

    if out_subblock_w is None:
        out_subblock_w = min(4, per_core_N)

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=compute_grid,
        fuse_batch=fuse_batch,
        mcast_in0=mcast_in0,
        in0_block_w=in0_block_w_tiles,  # K blocking (tiles)
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        # fused_activation=fused_activation,
        # You can add out_block_h/out_block_w later for more expliciticy.
        # untilize_out=Fase is default; keep tiles output for the next matmul.
    )

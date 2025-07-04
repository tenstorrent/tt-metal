# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Sequence

import torch

import ttnn

# Constants
TILE_SIZE = 32
NORM_CATEGORIES = {"attention_norm", "mlp_norm", "q_norm", "k_norm"}


# Compute kernel configurations
# FP32 acc does not appear to be needed for accuracy in model tests or demo runs.
COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

COMPUTE_KERNEL_CONFIG_HIFI2_FP16 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

COMPUTE_KERNEL_CONFIG_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

COMPUTE_KERNEL_CONFIG_HIFI2_NA = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)

COMPUTE_KERNEL_CONFIG_SDPA = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def get_dram_weight_grid(mesh_device):
    """Create DRAM weight grid from mesh device."""
    dram_grid_size = mesh_device.dram_grid_size()
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
            )
        }
    )


def get_out_subblock_w(per_core_N, out_subblock_h):
    """Calculate output subblock width for matmul config."""
    # Find the largest value <= 4 that evenly divides per_core_N
    max_val = 4 // out_subblock_h
    for i in range(max_val, 0, -1):
        if per_core_N % i == 0:
            return i
    return 1


def create_dram_sharded_mem_config(k, n, mesh_device):
    """Create DRAM-sharded memory config for width-sharded tensors"""
    dram_grid_size = mesh_device.dram_grid_size()
    dram_cores = dram_grid_size.x  # WH has 12 dram cores, P150 has 8, P100 has 7
    assert dram_grid_size.y == 1, "Current dram sharding assumes y dim is 1"
    padded_size = math.ceil(n / (TILE_SIZE * dram_cores)) * (TILE_SIZE * dram_cores)
    shard_spec = ttnn.ShardSpec(
        get_dram_weight_grid(mesh_device), (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def matmul_config(
    m: int,
    k: int,
    n: int,
    grid_size: tuple[int, int],
    in0_block_w: int = None,
    fuse_batch: bool = False,
    fused_activation=None,
    per_core_M=None,
    per_core_N=None,
):
    """Generate matmul multi-core reuse multicast program config."""
    if per_core_M is None:
        per_core_M = math.ceil(m / (TILE_SIZE * grid_size[1]))
    if per_core_N is None:
        per_core_N = math.ceil(n / (TILE_SIZE * grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = get_out_subblock_w(per_core_N, out_subblock_h)

    if in0_block_w is None:
        in0_block_w = find_largest_divisor(k // (TILE_SIZE * grid_size[1]))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=fuse_batch,
    )


def dram_shard_core_grid_for_k(k: int) -> ttnn.CoreGrid:
    """Calculate core grid for DRAM sharding based on k dimension."""
    rows, cols = find_grid(k // TILE_SIZE)
    return ttnn.CoreGrid(x=cols, y=rows)


def find_grid(N):
    """
    Find the number of rows and columns for a grid of cores such that
    the total number of tiles N can be evenly divided among the cores.
    Each core will have the same integer number of tiles.
    The grid size is limited to a maximum of 8 rows and 8 columns.

    Parameters:
        N (int): Total number of tiles to be distributed.

    Returns:
        tuple: A tuple (rows, cols) representing the grid dimensions.

    Raises:
        AssertionError: If it's not possible to find such a grid configuration.
    """
    max_rows = 8
    max_cols = 8
    max_cores = max_rows * max_cols

    # Find all possible numbers of cores that divide N and are less than or equal to max_cores
    target = 32
    possible_cores = [k for k in range(1, max_cores + 1) if N % k == 0]
    possible_cores.sort(key=lambda x: abs(x - target))  # Sort by closest to target

    for cores in possible_cores:
        # Try to find a grid configuration with the current number of cores
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols

    # If no configuration is found, assert an error
    raise AssertionError(
        f"Cannot find a grid configuration for {N} tiles that evenly divides into {max_cores} cores of max size {max_rows}x{max_cols}."
    )


def find_prefill_grid(row_tiles, col_tiles):
    """Find a grid such that the number of row tiles evenly divides into the number
    of rows and the number of column tiles evenly divides into the number of columns
    """
    max_rows = 8
    max_cols = 8

    # Find number of cols that evenly divides into the number of columns
    cols = None
    rows = None

    for i in range(max_cols, 0, -1):
        if col_tiles % i == 0:
            cols = i
            break

    for i in range(max_rows, 0, -1):
        if row_tiles % i == 0:
            rows = i
            break

    assert cols is not None, f"Cannot find a number of columns that evenly divides into {col_tiles}, not even 1(!)."
    assert rows is not None, f"Cannot find a number of rows that evenly divides into {row_tiles}, not even 1(!)."
    return rows, cols


def dram_shard_core_grid_for_k_and_n(k: int, n: int) -> ttnn.CoreGrid:
    """Calculate core grid for DRAM sharding based on k and n dimensions."""
    rows, cols = find_grid_k_n(k // TILE_SIZE, n // TILE_SIZE)
    return ttnn.CoreGrid(x=cols, y=rows)


def find_grid_k_n(K, N):
    """
    Find the number of rows and columns for a grid of cores such that
    the total number of tiles K and N can be evenly divided among the cores.
    Each core will have the same integer number of tiles.

    Parameters:
        K (int): Total number of K tiles to be distributed.
        N (int): Total number of N tiles to be distributed.

    Returns:
        tuple: A tuple (rows, cols) representing the grid dimensions.

    Raises:
        AssertionError: If it's not possible to find such a grid configuration.
    """
    max_rows = 8
    max_cols = 8  # Maximum number of rows or columns
    max_cores = max_rows * max_cols  # Maximum number of cores

    # Find all possible numbers of cores that divide both K and N and are less than or equal to max_cores
    possible_cores = [c for c in range(1, max_cores + 1) if K % c == 0 and N % c == 0]
    possible_cores.sort(reverse=True)  # Start checking from the largest number of cores

    for cores in possible_cores:
        # Try to find a grid configuration with the current number of cores
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols

    # If no configuration is found, assert an error
    raise AssertionError(
        f"Cannot find a grid configuration such that both {K} and {N} tiles evenly divide into cores of max size {max_rows}x{max_cols}."
    )


def find_largest_divisor(n, max_divisor=8):
    """Find the largest divisor of n that is <= max_divisor."""
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1  # Fallback to 1 if no divisor found


def dram_matmul_config(m: int, k: int, n: int, num_cores=None):
    """Generate DRAM sharded matmul program config."""
    # in0_block_w must evenly divide k and be no larger than tile_size * num_cores
    if num_cores is None:
        num_cores = dram_shard_core_grid_for_k_and_n(k, n).num_cores
        assert (
            k % (TILE_SIZE * num_cores) == 0
        ), f"k must be divisible by tile_size * num_cores: {k} % {TILE_SIZE * num_cores} != 0"

    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=find_largest_divisor(k // (TILE_SIZE * num_cores)),
        per_core_M=math.ceil(m / TILE_SIZE),
        per_core_N=math.ceil(n / (TILE_SIZE * num_cores)),
        fused_activation=None,
    )


def matmul_1d_config(
    m,
    k,
    n,
    grid=ttnn.CoreGrid(x=8, y=8),
    act=None,
    is_fp32_accumulate=False,
    overwrite_per_core_k=None,
    overwrite_subblock_w=None,
    overwrite_subblock_h=None,
):
    """Generate 1D matmul program config."""
    tile_width = 32
    tile_height = 32

    if n // tile_width // grid.num_cores < 1:
        # use less number of cores in case we have more N num tiles than cores
        grid_y = n // tile_width // grid.x
        grid = ttnn.CoreGrid(x=grid.x, y=grid_y)

    per_core_m = m // tile_height
    per_core_k = find_largest_divisor(k // (TILE_SIZE * grid.num_cores))
    per_core_n = math.ceil(n / tile_width / grid.num_cores)

    if is_fp32_accumulate:
        max_subblock_w_h = 4
    else:
        max_subblock_w_h = 8

    # find the largest value between 1 and 8 that is a factor of per_core_n
    # e.g. if per_core_n is 14, then out_subblock_w = 7
    out_subblock_w = max([i for i in range(1, max_subblock_w_h + 1) if per_core_n % i == 0])

    # find the largest value that is a factor of per_core_m such that
    # out_subblock_w * out_subblock_h <= 8
    out_subblock_h = max(
        [i for i in range(1, max_subblock_w_h + 1) if per_core_m % i == 0 and i * out_subblock_w <= max_subblock_w_h]
    )

    if overwrite_per_core_k is not None:
        per_core_k = overwrite_per_core_k

    if overwrite_subblock_w is not None:
        out_subblock_w = overwrite_subblock_w

    if overwrite_subblock_h is not None:
        out_subblock_h = overwrite_subblock_h

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=per_core_k,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )


def matmul_1d_config_from_tensor_shapes(
    in0_shape,
    in1_shape,
    grid=ttnn.CoreGrid(x=8, y=8),
    act=None,
    is_fp32_accumulate=False,
    overwrite_subblock_w=None,
    overwrite_subblock_h=None,
):
    """Generate 1D matmul program config from tensor shapes."""
    m, k, n = in0_shape[0] * in0_shape[1] * in0_shape[2], in0_shape[3], in1_shape[3]
    return matmul_1d_config(
        m,
        k,
        n,
        grid,
        act,
        is_fp32_accumulate,
        overwrite_subblock_w=overwrite_subblock_w,
        overwrite_subblock_h=overwrite_subblock_h,
    )


def create_sharded_norm_config(grid, dim, tile_padded_batch_rows):
    """Helper function to create LayerNormShardedMultiCoreProgramConfig for RMS NORM.

    Args:
        grid (ttnn.CoreGrid): Grid specification for the norm operation
        dim (int): Model dimension
        tile_padded_batch_rows (int): Padded batch size to tile size
    """
    block_w = dim // grid.num_cores // TILE_SIZE
    # Find largest value <= 4 that evenly divides block_w
    subblock_w = 4
    while subblock_w > 0:
        if block_w % subblock_w == 0:
            break
        subblock_w -= 1
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[grid.x, grid.y],
        subblock_w=subblock_w,
        block_h=tile_padded_batch_rows // TILE_SIZE,
        block_w=block_w,
        inplace=False,
    )


def dram_sharded_weight_config(k, n, dram_grid_size):
    """Create DRAM-sharded memory config for width-sharded tensors"""
    dram_cores = dram_grid_size.x  # WH has 12 dram cores, P150 has 8, P100 has 7
    assert dram_grid_size.y == 1, "Current dram sharding assumes y dim is 1"
    dram_weight_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
            )
        }
    )
    padding_multiple = TILE_SIZE * dram_cores
    padded_size = math.ceil(n / padding_multiple) * padding_multiple
    shard_spec = ttnn.ShardSpec(dram_weight_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def matmul_config(
    m: int,
    k: int,
    n: int,
    grid_size: tuple[int, int],
    in0_block_w: int = None,
    fuse_batch: bool = False,
    fused_activation=None,
    per_core_M=None,
    per_core_N=None,
):
    if per_core_M is None:
        per_core_M = math.ceil(m / (TILE_SIZE * grid_size[1]))
    if per_core_N is None:
        per_core_N = math.ceil(n / (TILE_SIZE * grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = get_out_subblock_w(per_core_N, out_subblock_h)  # TODO: Needed for TG hang workaround

    in0_block_w = find_largest_divisor(k // (TILE_SIZE * grid_size[1])) if not in0_block_w else in0_block_w

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=fuse_batch,
    )


def dram_shard_core_grid_for_k_and_n(k: int, n: int) -> ttnn.CoreGrid:
    rows, cols = find_grid_k_n(k // TILE_SIZE, n // TILE_SIZE)
    return ttnn.CoreGrid(x=cols, y=rows)


def find_grid_k_n(K, N):
    """
    Find the number of rows and columns for a grid of cores such that
    the total number of tiles N can be evenly divided among the cores.
    Each core will have the same integer number of tiles.

    Parameters:
        N (int): Total number of tiles to be distributed.

    Returns:
        tuple: A tuple (rows, cols) representing the grid dimensions.

    Raises:
        AssertionError: If it's not possible to find such a grid configuration.
    """
    max_rows = 8
    max_cols = 8  # Maximum number of rows or columns
    max_cores = max_rows * max_cols  # Maximum number of cores

    # Find all possible numbers of cores that divide N and are less than or equal to max_cores
    possible_cores = [c for c in range(1, max_cores + 1) if K % c == 0 and N % c == 0]
    possible_cores.sort(reverse=True)  # Start checking from the largest number of cores

    for cores in possible_cores:
        # Try to find a grid configuration with the current number of cores
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols

    # If no configuration is found, assert an error
    raise AssertionError(
        f"Cannot find a grid configuration such that both {K} and {N} tiles evenly divide into cores of max size {max_rows}x{max_cols}."
    )


def base_model_name(hf_config):
    """Get the base model name from the HuggingFace config."""
    model_name = hf_config.name_or_path.split("/")[-1]
    return model_name.split("B-")[0] + "B" if "B-" in model_name else model_name


def dequantize(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: Sequence[int] | None = None) -> torch.Tensor:
    """Dequantize a pytorch tensor using the provided scale."""
    assert tensor.ndim == inv_scale.ndim and tensor.dtype == torch.float8_e4m3fn and inv_scale.dtype == torch.float32
    if block_shape is not None:
        assert len(block_shape) == tensor.ndim and all(
            inv_scale.shape[i] * block_shape[i] == tensor.shape[i] for i in range(tensor.ndim)
        )
    else:
        assert all(tensor.shape[i] % inv_scale.shape[i] == 0 for i in range(tensor.ndim))
    for i in range(inv_scale.ndim):
        inv_scale = inv_scale.repeat_interleave(tensor.shape[i] // inv_scale.shape[i], dim=i)
    return tensor.bfloat16() * inv_scale.bfloat16()


def sub_state_dict(state_dict, prefix):
    """Get a subset of the state dict with a given prefix."""
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def save_and_get_path(path, tensor):
    """Save a tensor to a file and return the path."""
    ttnn.dump_tensor(path, tensor)
    ttnn.deallocate(tensor)
    return str(path)


def round_to_nearest_tile_size(value):
    """Round a value to the nearest multiple of TILE_SIZE."""
    return math.ceil(value / TILE_SIZE) * TILE_SIZE

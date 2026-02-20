# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import itertools
import json
import math
import os
from itertools import takewhile
from pathlib import Path
from types import NoneType
from typing import Any, Sequence

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import SavedWeight
from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict

# Constants
NORM_CATEGORIES = {"attention_norm", "mlp_norm", "q_norm", "k_norm"}
USERS_PER_ROW = 32
SEQ_LEN_CHUNK_SIZE = 1024  # NOTE: should be 512 for blackhole (in case of future bring-up)
TOPK_MIN_WIDTH = 64  # Minimum width of the topk input tensor


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


# Helper math functions
def even_int_div(a: int, b: int) -> int:
    """Integer division that raises an error if b does not divide a without a remainder."""
    assert a % b == 0
    return a // b


def find_all_divisors(n):
    """Generates all the divisors of n, from smallest to largest."""
    complementary_divisors = []  # This is all the divisors larger than sqrt(n)
    for divisor in range(1, int(math.sqrt(n)) + 1):
        if n % divisor == 0:
            complementary_divisors.append(n // divisor)
            yield divisor
    for complementary_divisor in reversed(complementary_divisors):
        yield complementary_divisor


def find_largest_divisor(n, max_divisor=8):
    """Find the largest divisor of n that is <= max_divisor."""
    return max(takewhile(lambda x: x <= max_divisor, find_all_divisors(n)), default=1)


# DRAM-sharded-matmul helper functions
def get_activation_sharding_core_counts_for_dram_matmul(activation_width: int, max_num_cores: int) -> set[int]:
    """Get the set of core counts on which the activation tensor can be sharded for DRAM matmul.
    Currently, the DRAM sharded matmul does not yet support padded activation shards. This means that,
    since the activation tensor is width sharded, the width dimension `activation_width` has to be divisible by
    the number of cores the activation is sharded over. This can however be different for input and output activations.
    """
    return set(
        takewhile(lambda x: x <= max_num_cores, find_all_divisors(ttnn.core.divup(activation_width, ttnn.TILE_SIZE)))
    )


def get_dram_sharded_matmul_config(m: int, k: int, n: int, input_num_shards: int, output_num_shards: int):
    # TODO: add documentation
    m_tiles = ttnn.core.divup(m, ttnn.TILE_SIZE)
    k_tiles = ttnn.core.divup(k, ttnn.TILE_SIZE)
    n_tiles = ttnn.core.divup(n, ttnn.TILE_SIZE)

    assert (
        k_tiles % input_num_shards == 0
    ), "The input tensor must evenly shard across input_num_shards (without padding)"
    assert (
        n_tiles % output_num_shards == 0
    ), "The output tensor must evenly shard across output_num_shards (without padding)"
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=find_largest_divisor(
            even_int_div(k_tiles, input_num_shards)
        ),  # in0_block_w has to divide k_tiles evenly
        per_core_M=m_tiles,
        per_core_N=even_int_div(n_tiles, output_num_shards),
        fused_activation=None,
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

    shard_spec = ttnn.ShardSpec(
        dram_weight_grid,
        (k, ttnn.core.roundup(ttnn.core.divup(n, dram_cores), ttnn.TILE_SIZE)),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


# Helper functions for other matmul configurations
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
        per_core_M = ttnn.core.divup(m, ttnn.TILE_SIZE * grid_size[1])
    if per_core_N is None:
        per_core_N = ttnn.core.divup(n, ttnn.TILE_SIZE * grid_size[0])

    out_subblock_h = 1
    out_subblock_w = get_out_subblock_w(per_core_N, out_subblock_h)  # TODO: Needed for TG hang workaround

    in0_block_w = find_largest_divisor(k // (ttnn.TILE_SIZE * grid_size[1])) if not in0_block_w else in0_block_w

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
    per_core_k = find_largest_divisor(k // (ttnn.TILE_SIZE * grid.num_cores))
    per_core_n = ttnn.core.divup(n, tile_width * grid.num_cores)

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
    padded_size = ttnn.core.roundup(n, ttnn.TILE_SIZE * dram_cores)
    shard_spec = ttnn.ShardSpec(
        get_dram_weight_grid(mesh_device), (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def dram_shard_core_grid_for_k(k: int) -> ttnn.CoreGrid:
    """Calculate core grid for DRAM sharding based on k dimension."""
    rows, cols = find_grid(k // ttnn.TILE_SIZE)
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
    rows, cols = find_grid_k_n(k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE)
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


def create_sharded_norm_config(grid, dim, tile_padded_batch_rows):
    """Helper function to create LayerNormShardedMultiCoreProgramConfig for RMS NORM.

    Args:
        grid (ttnn.CoreGrid): Grid specification for the norm operation
        dim (int): Model dimension
        tile_padded_batch_rows (int): Padded batch size to tile size
    """
    block_w = dim // grid.num_cores // ttnn.TILE_SIZE
    # Find largest value <= 4 that evenly divides block_w
    subblock_w = 4
    while subblock_w > 0:
        if block_w % subblock_w == 0:
            break
        subblock_w -= 1
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[grid.x, grid.y],
        subblock_w=subblock_w,
        block_h=tile_padded_batch_rows // ttnn.TILE_SIZE,
        block_w=block_w,
        inplace=False,
    )


def dram_shard_core_grid_for_k_and_n(k: int, n: int) -> ttnn.CoreGrid:
    rows, cols = find_grid_k_n(k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE)
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


def dequantize(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: Sequence[int]) -> torch.Tensor:
    """Dequantize a pytorch tensor using the provided scale."""
    assert tensor.ndim == inv_scale.ndim
    assert len(block_shape) == tensor.ndim and all(
        inv_scale.shape[i] * block_shape[i] >= tensor.shape[i] for i in range(tensor.ndim)
    )
    return dequantize_tensor(tensor, inv_scale, block_shape)


def get_state_dicts(
    dicts: Sequence[dict[str, torch.Tensor] | None],
    key: Any,
    shape: Sequence[int] | None = None,
    dtype: torch.dtype | None = None,
    concat_dim: int = 0,
    concat: bool = False,
) -> torch.Tensor:
    """Get a weight from a list of state dictionaries and combine them into a single tensor.

    Args:
        key (str): The key to look for in the dictionaries.
        dicts (Sequence[dict[str, torch.Tensor]]): A sequence of state dicts
        dim (int, optional): The dimension along which to combine the tensors. Defaults to 0.
        concat (bool, optional): Whether to concatenate the tensors or stack them. Defaults to False.
    Returns:
        torch.Tensor: The combined tensor.
    """
    if not dicts:
        return torch.empty()

    expected_shape = (
        next(map(lambda d: d[key].shape, filter(lambda d: d is not None, dicts)), None) if shape is None else shape
    )
    assert expected_shape is not None, "At least one dictionary must be non-empty, or a shape must be provided"

    expected_dtype = (
        next(map(lambda d: d[key].dtype, filter(lambda d: d is not None, dicts)), None) if dtype is None else dtype
    )
    assert expected_dtype is not None, "At least one dictionary must be non-empty, or a dtype must be provided"

    assert all(key in d for d in dicts if d is not None), f"Key {key} not found in all dictionaries"
    assert all(
        d[key].shape == expected_shape for d in dicts if d is not None
    ), f"Key {key} must have the value shaped as {expected_shape} in all dictionaries; instead got {[d[key].shape if d is not None else None for d in dicts]}"
    assert all(
        d[key].dtype == expected_dtype for d in dicts if d is not None
    ), f"Key {key} must have the dtype as {expected_dtype} in all dictionaries; instead got {[d[key].dtype if d is not None else None for d in dicts]}"

    tensors = [torch.zeros(expected_shape).to(dtype) if d is None else d[key] for d in dicts]

    if concat:
        return torch.concat(tensors, dim=concat_dim)
    return torch.stack(tensors, dim=concat_dim)


def sub_state_dict(state_dict: dict[str, torch.Tensor], prefix: str, num_layers: int | None = None):
    """Get a subset of the state dict with a given prefix."""
    # Preserve laziness when applicable by returning a LazyStateDict view.
    if isinstance(state_dict, LazyStateDict):
        return state_dict.view_with_prefix(prefix, num_layers)
    if num_layers is None:
        return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    else:
        return {
            k[len(prefix) :]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
            for layer_idx_str in ["".join(itertools.takewhile(str.isdigit, k.removeprefix("model.layers.")))]
            if not layer_idx_str or int(layer_idx_str) < num_layers
        }


def sub_state_dicts(
    state_dicts: Sequence[dict[str, torch.Tensor] | None], prefix: str
) -> tuple[dict[str, torch.Tensor] | None, ...]:
    """Get a subset of the state dict with a given prefix."""
    return tuple(None if d is None else sub_state_dict(d, prefix) for d in state_dicts)


TENSOR_CACHE_EXTENSION = ".tensorbin"

# Cache specs dumping for conversion optimization
_CACHE_SPECS_DUMP_ENV_VAR = "DEEPSEEK_V3_CACHE_SPECS_JSONL"
_CACHE_SPECS_DUMP_ENV_VAR_LEGACY = "DEEPSEEK_V3_DUMP_CACHE_SPECS"


def _enum_name_or_str(obj: Any) -> str | None:
    """Get the name of an enum or return the string representation."""
    if obj is None:
        return None
    if hasattr(obj, "name"):
        return obj.name
    return str(obj)


def _memory_config_to_dict(memory_config: ttnn.MemoryConfig | None) -> dict[str, Any] | None:
    """Convert a MemoryConfig to a dictionary for JSON serialization."""
    if memory_config is None:
        return None
    # Use the built-in to_json() method for proper serialization, then parse it
    # This handles CoreRangeSet and other complex types correctly
    try:
        return json.loads(memory_config.to_json())
    except (AttributeError, TypeError):
        # Fallback to manual conversion if to_json() is not available
        # This handles the case where grid might be a CoreRangeSet
        grid_dict = None
        if memory_config.shard_spec is not None and memory_config.shard_spec.grid is not None:
            grid = memory_config.shard_spec.grid
            # Handle CoreRangeSet - convert to list of ranges
            if hasattr(grid, "__iter__"):
                # It's a CoreRangeSet (iterable of CoreRange objects)
                grid_dict = [
                    {
                        "start": (core_range.start.x, core_range.start.y),
                        "end": (core_range.end.x, core_range.end.y),
                    }
                    for core_range in grid
                ]
            elif hasattr(grid, "start") and hasattr(grid, "end"):
                # It's a single CoreRange
                grid_dict = {
                    "start": (grid.start.x, grid.start.y),
                    "end": (grid.end.x, grid.end.y),
                }

        return {
            "memory_layout": _enum_name_or_str(memory_config.memory_layout),
            "buffer_type": _enum_name_or_str(memory_config.buffer_type),
            "shard_spec": (
                {
                    "grid": grid_dict,
                    "shape": list(memory_config.shard_spec.shape) if memory_config.shard_spec.shape else None,
                    "orientation": _enum_name_or_str(memory_config.shard_spec.orientation),
                }
                if memory_config.shard_spec is not None
                else None
            ),
        }


def _get_relative_cache_path(path: Path) -> str | None:
    """Extract the relative cache path from an absolute path."""
    if not path.is_absolute():
        return str(path)
    path_str = str(path)
    mesh_idx = path_str.find("mesh_")
    if mesh_idx == -1:
        return None
    parts = path_str[mesh_idx:].split("/", 1)
    if len(parts) < 2:
        return None
    return parts[1]


def _append_cache_specs_record(record: dict[str, Any]) -> None:
    """Append a cache specs record to the JSONL file specified by the environment variable."""
    dump_path_str = os.getenv(_CACHE_SPECS_DUMP_ENV_VAR) or os.getenv(_CACHE_SPECS_DUMP_ENV_VAR_LEGACY)
    if not dump_path_str:
        return

    dump_path = Path(dump_path_str)
    try:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(dump_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            fcntl_module = None
            try:
                import fcntl as fcntl_module
            except Exception:
                fcntl_module = None
            if fcntl_module is not None:
                try:
                    fcntl_module.flock(fd, fcntl_module.LOCK_EX)
                except Exception as e:
                    # Best-effort locking: ignore failures but log for diagnostics.
                    logger.debug(f"Failed to acquire file lock on {dump_path}: {e}")
            bytes_written = os.write(fd, data)
            if bytes_written != len(data):
                raise OSError(f"Short write while appending cache specs to {dump_path}")
            if fcntl_module is not None:
                try:
                    fcntl_module.flock(fd, fcntl_module.LOCK_UN)
                except Exception as e:
                    # Best-effort unlocking: ignore failures but log for diagnostics.
                    logger.debug(f"Failed to release file lock on {dump_path}: {e}")
        finally:
            os.close(fd)
    except Exception as e:
        logger.warning(f"Failed to append cache specs record to {dump_path}: {e}")


def shard_and_save(
    path: Path,
    tensor: torch.Tensor,
    shard_dims: tuple[int | None, int | None],
    mesh_device: ttnn.MeshDevice,
    remove_dims: tuple[bool, bool] | bool = False,
    *,
    dtype: ttnn.DataType | None = None,
    layout: ttnn.Layout | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    _torch_impl: bool = False,
) -> SavedWeight:
    """Shard a tensor and save it to a file."""
    assert all(isinstance(shard_dim, (int, NoneType)) for shard_dim in shard_dims)
    assert isinstance(remove_dims, bool) or all(isinstance(remove_dim, bool) for remove_dim in remove_dims)
    assert len(shard_dims) == 2, "shard_dims must be exactly 2 dimensions (can repeat)"

    if isinstance(remove_dims, bool):
        remove_dims = (remove_dims, remove_dims)

    assert (
        shard_dims[0] != shard_dims[1] or remove_dims[0] == remove_dims[1]
    ), "If sharding a single dim, both remove_dim values must be the same"

    for remove_dim, shard_dim, mesh_dim in zip(remove_dims, shard_dims, mesh_device.shape, strict=True):
        assert (
            shard_dim is None or tensor.shape[shard_dim] % mesh_dim == 0
        ), f"Cannot shard dimension {shard_dim} of size {tensor.shape[shard_dim]} into {mesh_dim} shards"
        assert not (remove_dim and shard_dim is None), f"Cannot remove unsharded dimension {shard_dim}"

    if shard_dims[0] == shard_dims[1] and shard_dims[0] is not None:
        assert remove_dims[0] == remove_dims[1], "If sharding a single dim, both remove_dim values must be the same"
        assert (
            tensor.shape[shard_dims[0]] % mesh_device.get_num_devices() == 0
        ), f"Cannot shard dimension {shard_dims[0]} of size {tensor.shape[shard_dims[0]]} into {mesh_device.get_num_devices()} shards"
        assert (
            not remove_dims[0] or tensor.shape[shard_dims[0]] == mesh_device.get_num_devices()
        ), f"The removed dim {shard_dims[0]} must be fully sharded"
    else:
        for remove_dim, shard_dim, mesh_dim in zip(remove_dims, shard_dims, mesh_device.shape, strict=True):
            assert (
                shard_dim is None or tensor.shape[shard_dim] % mesh_dim == 0
            ), f"Cannot shard dimension {shard_dim} of size {tensor.shape[shard_dim]} into {mesh_dim} shards"
            assert not (remove_dim and shard_dim is None), f"Cannot remove unsharded dimension {shard_dim}"
            assert (
                not remove_dim or tensor.shape[shard_dim] == mesh_dim
            ), f"The removed dim {shard_dim} must be fully sharded"

    if _torch_impl:
        ttnn_tensor = _shard_torch_impl(
            path=path,
            tensor=tensor,
            shard_dims=shard_dims,
            mesh_device=mesh_device,
            remove_dims=remove_dims,
            dtype=dtype,
            layout=layout,
            memory_config=memory_config,
        )
    else:
        ttnn_tensor = _shard_device_impl(
            path=path,
            tensor=tensor,
            shard_dims=shard_dims,
            mesh_device=mesh_device,
            remove_dims=remove_dims,
            dtype=dtype,
            layout=layout,
            memory_config=memory_config,
        )

    if not path.name.endswith(TENSOR_CACHE_EXTENSION):
        path = path.with_name(f"{path.name}{TENSOR_CACHE_EXTENSION}")

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.warning(f"Overwriting existing cache file: {path}")
    record = {
        "event": "deepseek_v3.cache_tensor_spec",
        "pid": os.getpid(),
        "cache_file_path": str(path),
        "cache_file_relpath": _get_relative_cache_path(path),
        "torch_shape": list(tensor.shape),
        "torch_dtype": str(tensor.dtype),
        "requested_dtype": _enum_name_or_str(dtype),
        "requested_layout": _enum_name_or_str(layout),
        "requested_memory_config": _memory_config_to_dict(memory_config),
        "shard_dims": list(shard_dims),
        "remove_dims": list(remove_dims),
        "mesh_shape": list(mesh_device.shape),
        "mesh_num_devices": mesh_device.get_num_devices(),
        "dtype_is_tilized": dtype in {ttnn.bfloat4_b, ttnn.bfloat8_b},
        "shard_device_impl_uses_dram_interleaved_workaround": memory_config == ttnn.DRAM_MEMORY_CONFIG,
        "torch_impl": _torch_impl,
        "status": "ok",
        "result_shape": list(ttnn_tensor.shape),
        "result_dtype": _enum_name_or_str(ttnn_tensor.dtype),
        "result_layout": _enum_name_or_str(ttnn_tensor.layout),
        "result_memory_config": _memory_config_to_dict(ttnn_tensor.memory_config()),
    }
    try:
        ttnn.dump_tensor(path, ttnn_tensor)
    except Exception as e:
        record["status"] = f"error({type(e).__name__}: {e})"
        _append_cache_specs_record(record)
        raise
    else:
        _append_cache_specs_record(record)

    # Always convert absolute paths to relative paths for portability
    # This ensures SavedWeight objects always have relative paths
    if path.is_absolute():
        path_str = str(path)
        mesh_idx = path_str.find("mesh_")
        if mesh_idx == -1:
            raise ValueError(f"Expected 'mesh_' in path: {path}")
        # Skip past "mesh_<rows>x<cols>/" to get relative path
        parts = path_str[mesh_idx:].split("/", 1)
        if len(parts) < 2:
            raise ValueError(f"Invalid path structure after 'mesh_': {path}")
        path = Path(parts[1])

    return SavedWeight(path, memory_config)


def _shard_device_impl(
    *,
    path: Path,
    tensor: torch.Tensor,
    shard_dims: tuple[int | None, int | None],
    mesh_device: ttnn.MeshDevice,
    remove_dims: tuple[bool, bool] | bool,
    dtype: ttnn.DataType | None,
    layout: ttnn.Layout | None,
    memory_config: ttnn.MemoryConfig | None,
) -> SavedWeight:
    assert layout in {
        None,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    }, "Device implementation only supports row-major and tiled layouts"

    dtype_is_tilized = dtype in {ttnn.bfloat4_b, ttnn.bfloat8_b}
    assert not (
        layout == ttnn.ROW_MAJOR_LAYOUT and dtype_is_tilized
    ), "Row-major layout is not supported for tilized dtypes"
    if dtype_is_tilized:
        layout = ttnn.TILE_LAYOUT

    if isinstance(remove_dims, bool):
        remove_dims = (remove_dims, remove_dims)

    if shard_dims[0] is None and shard_dims[1] is None:
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    if shard_dims[0] == shard_dims[1] and shard_dims[0] is not None:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=shard_dims[0])
    else:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=shard_dims)

    if memory_config != ttnn.DRAM_MEMORY_CONFIG:
        ttnn_tensor = ttnn.from_torch(
            tensor, layout=layout, memory_config=memory_config, mesh_mapper=mesh_mapper, device=mesh_device, dtype=dtype
        )
    else:
        ttnn_tensor = ttnn.from_torch(
            tensor,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        ttnn_tensor = ttnn.to_dtype(ttnn_tensor, dtype)
        ttnn_tensor = ttnn_tensor.to(layout)

    assert memory_config == ttnn_tensor.memory_config()
    assert dtype == ttnn_tensor.dtype
    assert layout == ttnn_tensor.layout

    new_tensor_shape = list(ttnn_tensor.shape)
    if shard_dims[0] == shard_dims[1]:
        if remove_dims[0]:
            new_tensor_shape.pop(shard_dims[0])
    else:
        if None not in shard_dims and shard_dims[0] > shard_dims[1]:
            shard_dims = (shard_dims[1], shard_dims[0])
            remove_dims = (remove_dims[1], remove_dims[0])
        if remove_dims[1]:
            new_tensor_shape.pop(shard_dims[1])
        if remove_dims[0]:
            new_tensor_shape.pop(shard_dims[0])

    new_tensor_shape = [1] * sum(remove_dims) + new_tensor_shape
    ttnn_tensor = ttnn_tensor.reshape(new_tensor_shape)

    return ttnn_tensor


def _shard_torch_impl(
    *,
    path: Path,
    tensor: torch.Tensor,
    shard_dims: tuple[int | None, ...],
    mesh_device: ttnn.MeshDevice,
    remove_dims: tuple[bool, ...],
    dtype: ttnn.DataType | None = None,
    layout: ttnn.Layout | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> SavedWeight:
    if shard_dims[0] == shard_dims[1]:
        assert remove_dims[0] == remove_dims[1], "If sharding a single dim, both remove_dim values must be the same"
        remove_dims = (remove_dims[0],)
        shard_dims = (shard_dims[0],)
        sharding_shape = (mesh_device.shape[0] * mesh_device.shape[1],)
    else:
        sharding_shape = (mesh_device.shape[0], mesh_device.shape[1])

    return ttnn.from_host_shards(
        [
            ttnn.from_torch(
                tensor[_get_shard_slices(tensor.shape, shard_dims, sharding_shape, shard_coords)][
                    _get_remove_dim_slices(tensor.shape, shard_dims, remove_dims)
                ],
                dtype=dtype,
                layout=layout,
                memory_config=memory_config,
            )
            for shard_coords in itertools.product(*map(range, sharding_shape))
        ],
        mesh_shape=mesh_device.shape,
    )


def _get_shard_slices(
    tensor_shape: Sequence[int],
    shard_dims: tuple[int | None, ...],
    mesh_shape: tuple[int, ...],
    device_coords: tuple[int, ...],
) -> tuple[slice, ...]:
    """Get the slices for a shard of a tensor given the sharding dimensions, mesh shape, and device coordinates."""
    slices = [slice(None) for _ in tensor_shape]
    for dim_idx, (device_coord, mesh_dim, shard_dim) in enumerate(
        zip(device_coords, mesh_shape, shard_dims, strict=True)
    ):
        assert 0 <= device_coord < mesh_dim, f"device_coords[{dim_idx}] out of range"
        if shard_dim is None:
            continue
        shard_size = even_int_div(tensor_shape[shard_dim], mesh_dim)
        slices[shard_dim] = slice(shard_size * device_coord, shard_size * (device_coord + 1))
    return tuple(slices)


def _get_remove_dim_slices(
    tensor_shape: Sequence[int],
    shard_dims: tuple[int | None, ...],
    remove_dims: tuple[bool, ...],
) -> tuple[slice | int, ...]:
    """Get the slices to remove the sharded dimensions if specified."""
    slices: list[slice | int] = [slice(None) for _ in tensor_shape]
    for shard_dim, remove_dim in zip(shard_dims, remove_dims, strict=True):
        if not remove_dim:
            continue
        assert shard_dim is not None
        slices[shard_dim] = 0
    return tuple(slices)


def get_mesh_coords(mesh_shape: list[int], row: int = None, col: int = None) -> list[ttnn.MeshCoordinate]:
    """Get mesh coordinates for a given mesh shape and optional row and column indices."""
    if row:
        assert 0 <= row < mesh_shape[0], "Row index out of bounds"
    if col:
        assert 0 <= col < mesh_shape[1], "Column index out of bounds"

    row_select = range(mesh_shape[0]) if row is None else [row]
    col_select = range(mesh_shape[1]) if col is None else [col]
    return [ttnn.MeshCoordinate(r, c) for r in row_select for c in col_select]

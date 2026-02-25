# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


# Parameters provided to the test vector generator are defined here.
parameters = {
    "nightly": {
        "prefill_spec": [
            # Various prefill configurations: [1, num_heads, seq_len, head_dim]
            {"input_shape": [1, 8, 32, 64], "cache_size": 128, "is_decode": False},
            {"input_shape": [1, 16, 64, 64], "cache_size": 256, "is_decode": False},
            {"input_shape": [1, 32, 128, 128], "cache_size": 512, "is_decode": False},
            {"input_shape": [1, 12, 256, 64], "cache_size": 512, "is_decode": False},
        ],
        "decode_spec": [
            # Various decode configurations: [1, batch, num_heads, head_dim]
            {"input_shape": [1, 8, 8, 64], "is_decode": True},
            {"input_shape": [1, 16, 16, 64], "is_decode": True},
            {"input_shape": [1, 32, 32, 128], "is_decode": True},
        ],
        "input_dtype": [ttnn.bfloat16],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b requires TILE_LAYOUT!"

    # Check which spec we're using
    if "prefill_spec" in test_vector and test_vector["prefill_spec"]:
        spec = test_vector["prefill_spec"]
        if spec["input_shape"][-1] % 64 != 0:
            return True, f"Input head_dim ({spec['input_shape'][-1]}) must be divisible by 64 for tiling"
    elif "decode_spec" in test_vector and test_vector["decode_spec"]:
        spec = test_vector["decode_spec"]
        if spec["input_shape"][-1] % 64 != 0:
            return True, f"Input head_dim ({spec['input_shape'][-1]}) must be divisible by 64 for tiling"

    return False, None


def rotate_half(x):
    """Rotates half the hidden dims of the input (HF style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_hf(x, cos, sin):
    """Golden function for HF-style rotary embedding."""
    return (x * cos) + (rotate_half(x) * sin)


def run(
    prefill_spec,
    decode_spec,
    input_dtype,
    input_layout,
    input_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    # Determine which mode we're testing
    if prefill_spec:
        spec = prefill_spec
        input_shape = spec["input_shape"]
        cache_size = spec["cache_size"]
        is_decode = False

        # Prefill: cos/sin shape is [1, 1, cache_size, head_dim]
        cos_sin_shape = [1, 1, cache_size, input_shape[-1]]
    else:
        spec = decode_spec
        input_shape = spec["input_shape"]
        is_decode = True

        # Decode: cos/sin shape is [1, batch, 1, head_dim]
        batch_size = input_shape[1]
        cos_sin_shape = [1, batch_size, 1, input_shape[-1]]

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape)
    torch_cos_cache_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_dtype
    )(cos_sin_shape)
    torch_sin_cache_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_dtype
    )(cos_sin_shape)

    if is_decode:
        # In decode mode, input is [1, batch, num_heads, head_dim]
        # cos/sin are [1, batch, 1, head_dim]
        # For golden, we need to broadcast cos/sin properly
        cos_expanded = torch_cos_cache_tensor.expand(-1, -1, input_shape[2], -1)
        sin_expanded = torch_sin_cache_tensor.expand(-1, -1, input_shape[2], -1)
        torch_output_tensor = apply_rotary_pos_emb_hf(torch_input_tensor, cos_expanded, sin_expanded)
    else:
        # In prefill mode, input is [1, num_heads, seq_len, head_dim]
        # cos/sin are [1, 1, cache_size, head_dim]
        # Extract the needed sequence length
        seq_len = input_shape[2]
        cos_sliced = torch_cos_cache_tensor[:, :, :seq_len, :]
        sin_sliced = torch_sin_cache_tensor[:, :, :seq_len, :]
        torch_output_tensor = apply_rotary_pos_emb_hf(torch_input_tensor, cos_sliced, sin_sliced)

    # For decode mode, we need HEIGHT_SHARDED memory config
    if is_decode:
        # Create sharded memory config
        batch = input_shape[1]
        num_heads = input_shape[2]
        head_dim = input_shape[3]

        # Each core handles one batch element
        # Input tensor: [1, batch, num_heads, head_dim] -> shard_height = num_heads * TILE_HEIGHT
        input_shard_height = num_heads * 32  # 32 is TILE_HEIGHT
        input_shard_width = head_dim

        # Cos/sin tensors: [1, batch, 1, head_dim] -> shard_height = 1 * TILE_HEIGHT
        cos_sin_shard_height = 32  # 32 is TILE_HEIGHT
        cos_sin_shard_width = head_dim

        num_cores = min(batch, device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y)
        core_grid = ttnn.CoreGrid(x=num_cores, y=1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})

        input_shard_spec = ttnn.ShardSpec(
            shard_grid, [input_shard_height, input_shard_width], ttnn.ShardOrientation.ROW_MAJOR, False
        )
        input_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
        )

        cos_sin_shard_spec = ttnn.ShardSpec(
            shard_grid, [cos_sin_shard_height, cos_sin_shard_width], ttnn.ShardOrientation.ROW_MAJOR, False
        )
        cos_sin_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, cos_sin_shard_spec
        )

        input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=input_dtype,
            layout=input_layout,
            device=device,
            memory_config=input_sharded_mem_config,
        )
        cos_cache_tensor = ttnn.from_torch(
            torch_cos_cache_tensor,
            dtype=input_dtype,
            layout=input_layout,
            device=device,
            memory_config=cos_sin_sharded_mem_config,
        )
        sin_cache_tensor = ttnn.from_torch(
            torch_sin_cache_tensor,
            dtype=input_dtype,
            layout=input_layout,
            device=device,
            memory_config=cos_sin_sharded_mem_config,
        )
    else:
        input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=input_dtype,
            layout=input_layout,
            device=device,
            memory_config=input_memory_config,
        )
        cos_cache_tensor = ttnn.from_torch(
            torch_cos_cache_tensor,
            dtype=input_dtype,
            layout=input_layout,
            device=device,
            memory_config=input_memory_config,
        )
        sin_cache_tensor = ttnn.from_torch(
            torch_sin_cache_tensor,
            dtype=input_dtype,
            layout=input_layout,
            device=device,
            memory_config=input_memory_config,
        )

    start_time = start_measuring_time()
    output_tensor = ttnn.experimental.rotary_embedding_hf(
        input_tensor, cos_cache_tensor, sin_cache_tensor, is_decode, memory_config=output_memory_config
    )
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]

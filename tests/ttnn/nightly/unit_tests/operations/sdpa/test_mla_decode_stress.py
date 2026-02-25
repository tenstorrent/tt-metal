# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
    run_flash_mla_decode_impl,
)


# When Q is pre-sharded and replicated across cores for MLA parallelization:
# - Original Q shape: (1, batch, num_heads, head_dim) e.g. (1, 4, 128, 576)
# - Each core gets a shard of (32, head_dim) = one tile row of heads
# - Q is replicated so each reducer group has its Q locally available
#
# To use this feature, set in SDPAProgramConfig:
#   q_locally_available=True
#   max_cores_per_head_batch=<cores per reducer group>
#
# The program factory will deduce:
#   - batch size from K/V tensor shape
#   - num_groups = total_shards / max_cores_per_head_batch
#   - q_heads_parallel_factor = num_groups / batch
#   - num_q_heads = q_heads_parallel_factor * shard_height


def create_replicated_q_shard_spec(device, batch, nh, d, num_cores_per_head=4):
    """
    Creates a memory config where Q is replicated within each reducer group.

    The core list follows the SDPA program factory's interleaved pattern:
    - Output cores (reducers) are at logical indices 0..num_output_cores-1
    - Worker cores start at index num_output_cores
    - For each virtual batch: [output_core, worker1, worker2, worker3]

    When this config is used, the program config should have:
        q_locally_available=True
        max_cores_per_head_batch=num_cores_per_head
    """
    q_heads_parallel_factor = (nh + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE  # ceil(nh / 32)
    num_virtual_batches = batch * q_heads_parallel_factor
    num_output_cores = num_virtual_batches

    # Total cores needed = num_virtual_batches * num_cores_per_head
    total_cores = num_virtual_batches * num_cores_per_head

    # Shard shape: each core gets (32, D) = (TILE_HEIGHT, head_dim)
    shard_height = ttnn.TILE_SIZE  # 32
    shard_width = d

    # Use device's actual grid size
    grid_x = device.compute_with_storage_grid_size().x
    grid_y = device.compute_with_storage_grid_size().y

    print(f"grid_x: {grid_x}, grid_y: {grid_y}, total_cores_needed: {total_cores}")

    if total_cores > grid_x * grid_y:
        raise ValueError(f"Not enough cores: need {total_cores}, have {grid_x * grid_y}")

    # Build core list in the exact SDPA order (matching program factory)
    tile_h = 4
    tile_w = 4

    num_tiles_y = grid_y // tile_h
    num_tiles_x = grid_x // tile_w

    core_list = []
    for tile_x in range(num_tiles_x):  # quadrant cols
        for tile_y in range(num_tiles_y):  # quadrant rows
            for local_y in range(tile_h):  # row inside quadrant
                for local_x in range(tile_w):  # col inside quadrant
                    x = tile_x * tile_w + local_x
                    y = tile_y * tile_h + local_y
                    core_list.append((x, y))
    print(f"core_list: {core_list[:total_cores]}")

    # Create core range set from the list (only use cores we need)
    core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in core_list[:total_cores]]
    )
    return ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=core_range_set,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize(
    "batch",
    [
        # 1,  # Single batch
        # 2,  # Multiple batches # Removing to reduce CI load
        4,  # Multiple batches # Removing to reduce CI load
        # 8,  # Even larger batch size
    ],
)
@pytest.mark.parametrize(
    "seq_len",
    [
        1 * 1024,  # Long sequence length
    ],
)
@pytest.mark.parametrize(
    "nh",
    [
        # 16,
        # 32,
        128,
    ],
)
@pytest.mark.parametrize(
    "nkv",
    [
        1,
        # 8, # Removing to reduce CI load
        # 16,
    ],
)
@pytest.mark.parametrize(
    "kv_lora_rank",
    [
        # 64,
        512,
    ],
)
@pytest.mark.parametrize(
    "d_rope",
    [
        # 0,
        # 32,
        64,
        # 128,
    ],
)
@pytest.mark.parametrize(
    "q_num_cores",
    [
        # 0,  # No sharding
        # 8,  # Shard across 8 cores
        64,  # Shard across all cores
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "q_custom_shard",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "use_paged_attention",
    [
        # False,
        True,
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        32,
        # 128,
    ],
)
def test_flash_mla_decode_stress(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_num_cores,
    q_dtype,
    q_custom_shard,
    dtype,
    use_paged_attention,
    block_size,
    function_level_defaults,
    reset_seeds,
):
    # If GQA (nkv > 1), then only batch shard
    # If nkv == 1, then batch * nh shard, unless q_num_cores is 0 (ie, in DRAM)
    num_sharding_cores = batch if nkv > 1 else max(batch, q_num_cores)
    if nh * (kv_lora_rank + d_rope) * nkv / num_sharding_cores >= 8 * 1024:  # found experimentally
        pytest.skip(
            f"Skipping test with large values, due to memory constraints. Got {nh=}, {kv_lora_rank=}, {nkv=}, {batch=}, {q_num_cores=}, {d_rope=}"
        )

    if batch * nh < ttnn.TILE_SIZE and q_num_cores > 0:
        pytest.skip("Skipping test with small batch and nh with q_num_cores > 0.")

    effective_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    if nkv > 1 and nkv % (effective_num_cores / batch) != 0:
        pytest.skip(
            f"Skipping test with nkv {nkv} not divisible by effective_num_cores {effective_num_cores} / batch {batch}."
        )

    q_mem_config = (
        create_replicated_q_shard_spec(device, batch, nh, kv_lora_rank + d_rope, 4) if q_custom_shard else None
    )

    run_flash_mla_decode_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_num_cores,
        q_dtype,
        q_mem_config,
        dtype,
        use_paged_attention,
        block_size,
        True,
    )

    ttnn.ReadDeviceProfiler(device)

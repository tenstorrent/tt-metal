# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.common.utility_functions import is_wormhole_b0
import ttnn
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
    run_flash_mla_decode_impl,
)


def create_replicated_q_shard_spec(device, batch, nh, d, num_cores_per_head=4):
    q_heads_parallel_factor = (nh + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    num_virtual_batches = batch * q_heads_parallel_factor
    total_cores = num_virtual_batches * num_cores_per_head

    shard_height = ttnn.TILE_SIZE
    shard_width = d

    grid_x = device.compute_with_storage_grid_size().x
    grid_y = device.compute_with_storage_grid_size().y

    if total_cores > grid_x * grid_y:
        raise ValueError(f"Not enough cores: need {total_cores}, have {grid_x * grid_y}")

    tile_h = 4
    tile_w = 4
    num_tiles_y = grid_y // tile_h
    num_tiles_x = grid_x // tile_w

    core_list = []
    for tile_x in range(num_tiles_x):
        for tile_y in range(num_tiles_y):
            for local_y in range(tile_h):
                for local_x in range(tile_w):
                    x = tile_x * tile_w + local_x
                    y = tile_y * tile_h + local_y
                    core_list.append((x, y))

    core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in core_list[:total_cores]]
    )
    return ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=core_range_set,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("seq_len", [1 * 1024])
@pytest.mark.parametrize("nh", [128])
@pytest.mark.parametrize("nkv", [1])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("d_rope", [64])
@pytest.mark.parametrize("q_num_cores", [64])
@pytest.mark.parametrize("q_dtype, dtype", [(ttnn.bfloat16, ttnn.bfloat8_b)])
@pytest.mark.parametrize("q_custom_shard", [False])
@pytest.mark.parametrize("reuse_k", [True])
@pytest.mark.parametrize("use_paged_attention", [True])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("max_cores_per_head_batch", [4])
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
    reuse_k,
    max_cores_per_head_batch,
    function_level_defaults,
    reset_seeds,
):
    num_sharding_cores = batch if nkv > 1 else max(batch, q_num_cores)
    if nh * (kv_lora_rank + d_rope) * nkv / num_sharding_cores >= 8 * 1024:
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
        create_replicated_q_shard_spec(device, batch, nh, kv_lora_rank + d_rope, 4)
        if q_custom_shard and is_wormhole_b0()
        else None
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
        reuse_k,
        max_cores_per_head_batch,
    )

    ttnn.ReadDeviceProfiler(device)

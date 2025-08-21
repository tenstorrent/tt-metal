# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from tests.tt_eager.python_api_testing.unit_testing.misc.test_flash_multi_latent_attention_decode import (
    run_flash_mla_decode_impl,
)


@pytest.mark.parametrize(
    "batch",
    [
        1,  # Single batch
        # 2,  # Multiple batches # Removing to reduce CI load
        8,  # Even larger batch size
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
        16,
        32,
        128,
    ],
)
@pytest.mark.parametrize(
    "nkv",
    [
        1,
        # 8, # Removing to reduce CI load
        16,
    ],
)
@pytest.mark.parametrize(
    "kv_lora_rank",
    [
        64,
        512,
    ],
)
@pytest.mark.parametrize(
    "d_rope",
    [
        0,
        32,
        128,
    ],
)
@pytest.mark.parametrize(
    "q_num_cores",
    [
        0,  # No sharding
        8,  # Shard across 8 cores
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
    "use_paged_attention",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        32,
        128,
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
        dtype,
        use_paged_attention,
        block_size,
    )

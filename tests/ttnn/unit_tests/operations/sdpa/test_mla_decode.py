# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import run_flash_mla_decode_impl


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope, q_num_cores, q_mem_config",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope, number of cores to shard q on
    [
        (4, 1024, 128, 1, 512, 64, 64, None),  # DeepSeek V3 TG full DP
        (2, 1024, 8, 1, 128, 64, 0, None),  # small config, DRAM Q
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
        True,
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        64,
    ],
)
@pytest.mark.parametrize(
    "reuse_k",
    [
        True,
    ],
)
def test_flash_mla_decode(
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
    function_level_defaults,
    reset_seeds,
):
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
        max_cores_per_head_batch=4,
    )


# k_chunk_size=0 takes the dynamic-chunk path, whose QK matmul used to produce only the first tile-row of Q
# heads. With more than 32 heads on a core (64 or 128 with DRAM Q: 2 or 4 tile-rows) the next step waited
# forever. nh=8 fits in one tile-row and is the control. kv_lora_rank=128 keeps the CBs within L1.
@pytest.mark.parametrize("nh, kv_lora_rank", [(8, 512), (64, 128), (128, 128)])
def test_flash_mla_decode_dynamic_chunk(device, nh, kv_lora_rank, function_level_defaults, reset_seeds):
    run_flash_mla_decode_impl(
        device,
        batch=4,
        seq_len=1024,
        nh=nh,
        nkv=1,
        kv_lora_rank=kv_lora_rank,
        d_rope=64,
        q_num_cores=0,
        q_dtype=ttnn.bfloat16,
        q_mem_config=None,
        dtype=ttnn.bfloat8_b,
        use_paged_attention=True,
        block_size=64,
        reuse_k=True,
        max_cores_per_head_batch=4,
        k_chunk_size=0,
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import run_flash_mla_decode_impl


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope, q_num_cores",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope, number of cores to shard q on
    [
        (4, 1024, 128, 1, 512, 64, 64),  # DeepSeek V3 TG full DP
        (2, 1024, 8, 1, 128, 64, 0),  # small config, DRAM Q
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
        dtype,
        use_paged_attention,
        block_size,
        reuse_k,
    )

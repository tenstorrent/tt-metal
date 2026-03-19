# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import run_flash_mla_prefill_impl


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope
    [
        (2, 1024, 128, 1, 512, 64),  # large head count
        (2, 4096, 64, 1, 256, 0),  # zero d_rope
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype, use_paged_attention, block_size",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b, True, 128),
        (ttnn.bfloat8_b, ttnn.bfloat4_b, True, 32),
    ],
)
def test_flash_mla_prefill(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    use_paged_attention,
    block_size,
    function_level_defaults,
    reset_seeds,
):
    # Skip specific failing combination with LLK asserts
    if (
        batch == 2
        and seq_len == 4096
        and nh == 64
        and nkv == 1
        and kv_lora_rank == 256
        and d_rope == 0
        and is_llk_assert_enabled()
    ):
        pytest.skip("Hits LLK_ASSERT for num_faces check, due to kernel being large and num_faces get ? as value.")

    run_flash_mla_prefill_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
        use_paged_attention,
        block_size,
    )

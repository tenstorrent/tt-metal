# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
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


def test_chunked_flash_mla_prefill_partial_single_page_paged_cache(
    device,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_prefill_impl(
        device,
        batch=1,
        seq_len=64,
        nh=32,
        nkv=1,
        kv_lora_rank=64,
        d_rope=64,
        q_dtype=ttnn.bfloat16,
        dtype=ttnn.bfloat8_b,
        use_paged_attention=True,
        block_size=64,
    )


# The head_dim_v <= K head dim check used to print Q's head dim: "512 and 576" here, instead of K's 256.
def test_flash_mla_prefill_head_dim_v_wider_than_k(device, expect_error):
    def to_device(shape):
        return ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    tt_q = to_device([1, 8, 128, 576])
    tt_k = to_device([1, 1, 128, 256])
    with expect_error(RuntimeError, "head dim of K, got 512 and 256"):
        ttnn.transformer.flash_mla_prefill(tt_q, tt_k, head_dim_v=512, is_causal=True)

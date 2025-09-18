# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from tests.tt_eager.python_api_testing.unit_testing.misc.test_flash_multi_latent_attention_prefill import (
    run_flash_mla_prefill_impl,
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
def test_flash_mla_prefill_stress(
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

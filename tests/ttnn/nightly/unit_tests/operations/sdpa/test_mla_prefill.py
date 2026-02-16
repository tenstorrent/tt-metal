# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
    run_flash_mla_prefill_impl,
)


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope
    [
        (2, 1024, 128, 1, 512, 64),
        (2, 8 * 1024, 8, 1, 128, 64),
        (2, 4 * 1024, 64, 1, 256, 0),
        (2, 4 * 1024, 64, 1, 32, 64),
        (8, 4 * 1024, 8, 1, 128, 32),
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat8_b, ttnn.bfloat4_b),
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
        128,
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

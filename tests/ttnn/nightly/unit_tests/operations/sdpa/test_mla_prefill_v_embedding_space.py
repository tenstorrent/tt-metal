# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.sdpa.test_mla_prefill_v_embedding_space import run_flash_mla_prefill_impl


@pytest.mark.parametrize(
    "batch, seq_len_q, seq_len_kv, nh, nkv, kv_lora_rank, d_rope, v_head_dim, q_chunk_first, v_chunk_first, q_chunk_second, v_chunk_second",
    [
        # cases from deepseek
        # TP=8, SP=1, seq_len = 1k
        (1, 1024, 1024, 16, 16, 512, 64, 128, 32, 32, 32, 32),
        # TP=8, SP=1, seq_len = 4k
        (1, 4096, 4096, 16, 16, 512, 64, 128, 32, 32, 32, 32),
        # TP=4, SP=1, seq_len = 1k
        (1, 1024, 1024, 32, 32, 512, 64, 128, 32, 32, 32, 32),
        # TP=4, SP=1, seq_len = 4k
        (1, 4096, 4096, 32, 32, 512, 64, 128, 32, 32, 32, 32),
        # TP=4, SP=32, seq_len = 16k
        (1, 512, 16384, 32, 32, 512, 64, 128, 32, 32, 32, 32),
        # TP=4, SP=32, seq_len = 32k
        (1, 1024, 32768, 32, 32, 512, 64, 128, 32, 32, 32, 32),
        # TP=4, SP=32, seq_len = 128k
        (1, 4096, 131072, 32, 32, 512, 64, 128, 128, 256, 128, 512),
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_flash_mla_prefill(
    device,
    batch,
    seq_len_q,
    seq_len_kv,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    v_head_dim,
    q_chunk_first,
    v_chunk_first,
    q_chunk_second,
    v_chunk_second,
    function_level_defaults,
    reset_seeds,
):
    if device.arch() == ttnn.device.Arch.WORMHOLE_B0:
        if seq_len_kv == 131072:
            pytest.skip("Skip WH test due to pcc failure. Should properly set k/v chunk sizes for this case.")
        q_chunk_first = 32
        v_chunk_first = 32
        q_chunk_second = 32
        v_chunk_second = 32
        logger.info("Using wormhole device, setting chunk sizes to 32")

    run_flash_mla_prefill_impl(
        device,
        batch,
        seq_len_q,
        seq_len_kv,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
        v_head_dim,
        q_chunk_first,
        v_chunk_first,
        q_chunk_second,
        v_chunk_second,
    )

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Repro: ``ttnn.transformer.paged_scaled_dot_product_attention_decode`` runs out
of per-core L1 under its default schedule when ``head_dim >= 256``.

Shape below is the Gemma-4 31B global-attention shape at max_model_len=2048
(head_dim=512). Llama-class shapes (head_dim=128) fit the default schedule.
"""

import random

import numpy as np
import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    run_test_sdpa_decode_paged_attention,
)


@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)
    yield


@pytest.mark.parametrize("kv_dtype", [ttnn.bfloat8_b], ids=["kvb8"])
@pytest.mark.parametrize("q_dtype", [ttnn.bfloat16], ids=["qb16"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, cur_pos_tensor, sliding_window_size",
    # Gemma-4 31B global attention: b=1, num_heads=32, head_dim=512, max_model_len=1024.
    ([1, 8, 1, 1024, 512, (8, 8), True, None],),  # (11, 10) bh also same issue
    ids=["gemma4-31b-global-ml1024"],
)
@pytest.mark.parametrize("block_size", (32,), ids=["paged_32"])
def test_paged_sdpa_decode_l1_overflow(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    kv_dtype,
    grid_size,
    q_dtype,
    cur_pos_tensor,
    sliding_window_size,
    block_size,
    reset_seeds,
):
    run_test_sdpa_decode_paged_attention(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        kv_dtype,
        grid_size,
        q_dtype,
        cur_pos_tensor,
        block_size=block_size,
        sharded_in=True,
        sharded_out=False,
        sliding_window_size=sliding_window_size,
    )

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    skip_for_grayskull,
    get_devices_for_t3000,
    nearest_32,
)


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )


def run_test_concat_head(devices, n_local_heads, padded_local_heads, head_dim, batch):
    ## Split Heads
    padded_batch = nearest_32(batch)
    seq_len = 1

    # Prepare input
    concat_head_input = torch.rand(1, batch, padded_local_heads, head_dim)

    shard_grid = ttnn.CoreRangeSet({num_to_corerange(batch)})
    SCORES_BATCHED_MM_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_grid,
            [
                padded_local_heads,  # Each core has padded_local_heads
                head_dim,  # head dim
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    WIDTH_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)

    # Prepare tt input
    concat_head_input_tt = torch2tt_tensor(concat_head_input, tt_device=None).to(
        device=devices[0], mem_config=SCORES_BATCHED_MM_OUTPUT_MEMCFG
    )

    concat_head_output = ttnn.experimental.nlp_concat_heads_decode(
        concat_head_input_tt,
        num_heads=n_local_heads,
    )  # seqlen, 1, batch, hidden_size

    logger.info(f"concat_head_output: {concat_head_output.memory_config()}")

    # Input: (1, 32, 32(8), 128)
    # Output: (1, 1, 32, 1024)
    concat_head_output_torch = concat_head_input[:, :, :n_local_heads].reshape(1, 1, batch, head_dim * n_local_heads)

    # compare
    concat_head_output_tt_cpu = tt2torch_tensor(concat_head_output)
    concat_head_output_tt_unpadded = concat_head_output_tt_cpu[:, :, :batch, :]
    out_pass_q, output_pcc_q = comp_pcc(concat_head_output_tt_unpadded, concat_head_output_torch)
    logger.info(f"PCC value: {output_pcc_q}")
    assert out_pass_q


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "n_local_heads, padded_local_heads, head_dim, batch_size",
    ((8, 32, 128, 32), (17, 32, 96, 32), (32, 32, 64, 32), (8, 32, 128, 16)),
)
def test_concat_head(
    n_local_heads,
    padded_local_heads,
    head_dim,
    batch_size,
    all_devices,
    use_program_cache,
):
    devices = get_devices_for_t3000(all_devices, num_devices=1)
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_concat_head(devices, n_local_heads, padded_local_heads, head_dim, batch_size)

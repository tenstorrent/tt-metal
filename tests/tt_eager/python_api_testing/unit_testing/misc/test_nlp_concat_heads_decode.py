# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000


def run_test_concat_head(
    devices,
    n_local_heads,
    padded_local_heads,
    head_dim,
):
    ## Split Heads
    batch = 32
    seq_len = 1

    # Prepare input
    concat_head_input = torch.rand(1, batch, padded_local_heads, head_dim)

    shard_spec_32_cores_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(7, 3),
            ),
        }
    )
    SCORES_BATCHED_MM_OUTPUT_MEMCFG = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_32_cores_grid,
            [
                batch,  # Each core has padded_local_heads
                head_dim,  # head dim
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    WIDTH_SHARDED_MEMCFG = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, ttl.tensor.BufferType.L1
    )

    # Prepare tt input
    concat_head_input_tt = torch2tt_tensor(concat_head_input, tt_device=None).to(
        device=devices[0], mem_config=SCORES_BATCHED_MM_OUTPUT_MEMCFG
    )

    concat_head_output = ttl.tensor.nlp_concat_heads_decode(
        concat_head_input_tt,
        num_heads=n_local_heads,
    )  # seqlen, 1, batch, hidden_size

    logger.info(f"concat_head_output: {concat_head_output.memory_config()}")

    # Input: (1, 32, 32(8), 128)
    # Output: (1, 1, 32, 1024)
    concat_head_output_torch = concat_head_input[:, :, :n_local_heads].reshape(1, 1, batch, head_dim * n_local_heads)

    # compare
    concat_head_output_tt_cpu = tt2torch_tensor(concat_head_output)
    out_pass_q, output_pcc_q = comp_pcc(concat_head_output_tt_cpu, concat_head_output_torch)
    logger.info(f"PCC value: {output_pcc_q}")
    assert out_pass_q


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "n_local_heads, padded_local_heads, head_dim",
    ((8, 32, 128), (17, 32, 96), (32, 32, 64)),
)
def test_concat_head(
    n_local_heads,
    padded_local_heads,
    head_dim,
    all_devices,
    use_program_cache,
):
    devices = get_devices_for_t3000(all_devices, num_devices=1)
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_concat_head(
            devices,
            n_local_heads,
            padded_local_heads,
            head_dim,
        )

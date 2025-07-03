# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import get_devices_for_t3000, skip_for_grayskull, torch2tt_tensor, tt2torch_tensor
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_test_concat_head1(
    devices,
    batch,
    seq_len,
):
    ## Split Heads
    n_local_heads = 8
    n_local_kv_heads = 1
    head_dim = 128
    # Prepare input
    concat_head_input = torch.rand(1, n_local_heads, batch, head_dim)

    shard_spec_8_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 0),
            ),
        }
    )

    SCORES_TRANSPOSED_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_8_cores_grid,  # Volume must match # of attn heads
            [
                32,  # Each core has 32 users
                head_dim,  # head dim
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    WIDTH_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)

    # Prepare tt input
    concat_head_input_tt = torch2tt_tensor(concat_head_input, tt_device=None).to(
        device=devices[0], mem_config=SCORES_TRANSPOSED_OUTPUT_MEMCFG
    )

    concat_head_output = ttnn.experimental.nlp_concat_heads(
        concat_head_input_tt, memory_config=WIDTH_SHARDED_MEMCFG
    )  # seqlen, 1, batch, hidden_size

    logger.info(f"concat_head_output: {concat_head_output.memory_config()}")

    concat_head_output_torch = concat_head_input.permute(0, 2, 1, 3).reshape(1, 1, batch, head_dim * 8)

    # compare
    concat_head_output_tt_cpu = tt2torch_tensor(concat_head_output)
    out_pass_q, output_pcc_q = comp_pcc(concat_head_output_tt_cpu, concat_head_output_torch)
    logger.info(f"PCC value: {output_pcc_q}")
    assert out_pass_q


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1),),
)
def test_concat_head1(
    batch,
    seq_len,
    all_devices,
):
    n_devices = 8
    devices = get_devices_for_t3000(all_devices, num_devices=1)
    torch.manual_seed(0)

    run_test_concat_head1(
        devices,
        batch,
        seq_len,
    )


def run_test_concat_head2(
    devices,
    batch,
    seq_len,
):
    ## Split Heads
    n_local_heads = 8
    padded_local_heads = 32
    head_dim = 128
    # Prepare input
    concat_head_input = torch.rand(1, batch, padded_local_heads, head_dim)

    shard_spec_32_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 3),
            ),
        }
    )
    SCORES_BATCHED_MM_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                batch,  # Each core has 32 users
                head_dim,  # head dim
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
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
    out_pass_q, output_pcc_q = comp_pcc(concat_head_output_tt_cpu, concat_head_output_torch)
    logger.info(f"PCC value: {output_pcc_q}")
    assert out_pass_q


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1),),
)
def test_concat_head2(
    batch,
    seq_len,
    all_devices,
):
    n_devices = 8
    devices = get_devices_for_t3000(all_devices, num_devices=1)
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_concat_head2(
            devices,
            batch,
            seq_len,
        )

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc, tt2torch_tensor
import torch
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0


def run_nlp_concat_heads_test(
    batch, seq_len, dtype, in0_mem_config, out_mem_config, device
):
    torch.manual_seed(1234)

    num_heads = 71
    head_dim = 64
    in0_shape = [batch, num_heads, seq_len, head_dim]

    A = torch.randn(in0_shape)

    in0_t = (
        ttl.tensor.Tensor(A, dtype)
        .to(ttl.tensor.Layout.TILE)
        .to(device, in0_mem_config)
    )

    out = ttl.tensor.nlp_concat_heads(in0_t, out_mem_config)

    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_storage == in0_mem_config.buffer_storage
    assert out.memory_config().buffer_storage == out_mem_config.buffer_storage
    logger.debug(f"in0: {in0_t.memory_config().buffer_storage} and {in0_t.dtype()}")
    logger.debug(f"out: {out.memory_config().buffer_storage} and {out.dtype()}")

    assert out.shape() == [batch, 1, seq_len, num_heads * head_dim]

    pyt_got_back_rm_out = tt2torch_tensor(out)

    ref_out = torch.transpose(A, -3, -2).reshape(
        [batch, 1, seq_len, num_heads * head_dim]
    )

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm_out, ref_out, pcc)
    logger.info(f"passing={passing_pcc}")
    logger.info(f"output pcc={output_pcc}")
    assert passing_pcc

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch, seq_len",
    ((1, 32), (1, 64), (1, 128)),
    ids=[
        "batch1_seq32",
        "batch1_seq64",
        "batch1_seq128",
    ],
)
def test_nlp_concat_heads_test(
    batch, seq_len, dtype, in0_mem_config, out_mem_config, request, device
):
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/nlp_concat_heads_tm_{request.node.callspec.id}"
    )
    run_nlp_concat_heads_test(
        batch, seq_len, dtype, in0_mem_config, out_mem_config, device
    )

@skip_for_wormhole_b0
def test_nlp_concat_heads_with_program_cache(use_program_cache, device):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    dram_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)
    for _ in range(2):
        run_nlp_concat_heads_test(
            1, 32, dtype, dram_mem_config, dram_mem_config, device
        )

    dram_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1)
    for _ in range(2):
        run_nlp_concat_heads_test(
            1, 32, dtype, dram_mem_config, dram_mem_config, device
        )

    assert ttl.program_cache.num_entries() == 2

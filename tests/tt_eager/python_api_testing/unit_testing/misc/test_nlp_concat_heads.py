# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc, tt2torch_tensor
import torch
from models.utility_functions import is_wormhole_b0


def run_nlp_concat_heads_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    num_heads = 71
    head_dim = 64
    in0_shape = [batch, num_heads, seq_len, head_dim]

    A = torch.randn(in0_shape)

    in0_t = ttl.tensor.Tensor(A, dtype).to(ttl.tensor.Layout.TILE).to(device, in0_mem_config)

    out = ttl.tensor.nlp_concat_heads(in0_t, out_mem_config)

    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert out.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {in0_t.memory_config().buffer_type} and {in0_t.get_dtype()}")
    logger.debug(f"out: {out.memory_config().buffer_type} and {out.get_dtype()}")

    assert list(out.get_legacy_shape()) == [batch, 1, seq_len, num_heads * head_dim]

    pyt_got_back_rm_out = tt2torch_tensor(out)

    ref_out = torch.transpose(A, -3, -2).reshape([batch, 1, seq_len, num_heads * head_dim])

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm_out, ref_out, pcc)
    logger.debug(f"passing={passing_pcc}")
    logger.debug(f"output pcc={output_pcc}")
    assert passing_pcc


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
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
def test_nlp_concat_heads_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, request, device):
    ttl.profiler.set_profiler_location(f"nlp_concat_heads_tm_{request.node.callspec.id}")
    run_nlp_concat_heads_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, device)


def test_nlp_concat_heads_with_program_cache(device, use_program_cache):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    for _ in range(2):
        run_nlp_concat_heads_test(1, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    for _ in range(2):
        run_nlp_concat_heads_test(1, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    assert device.num_program_cache_entries() == 2

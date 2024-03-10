# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger


import numpy as np

import tt_lib as ttl
from models.utility_functions import (
    comp_pcc,
)
import torch


def run_split_query_key_value_and_split_heads_test(device, batch, dtype, in0_mem_config, out_mem_config):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x * compute_grid_size.y < (12 * 9):
        logger.info(f"Grid size {compute_grid_size} is not supported")
        pytest.skip()

    torch.manual_seed(1234)

    a_shape = [batch, 1, 384, 3072]

    A = torch.randn(a_shape)

    a_t = (
        ttl.tensor.Tensor(
            A.flatten().tolist(),
            a_shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in0_mem_config)
    )

    q, k, v = ttl.operations.primary.transformers.split_query_key_value_and_split_heads(
        a_t, ttl.tensor.CoreCoord(12, 9), out_mem_config
    )

    # Check memory of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    assert k.memory_config().buffer_type == out_mem_config.buffer_type
    assert v.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {a_t.memory_config().buffer_type} and {a_t.get_dtype()}")
    logger.debug(f"q: {q.memory_config().buffer_type} and {q.get_dtype()}")
    logger.debug(f"k: {k.memory_config().buffer_type} and {k.get_dtype()}")
    logger.debug(f"v: {v.memory_config().buffer_type} and {v.get_dtype()}")

    assert q.get_legacy_shape() == [batch, 16, 384, 64]
    assert k.get_legacy_shape() == [batch, 16, 64, 384]
    assert v.get_legacy_shape() == [batch, 16, 384, 64]

    tt_host_rm_q = q.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_q = tt_host_rm_q.to_torch()
    tt_host_rm_k = k.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_k = tt_host_rm_k.to_torch()
    tt_host_rm_v = v.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_v = tt_host_rm_v.to_torch()

    (ref_q, ref_k, ref_v) = torch.split(A, 1024, dim=-1)

    ref_q = ref_q.reshape([batch, 384, 16, 64]).transpose(-3, -2)
    ref_k = ref_k.reshape([batch, 384, 16, 64]).transpose(-3, -2).transpose(-2, -1)
    ref_v = ref_v.reshape([batch, 384, 16, 64]).transpose(-3, -2)

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, 0.99)
    logger.debug(f"Q passing={passing_pcc_q}")
    logger.debug(f"Q output pcc={output_pcc_q}")
    assert passing_pcc_q
    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, 0.99)
    logger.debug(f"K passing={passing_pcc_k}")
    logger.debug(f"K output pcc={output_pcc_k}")
    assert passing_pcc_k
    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, 0.99)
    logger.debug(f"V passing={passing_pcc_v}")
    logger.debug(f"V output pcc={output_pcc_v}")
    assert passing_pcc_v


import pytest


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
    "batch",
    (9, 8, 7),
    ids=[
        "batch_9",
        "batch_8",
        "batch_7",
    ],
)
def test_split_query_key_value_and_split_heads(device, batch, dtype, in0_mem_config, out_mem_config, request):
    ttl.profiler.set_profiler_location(f"BERT_large_create_qvk_heads_tm_{request.node.callspec.id}")
    run_split_query_key_value_and_split_heads_test(device, batch, dtype, in0_mem_config, out_mem_config)


def test_split_query_key_value_and_split_heads_with_program_cache(device, use_program_cache):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    for _ in range(2):
        run_split_query_key_value_and_split_heads_test(device, 9, dtype, mem_config, mem_config)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    for _ in range(2):
        run_split_query_key_value_and_split_heads_test(device, 9, dtype, mem_config, mem_config)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    assert device.num_program_cache_entries() == 2

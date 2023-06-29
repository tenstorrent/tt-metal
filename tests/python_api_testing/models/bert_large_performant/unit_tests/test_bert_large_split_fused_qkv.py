from pathlib import Path
import sys
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

import numpy as np

from libs import tt_lib as ttl
from python_api_testing.models.utility_functions import (
    comp_pcc,
)
import torch


def run_bert_large_split_fused_qkv_test(dtype, in0_mem_config, out_mem_config):
    torch.manual_seed(1234)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device, ttl.device.MemoryAllocator.L1_BANKING)
    host = ttl.device.GetHost()
    a_shape = [9, 1, 384, 3072]

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

    q, k, v = ttl.tensor.bert_large_split_fused_qkv(a_t, out_mem_config)

    # Check memory of inputs and outputs
    assert a_t.buffer_type() == in0_mem_config.buffer_type
    assert q.buffer_type() == out_mem_config.buffer_type
    assert k.buffer_type() == out_mem_config.buffer_type
    assert v.buffer_type() == out_mem_config.buffer_type
    logger.debug(f"in0 is on: {a_t.buffer_type()}")
    logger.debug(f"q is on: {q.buffer_type()}")
    logger.debug(f"k is on: {k.buffer_type()}")
    logger.debug(f"v is on: {v.buffer_type()}")

    assert q.shape() == [9, 1, 384, 1024]
    assert k.shape() == [9, 1, 384, 1024]
    assert v.shape() == [9, 1, 384, 1024]

    tt_host_rm_q = q.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_q = torch.Tensor(tt_host_rm_q.data()).reshape(tt_host_rm_q.shape())
    tt_host_rm_k = k.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_k = torch.Tensor(tt_host_rm_k.data()).reshape(tt_host_rm_k.shape())
    tt_host_rm_v = v.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_v = torch.Tensor(tt_host_rm_v.data()).reshape(tt_host_rm_v.shape())

    (ref_q, ref_k, ref_v) = torch.split(A, 1024, dim=-1)

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, 0.99)
    logger.info(f"Q passing={passing_pcc_q}")
    logger.info(f"Q output pcc={output_pcc_q}")
    assert passing_pcc_q
    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, 0.99)
    logger.info(f"K passing={passing_pcc_k}")
    logger.info(f"K output pcc={output_pcc_k}")
    assert passing_pcc_k
    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, 0.99)
    logger.info(f"V passing={passing_pcc_v}")
    logger.info(f"V output pcc={output_pcc_v}")
    assert passing_pcc_v

    ttl.device.CloseDevice(device)


import pytest


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
def test_bert_large_split_fused_qkv_test(
    dtype, in0_mem_config, out_mem_config, request
):
    ttl.profiler.set_profiler_flag(False)
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/BERT_large_split_fused_qkv_tm_{request.node.callspec.id}"
    )
    run_bert_large_split_fused_qkv_test(dtype, in0_mem_config, out_mem_config)


def test_bert_large_split_fused_qkv_with_program_cache(use_program_cache):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    dram_mem_config = ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.DRAM)
    for _ in range(2):
        run_bert_large_split_fused_qkv_test(dtype, dram_mem_config, dram_mem_config)

    dram_mem_config = ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.L1)
    for _ in range(2):
        run_bert_large_split_fused_qkv_test(dtype, dram_mem_config, dram_mem_config)

    assert ttl.program_cache.num_entries() == 2

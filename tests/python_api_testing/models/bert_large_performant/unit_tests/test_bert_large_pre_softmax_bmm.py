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


def run_bert_large_pre_softmax_bmm_test(
    dtype, in0_mem_config, in1_mem_config, out_mem_config
):
    torch.manual_seed(1234)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device, ttl.device.MemoryAllocator.L1_BANKING)
    host = ttl.device.GetHost()
    a_shape = [9, 16, 384, 64]
    b_shape = [9, 16, 64, 384]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

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
    b_t = (
        ttl.tensor.Tensor(
            B.flatten().tolist(),
            b_shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in1_mem_config)
    )

    t2 = ttl.tensor.bert_large_pre_softmax_bmm(a_t, b_t, out_mem_config)

    # Check memory of inputs and outputs
    assert a_t.buffer_type() == in0_mem_config.buffer_type
    assert b_t.buffer_type() == in1_mem_config.buffer_type
    assert t2.buffer_type() == out_mem_config.buffer_type
    logger.debug(f"in0 is on: {a_t.buffer_type()}")
    logger.debug(f"in1 is on: {b_t.buffer_type()}")
    logger.debug(f"out is on: {t2.buffer_type()}")

    assert t2.shape() == [9, 16, 384, 384]
    tt_host_rm = t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm = torch.Tensor(tt_host_rm.data()).reshape(tt_host_rm.shape())

    ref_bmm = torch.matmul(A, B)
    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    logger.info(f"Passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")
    ttl.device.CloseDevice(device)
    assert passing_pcc


import pytest


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.L1),
    ),
)
@pytest.mark.parametrize(
    "in1_mem_config",
    (
        ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.L1),
    ),
)
@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, -1, ttl.tensor.BufferType.L1),
    ),
)
def test_bert_large_pre_softmax_bmm_test(
    dtype, in0_mem_config, in1_mem_config, out_mem_config
):
    run_bert_large_pre_softmax_bmm_test(
        dtype, in0_mem_config, in1_mem_config, out_mem_config
    )

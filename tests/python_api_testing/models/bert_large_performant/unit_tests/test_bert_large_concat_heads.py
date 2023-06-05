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


def run_bert_large_concat_heads_test(dtype, in0_mem_config, out_mem_config):
    torch.manual_seed(1234)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device, ttl.device.MemoryAllocator.L1_BANKING)
    host = ttl.device.GetHost()
    a_shape = [9, 16, 384, 64]

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

    out = ttl.tensor.bert_large_concat_heads(a_t, out_mem_config)

    # Check memory of inputs and outputs
    assert a_t.buffer_type() == in0_mem_config.buffer_type
    assert out.buffer_type() == out_mem_config.buffer_type

    logger.debug(f"in0 is on: {a_t.buffer_type()}")
    logger.debug(f"out is on: {out.buffer_type()}")

    assert out.shape() == [9, 1, 384, 1024]
    tt_host_rm_out = out.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_out = torch.Tensor(tt_host_rm_out.data()).reshape(
        tt_host_rm_out.shape()
    )

    ref_out = torch.transpose(A, -3, -2).reshape([9, 1, 384, 1024])
    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm_out, ref_out, 0.99)
    logger.info(f"passing={passing_pcc}")
    logger.info(f"output pcc={output_pcc}")
    assert passing_pcc

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
def test_bert_large_concat_heads_test(dtype, in0_mem_config, out_mem_config):
    run_bert_large_concat_heads_test(dtype, in0_mem_config, out_mem_config)

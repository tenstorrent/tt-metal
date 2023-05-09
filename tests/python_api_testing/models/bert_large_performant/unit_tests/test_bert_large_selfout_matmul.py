from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

import numpy as np

from libs import tt_lib as ttl
from python_api_testing.models.utility_functions import (
    comp_pcc,
)
import torch


def run_bert_large_selfout_matmul_test(dtype):
    torch.manual_seed(1234)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device, ttl.device.MemoryAllocator.L1_BANKING)
    host = ttl.device.GetHost()
    a_shape = [9, 1, 384, 1024]
    b_shape = [1, 1, 1024, 1024]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    memory_config = ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1)
    a_t = (
        ttl.tensor.Tensor(
            A.flatten().tolist(),
            a_shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, memory_config)
    )
    b_t = (
        ttl.tensor.Tensor(
            B.flatten().tolist(),
            b_shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.bert_large_selfout_matmul(a_t, b_t)
    assert t2.shape() == [9, 1, 384, 1024]
    tt_host_rm = t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm = torch.Tensor(tt_host_rm.data()).reshape(tt_host_rm.shape())

    ref_bmm = torch.matmul(A, B)
    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    ttl.device.CloseDevice(device)
    assert passing_pcc


import pytest


@pytest.mark.parametrize(
    "dtype",
    ((ttl.tensor.DataType.BFLOAT8_B),),
)
def test_bert_large_selfout_matmul_test(dtype):
    run_bert_large_selfout_matmul_test(dtype)

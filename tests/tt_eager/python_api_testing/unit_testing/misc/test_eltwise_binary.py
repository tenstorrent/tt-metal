# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import untilize, comp_pcc
from models.utility_functions import is_grayskull
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize(
    "dtype",
    [ttl.tensor.DataType.FLOAT32],
    ids=["float32"],
)
@pytest.mark.parametrize(
    "test_func_name, torch_func_name",
    [(ttl.tensor.add, torch.add), (ttl.tensor.sub, torch.sub), (ttl.tensor.mul, torch.mul)],
)
def test_run_elt_binary(dtype, test_func_name, torch_func_name, device):
    shape = [2, 16, 256, 256]

    torch.manual_seed(10)

    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)

    in0 = torch.randn(shape).bfloat16().float()
    in1 = torch.randn(shape).bfloat16().float()
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=mem_config, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=mem_config, tt_dtype=dtype)

    out_t = test_func_name(in0_t, in1_t)
    out = tt2torch_tensor(out_t)

    passing, output = comp_pcc(out, torch_func_name(in0, in1), 0.9999)
    logger.info(output)
    assert passing

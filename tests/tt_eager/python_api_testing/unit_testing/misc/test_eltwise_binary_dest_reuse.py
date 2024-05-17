# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc
from models.utility_functions import is_grayskull
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


@pytest.mark.parametrize(
    "ttl_dtype, torch_dtype",
    [
        (ttl.tensor.DataType.FLOAT32, torch.float32),
        (ttl.tensor.DataType.BFLOAT16, torch.bfloat16),
    ],
    ids=[
        "float32",
        "bfloat16",
    ],
)
@pytest.mark.parametrize(
    "test_func_name, torch_func_name",
    [
        (ttl.tensor.add_with_dest, torch.add),
        (ttl.tensor.sub_with_dest, torch.sub),
        (ttl.tensor.mul_with_dest, torch.mul),
    ],
    ids=["add", "sub", "mul"],
)
@pytest.mark.parametrize(
    "shape",
    [
        [4, 4, 256, 256],
        # [1, 1, 256, 256],
        [1, 1, 32, 32],
    ],
)
def test_run_elt_binary(ttl_dtype, torch_dtype, test_func_name, torch_func_name, shape, device):
    torch.manual_seed(0)
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("GS does not support fp32")
    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)

    in0 = torch.randn(shape, dtype=torch_dtype).float()
    in1 = torch.randn(shape, dtype=torch_dtype).float()
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=mem_config, tt_dtype=ttl_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=mem_config, tt_dtype=ttl_dtype)

    out_t = test_func_name(in0_t, in1_t)
    out = tt2torch_tensor(out_t)
    print(f"{torch_func_name(in0, in1)}")
    print(f"{out}")

    passing, output = comp_pcc(out, torch_func_name(in0, in1), 0.9999)
    logger.info(output)
    assert passing

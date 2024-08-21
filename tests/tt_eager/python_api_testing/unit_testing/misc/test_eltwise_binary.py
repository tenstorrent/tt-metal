# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import untilize, comp_pcc
from models.utility_functions import is_grayskull
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "bfloat16",
    ],
)
@pytest.mark.parametrize(
    "test_func_name, torch_func_name",
    [(ttnn.add, torch.add), (ttnn.sub, torch.sub), (ttnn.mul, torch.mul)],
)
@pytest.mark.parametrize(
    "pre_in0_silu",
    [True, False],
    ids=["silu", "no-silu"],
)
@pytest.mark.parametrize(
    "shard",
    [False, True],
    ids=["interleaved", "sharded"],
)
def test_run_elt_binary(dtype, test_func_name, torch_func_name, pre_in0_silu, device, shard):
    shape = (1, 1, 32, 1024)

    torch.manual_seed(10)

    if shard:
        mem_config = ttnn.create_sharded_memory_config(
            shape=shape,
            core_grid=ttnn.CoreGrid(y=1, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    in0 = torch.randn(shape).bfloat16().float()
    in1 = torch.randn(shape).bfloat16().float()
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=mem_config, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=mem_config, tt_dtype=dtype)

    if pre_in0_silu:
        torch_silu = torch.nn.SiLU()
        out_t = test_func_name(in0_t, in1_t, input_tensor_a_activation=ttnn.UnaryOpType.SILU, memory_config=mem_config)
    else:
        out_t = test_func_name(in0_t, in1_t)
    out = tt2torch_tensor(out_t)

    if pre_in0_silu:
        passing, output = comp_pcc(out, torch_func_name(torch_silu(in0), in1), 0.9999)
    else:
        passing, output = comp_pcc(out, torch_func_name(in0, in1), 0.9999)
    logger.info(output)
    assert passing

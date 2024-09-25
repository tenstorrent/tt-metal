# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import random
import numpy as np
import ttnn

from models.utility_functions import (
    comp_pcc,
)
import torch
import sys
import numpy
import pytest
import os
import itertools

debug = False

# TODO: test other dims/num_splits
two_chunk_dim_two_tests = list(
    zip(
        [[1, 2**k, 2] for k in range(10)],  # shapes
        itertools.repeat(2),  # chunks
        itertools.repeat(2),  # dim
    )
)

two_chunk_dim_two_ids = [
    "x".join(map(str, shape)) + f"@{dim}" + f"->{chunks}" for (shape, chunks, dim) in two_chunk_dim_two_tests
]


@pytest.mark.parametrize(
    "in_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
    ids=["in_DRAM", "in_L1"],
)
@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "refshape_chunks_dim",
    tuple(two_chunk_dim_two_tests),
    ids=two_chunk_dim_two_ids,
)
def test_split_rm(refshape_chunks_dim, in_mem_config, out_mem_config, device, dtype=ttnn.bfloat16):
    (refshape, num_splits, dim) = refshape_chunks_dim
    profile_location = "split_rm/"
    os.system(f"rm -rf {profile_location}")

    torch.manual_seed(1234)

    Z = refshape[0]
    Y = refshape[1]
    X = refshape[2]

    assert dim in [0, 1, 2]

    if dim == 2:
        chunk_shape = [Z, Y, X // num_splits]
    elif dim == 1:
        chunk_shape = [Z, Y // num_splits, X]
    elif dim == 0:
        chunk_shape = [Z // num_splits, Y, X]

    logger.info(f"Split tensor of size: {str(refshape)}")

    dtype_torch = torch.bfloat16

    A = torch.arange(Z * Y * X, dtype=dtype_torch).reshape(refshape)
    assert list(A.size()) == refshape

    a_t = ttnn.from_torch(A, layout=ttnn.Layout.ROW_MAJOR, dtype=dtype, memory_config=in_mem_config, device=device)

    # Check memory of inputs and outputs
    # logger.debug(f"input to rm split is on: {a_t.memory_config().buffer_type}")

    dev_buffers = ttnn.split(a_t, num_splits, dim, memory_config=out_mem_config)
    # dev_buffers come out tilized
    pyt_buff_list = []

    assert len(dev_buffers) == num_splits
    for index, buff in enumerate(dev_buffers):
        logger.debug(f"buff{index} is on: {buff.memory_config().buffer_type}")
        assert list(buff.shape) == chunk_shape
        tt_host_rm_buff = (
            buff.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(buff.shape.with_tile_padding().without_padding())
        )
        pyt_got_back_rm_buff = tt_host_rm_buff.to_torch()
        pyt_buff_list.append(pyt_got_back_rm_buff)

    golden_buffers = torch.chunk(A, num_splits, dim=dim)
    assert len(pyt_buff_list) == len(golden_buffers)
    if debug:
        for i in range(0, num_splits):
            print(f"torch result [{i+1}]: ", golden_buffers[i][0, 0, 0])
            print(f"our result [{i+1}]: ", pyt_buff_list[i][0, 0, 0])
            print()

    for index, pyt_buff in enumerate(pyt_buff_list):
        golden_buff = golden_buffers[index]
        passing_pcc, output_pcc = comp_pcc(pyt_buff, golden_buff, 1.0)
        logger.debug(f"Out passing={passing_pcc}")
        logger.debug(f"Output pcc={output_pcc}")
        assert passing_pcc

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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
@pytest.mark.parametrize("dim", (2, 3))  # 0, 1 ),
@pytest.mark.parametrize(
    "refshape",
    (
        [1, 2, 64, 128],
        [1, 2, 1024, 128],
        [1, 2, 256, 2560],
        [1, 2, 1024, 2560],
        [1, 2, 256, 5120],
        [1, 2, 64, 10240],
        [1, 1, 64, 64],
    ),
    ids=["1x2x64x64", "1x2x64x128", "1x2x256x2560", "1x2x1024x2560", "1x2x256x5120", "1x2x64x10240", "1x1x64x64"],
)
def test_split_tiled_w(dim, refshape, in_mem_config, out_mem_config, device, dtype=ttnn.bfloat16):
    profile_location = "splitTwoChunks/"
    os.system(f"rm -rf {profile_location}")

    _shape = refshape
    assert _shape[0] == 1
    num_splits = 2
    torch.manual_seed(1234)

    tile_size = 32
    W = _shape[0]  # *(random.choice([1,2]))
    Z = _shape[1]
    Y = _shape[2]
    X = _shape[3]

    assert dim in [0, 1, 2, 3]

    if dim == 0:
        W, Y = Y, W
    elif dim == 1:
        Z, Y = Y, Z
    elif dim in [2, 3]:
        Y, X = X, Y

    if Y % 32 != 0:
        Y = 64

    if dim == 3:
        chunk_shape = [W, Z, Y, X // 2]
    elif dim == 2:
        chunk_shape = [W, Z, Y // 2, X]
    elif dim == 1:
        chunk_shape = [W, Z // 2, Y, X]
    elif dim == 0:
        chunk_shape = [W // 2, Z, Y, X]

    a_shape = [W, Z, Y, X]
    logger.info(f"Split tensor of size: {str(a_shape)}")

    dtype_torch = torch.bfloat16

    A = torch.arange(W * Z * Y * X, dtype=dtype_torch).reshape(a_shape)
    assert list(A.size()) == a_shape

    tiled = (a_shape[dim] % tile_size == 0) and (a_shape[3] % tile_size == 0)

    if tiled:
        a_t = (
            ttnn.Tensor(
                A.flatten().tolist(),
                a_shape,
                dtype,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device, in_mem_config)
        )
    else:
        assert False
        a_t = ttnn.Tensor(
            A.flatten().tolist(),
            a_shape,
            dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        ).to(device)

    dev_buffers = ttnn.split(a_t, 2, dim, memory_config=out_mem_config)

    # Check memory of inputs and outputs
    logger.debug(f"in0 is on: {a_t.memory_config().buffer_type}")

    pyt_buff_list = []

    assert len(dev_buffers) == num_splits
    for index, buff in enumerate(dev_buffers):
        logger.debug(f"buff{index} is on: {buff.memory_config().buffer_type}")
        assert list(buff.get_legacy_shape()) == chunk_shape
        tt_host_rm_buff = buff.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
        pyt_got_back_rm_buff = tt_host_rm_buff.to_torch()
        pyt_buff_list.append(pyt_got_back_rm_buff)

    golden_buffers = torch.chunk(A, num_splits, dim=dim)
    assert len(pyt_buff_list) == len(golden_buffers)

    for index, pyt_buff in enumerate(pyt_buff_list):
        golden_buff = golden_buffers[index]
        passing_pcc, output_pcc = comp_pcc(pyt_buff, golden_buff, 1.0)
        logger.debug(f"Out passing={passing_pcc}")
        logger.debug(f"Output pcc={output_pcc}")
        assert passing_pcc

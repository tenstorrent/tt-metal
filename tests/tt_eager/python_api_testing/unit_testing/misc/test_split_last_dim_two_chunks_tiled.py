# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from loguru import logger


# TODO: need to use old style of importing until
# the rest of the unit tests have been uplifted
import tt_lib as ttl

# TODO: need to use old utility_functions until the
# rest of the unit tests use utility_functions
from models.utility_functions import (
    comp_pcc,
)
import torch
import pytest
import os


def getTensorFromBuff(buff):
    tensor_from_buff = buff.clone()
    tensor_from_buff[
        torch.logical_or(
            torch.isnan(tensor_from_buff),
            torch.logical_or(torch.isinf(tensor_from_buff), torch.isneginf(tensor_from_buff)),
        )
    ] = 0
    return tensor_from_buff


@pytest.mark.parametrize(
    "in_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["in_DRAM", "in_L1"],
)
@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "shape",
    (
        [1, 2, 32, 64],
        [1, 2, 64, 64],
        [1, 2, 64, 128],
        [1, 2, 1024, 128],
        [1, 2, 256, 2560],
        [1, 2, 1024, 2560],
        [1, 2, 256, 5120],
        [1, 2, 64, 10240],
        [1, 2, 16, 10240],
    ),
    ids=[
        "1x2x32x64",
        "1x2x64x64",
        "1x2x64x128",
        "1x2x1024x128",
        "1x2x256x2560",
        "1x2x1024x2560",
        "1x2x256x5120",
        "1x2x64x10240",
        "1x2x16x10240",
    ],
)
def test_split_tiled_w(shape, in_mem_config, out_mem_config, device, dtype=ttl.tensor.DataType.BFLOAT16):
    profile_location = "splitTwoChunks/"
    os.system(f"rm -rf {profile_location}")

    ttl.profiler.set_profiler_location(profile_location)

    assert shape[0] == 1
    untiled_shape = [1, 2, 16, 10240]

    if shape == untiled_shape and (dtype == ttl.tensor.DataType.BFLOAT8_B):
        pytest.skip("BFLOAT8_B only supported with tile")

    num_splits = 2
    torch.manual_seed(1234)

    tile_size = 32
    W = 1
    Z = shape[1]
    Y = shape[2]
    X = shape[3]

    a_shape = [W, Z, Y, X]
    logger.info(f"Split tensor of size: {str(a_shape)}")

    dtype_torch = torch.bfloat16

    A = torch.arange(W * Z * Y * X, dtype=dtype_torch).reshape(a_shape)
    assert list(A.size()) == a_shape

    tiled = (shape[2] % tile_size == 0) and (shape[3] % tile_size == 0)

    if tiled:
        a_t = (
            ttl.tensor.Tensor(
                A.flatten().tolist(),
                a_shape,
                dtype,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device, in_mem_config)
        )
    else:
        a_t = ttl.tensor.Tensor(
            A.flatten().tolist(),
            a_shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        ).to(device)

    ttl.profiler.start_profiling("Run")
    dev_buffers = ttl.tensor.split_last_dim_two_chunks_tiled(a_t, out_mem_config)
    ttl.profiler.stop_profiling("Run")

    # Check memory of inputs and outputs
    logger.debug(f"in0 is on: {a_t.memory_config().buffer_type}")

    pyt_buff_list = []

    assert len(dev_buffers) == num_splits
    for index, buff in enumerate(dev_buffers):
        logger.debug(f"buff{index} is on: {buff.memory_config().buffer_type}")
        assert list(buff.get_legacy_shape()) == [W, Z, Y, int(X / num_splits)]
        tt_host_rm_buff = buff.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
        pyt_got_back_rm_buff = tt_host_rm_buff.to_torch()
        pyt_buff_list.append(pyt_got_back_rm_buff)

    golden_buffers = torch.chunk(A, num_splits, dim=-1)
    assert len(pyt_buff_list) == len(golden_buffers)

    for index, pyt_buff in enumerate(pyt_buff_list):
        golden_buff = golden_buffers[index]
        passing_pcc, output_pcc = comp_pcc(pyt_buff, golden_buff, 1.0)
        logger.debug(f"Out passing={passing_pcc}")
        logger.debug(f"Output pcc={output_pcc}")
        assert passing_pcc

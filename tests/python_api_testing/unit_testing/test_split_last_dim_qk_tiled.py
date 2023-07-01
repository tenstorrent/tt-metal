from pathlib import Path
import sys
from loguru import logger

import numpy as np

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

# TODO: need to use old style of importing until
# the rest of the unit tests have been uplifted
import tt_lib as ttl

# TODO: need to use old utility_functions until the
# rest of the unit tests use utility_functions_new
from python_api_testing.models.utility_functions import (
    comp_pcc,
)
import torch
import sys
import numpy
import pytest


@pytest.mark.parametrize(
    "in_mem_config",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["in_DRAM", "in_L1"],
)
@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "shape",
    ([1, 2, 1024, 2560], [1, 2, 256, 5120], [1, 2, 64, 10240], [1, 2, 16, 10240]),
    ids=["1x2x1024x2560", "1x2x256x5120", "1x2x64x10240", "1x2x16x10240"],
)
def test_split_tiled_single_core_test_w(shape, in_mem_config, out_mem_config):
    dtype = ttl.tensor.DataType.BFLOAT16
    assert shape[0] == 1
    untiled_shape = [1, 2, 16, 10240]
    if shape == untiled_shape and (
        out_mem_config.buffer_type == ttl.tensor.BufferType.L1
        or in_mem_config.buffer_type == ttl.tensor.BufferType.L1
    ):
        pytest.skip("No Autoformat support for L1 buffers")
    num_splits = 2
    torch.manual_seed(1234)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device, ttl.device.MemoryAllocator.L1_BANKING)
    # ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()
    tile_size = 32
    num_tiles_per_tensor_h = int(shape[2] / tile_size)
    num_tiles_per_tensor_w = int((shape[3] / tile_size) / num_splits)
    n = 1
    c = shape[1]
    h = shape[2]
    w = shape[3]
    dim = 3

    a_shape = [n, c, h, w]
    logger.info(f"Split tensor of size: {str(a_shape)}")

    A = torch.arange(c * h * w, dtype=torch.bfloat16).reshape(a_shape)
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

    dev_buffers = ttl.tensor.split_last_dim_qk_tiled(a_t, out_mem_config)

    # Check memory of inputs and outputs
    logger.debug(f"in0 is on: {a_t.memory_config().buffer_type}")

    pyt_buff_list = []

    assert len(dev_buffers) == num_splits
    for index, buff in enumerate(dev_buffers):
        logger.debug(f"buff{index} is on: {buff.memory_config().buffer_type}")
        assert buff.shape() == [n, c, h, int(w / num_splits)]
        tt_host_rm_buff = buff.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
        pyt_got_back_rm_buff = torch.Tensor(tt_host_rm_buff.data()).reshape(
            tt_host_rm_buff.shape()
        )
        pyt_buff_list.append(pyt_got_back_rm_buff)

    golden_buffers = torch.chunk(A, num_splits, dim=-1)
    assert len(pyt_buff_list) == len(golden_buffers)

    for index, pyt_buff in enumerate(pyt_buff_list):
        golden_buff = golden_buffers[index]
        passing_pcc_q, output_pcc_q = comp_pcc(pyt_buff, golden_buff, 0.99)
        logger.info(f"Q passing={passing_pcc_q}")
        logger.info(f"Q output pcc={output_pcc_q}")
        assert passing_pcc_q

    ttl.device.CloseDevice(device)

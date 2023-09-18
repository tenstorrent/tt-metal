# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
)
from loguru import logger


def test_sharded_tile(device):
    N = 1
    C = 1
    H = 100352
    W = 64
    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttl.tensor.BufferType.L1,
            ),
        )
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt, 98, [1024, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_got_back = zt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    assert torch.equal(tt_og, tt_got_back)


def test_sharded_rm(device):
    N = 1
    C = 1
    H = 100352
    W = 64
    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16().float()

    xt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(
        device,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt, 98, [1024, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to_torch()

    tt_got_back = zt.cpu().to_torch()

    passing, output = comp_equal(tt_og, tt_got_back)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("H, num_cores", [[100352, 98], [25088, 98]])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_sharded_untilize(H, num_cores, in_sharded, out_sharded, device):
    N = 1
    C = 1
    H = H
    W = 64
    if out_sharded and not in_sharded and H == 100352:
        pytest.skip("Unsupported config for sharding")

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            num_cores,
            [H // num_cores, 64],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        )

    yt = ttl.tensor.untilize(
        xt,
        output_mem_config=out_mem_config,
        use_multicore=True,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    passing, output = comp_equal(x, tt_got_back)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("H, num_cores", [[25088, 98]])
def test_sharded_tilize(H, num_cores, device):
    N = 1
    C = 1
    W = 64
    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(
        device,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt,
        num_cores,
        [H // num_cores, 64],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
    )

    yt_tilized = ttl.tensor.tilize(
        yt,
        output_mem_config=ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
        use_multicore=True,
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt_tilized,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_got_back = zt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    passing, output = comp_equal(x, tt_got_back)
    logger.info(output)

    assert passing

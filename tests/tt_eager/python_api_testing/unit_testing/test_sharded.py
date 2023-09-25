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

    assert torch.equal(tt_og, tt_got_back)

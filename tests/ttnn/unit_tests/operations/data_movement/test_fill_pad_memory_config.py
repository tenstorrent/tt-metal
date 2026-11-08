# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

# The binding documents memory_config for the output, but prim::fill_pad is in-place and returned the input
# tensor with the input's config, so an explicit request was silently dropped. These cases request L1 and
# check the returned config; the controls keep the input's config on both the pre-fix and post-fix code.
L1_SHARDED = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]),
        (32, 64),
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)


def _check_values(host, x_torch):
    assert torch.equal(host.float().bfloat16(), x_torch.reshape(host.shape))


@pytest.mark.parametrize(
    "shape, memory_config",
    [
        ((1, 1, 33, 35), ttnn.L1_MEMORY_CONFIG),
        ((1, 1, 33, 64), L1_SHARDED),
        ((1, 1, 33, 35), None),
        ((1, 1, 33, 35), ttnn.DRAM_MEMORY_CONFIG),
        ((1, 1, 32, 32), ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_fill_implicit_tile_padding_honors_memory_config(device, shape, memory_config):
    torch.manual_seed(0)
    x_torch = torch.randn(shape).bfloat16()
    x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    y = ttnn.fill_implicit_tile_padding(x, 0.0, memory_config=memory_config)
    y_torch = ttnn.to_torch(y)

    expected = memory_config if memory_config is not None else x.memory_config()
    assert y.memory_config().buffer_type == expected.buffer_type
    assert y.memory_config().memory_layout == expected.memory_layout
    _check_values(y_torch, x_torch)

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
import torch


def test_open_device():
    """Simple unit test to test device open/close APIs"""
    device = ttnn.open_device(device_id=0)
    ttnn.close_device(device)


def test_manage_device():
    """Simple unit test to test device context manager APIs"""
    with ttnn.manage_device(0) as device:
        pass


def test_l1_size():
    assert ttnn.get_max_worker_l1_unreserved_size() > 1024 * 1024


@pytest.mark.parametrize(
    "device_params",
    [{"worker_l1_size": 16385}],
    indirect=True,
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_worker_l1_size(device, layout, dtype):
    torch_tensor = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)

    core_grid = ttnn.CoreGrid(y=1, x=1)
    memory_config = ttnn.create_sharded_memory_config(torch_tensor.shape, core_grid, ttnn.ShardStrategy.BLOCK)
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=memory_config)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    torch_loop_back_tensor = ttnn.to_torch(ttnn_loop_back_tensor)


@pytest.mark.parametrize(
    "device_params",
    [{"worker_l1_size": 16385}],
    indirect=True,
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_worker_l1_fail(device, layout, dtype):
    torch_tensor = torch.rand((1, 1, 32, 1024), dtype=torch.bfloat16)

    core_grid = ttnn.CoreGrid(y=1, x=1)
    memory_config = ttnn.create_sharded_memory_config(torch_tensor.shape, core_grid, ttnn.ShardStrategy.BLOCK)
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout)
    with pytest.raises(RuntimeError, match=".*Out of Memory:.*"):
        ttnn_tensor = ttnn.to_device(
            ttnn_tensor,
            device,
            memory_config=memory_config,
        )

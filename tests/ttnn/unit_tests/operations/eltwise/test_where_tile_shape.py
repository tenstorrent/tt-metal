# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("tile_height", [4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_where_tile_shape(device, tile_height, dtype):
    host = (torch.arange(32 * 64, dtype=torch.float32) % 127).reshape(1, 1, 32, 64)
    predicate = host % 2

    def make(value):
        return ttnn.from_torch(
            value, dtype=dtype, tile=ttnn.Tile([tile_height, 32]), layout=ttnn.TILE_LAYOUT, device=device
        )

    condition = make(predicate)
    true_value = make(host)
    false_value = make(-host)
    expected = torch.where(predicate.bool(), host, -host)
    for _ in range(2):
        if tile_height != 32:
            with pytest.raises(RuntimeError, match="Ternary operations support only 32x32 tiles"):
                ttnn.where(condition, true_value, false_value)
        else:
            output = ttnn.where(condition, true_value, false_value)
            assert torch.equal(ttnn.to_torch(output).float(), expected)


@pytest.mark.parametrize("small_tensor", [0, 1, 2, 3])
def test_where_rejects_each_small_tile_tensor(device, small_tensor):
    values = [
        torch.ones((1, 1, 32, 64)),
        torch.full((1, 1, 32, 64), 3.0),
        torch.full((1, 1, 32, 64), -5.0),
        torch.zeros((1, 1, 32, 64)),
    ]
    tensors = [
        ttnn.from_torch(
            value,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            tile=ttnn.Tile([16 if index == small_tensor else 32, 32]),
        )
        for index, value in enumerate(values)
    ]
    with pytest.raises(RuntimeError, match="Ternary operations support only 32x32 tiles"):
        ttnn.where(*tensors[:3], output_tensor=tensors[3])

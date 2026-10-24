# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("tile_height", [4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_tile_shape(device, tile_height, dtype, descending, expect_error):
    host = torch.arange(63, -1, -1, dtype=torch.float32).reshape(1, 1, 1, 64).repeat(1, 1, 32, 1)
    source = ttnn.from_torch(
        host, dtype=dtype, tile=ttnn.Tile([tile_height, 32]), layout=ttnn.TILE_LAYOUT, device=device
    )
    for _ in range(2):
        if tile_height != 32:
            with expect_error(RuntimeError, "Sort supports only 32x32 tiles"):
                ttnn.sort(source, dim=-1, descending=descending)
        else:
            values, indices = ttnn.sort(source, dim=-1, descending=descending)
            expected_values, expected_indices = torch.sort(host, dim=-1, descending=descending)
            assert torch.equal(ttnn.to_torch(values).float(), expected_values)
            assert torch.equal(ttnn.to_torch(indices).long(), expected_indices)
        assert torch.equal(ttnn.to_torch(source).float(), host)


@pytest.mark.parametrize("small_output", [0, 1])
def test_sort_rejects_small_output_tile(device, small_output, expect_error):
    host = torch.arange(63, -1, -1, dtype=torch.float32).reshape(1, 1, 1, 64).repeat(1, 1, 32, 1)
    source = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    outputs = [
        ttnn.from_torch(
            torch.zeros_like(host),
            dtype=dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile([16 if index == small_output else 32, 32]),
        )
        for index, dtype in enumerate([ttnn.bfloat16, ttnn.uint16])
    ]
    with expect_error(RuntimeError, "Sort supports only 32x32 tiles"):
        ttnn.sort(source, dim=-1, out=outputs)

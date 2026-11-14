# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("tile_height", [4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("multicore", [False, True])
def test_pad_tile_shape(device, tile_height, dtype, multicore, expect_error):
    host = (torch.arange(32 * 64, dtype=torch.float32) % 127).reshape(1, 1, 32, 64)
    source = ttnn.from_torch(
        host, dtype=dtype, tile=ttnn.Tile([tile_height, 32]), layout=ttnn.TILE_LAYOUT, device=device
    )
    for _ in range(2):
        if tile_height != 32:
            with expect_error(RuntimeError, "Pad supports only 32x32 tiles"):
                ttnn.pad(source, [1, 1, 64, 96], [0, 0, 0, 0], 0, use_multicore=multicore)
        else:
            output = ttnn.pad(source, [1, 1, 64, 96], [0, 0, 0, 0], 0, use_multicore=multicore)
            expected = torch.zeros((1, 1, 64, 96))
            expected[..., :32, :64] = host
            assert torch.equal(ttnn.to_torch(output).float(), expected)
        assert torch.equal(ttnn.to_torch(source).float(), host)

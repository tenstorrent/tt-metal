# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

SMALL = ttnn.Tile([16, 32])
FULL = ttnn.Tile([32, 32])


def _tile_tensor(device, shape, tile, dtype=ttnn.bfloat16, fill=None):
    host = torch.randn(shape).to(torch.bfloat16) if fill is None else torch.full(shape, fill, dtype=torch.bfloat16)
    return ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, tile=tile, device=device)


def test_softmax_rejects_a_small_input_tile(device, expect_error):
    with expect_error(RuntimeError, "supports only 32x32 tiles"):
        ttnn.softmax(_tile_tensor(device, [1, 1, 64, 256], SMALL), dim=-1)
    ttnn.softmax(_tile_tensor(device, [1, 1, 64, 256], FULL), dim=-1)  # control


def test_typecast_rejects_small_input_and_output_tiles(device, expect_error):
    with expect_error(RuntimeError, "supports only 32x32 tiles"):
        ttnn.typecast(_tile_tensor(device, [1, 1, 64, 256], SMALL), ttnn.float32)
    out = _tile_tensor(device, [1, 1, 64, 256], SMALL, dtype=ttnn.float32, fill=0.0)
    with expect_error(RuntimeError, "supports only 32x32 tiles"):
        ttnn.typecast(_tile_tensor(device, [1, 1, 64, 256], FULL), ttnn.float32, output_tensor=out)
    ttnn.typecast(_tile_tensor(device, [1, 1, 64, 256], FULL), ttnn.float32)  # control

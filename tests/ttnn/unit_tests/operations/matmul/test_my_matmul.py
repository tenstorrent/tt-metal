# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


# (M, K, N) shapes, each chosen to exercise a different part of the op:
#   32x32x32        - single output tile -> only one core does work
#   256x256x256     - 8x8 = 64 output tiles -> spreads across the whole grid
#   128x256x512     - non-square, deep-ish K
#   160x64x224      - 5x7 = 35 output tiles -> ragged split (doesn't divide the grid evenly)
#   96x512x96       - large K (16 k-tiles) with a small output block
#   48x96x80        - logical dims NOT tile-aligned -> exercises auto zero-padding
SHAPES = [
    (32, 32, 32),
    (256, 256, 256),
    (128, 256, 512),
    (160, 64, 224),
    (96, 512, 96),
    (48, 96, 80),
    (3, 17, 25),
    (124, 123, 122),
    (253, 24, 134),
    (512, 512, 512),
    (32, 512, 2048),
    (2048, 256, 32),
    (33, 33, 33),
    (65, 129, 97),
    (1000, 700, 1300),
    (255, 257, 256),
    (4096, 64, 512),
    (128, 4096, 96),
    (3072, 3072, 128),
    (17, 2049, 4096),
    (777, 333, 999),
    (4096, 4096, 4096),
    (4096, 16384, 4096),
]


@pytest.mark.parametrize("M, K, N", SHAPES)
def test_my_matmul(device, M, K, N):
    torch.manual_seed(42)

    a = torch.randn((M, K), dtype=torch.bfloat16)
    b = torch.randn((K, N), dtype=torch.bfloat16)
    golden = a.float() @ b.float()

    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    tc = ttnn.my_matmul(ta, tb)
    result = ttnn.to_torch(tc)

    assert result.shape == golden.shape, f"shape {tuple(result.shape)} != {tuple(golden.shape)}"
    assert_with_pcc(golden, result.float(), 0.999)

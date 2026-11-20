# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def test_from_torch_large(device):
    # A 4 GiB bfloat16 host tensor through the host TILE tilize and untilize. Nightly only because of its size.
    torch_x = torch.rand((2048, 1024, 32, 32), dtype=torch.bfloat16)
    x_tensor = ttnn.from_torch(torch_x, layout=ttnn.TILE_LAYOUT)
    x_tensor = ttnn.to_torch(x_tensor)
    assert torch.equal(torch_x, x_tensor)

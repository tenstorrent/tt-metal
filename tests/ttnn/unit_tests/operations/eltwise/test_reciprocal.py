# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


def test_reciprocal_fp32_faithful(device):
    if device.arch() != ttnn.device.Arch.WORMHOLE_B0:
        pytest.skip("Wormhole reciprocal precision regression")

    # Exhaust every FP32 mantissa in [1, 2), in eight chunks to bound memory use.
    chunk_size = 1 << 20
    for start in range(0x3F800000, 0x40000000, chunk_size):
        bits = torch.arange(start, start + chunk_size, dtype=torch.int32)
        values = bits.view(torch.float32).reshape(1024, 1024)
        input_tensor = ttnn.from_torch(values, layout=ttnn.TILE_LAYOUT, device=device)
        actual = ttnn.to_torch(ttnn.reciprocal(input_tensor)).to(torch.float64)
        exact = 1.0 / values.to(torch.float64)
        # Reciprocal results lie in [0.5, 1], with spacing 2**-24 below 1.
        assert torch.all(
            (actual - exact).abs() < 2.0**-24
        ), f"Unfaithful reciprocal in chunk starting at {start:#010x}"

# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf shapes for --profile runs (device kernel time); correctness is a smoke check only."""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm import layer_norm

WIDTH = 768
PERF_SHAPES = [
    (32, 1, 32, WIDTH),  # R = 32
    (8, 1, 512, WIDTH),  # R = 128
    (124, 1, 33, WIDTH),  # R = 248
]


@pytest.mark.parametrize("affine_form", ["tile_dram", "rm_l1"])
@pytest.mark.parametrize("shape", PERF_SHAPES, ids=lambda s: "x".join(str(d) for d in s))
def test_layer_norm_tile_perf(device, shape, affine_form):
    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn(shape).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    vec = torch.randn(WIDTH).to(torch.bfloat16)
    if affine_form == "tile_dram":
        g = ttnn.from_torch(
            vec, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    else:
        g = ttnn.from_torch(
            vec.reshape(1, 1, WIDTH // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    out = layer_norm(x, g, g)
    assert tuple(out.shape) == shape

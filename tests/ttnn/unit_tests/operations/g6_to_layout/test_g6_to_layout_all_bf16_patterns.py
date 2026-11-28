# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every bfloat16 bit pattern through the Tensix reorder of g6_to_layout, in both directions.

The bf16 TILE -> ROW_MAJOR (staged_reorder) and ROW_MAJOR -> TILE (row_major_to_tile) regimes run
unpack / MOVA2D / pack on UInt16 views of the bf16 bytes. A Float16_b view would turn -0.0 and
subnormals into +0 and NaN into +-inf, so the gate is bit identity on all 65536 patterns, each shape
holding a random permutation of them (every pattern lands in many tile and face positions).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_layout import g6_to_layout, select_plan

SHAPES = [
    pytest.param((1, 256, 256), id="aligned-256x256"),
    pytest.param((4, 128, 160), id="aligned-4-images"),
    pytest.param((3, 97, 233), id="ragged-h-and-w"),
    pytest.param((2, 40, 1000), id="ragged-h-wide"),
    pytest.param((16, 1, 4100), id="h1-multi-image"),
]
MEMORY = [
    pytest.param(ttnn.DRAM_MEMORY_CONFIG, id="dram"),
    pytest.param(ttnn.L1_MEMORY_CONFIG, id="l1"),
]


def _all_patterns(shape, seed):
    """Every bf16 pattern once, the rest of the tensor random patterns, in a random order."""
    numel = 1
    for d in shape:
        numel *= d
    assert numel >= 1 << 16
    generator = torch.Generator().manual_seed(seed)
    patterns = torch.arange(-0x8000, 0x8000, dtype=torch.int32)
    extra = torch.randint(-0x8000, 0x8000, (numel - patterns.numel(),), generator=generator, dtype=torch.int32)
    bits = torch.cat([patterns, extra])[torch.randperm(numel, generator=generator)]
    return bits.to(torch.int16).view(torch.bfloat16).reshape(shape)


@pytest.mark.parametrize("memory_config", MEMORY)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "source_layout,regime",
    [
        pytest.param(ttnn.TILE_LAYOUT, "staged_reorder", id="tile-to-rm"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, "row_major_to_tile", id="rm-to-tile"),
    ],
)
def test_all_bf16_patterns(device, shape, memory_config, source_layout, regime):
    x = _all_patterns(shape, seed=sum(shape))
    target = ttnn.ROW_MAJOR_LAYOUT if source_layout == ttnn.TILE_LAYOUT else ttnn.TILE_LAYOUT
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=source_layout, device=device, memory_config=memory_config)
    assert select_plan(tt_x, target)["regime"] == regime
    out = g6_to_layout(tt_x, target)
    got = ttnn.to_torch(out).reshape(shape)
    mismatches = int((got.view(torch.int16) != x.view(torch.int16)).sum())
    assert mismatches == 0, f"{mismatches} of {x.numel()} bf16 patterns changed"

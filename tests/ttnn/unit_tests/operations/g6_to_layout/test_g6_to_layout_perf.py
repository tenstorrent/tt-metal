# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-measurement shapes for g6_to_layout (run under run_safe_pytest.sh --profile).

Production cases only (BB-22): the flagged perf cases p1/p2/p4 plus one guard per distinct
R1 path (tall rows, column-chunked short rows, one long row, multi-image H = 1, L1 placement).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_layout import g6_to_layout


@pytest.mark.parametrize(
    "shape, memory_config",
    [
        pytest.param((1, 2048, 2048, 3), ttnn.DRAM_MEMORY_CONFIG, id="c308-p1"),
        pytest.param((1, 1024, 2048, 128), ttnn.DRAM_MEMORY_CONFIG, id="c304-p2"),
        pytest.param((1, 1024, 1024, 3), ttnn.DRAM_MEMORY_CONFIG, id="c286-p4"),
        pytest.param((16, 4096, 2560), ttnn.DRAM_MEMORY_CONFIG, id="c1978-tall"),
        pytest.param((1, 4096, 2560), ttnn.DRAM_MEMORY_CONFIG, id="c2031-chunked"),
        pytest.param((1, 131072), ttnn.DRAM_MEMORY_CONFIG, id="c294-long-row"),
        pytest.param((16, 1, 2560), ttnn.DRAM_MEMORY_CONFIG, id="c1993-h1-multi-image"),
        pytest.param((1, 32, 2560), ttnn.L1_MEMORY_CONFIG, id="c2038-l1"),
    ],
)
def test_perf(device, shape, memory_config):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    out = g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(out).reshape(shape).view(torch.int16), x.view(torch.int16))


def _sharded(memory_layout, ranges, shard_shape):
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in ranges]
    )
    spec = ttnn.ShardSpec(grid, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, spec)


_HEIGHT_GRID = [((0, 0), (10, 6)), ((0, 7), (8, 7))]


# Refinement 1 paths: packer_untilize (cases 29, 43) and sharded_in_place (cases 20, 37, 48).
@pytest.mark.parametrize(
    "shape, memory_config, out_memory_config",
    [
        pytest.param((1, 32, 64, 256), ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, id="c29-bf8b-l1"),
        pytest.param((1, 128, 256, 32), ttnn.L1_MEMORY_CONFIG, None, id="c43-bf8b-l1"),
        pytest.param(
            (1, 1, 1, 256),
            _sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, [((0, 0), (7, 0))], (32, 32)),
            ttnn.L1_MEMORY_CONFIG,
            id="c20-bf8b-block",
        ),
        pytest.param(
            (1, 64, 128, 256),
            _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _HEIGHT_GRID, (96, 256)),
            ttnn.L1_MEMORY_CONFIG,
            id="c37-bf8b-height",
        ),
        pytest.param(
            (1, 64, 128, 128),
            _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _HEIGHT_GRID, (96, 128)),
            ttnn.L1_MEMORY_CONFIG,
            id="c48-bf8b-height",
        ),
    ],
)
def test_perf_bf8b(device, shape, memory_config, out_memory_config):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    kwargs = {} if out_memory_config is None else {"memory_config": out_memory_config}
    out = g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT, **kwargs)
    expected = ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).to(torch.bfloat16)
    assert torch.equal(ttnn.to_torch(out).reshape(shape).view(torch.int16), expected.reshape(shape).view(torch.int16))

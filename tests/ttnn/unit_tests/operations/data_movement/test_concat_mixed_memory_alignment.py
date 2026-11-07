# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("first_width", [32, 36, 40])
@pytest.mark.parametrize("memories", [("L1", "DRAM"), ("DRAM", "DRAM"), ("L1", "L1"), ("DRAM", "L1")])
def test_concat_mixed_memory_row_alignment(device, rank, first_width, memories):
    shape_a = (first_width,) if rank == 1 else (4, first_width)
    shape_b = (32,) if rank == 1 else (4, 32)
    a = torch.arange(torch.Size(shape_a).numel(), dtype=torch.float32).reshape(shape_a)
    b = torch.arange(torch.Size(shape_b).numel(), dtype=torch.float32).reshape(shape_b) + 1000
    configs = {"L1": ttnn.L1_MEMORY_CONFIG, "DRAM": ttnn.DRAM_MEMORY_CONFIG}
    inputs = [
        ttnn.from_torch(
            x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=configs[memory]
        )
        for x, memory in zip((a, b), memories)
    ]
    output = ttnn.concat(inputs, dim=-1)
    assert torch.equal(ttnn.to_torch(output), torch.cat((a, b), dim=-1))


@pytest.mark.parametrize("height", [32, 64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_concat_mixed_memory_tiled_bfloat8_keeps_values(device, height, seed):
    torch.manual_seed(seed)
    tensors = [
        ttnn.from_torch(
            torch.randn((1, 1, height, width)),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory,
        )
        for width, memory in [
            (32, ttnn.L1_MEMORY_CONFIG),
            (64, ttnn.DRAM_MEMORY_CONFIG),
            (64, ttnn.L1_MEMORY_CONFIG),
        ]
    ]
    expected = torch.cat([ttnn.to_torch(tensor) for tensor in tensors], dim=3)
    # Repeat with cached programs to check that rebinding preserves exact values.
    for _ in range(2):
        result = ttnn.concat(tensors, dim=3)
        assert torch.equal(ttnn.to_torch(result), expected)


@pytest.mark.parametrize("width", [36, 40])
@pytest.mark.parametrize("case", ["height_stride", "width_stride", "mixed_layout", "sub_core", "sharded"])
def test_concat_native_stick_alignment(device, width, case):
    configs = {"L1": ttnn.L1_MEMORY_CONFIG, "DRAM": ttnn.DRAM_MEMORY_CONFIG}
    dim = -1
    output_memory = ttnn.DRAM_MEMORY_CONFIG
    kwargs = {}
    if case == "height_stride":
        shapes, memories = [(1024, width), (1024, width)], ["L1", "DRAM"]
        dim, output_memory = 0, ttnn.L1_MEMORY_CONFIG
    elif case == "width_stride":
        shapes, memories = [(1024, 32), (1024, 32), (1024, width)], ["L1", "DRAM", "L1"]
        output_memory = ttnn.L1_MEMORY_CONFIG
    elif case == "sharded":
        shapes = [(1, 1, 128, 32), (1, 1, 128, width), (1, 1, 128, 32)]
        memories = ["DRAM", "L1", "DRAM"]
    else:
        shapes, memories = [(4, width), (4, 32)], ["L1", "DRAM"]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    if case == "sub_core":
        kwargs["sub_core_grids"] = grid
    references, inputs = [], []
    for i, (shape, memory) in enumerate(zip(shapes, memories)):
        value = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape) + i * 1000
        config = configs[memory]
        if case == "sharded":
            config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                getattr(ttnn.BufferType, memory),
                ttnn.ShardSpec(grid, [128, shape[-1]], ttnn.ShardOrientation.ROW_MAJOR),
            )
        layout = ttnn.TILE_LAYOUT if case == "mixed_layout" and i == 1 else ttnn.ROW_MAJOR_LAYOUT
        inputs.append(ttnn.from_torch(value, dtype=ttnn.float32, layout=layout, device=device, memory_config=config))
        references.append(value)
    output = ttnn.concat(inputs, dim=dim, memory_config=output_memory, **kwargs)
    assert torch.equal(ttnn.to_torch(output), torch.cat(references, dim=dim))


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("num_inputs", [2, 3])
@pytest.mark.parametrize("output_memory", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("mixed_memory", [False, True])
def test_concat_aligned_row_major(device, dim, num_inputs, output_memory, mixed_memory):
    references = [torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32) + i * 1000 for i in range(num_inputs)]
    inputs = [
        ttnn.from_torch(
            value,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG if mixed_memory and i % 2 == 0 else ttnn.DRAM_MEMORY_CONFIG,
        )
        for i, value in enumerate(references)
    ]
    output = ttnn.concat(inputs, dim=dim, memory_config=output_memory)
    assert torch.equal(ttnn.to_torch(output), torch.cat(references, dim=dim))


@pytest.mark.parametrize("tile_first", [False, True])
def test_concat_mixed_layout_bfloat16_halfword_sticks(device, tile_first):
    widths = [32, 33, 32] if tile_first else [24, 33]
    memories = ["DRAM", "L1", "DRAM"] if tile_first else ["L1", "DRAM"]
    references = [(torch.arange(4 * w).reshape(4, w) % 97 + i * 100).to(torch.bfloat16) for i, w in enumerate(widths)]
    inputs = [
        ttnn.from_torch(
            value,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT if i == (0 if tile_first else 1) else ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=getattr(ttnn, memory + "_MEMORY_CONFIG"),
        )
        for i, (value, memory) in enumerate(zip(references, memories))
    ]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    output = ttnn.concat(inputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=grid)
    assert torch.equal(ttnn.to_torch(output), torch.cat(references, dim=1))


def test_concat_wide_mixed_memory_uses_bounded_staging(device):
    references = [torch.arange(4 * width, dtype=torch.float32).reshape(4, width) % 97 for width in [36, 262144]]
    inputs = [
        ttnn.from_torch(
            value,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=memory,
        )
        for value, memory in zip(references, [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
    ]
    output = ttnn.concat(inputs, dim=1)
    assert torch.equal(ttnn.to_torch(output), torch.cat(references, dim=1))


def test_concat_empty_width_input_with_subgrid(device):
    expected = torch.arange(128, dtype=torch.float32).reshape(4, 32)
    inputs = [
        ttnn.from_torch(
            value,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=memory,
        )
        for value, memory in [(torch.empty((4, 0)), ttnn.L1_MEMORY_CONFIG), (expected, ttnn.DRAM_MEMORY_CONFIG)]
    ]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    output = ttnn.concat(inputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=grid)
    assert torch.equal(ttnn.to_torch(output), expected)


@pytest.mark.parametrize("prefix_width", [16, 17])
@pytest.mark.parametrize("last_width", [1, 33, 257])
def test_concat_mixed_memory_uint8_odd_sticks(device, prefix_width, last_width):
    widths = [0, prefix_width, last_width]
    references = [(torch.arange(4 * w).reshape(4, w) % 97 + i * 100).to(torch.uint8) for i, w in enumerate(widths)]
    inputs = [
        ttnn.from_torch(
            value,
            dtype=ttnn.uint8,
            layout=ttnn.TILE_LAYOUT if i == 0 else ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG if i == 1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        for i, value in enumerate(references)
    ]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    output = ttnn.concat(inputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=grid)
    assert torch.equal(ttnn.to_torch(output), torch.cat(references, dim=1))


def test_concat_mixed_layout_bfloat16_short_final_chunk(device):
    references = [
        (torch.arange(4 * w).reshape(4, w) % 97 + i * 100).to(torch.bfloat16) for i, w in enumerate([24, 129])
    ]
    inputs = [
        ttnn.from_torch(
            value,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if i == 0 else ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG if i == 0 else ttnn.DRAM_MEMORY_CONFIG,
        )
        for i, value in enumerate(references)
    ]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    output = ttnn.concat(inputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=grid)
    assert torch.equal(ttnn.to_torch(output), torch.cat(references, dim=1))


@pytest.mark.parametrize("case", ["staged_l1", "uint8_nonempty"])
def test_concat_mixed_layout_staged_source(device, case):
    is_uint8 = case == "uint8_nonempty"
    widths = [32, 33] if is_uint8 else [33, 32]
    dtype = ttnn.uint8 if is_uint8 else ttnn.bfloat16
    height = 32 if is_uint8 else 4
    references = [
        (torch.arange(height * width).reshape(height, width) % 97 + i * 100).to(
            torch.uint8 if is_uint8 else torch.bfloat16
        )
        for i, width in enumerate(widths)
    ]
    if is_uint8:
        # Isolate concat from the existing UINT8 untilize value-conversion defect.
        references[0].zero_()
    inputs = [
        ttnn.from_torch(
            value,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT if i == 0 else ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if i == (1 if is_uint8 else 0) else ttnn.L1_MEMORY_CONFIG,
        )
        for i, value in enumerate(references)
    ]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    output = ttnn.concat(inputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=grid)
    assert torch.equal(ttnn.to_torch(output), torch.cat(references, dim=1))

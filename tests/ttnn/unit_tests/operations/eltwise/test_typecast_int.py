# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


# use case for TG Llama : need to achieve (int32 + int32) addition with (uint16 + int32) inputs
def test_typecast_uint16(device):
    torch.manual_seed(0)

    in_data1 = torch.tensor([[[[700, 100, 65000, 9500]]]], dtype=torch.int32)
    in_data2 = torch.tensor([[[[70000, 1000, 65000, 95000]]]], dtype=torch.int32)

    input_mem_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )
    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    input_tensor3 = ttnn.typecast(
        input_tensor1,
        ttnn.uint32,
        memory_config=input_mem_config,
    )

    input_tensor3 = ttnn.typecast(
        input_tensor3,
        ttnn.int32,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.add(input_tensor3, input_tensor2)

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)
    golden_function = ttnn.get_golden_function(ttnn.add)
    golden_tensor = golden_function(in_data1, in_data2)

    assert torch.equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "shape, sub_core_grid",
    [
        (
            (torch.Size([1, 2, 32, 960])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 7, 32, 96])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 8, 32, 128])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 17, 32, 32])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
    ],
)
def test_typecast_subcore_grid(device, shape, sub_core_grid):
    torch.manual_seed(0)

    in_data1 = torch.randint(0, 65500, (shape), dtype=torch.int32)
    in_data2 = torch.randint(0, 128000, (shape), dtype=torch.int32)

    input_mem_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )
    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    input_tensor3 = ttnn.typecast(
        input_tensor1,
        ttnn.uint32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )

    input_tensor3 = ttnn.typecast(
        input_tensor3,
        ttnn.int32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )

    output_tensor = ttnn.add(input_tensor3, input_tensor2)

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)
    golden_function = ttnn.get_golden_function(ttnn.add)
    golden_tensor = golden_function(in_data1, in_data2)

    assert torch.equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "shape, sub_core_grid",
    [
        # Large tensors: many tiles per core stresses the CB allocation.
        # Before the fix, TypecastSubgridProgramFactory allocated all per-core tiles
        # into CBs at once (ntiles_per_block * 2), overflowing L1 for large tensors.
        (
            torch.Size([1, 1, 1024, 2048]),  # 2048 tiles, ~1024 tiles/core with 2 cores
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))]),
        ),
        (
            torch.Size([1, 2, 2048, 2048]),  # 8192 tiles across 7 cores
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 6))]),
        ),
        (
            torch.Size([1, 4, 1024, 2048]),  # 8192 tiles across a multi-range sub_core_grid
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
    ],
)
def test_typecast_subcore_grid_large_tensor(device, shape, sub_core_grid):
    """Regression test: large tensors with sub_core_grids must not overflow L1."""
    torch.manual_seed(0)

    in_data = torch.randint(0, 65500, shape, dtype=torch.int32)
    input_mem_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.typecast(
        input_tensor,
        ttnn.uint32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )

    output_tensor = ttnn.typecast(
        output_tensor,
        ttnn.int32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )

    result = ttnn.to_torch(output_tensor, dtype=torch.int32)
    assert torch.equal(in_data, result)


@pytest.mark.parametrize(
    "shape, output_dtype, sub_core_grid",
    [
        (
            torch.Size([1, 1, 1024, 2048]),
            ttnn.bfloat8_b,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 6))]),
        ),
        (
            torch.Size([1, 1, 1024, 2048]),
            ttnn.bfloat4_b,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 6))]),
        ),
        (
            torch.Size([1, 4, 2048, 2048]),
            ttnn.bfloat8_b,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
        (
            torch.Size([1, 4, 2048, 2048]),
            ttnn.bfloat4_b,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
    ],
)
def test_typecast_bfloat_subcore_grid_large_tensor(device, shape, output_dtype, sub_core_grid):
    """Regression test: bfloat16 -> bfloat8_b/bfloat4_b typecast with sub_core_grids on large tensors."""
    torch.manual_seed(0)

    in_data = torch.randn(shape, dtype=torch.bfloat16)
    input_mem_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.typecast(
        input_tensor,
        output_dtype,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )
    assert output_tensor.dtype == output_dtype
    assert list(output_tensor.shape) == list(shape)

    roundtrip = ttnn.typecast(
        output_tensor,
        ttnn.bfloat16,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )
    result = ttnn.to_torch(roundtrip).to(torch.float32)
    reference = in_data.to(torch.float32)

    pcc = torch.corrcoef(torch.stack([reference.flatten(), result.flatten()]))[0, 1].item()
    min_pcc = 0.95 if output_dtype == ttnn.bfloat4_b else 0.99
    assert pcc >= min_pcc, f"PCC {pcc:.4f} below threshold {min_pcc} for {output_dtype}"


# for range verification in conversions
def test_typecast_uint16_subcore_grid(device):
    in_data1 = torch.tensor([[[[700, 100, 65000, 9500]]]], dtype=torch.int32)
    in_data2 = torch.tensor([[[[70000, 1000, 65000, 95000]]]], dtype=torch.int32)

    input_mem_config = ttnn.DRAM_MEMORY_CONFIG
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
        ]
    )

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )
    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    input_tensor3 = ttnn.typecast(
        input_tensor1,
        ttnn.uint32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grids,
    )
    input_tensor3 = ttnn.typecast(
        input_tensor3,
        ttnn.int32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grids,
    )
    output_tensor = ttnn.add(input_tensor3, input_tensor2)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)
    golden_function = ttnn.get_golden_function(ttnn.add)
    golden_tensor = golden_function(in_data1, in_data2)

    assert torch.equal(golden_tensor, output_tensor)

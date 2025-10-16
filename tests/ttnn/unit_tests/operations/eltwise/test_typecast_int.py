# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

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

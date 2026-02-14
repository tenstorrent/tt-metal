# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

torch.manual_seed(0)


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 2**15, shape, dtype=torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


# ────────────────────────────────────────────────────────────────────────────────
# Test: Tile-layout pad with sub_core_grids
# ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "shape, padding, torch_padding",
    [
        # Height padding only (TILE-aligned)
        ((1, 1, 32, 32), ((0, 0), (0, 0), (0, 32), (0, 0)), (0, 0, 0, 32, 0, 0, 0, 0)),
        # Width padding only (TILE-aligned)
        ((1, 1, 32, 32), ((0, 0), (0, 0), (0, 0), (0, 32)), (0, 32, 0, 0, 0, 0, 0, 0)),
        # Height and width padding
        ((1, 1, 32, 32), ((0, 0), (0, 0), (0, 32), (0, 32)), (0, 32, 0, 32, 0, 0, 0, 0)),
        # Channel padding
        ((1, 1, 32, 64), ((0, 0), (0, 1), (0, 0), (0, 0)), (0, 0, 0, 0, 0, 1, 0, 0)),
        # Larger tensor with height+width padding
        ((2, 3, 64, 64), ((0, 0), (0, 0), (0, 64), (0, 64)), (0, 64, 0, 64, 0, 0, 0, 0)),
        # Small tensor padded significantly
        ((1, 1, 32, 32), ((0, 0), (0, 0), (0, 96), (0, 96)), (0, 96, 0, 96, 0, 0, 0, 0)),
    ],
)
@pytest.mark.parametrize(
    "sub_core_grid",
    [
        # Single contiguous range
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))]),
        # Two disjoint ranges (galaxy-style)
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            ]
        ),
        # Smaller single range
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))]),
    ],
)
@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_pad_tile_subcoregrids(device, shape, padding, torch_padding, sub_core_grid, value, dtype):
    """Test tile-layout pad with sub_core_grids parameter."""
    torch.manual_seed(0)

    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.pad(
        input_tensor,
        padding=padding,
        value=value,
        use_multicore=True,
        sub_core_grids=sub_core_grid,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


# ────────────────────────────────────────────────────────────────────────────────
# Test: Row-major-layout pad with sub_core_grids
# ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "shape, padding, torch_padding",
    [
        # Height and channel padding (RM)
        ((1, 3, 230, 224), ((0, 0), (0, 1), (0, 25), (0, 32)), (0, 32, 0, 25, 0, 1, 0, 0)),
        # Width only padding
        ((1, 1, 32, 100), ((0, 0), (0, 0), (0, 0), (0, 28)), (0, 28, 0, 0, 0, 0, 0, 0)),
        # Height only padding
        ((1, 1, 50, 64), ((0, 0), (0, 0), (0, 14), (0, 0)), (0, 0, 0, 14, 0, 0, 0, 0)),
        # Small padding
        ((1, 1, 16, 16), ((0, 0), (0, 0), (0, 4), (0, 4)), (0, 4, 0, 4, 0, 0, 0, 0)),
    ],
)
@pytest.mark.parametrize(
    "sub_core_grid",
    [
        # Single contiguous range
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))]),
        # Two disjoint ranges (galaxy-style)
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            ]
        ),
    ],
)
@pytest.mark.parametrize("value", [0])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_pad_rm_subcoregrids(device, shape, padding, torch_padding, sub_core_grid, value, dtype):
    """Test row-major-layout pad with sub_core_grids parameter."""
    torch.manual_seed(0)

    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.pad(
        input_tensor,
        padding=padding,
        value=value,
        use_multicore=True,
        sub_core_grids=sub_core_grid,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


# ────────────────────────────────────────────────────────────────────────────────
# Test: Tile-layout pad with sub_core_grids using legacy API (shape-based)
# ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "shape, padded_shape",
    [
        [(1, 1, 32, 32), (1, 1, 64, 64)],
        [(2, 3, 64, 64), (2, 3, 128, 128)],
    ],
)
@pytest.mark.parametrize(
    "sub_core_grid",
    [
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))]),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            ]
        ),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_pad_tile_subcoregrids_legacy_api(device, shape, padded_shape, sub_core_grid, dtype):
    """Test tile-layout pad with sub_core_grids using legacy shape-based API."""
    torch.manual_seed(0)

    torch_input_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.pad(
        input_tensor,
        padded_shape,
        [0, 0, 0, 0],
        0,
        use_multicore=True,
        sub_core_grids=sub_core_grid,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch.Size(padded_shape)
    # Verify original data is preserved in the non-padded region
    output_slice = output_tensor[: shape[0], : shape[1], : shape[2], : shape[3]]
    assert torch.equal(torch_input_tensor, output_slice)


# ────────────────────────────────────────────────────────────────────────────────
# Test: sub_core_grids with different data types
# ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.float32],
)
def test_pad_tile_subcoregrids_dtypes(device, dtype):
    """Test tile-layout pad with sub_core_grids across all supported dtypes."""
    torch.manual_seed(0)

    shape = (1, 1, 32, 32)
    padding = ((0, 0), (0, 0), (0, 32), (0, 32))
    torch_padding = (0, 32, 0, 32, 0, 0, 0, 0)
    value = 0
    sub_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))])

    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.pad(
        input_tensor,
        padding=padding,
        value=value,
        use_multicore=True,
        sub_core_grids=sub_core_grid,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


# ────────────────────────────────────────────────────────────────────────────────
# Test: Program cache hit with sub_core_grids
# ────────────────────────────────────────────────────────────────────────────────
def test_pad_subcoregrids_program_cache(device):
    """Verify program cache hit when calling pad with sub_core_grids twice."""
    torch.manual_seed(0)
    shape = (1, 1, 32, 32)
    padding = ((0, 0), (0, 0), (0, 32), (0, 32))
    sub_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))])

    for _ in range(2):
        torch_input = torch.rand(shape).bfloat16().float()
        input_tensor = ttnn.from_torch(
            torch_input,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output_tensor = ttnn.pad(
            input_tensor,
            padding=padding,
            value=0,
            use_multicore=True,
            sub_core_grids=sub_core_grid,
        )
        output_tensor = ttnn.to_torch(output_tensor)

        # Insert a dummy tensor to change allocation
        dummy = ttnn.from_torch(
            torch.randn(1, 1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    assert device.num_program_cache_entries() == 1


# ────────────────────────────────────────────────────────────────────────────────
# Test: Validation — sub_core_grids with sharded input should fail
# ────────────────────────────────────────────────────────────────────────────────
def test_pad_subcoregrids_rejects_sharded(device):
    """Verify that sub_core_grids with sharded input raises an error."""
    torch_input = torch.rand(1, 1, 64, 64).bfloat16()
    input_tensor = ttnn.from_torch(
        torch_input,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Shard the input
    num_cores_y = min(4, device.core_grid.y)
    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, num_cores_y - 1))])
    num_cores = 4 * num_cores_y
    shard_h = (64 + num_cores - 1) // num_cores
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 64), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem)

    sub_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    with pytest.raises(RuntimeError):
        ttnn.pad(
            input_tensor,
            padding=((0, 0), (0, 0), (0, 32), (0, 0)),
            value=0,
            use_multicore=True,
            sub_core_grids=sub_core_grid,
        )


# ────────────────────────────────────────────────────────────────────────────────
# Test: Validation — sub_core_grids with use_multicore=False should fail
# ────────────────────────────────────────────────────────────────────────────────
def test_pad_subcoregrids_rejects_singlecore(device):
    """Verify that sub_core_grids with use_multicore=False raises an error."""
    torch_input = torch.rand(1, 1, 32, 32).bfloat16()
    input_tensor = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sub_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    with pytest.raises(RuntimeError):
        ttnn.pad(
            input_tensor,
            padding=((0, 0), (0, 0), (0, 32), (0, 32)),
            value=0,
            use_multicore=False,
            sub_core_grids=sub_core_grid,
        )


# ────────────────────────────────────────────────────────────────────────────────
# Test: sub_core_grids=None should behave identically to the default path
# ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_pad_subcoregrids_none_matches_default(device, layout):
    """Verify that passing sub_core_grids=None produces the same result as the default."""
    torch.manual_seed(0)
    shape = (1, 1, 32, 64)
    padding = ((0, 0), (0, 0), (0, 32), (0, 32))
    torch_padding = (0, 32, 0, 32, 0, 0, 0, 0)
    value = 0

    torch_input = torch.rand(shape).bfloat16().float()
    torch_output = torch.nn.functional.pad(torch_input, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(
        torch_input,
        layout=layout,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # With sub_core_grids=None (explicit)
    output_none = ttnn.pad(
        input_tensor,
        padding=padding,
        value=value,
        use_multicore=True,
        sub_core_grids=None,
    )
    output_none = ttnn.to_torch(output_none)

    # Without sub_core_grids at all (default)
    output_default = ttnn.pad(
        input_tensor,
        padding=padding,
        value=value,
        use_multicore=True,
    )
    output_default = ttnn.to_torch(output_default)

    assert torch.equal(output_none, output_default)
    assert output_none.shape == torch_output.shape
    assert_with_pcc(torch_output, output_none, 0.9999)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal


def _bf16_pattern(shape, offset):
    numel = 1
    for dim in shape:
        numel *= dim
    return ((torch.arange(numel) + offset) % 64).to(torch.bfloat16).reshape(shape)


def _height_sharded_memory_config(device, shard_height, shard_width, num_cores):
    core_grid = device.compute_with_storage_grid_size()
    if core_grid.x * core_grid.y < num_cores:
        pytest.skip(f"Need {num_cores} cores, device has {core_grid.x * core_grid.y}")
    grid = ttnn.num_cores_to_corerangeset(num_cores, core_grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(grid, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


@pytest.mark.parametrize(
    "layout, shape, slice_ends",
    [
        # Full row-major bounds with shard width == input width hit the no-op in
        # padded_slice.cpp and return through to_memory_config. Cut the height so both
        # calls dispatch PaddedSliceRMProgramFactory.
        (ttnn.ROW_MAJOR_LAYOUT, (1, 1, 16, 32), (1, 1, 8, 32)),
        (ttnn.TILE_LAYOUT, (1, 1, 64, 32), (1, 1, 64, 32)),
    ],
    ids=["rm", "tile"],
)
def test_padded_slice_cache_hit(layout, shape, slice_ends, device):
    """A second allocation with the same slice bounds must hit the cached program and read the new buffers.

    Row-major selects PaddedSliceRMProgramFactory, which rewrites the reader address and the output CB.
    Tiled selects PaddedSliceTileProgramFactory, which binds the reader buffer and the output CB.
    The row-major case slices height so it cannot take the full-tensor no-op path.
    """
    num_cores = 2
    begins = [0, 0, 0, 0]
    ends = list(slice_ends)
    step = [1, 1, 1, 1]
    sliced_shape = [end - begin for begin, end in zip(begins, ends)]
    output_height = sliced_shape[0] * sliced_shape[1] * sliced_shape[2]
    output_width = sliced_shape[3]
    shard_height = output_height // num_cores
    output_mem_config = _height_sharded_memory_config(device, shard_height, output_width, num_cores)

    def allocate_input(offset):
        src = _bf16_pattern(shape, offset)
        tt_in = ttnn.from_torch(
            src, device=device, layout=layout, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return src, tt_in

    def reference(src):
        return src[begins[0] : ends[0], begins[1] : ends[1], begins[2] : ends[2], begins[3] : ends[3]]

    src_a, tt_in_a = allocate_input(0)
    tt_out_a = ttnn.experimental.padded_slice(tt_in_a, begins, ends, step, memory_config=output_mem_config)
    assert list(tt_out_a.shape) == [1, 1, output_height, output_width]
    assert_equal(reference(src_a), ttnn.to_torch(tt_out_a))

    src_b, tt_in_b = allocate_input(17)
    assert tt_in_b.buffer_address() != tt_in_a.buffer_address(), "second input must land at a new address"
    entries = device.num_program_cache_entries()
    tt_out_b = ttnn.experimental.padded_slice(tt_in_b, begins, ends, step, memory_config=output_mem_config)
    assert list(tt_out_b.shape) == [1, 1, output_height, output_width]
    assert tt_out_b.buffer_address() != tt_out_a.buffer_address(), "second output must land at a new address"
    assert device.num_program_cache_entries() == entries
    assert_equal(reference(src_b), ttnn.to_torch(tt_out_b))


def _last_dim_padded_slice_setup(device, begins, ends, shard_shape, core_grid):
    requested_num_cores = core_grid.num_cores()
    device_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    if device_num_cores < requested_num_cores:
        pytest.skip(f"Not enough cores to run test case (need {requested_num_cores} but have {device_num_cores})")

    torch.manual_seed(0)
    torch_input = torch.randn([1, 1, 64, 64], dtype=torch.bfloat16)
    sliced = torch_input[..., begins[3] : ends[3]]
    pad = shard_shape[1] - sliced.shape[-1]
    expected = torch.nn.functional.pad(sliced, (0, pad)) if pad > 0 else sliced

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    return tt_input, output_mem_config, expected


# Last-dim slice with output padding.
@pytest.mark.parametrize(
    "begins, ends, shard_shape, core_grid",
    (
        (
            [0, 0, 0, 16],
            [1, 1, 64, 64],
            (8, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        ),
        (
            [0, 0, 0, 0],
            [1, 1, 64, 40],
            (8, 48),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        ),
    ),
)
def test_padded_slice_rm_last_dim_pads_sliced_row(device, begins, ends, shard_shape, core_grid):
    tt_input, output_mem_config, expected = _last_dim_padded_slice_setup(device, begins, ends, shard_shape, core_grid)
    actual = ttnn.experimental.padded_slice(tt_input, begins, ends, [1, 1, 1, 1], memory_config=output_mem_config)
    passed, message = assert_equal(expected, ttnn.to_torch(actual))
    assert passed, message


# Last-dim offset, shard width == sliced width: Quasar read bound without the pad writer.
def test_quasar_padded_slice_rm_last_dim_no_pad(device):
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    begins, ends, shard_shape = [0, 0, 0, 16], [1, 1, 64, 64], (8, 48)
    tt_input, output_mem_config, expected = _last_dim_padded_slice_setup(device, begins, ends, shard_shape, core_grid)
    actual = ttnn.experimental.quasar.padded_slice(
        tt_input, begins, ends, [1, 1, 1, 1], memory_config=output_mem_config
    )
    passed, message = assert_equal(expected, ttnn.to_torch(actual))
    assert passed, message


# Sliced last dim narrower than the output shard: Quasar pad-row path is not ported.
def test_quasar_padded_slice_rm_last_dim_pad_rejected(device, expect_error):
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    begins, ends, shard_shape = [0, 0, 0, 16], [1, 1, 64, 64], (8, 64)
    tt_input, output_mem_config, _ = _last_dim_padded_slice_setup(device, begins, ends, shard_shape, core_grid)
    with expect_error(RuntimeError, r"pad-row path \(output_row > slice_row\) not yet ported"):
        ttnn.experimental.quasar.padded_slice(tt_input, begins, ends, [1, 1, 1, 1], memory_config=output_mem_config)


# The row-major pad writer zero-filled its scratch page only for 2- and 4-byte elements, so a uint8
# output copied stale L1 into the padding columns.
@pytest.mark.requires_grid_size((8, 1))
def test_padded_slice_rm_uint8_pads_with_zeros(device):
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    begins, ends, shard_shape = [0, 0, 0, 0], [1, 1, 64, 40], (8, 64)

    # Leave non-zero data behind in L1 for a pad page that is never cleared to pick up.
    for _ in range(2):
        poison = ttnn.from_torch(
            torch.full([1, 1, 512, 512], 0x7B7B, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn.add(poison, poison))
        ttnn.deallocate(poison)

    torch.manual_seed(0)
    torch_input = torch.randint(1, 256, [1, 1, 64, 64], dtype=torch.uint8)
    expected = torch.nn.functional.pad(torch_input[..., :40], (0, shard_shape[1] - 40))
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    actual = ttnn.experimental.padded_slice(tt_input, begins, ends, [1, 1, 1, 1], memory_config=output_mem_config)
    passed, message = assert_equal(expected, ttnn.to_torch(actual))
    assert passed, message

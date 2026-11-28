# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import torch
import ttnn

from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    UnreadableDocstring,
    check_readable,
    docstring_blackhole_dtype_layout_params,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
    interleaved_memory_config,
)
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact, full_range_input


def _docstring_aligned_split_dtypes():
    """Dtypes the ttnn.split docstring says need a rank of 2 or more and, in the last two dims, chunks whose sizes are
    multiples of 32, except the last."""
    names = re.findall(r"``(\w+)``\s+tensors\s+need\s+a\s+rank\s+of\s+2\s+or\s+more", ttnn.split.__doc__)
    dtypes = {getattr(ttnn.DataType, name, None) for name in names}
    if not names or not all(isinstance(dtype, ttnn.DataType) for dtype in dtypes):
        return UnreadableDocstring(f"could not read the dtypes that need a rank of 2 or more from: {names}")
    return dtypes


def _chunk_sizes(length, split_size):
    if isinstance(split_size, int):
        return [min(split_size, length - start) for start in range(0, length, split_size)]
    return list(split_size)


# (shape, split_size, dim). For TILE inputs the first two run the native split kernel (an equal split of the last dim
# into chunks of 32: 3 chunks of a batch-1 input, 2 chunks of a batch-2 input) and the rest run slices: an outer dim
# with a smaller last chunk, the second-to-last dim into chunks of 32, a list whose chunks start at multiples of 32,
# chunks of sizes that are not multiples of 32 in each of the last two dims, and a rank-1 input.
_DOCSTRING_SPLIT_CASES = [
    ((1, 3, 64, 96), 32, -1),
    ((2, 3, 64, 64), 32, -1),
    ((2, 3, 64, 96), 2, 1),
    ((2, 3, 64, 96), 32, -2),
    ((2, 3, 64, 96), [64, 32], -1),
    ((2, 3, 64, 96), 24, -1),
    ((2, 3, 64, 96), [22, 42], -2),
    ((100,), 40, 0),
]


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.split))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.split))
def test_split_docstring_supported_inputs(dtype, layout, buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.split docstring lists, with int and list split
    sizes along outer and inner dims, and compares each chunk exactly with torch.split."""
    check_readable(dtype, layout, buffer_type)
    aligned_dtypes = _docstring_aligned_split_dtypes()
    check_readable(aligned_dtypes)
    memory_config = interleaved_memory_config(buffer_type)
    for shape, split_size, dim in _DOCSTRING_SPLIT_CASES:
        rank = len(shape)
        sizes = _chunk_sizes(shape[dim], split_size)
        aligned = dim % rank < rank - 2 or all(size % 32 == 0 for size in sizes[:-1])
        if dtype in aligned_dtypes and (rank == 1 or not aligned):
            continue
        # A TILE input of rank 1, or one split in its last two dims into chunks that do not start at multiples of 32,
        # is re-tiled through ROW_MAJOR by compute kernels that hold BFLOAT16 zeros by value. There a BFLOAT16 input
        # leaves out NaN, as the docstring requires, and -0.0 as well.
        bfloat16_retiled = dtype == ttnn.bfloat16 and layout == ttnn.TILE_LAYOUT and (rank == 1 or not aligned)
        # preserve_nan_values keeps full_range_input's NaN in the input: a BFLOAT16 input built on device can lose it.
        input_tensor = ttnn.from_torch(
            full_range_input(shape, dtype, nan=not bfloat16_retiled, negative_zero=not bfloat16_retiled),
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=memory_config,
            preserve_nan_values=True,
        )
        expected = torch.split(ttnn.to_torch(input_tensor), split_size, dim)

        output = ttnn.split(input_tensor, split_size, dim)

        case = f"shape {shape}, split_size {split_size}, dim {dim}"
        assert isinstance(output, list), case
        assert len(output) == len(expected), case
        for expected_chunk, output_chunk in zip(expected, output):
            assert output_chunk.dtype == dtype, case
            assert output_chunk.layout == layout, case
            assert output_chunk.memory_config().buffer_type == buffer_type, case
            assert_exact(expected_chunk, ttnn.to_torch(output_chunk))


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.split))
@pytest.mark.parametrize("dtype, layout", docstring_blackhole_dtype_layout_params(ttnn.split))
def test_split_docstring_blackhole_inputs(dtype, layout, buffer_type, device):
    """Runs the dtypes the ttnn.split docstring lists for Blackhole, in every interleaved buffer type it lists, on the
    splits of test_split_docstring_supported_inputs. Values are compared after a conversion to FLOAT32 on host, which
    holds every FP8_E4M3 value."""
    check_readable(dtype, layout, buffer_type)
    if not is_blackhole():
        pytest.skip("the docstring lists these dtypes for Blackhole")

    def read(tensor):
        return ttnn.to_torch(ttnn.to_dtype(tensor.cpu(), ttnn.float32))

    memory_config = interleaved_memory_config(buffer_type)
    for shape, split_size, dim in _DOCSTRING_SPLIT_CASES:
        input_tensor = ttnn.from_torch(
            full_range_input(shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=memory_config
        )
        expected = torch.split(read(input_tensor), split_size, dim)

        output = ttnn.split(input_tensor, split_size, dim)

        case = f"shape {shape}, split_size {split_size}, dim {dim}"
        assert len(output) == len(expected), case
        for expected_chunk, output_chunk in zip(expected, output):
            assert output_chunk.dtype == dtype, case
            assert output_chunk.layout == layout, case
            assert output_chunk.memory_config().buffer_type == buffer_type, case
            assert_exact(expected_chunk, read(output_chunk))


# A (1, 1, 64, 128) input on two or four cores: HEIGHT_SHARDED in (32, 128) shards, WIDTH_SHARDED in (64, 64) and
# BLOCK_SHARDED in (32, 64).
_SPLIT_SHARD_GRIDS = {
    ttnn.ShardStrategy.HEIGHT: ttnn.CoreGrid(y=2, x=1),
    ttnn.ShardStrategy.WIDTH: ttnn.CoreGrid(y=1, x=2),
    ttnn.ShardStrategy.BLOCK: ttnn.CoreGrid(y=2, x=2),
}


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("strategy", list(_SPLIT_SHARD_GRIDS), ids=lambda strategy: strategy.name)
def test_split_docstring_sharded_inputs(strategy, layout, device):
    """The Sharded item of the ttnn.split docstring's Memory Support and its ``memory_config`` default. With the default,
    an equal split of a HEIGHT_SHARDED, WIDTH_SHARDED or BLOCK_SHARDED L1 input keeps the memory layout, buffer type
    and cores, with the shard shape scaled like the chunk shape; a scaled shard that is not whole tiles for a TILE
    input, and an unequal split, give DRAM interleaved outputs. A given sharded or interleaved ``memory_config`` is
    kept, and an interleaved input splits into a given sharded one."""
    shape = (1, 1, 64, 128)
    grid = _SPLIT_SHARD_GRIDS[strategy]

    def sharded(shard_shape_of):
        return ttnn.create_sharded_memory_config(
            shard_shape_of, core_grid=grid, strategy=strategy, use_height_and_width_as_shard_shape=False
        )

    input_config = sharded(shape)
    input_tensor = ttnn.from_torch(
        full_range_input(shape, ttnn.bfloat16),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=input_config,
        preserve_nan_values=True,
    )
    input_shard = list(input_config.shard_spec.shape)

    def check(split_size, dim, expected_config_of, source=input_tensor, **kwargs):
        expected = torch.split(ttnn.to_torch(source), split_size, dim)
        output = ttnn.split(source, split_size, dim, **kwargs)
        assert len(output) == len(expected)
        for expected_chunk, output_chunk in zip(expected, output):
            want = expected_config_of(expected_chunk.shape)
            assert output_chunk.memory_config() == want, (split_size, dim, output_chunk.memory_config(), want)
            assert_exact(expected_chunk, ttnn.to_torch(output_chunk))

    def scaled(chunk_shape):
        # the input's shard shape, scaled like the chunk shape, on the same cores
        height = input_shard[0] * chunk_shape[-2] // shape[-2]
        width = input_shard[1] * chunk_shape[-1] // shape[-1]
        spec = ttnn.ShardSpec(input_config.shard_spec.grid, (height, width), input_config.shard_spec.orientation)
        return ttnn.MemoryConfig(input_config.memory_layout, ttnn.BufferType.L1, spec)

    # an equal split of the last dim: the scaled shards are whole tiles
    check(64, -1, scaled)
    # an equal split of the second-to-last dim halves the shard height: whole tiles only for BLOCK and WIDTH
    tile_rows_halved = input_shard[0] // 2 % 32 != 0
    dram = lambda chunk_shape: ttnn.DRAM_MEMORY_CONFIG
    check(32, -2, dram if layout == ttnn.TILE_LAYOUT and tile_rows_halved else scaled)
    # an unequal split
    check([96, 32], -1, dram)
    # a given sharded memory_config, built for the chunk shape, and a given interleaved one
    chunk_config = sharded((1, 1, 64, 64))
    check(64, -1, lambda chunk_shape: chunk_config, memory_config=chunk_config)
    check([96, 32], -1, lambda chunk_shape: ttnn.L1_MEMORY_CONFIG, memory_config=ttnn.L1_MEMORY_CONFIG)
    interleaved_input = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
    check(64, -1, lambda chunk_shape: chunk_config, source=interleaved_input, memory_config=chunk_config)

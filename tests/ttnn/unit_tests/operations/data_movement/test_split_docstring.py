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
        # leaves out NaN, as the docstring requires, and its zeros are compared by value.
        bfloat16_retiled = dtype == ttnn.bfloat16 and layout == ttnn.TILE_LAYOUT and (rank == 1 or not aligned)
        # preserve_nan_values keeps full_range_input's NaN in the input: a BFLOAT16 input built on device can lose it.
        input_tensor = ttnn.from_torch(
            full_range_input(shape, dtype, nan=not bfloat16_retiled),
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
            assert_exact(expected_chunk, ttnn.to_torch(output_chunk), signed_zeros=not bfloat16_retiled)


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
    torch.manual_seed(0)
    for shape, split_size, dim in _DOCSTRING_SPLIT_CASES:
        input_tensor = ttnn.from_torch(
            torch.randn(shape) * 8, dtype=dtype, layout=layout, device=device, memory_config=memory_config
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

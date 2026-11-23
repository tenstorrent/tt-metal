# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import re

import pytest
import torch
import ttnn
import math
from tests.ttnn.utils_for_testing import assert_equal
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    check_readable,
    docstring_blackhole_dtype_layout_params,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
    interleaved_memory_config,
)
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact, full_range_input


@pytest.mark.parametrize(
    "shape, num_chunks, dim",
    [
        ((50,), 7, 0),
        ((200,), 1, 0),
        ((4, 3), 5, 0),
        ((2, 4, 3), 2, 0),
        ((2, 4, 3), 2, 1),
        ((2, 3, 4, 2), 4, 2),
        ((2, 2, 2), 2, 0),
        ((10, 1, 1), 5, 0),
        ((2, 3, 4, 4, 2), 5, 3),
        ((1, 2, 2, 2, 2, 2), 3, 2),
        ((1, 2, 2, 2, 2, 2), 5, 4),
        ((1, 1, 1, 4, 2, 2), 5, 2),
        ((1, 1, 1, 2, 2, 3), 5, 3),
        ((1, 1, 1, 3, 2, 4), 5, 4),
        ((1, 1, 1, 3, 2, 4), 4, 2),
        ((1, 1, 1, 3, 2, 4), 6, 1),
        ((1, 1, 1, 3, 2, 4), 3, 5),
        ((1, 1, 1, 3, 2, 4), 7, 0),
        ((1, 1, 1, 4, 2, 4), 5, 4),
        ((1, 1, 1, 3, 2, 4), 8, 1),
    ],
)
def test_chunking(device, shape, num_chunks, dim):
    numel = math.prod(shape)
    tensor = torch.arange(numel, dtype=torch.float32).reshape(shape)
    ttnn_tensor = ttnn.from_torch(tensor)

    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    torch_chunks = torch.chunk(tensor, num_chunks, dim)
    ttnn_chunks = ttnn.chunk(ttnn_tensor, num_chunks, dim)

    ttnn_chunks_torch = [ttnn.to_torch(chunk) for chunk in ttnn_chunks]
    assert_equal(torch.cat(torch_chunks, dim=-1), torch.cat(ttnn_chunks_torch, dim=-1))


def _docstring_aligned_chunk_dtypes():
    """Dtypes the ttnn.chunk docstring says need a rank of 2 or more and, in the last two dims, a chunk size that is a
    multiple of 32."""
    names = re.findall(r"``(\w+)``\s+tensors\s+need\s+a\s+rank\s+of\s+2\s+or\s+more", ttnn.chunk.__doc__)
    return {getattr(ttnn.DataType, name) for name in names}


# (shape, dim, chunks): an outer dim split into chunks of 2 and 1, each of the last two dims split into chunks of 32
# and into chunks of a size that is not a multiple of 32, and a rank-1 input.
_DOCSTRING_CHUNK_CASES = [
    ((2, 3, 64, 96), 1, 2),
    ((2, 3, 64, 96), -1, 3),
    ((2, 3, 64, 96), -2, 2),
    ((2, 3, 64, 96), -1, 4),
    ((2, 3, 64, 96), -2, 3),
    ((100,), 0, 3),
]


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.chunk))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.chunk))
def test_chunk_docstring_supported_inputs(dtype, layout, buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.chunk docstring lists, splitting an outer dim,
    each of the last two dims and a rank-1 input, and compares each chunk exactly with torch.chunk."""
    check_readable(dtype, layout, buffer_type)
    memory_config = interleaved_memory_config(buffer_type)
    for shape, dim, chunks in _DOCSTRING_CHUNK_CASES:
        rank = len(shape)
        aligned = dim % rank < rank - 2 or math.ceil(shape[dim] / chunks) % 32 == 0
        if dtype in _docstring_aligned_chunk_dtypes() and (rank == 1 or not aligned):
            continue
        # A TILE input of rank 1, or one split in its last two dims into chunks of a size that is not a multiple of
        # 32, is re-tiled through ROW_MAJOR by compute kernels that hold BFLOAT16 zeros by value. There a BFLOAT16
        # input leaves out NaN, as the docstring requires, and its zeros are compared by value.
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
        expected = torch.chunk(ttnn.to_torch(input_tensor), chunks, dim)

        output = ttnn.chunk(input_tensor, chunks, dim)

        assert isinstance(output, list)
        assert len(output) == len(expected), f"shape {shape}, dim {dim}, chunks {chunks}"
        for expected_chunk, output_chunk in zip(expected, output):
            assert output_chunk.dtype == dtype
            assert output_chunk.layout == layout
            assert output_chunk.memory_config().buffer_type == buffer_type
            assert_exact(expected_chunk, ttnn.to_torch(output_chunk), signed_zeros=not bfloat16_retiled)


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.chunk))
@pytest.mark.parametrize("dtype, layout", docstring_blackhole_dtype_layout_params(ttnn.chunk))
def test_chunk_docstring_blackhole_inputs(dtype, layout, buffer_type, device):
    """Runs the dtypes the ttnn.chunk docstring lists for Blackhole, in every interleaved buffer type it lists, on the
    splits of test_chunk_docstring_supported_inputs. Values are compared after a conversion to FLOAT32 on host, which
    holds every FP8_E4M3 value."""
    check_readable(dtype, layout, buffer_type)
    if not is_blackhole():
        pytest.skip("the docstring lists these dtypes for Blackhole")

    def read(tensor):
        return ttnn.to_torch(ttnn.to_dtype(tensor.cpu(), ttnn.float32))

    memory_config = interleaved_memory_config(buffer_type)
    torch.manual_seed(0)
    for shape, dim, chunks in _DOCSTRING_CHUNK_CASES:
        input_tensor = ttnn.from_torch(
            torch.randn(shape) * 8, dtype=dtype, layout=layout, device=device, memory_config=memory_config
        )
        expected = torch.chunk(read(input_tensor), chunks, dim)

        output = ttnn.chunk(input_tensor, chunks, dim)

        case = f"shape {shape}, dim {dim}, chunks {chunks}"
        assert len(output) == len(expected), case
        for expected_chunk, output_chunk in zip(expected, output):
            assert output_chunk.dtype == dtype, case
            assert output_chunk.layout == layout, case
            assert output_chunk.memory_config() == memory_config, case
            assert_exact(expected_chunk, read(output_chunk))

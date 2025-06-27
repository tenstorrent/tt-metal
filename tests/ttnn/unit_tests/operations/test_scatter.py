# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def select_torch_dtype(ttnn_dtype):
    if ttnn_dtype is ttnn.bfloat16:
        return torch.bfloat16
    if ttnn_dtype is ttnn.float32:
        return torch.float32
    if ttnn_dtype is ttnn.uint8:
        return torch.uint8
    if ttnn_dtype is ttnn.uint16:
        return torch.int64
    if ttnn_dtype is ttnn.int32:
        return torch.int64
    if ttnn_dtype is ttnn.uint32:
        return (
            torch.int64
        )  # !!! there is a strict requirement for the index tensor in Torch to be int64, and there is no int64 in ttnn


@pytest.mark.parametrize(
    "input_shape, dim, index_and_source_shape, input_dtype, index_dtype, layout",
    [
        ([100], -1, [80], ttnn.float32, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([50, 200], 0, [50, 200], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([2, 30, 200], -1, [2, 30, 200], ttnn.float32, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([1, 1, 20, 20, 200], -1, [1, 1, 20, 20, 20], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([2, 2, 2, 2, 2, 2, 2, 2], -1, [2, 2, 2, 2, 2, 2, 2, 2], ttnn.float32, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([10, 1, 10, 1, 10], 0, [10, 1, 10, 1, 10], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([10, 50, 10, 50, 10], -1, [10, 50, 10, 50, 10], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([10, 50, 10, 50, 100], -1, [10, 50, 10, 50, 100], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([700, 10, 30, 10], -1, [700, 10, 30, 10], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([700, 10, 30, 100], -1, [700, 10, 30, 10], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([1, 151936], -1, [1, 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR),
        ([100, 151936], -1, [100, 151936], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR),
        ([2, 10, 151936], -1, [2, 10, 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR),
        ([1, 151936], -1, [1, 151936], ttnn.float32, ttnn.uint32, ttnn.Layout.ROW_MAJOR),
        ([100, 151936], -1, [100, 151936], ttnn.bfloat16, ttnn.uint32, ttnn.Layout.ROW_MAJOR),
        ([2, 10, 151936], -1, [2, 10, 151936], ttnn.float32, ttnn.uint32, ttnn.Layout.ROW_MAJOR),
        ([1, 2, 3, 4, 5, 6, 7, 8], 1, [1, 2, 3, 4, 5, 6, 7, 8], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE),
        ([1, 2, 3, 4, 5, 6, 7, 8], 2, [1, 2, 3, 4, 5, 6, 7, 8], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([1, 2, 3, 4, 5, 6, 7, 8], 3, [1, 2, 3, 4, 5, 6, 7, 8], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE),
        ([1, 2, 3, 4, 5, 6, 7, 8], 4, [1, 2, 3, 4, 5, 6, 7, 8], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([1, 2, 3, 4, 5, 6, 7, 8], 5, [1, 2, 3, 4, 5, 6, 7, 8], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE),
        ([1, 2, 3, 4, 5, 6, 7, 8], 6, [1, 2, 3, 4, 5, 6, 7, 8], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([1, 2, 3, 4, 5, 6, 7, 8], 7, [1, 2, 3, 4, 5, 6, 7, 8], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE),
        ([1, 2, 3, 4, 5, 6, 7, 8], 0, [1, 2, 3, 4, 5, 6, 7, 8], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([10, 50, 10, 50, 100], -1, [10, 50, 10, 50, 100], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        ([50, 200], -1, [50, 200], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE),
        ##################
        # these cases fail due to the int32 transpose issue
        # ([50, 200], 0, [50, 200], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        # ([10, 50, 10, 50, 100], 0, [10, 50, 10, 50, 100], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        # ([50, 200], 0, [50, 200], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        ##################
        # these cases fail due to the to_layout precision issue (fp32 tiled <-> row-major) : #23405
        # ([10, 50, 10, 50, 100], -1, [10, 50, 10, 50, 100], ttnn.float32, ttnn.uint16, ttnn.Layout.TILE),
        # ([2, 30, 200], -1, [2, 30, 200], ttnn.float32, ttnn.uint16, ttnn.Layout.TILE),
        ##################
        # these cases fail due to the to_layout integer issue (integer dtype size>256 tiled -> row-major): #23407
        # ([1, 151936], -1, [1, 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        # ([100, 151936], -1, [100, 151936], ttnn.float32, ttnn.int32, ttnn.Layout.TILE),
        # ([2, 10, 151936], -1, [2, 10, 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        # ([1, 151936], -1, [1, 151936], ttnn.float32, ttnn.uint32, ttnn.Layout.TILE),
        # ([100, 151936], -1, [100, 151936], ttnn.bfloat16, ttnn.uint32, ttnn.Layout.TILE),
        # ([2, 10, 151936], -1, [2, 10, 151936], ttnn.float32, ttnn.uint32, ttnn.Layout.TILE),
    ],
)
def test_scatter_normal(input_shape, dim, index_and_source_shape, input_dtype, index_dtype, layout, device):
    torch.manual_seed(22052025)
    # https://github.com/tenstorrent/tt-metal/issues/23205
    device.disable_and_clear_program_cache()
    torch_dtype = select_torch_dtype(input_dtype)
    torch_index_dtype = select_torch_dtype(index_dtype)
    ##
    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=layout, device=device)

    torch_index = torch.randint(0, input_shape[dim], index_and_source_shape, dtype=torch_index_dtype)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=layout, device=device)

    torch_src = torch.randn(index_and_source_shape, dtype=torch_dtype)
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=layout, device=device)

    for _ in range(2):
        torch_result = torch.scatter(torch_input, dim, index=torch_index, src=torch_src)
        ttnn_result = ttnn.experimental.scatter(ttnn_input, dim, ttnn_index, ttnn_src)

        torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
        assert torch_result_from_ttnn.shape == torch_result.shape
        assert torch_result_from_ttnn.dtype == torch_result.dtype
        torch.testing.assert_close(torch_result_from_ttnn, torch_result)

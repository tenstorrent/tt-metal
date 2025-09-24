# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import _ttml  # noqa: E402


@pytest.mark.parametrize(
    "tensor_data, numpy_type, autograd_type, layout",
    [
        (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.TILE),
        (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.TILE),
        (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.TILE),
        (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.ROW_MAJOR),
        (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.INT32, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.INT32, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.INT32, _ttml.Layout.TILE),
        (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.INT32, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.INT32, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.INT32, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.UINT32, _ttml.Layout.TILE),
        (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.UINT32, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.UINT32, _ttml.Layout.TILE),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.int32, _ttml.autograd.DataType.UINT32, _ttml.Layout.ROW_MAJOR),
        (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.uint32, _ttml.autograd.DataType.UINT32, _ttml.Layout.ROW_MAJOR),
        # (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.float32, _ttml.autograd.DataType.UINT32, _ttml.Layout.ROW_MAJOR),
    ],
)
def test_numpy_autograd_conversion(tensor_data, numpy_type, autograd_type, layout):
    numpy_tensor = np.array(tensor_data, dtype=numpy_type)
    autograd_tensor = _ttml.autograd.Tensor.from_numpy(numpy_tensor, layout=layout, new_type=autograd_type)
    assert (autograd_tensor.to_numpy() == np.array(tensor_data, dtype=numpy_type)).all()

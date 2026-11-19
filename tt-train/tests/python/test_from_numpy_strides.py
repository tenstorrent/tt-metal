# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor.from_numpy reads an array by its strides, so views that are not C-contiguous round-trip exactly (#62364)."""

import ml_dtypes
import numpy as np
import pytest

import ttnn
import ttml

_BASE = np.arange(4 * 64, dtype=np.float32).reshape(1, 1, 4, 64)

VIEWS = {
    "contiguous": lambda a: a,
    "last_dim_slice": lambda a: a[..., :32],
    "last_dim_step": lambda a: a[..., ::2],
    "swapaxes": lambda a: np.swapaxes(a, 2, 3),
    "fortran_order": np.asfortranarray,
    "negative_stride_rows": lambda a: a[:, :, ::-1, :],
    "negative_stride_cols": lambda a: a[..., ::-1],
    "broadcast_zero_stride": lambda a: np.broadcast_to(a[:, :, :1, :], a.shape),
    "read_only": lambda a: _read_only(a[..., :32]),
}


def _read_only(a):
    a = a.copy()
    a.flags.writeable = False
    return a


@pytest.mark.requires_device
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
@pytest.mark.parametrize("view", list(VIEWS))
@pytest.mark.parametrize(
    "numpy_dtype,new_type",
    [(np.float32, ttnn.DataType.FLOAT32), (np.int32, ttnn.DataType.INT32), (np.float32, ttnn.DataType.BFLOAT16)],
    ids=["float32", "int32", "float32_to_bfloat16"],
)
def test_from_numpy_strided_view(view, layout, numpy_dtype, new_type):
    array = VIEWS[view](_BASE.astype(numpy_dtype))
    got = ttml.autograd.Tensor.from_numpy(array, layout, new_type).to_numpy(new_type=ttnn.DataType.FLOAT32)
    np.testing.assert_array_equal(got, np.asarray(array, dtype=np.float32))


@pytest.mark.requires_device
@pytest.mark.parametrize("view", list(VIEWS))
def test_from_numpy_strided_view_ml_dtypes_bfloat16(view):
    array = VIEWS[view](_BASE.astype(ml_dtypes.bfloat16))
    got = ttml.autograd.Tensor.from_numpy(array, ttnn.Layout.TILE).to_numpy(new_type=ttnn.DataType.FLOAT32)
    np.testing.assert_array_equal(got, np.asarray(array, dtype=np.float32))

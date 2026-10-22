# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Loading numpy values into an existing tensor, in the dtype the tensor is stored in."""

import ml_dtypes
import numpy as np
import ttnn

# Note: _ttml is a top-level module, not a subpackage of ttml
import _ttml as cpp

_NUMPY_DTYPES = {ttnn.DataType.FLOAT32: np.float32, ttnn.DataType.BFLOAT16: ml_dtypes.bfloat16}


def assign_numpy(tensor, array: np.ndarray, *, layout=ttnn.Layout.TILE, mapper=None) -> None:
    """Load ``array`` into ``tensor`` in the dtype ``tensor`` is stored in.

    The array is converted on the host to the tensor's dtype: an fp32 tensor gets the exact values, a bf16 tensor the
    values rounded with ml_dtypes. An array with fewer dimensions than ``tensor`` gets leading unit ones: a 2D weight
    loads into a 4D parameter as ``[1, 1, rows, cols]``, and an array of the tensor's rank keeps its shape. ``mapper``
    distributes the array as in ``Tensor.from_numpy``.
    """
    dtype = tensor.dtype()
    if dtype not in _NUMPY_DTYPES:
        raise ValueError(f"assign_numpy: unsupported tensor dtype {dtype} (only FLOAT32 and BFLOAT16)")
    # from_numpy reads the buffer in C order and rejects read-only arrays; copy only when the array is neither.
    values = np.require(array, dtype=_NUMPY_DTYPES[dtype], requirements=["C", "W"])
    values = values.reshape((1,) * max(0, tensor.get_rank() - values.ndim) + values.shape)
    tensor.assign(cpp.autograd.Tensor.from_numpy(values, layout=layout, new_type=dtype, mapper=mapper))

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

    The array is converted on the host, to bf16 with ml_dtypes, so an fp32 tensor gets the exact values and a bf16
    tensor gets the same values as before. ``mapper`` distributes the array as in ``Tensor.from_numpy``.
    """
    dtype = tensor.get_value(cpp.autograd.PreferredPrecision.NATIVE).dtype
    if dtype not in _NUMPY_DTYPES:
        raise ValueError(f"assign_numpy: unsupported tensor dtype {dtype} (only FLOAT32 and BFLOAT16)")
    # A C-ordered, writable copy: from_numpy reads the buffer in C order and rejects read-only arrays.
    values = np.array(array, dtype=_NUMPY_DTYPES[dtype], order="C")
    tensor.assign(cpp.autograd.Tensor.from_numpy(values, layout=layout, new_type=dtype, mapper=mapper))

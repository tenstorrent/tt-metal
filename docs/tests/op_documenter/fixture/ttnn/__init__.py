# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from _op_documenter_bindings import Tensor, LayerNorm, overloaded as _overloaded
from .decorators import FastOperation, Operation

Tensor.__module__ = "ttnn._ttnn.tensor"

layer_norm = FastOperation(
    "layer_norm",
    LayerNorm(),
    """Normalize an input tensor.

Args:
  input_tensor (ttnn.Tensor): The input tensor.

Keyword args:
  epsilon (float): 1e-12.
  compute_kernel_config (object) The kernel configuration.

Returns:
  ttnn.Tensor: The normalized tensor.
""",
)

overloaded = Operation("overloaded", _overloaded, "Overloaded function.")


def _transpose(input_tensor, dim1, dim2, *, memory_config=None):
    return input_tensor


transpose = Operation(
    "transpose",
    _transpose,
    """Transpose an input tensor.

Args:
    * :attr:`input_tensor`: The input tensor.
    * :attr:`dim1`: The first dimension.
    * :attr:`dim2`: The second dimension.
""",
)

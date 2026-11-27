# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host <-> device tensor helpers for the tt-train op tests."""

from __future__ import annotations

import numpy as np

import ttnn
import ttml

NATIVE = ttml.autograd.PreferredPrecision.NATIVE


def to_device(array, dtype="bf16", requires_grad=False):
    """``array`` as a TILE autograd tensor stored in bf16, or in fp32 with ``dtype="fp32"``."""
    new_type = ttnn.DataType.FLOAT32 if dtype == "fp32" else ttnn.DataType.BFLOAT16
    tensor = ttml.autograd.Tensor.from_numpy(array.astype(np.float32), layout=ttnn.Layout.TILE, new_type=new_type)
    tensor.set_requires_grad(requires_grad)
    return tensor


def value(tensor):
    """The device tensor as stored, so an fp32 input reaches the op as fp32."""
    return tensor.get_value(NATIVE)


def to_host(ttnn_tensor):
    return ttml.autograd.create_tensor(ttnn_tensor).to_numpy().astype(np.float64)


def as_stored(array, dtype="bf16"):
    """``array`` as held on device, so the oracle sees what the kernel received."""
    return to_host(value(to_device(array, dtype)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

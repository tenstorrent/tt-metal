# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for host-io micro-op and tests.
"""
import torch

import ttnn

TORCH_TO_TTNN_DTYPE = {
    torch.bfloat16: ttnn.bfloat16,
    torch.float32: ttnn.float32,
    torch.int32: ttnn.int32,
    torch.uint32: ttnn.uint32,
    torch.uint16: ttnn.uint16,
    torch.uint8: ttnn.uint8,
}

TTNN_TO_TORCH_DTYPE = {v: k for k, v in TORCH_TO_TTNN_DTYPE.items()}


def ttnn_dtype_from_torch_dtype(torch_dtype):
    if torch_dtype not in TORCH_TO_TTNN_DTYPE:
        raise ValueError(f"No ttnn equivalent for torch dtype: {torch_dtype}")
    return TORCH_TO_TTNN_DTYPE[torch_dtype]


def torch_dtype_from_ttnn_dtype(ttnn_dtype):
    if ttnn_dtype not in TTNN_TO_TORCH_DTYPE:
        raise ValueError(f"No torch equivalent for ttnn dtype: {ttnn_dtype}")
    return TTNN_TO_TORCH_DTYPE[ttnn_dtype]


def dtype_size(dtype):
    """Get element size in bytes for a torch or ttnn dtype."""
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    elif dtype in TTNN_TO_TORCH_DTYPE:
        return torch.tensor([], dtype=TTNN_TO_TORCH_DTYPE[dtype]).element_size()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

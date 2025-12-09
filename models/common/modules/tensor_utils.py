# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tensor utility functions for TTTv2 modules.
"""


# todo)) add a on-device pad_dim_to_size function?
def pad_dim_to_size(x: "torch.Tensor", dim: int, size: int) -> "torch.Tensor":
    """Pads the specified dimension of the input tensor with zeros."""
    if dim < 0:
        dim = x.dim() + dim
    current_size = x.size(dim)
    pad_size = size - current_size

    if pad_size < 0:
        raise ValueError(f"Target size {size} is smaller than current size {current_size} on dim {dim}")

    if pad_size == 0:
        return x

    pad = [0] * (2 * x.dim())
    pad_index = 2 * (x.dim() - dim - 1)
    pad[pad_index + 1] = pad_size

    import torch

    return torch.nn.functional.pad(x, pad, mode="constant", value=0)

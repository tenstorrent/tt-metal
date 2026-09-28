# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared (input_shape, other_shape) parametrisation for binary_backward tests (#56601).

PyTorch autograd contract for every ``ttnn.<op>_bw`` in the binary family:

    grad_output.shape == torch.broadcast_shapes(input.shape, other.shape)
    input_grad.shape  == input.shape
    other_grad.shape  == other.shape

Every row here exercises one interesting divergence between ``input.shape`` and
``other.shape``. Consumers should build ``grad`` via ``torch.broadcast_shapes``.
"""

import torch


# rows chosen to exercise every axis-divergence class autograd's AccumulateGrad handles
BINARY_BACKWARD_BROADCAST_MATRIX = [
    # (input_shape, other_shape, label)
    ((1, 3, 32, 64), (1, 3, 32, 64), "no_broadcast"),
    ((1, 1, 32, 64), (1, 3, 32, 64), "channel_bcast_on_input"),
    ((1, 3, 32, 64), (1, 1, 32, 64), "channel_bcast_on_other"),
    ((1, 3, 1, 64), (1, 3, 32, 64), "row_bcast_on_input"),
    ((1, 3, 32, 1), (1, 3, 32, 64), "col_bcast_on_input"),
    ((1, 3, 32, 64), (1, 1, 1, 1), "scalar_like_other"),
    ((1, 1, 1, 1), (1, 3, 32, 64), "scalar_like_input"),
    ((32, 64), (1, 3, 32, 64), "cross_rank_input_lower"),
    ((1, 3, 32, 64), (32, 64), "cross_rank_other_lower"),
]


def broadcast_grad_shape(input_shape, other_shape):
    """Autograd invariant: grad_output.shape == broadcast(input.shape, other.shape)."""
    return torch.broadcast_shapes(input_shape, other_shape)


def matrix_ids():
    """pytest ids matching BINARY_BACKWARD_BROADCAST_MATRIX for readable test names."""
    return [row[2] for row in BINARY_BACKWARD_BROADCAST_MATRIX]


def matrix_shapes():
    """Yield (input_shape, other_shape) pairs without the label — for parametrize values."""
    return [(row[0], row[1]) for row in BINARY_BACKWARD_BROADCAST_MATRIX]

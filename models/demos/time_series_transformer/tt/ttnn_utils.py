# tt/ttnn_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import ttnn

TILE = 32


def layer_norm_padded(x, weight, bias, orig_dim):
    """
    Run ttnn.layer_norm correctly on a tile-padded tensor.

    x: ttnn tensor [..., padded_dim] where only [..., :orig_dim] is real data.
    weight/bias: ttnn tensors padded to the same padded_dim; sliced back to
                 orig_dim before use so layer_norm normalizes over the true
                 width only (padding lanes would otherwise corrupt mean/var).
    orig_dim: the true (unpadded) feature width, e.g. D_MODEL=26.

    FIX: ttnn.slice() kwargs changed between TTNN versions.
         Old API: begins=, ends=  (removed)
         New API: slice_start=, slice_end=
    """
    last_dim = x.shape[-1]

    # FIX: was ttnn.slice(x, [0, 0, 0], [x.shape[0], x.shape[1], orig_dim])
    #      with begins=/ends= kwargs — those no longer exist.
    #      Use slice_start=/slice_end= instead.
    x_unpadded = ttnn.slice(x, slice_start=[0, 0, 0], slice_end=[x.shape[0], x.shape[1], orig_dim])
    w_unpadded = ttnn.slice(weight, slice_start=[0], slice_end=[orig_dim])
    b_unpadded = ttnn.slice(bias, slice_start=[0], slice_end=[orig_dim])

    out = ttnn.layer_norm(x_unpadded, weight=w_unpadded, bias=b_unpadded)

    pad_amount = last_dim - orig_dim
    if pad_amount > 0:
        out = ttnn.pad(out, padding=[(0, 0), (0, 0), (0, pad_amount)], value=0.0)

    return out

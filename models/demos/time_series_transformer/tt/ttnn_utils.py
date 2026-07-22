# tt/ttnn_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import ttnn


def layer_norm_padded(x, weight, bias, orig_dim):
    """
    Run ttnn.layer_norm correctly on a tile-padded tensor.

    x: ttnn tensor [..., padded_dim] where only [..., :orig_dim] is real data.
    weight/bias: ttnn tensors at native orig_dim width — never padded to
        padded_dim, unlike qkv/out_proj/ffn weights (see _build_layer_norm
        in tst_model.py). Do not slice or pad them here; if a future change
        pads them to padded_dim, this function's shapes will mismatch.
    orig_dim: the true (unpadded) feature width, e.g. D_MODEL=26.

    Slices x down to orig_dim before normalizing, so padding lanes don't
    corrupt mean/var, then pads the result back to padded_dim.
    """
    last_dim = x.shape[-1]
    x_unpadded = ttnn.slice(x, slice_start=[0, 0, 0], slice_end=[x.shape[0], x.shape[1], orig_dim])

    out = ttnn.layer_norm(x_unpadded, weight=weight, bias=bias)

    pad_amount = last_dim - orig_dim
    if pad_amount > 0:
        out = ttnn.pad(out, padding=[(0, 0), (0, 0), (0, pad_amount)], value=0.0)
    return out

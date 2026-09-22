# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.pad``.

Model call site (modules/rope/rope_1d.py):
  * L224  cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
  * L225  sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)

The rope table slice [1, 1, seq_len, head_dim] is zero-padded along the sequence
dim (dim 2) up to ``pad_to`` for SDPA alignment (rope_1d.py:218-225):

    padding = [(0, 0)] * 4
    padding[2] = (0, pad_len)        # pad_len = pad_to - seq_len

Value-preserving (pad value 0.0) -> torch reference is ``F.pad`` on dim 2; PCC 0.999.

NOTE: the task also mentions vocab / tile padding, but no other ``ttnn.pad`` call
site exists in the model source (vocab width is padded at weight-prep time, not via
ttnn.pad). Only the rope pad pattern is covered here.
"""

import pytest
import torch.nn.functional as F

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

# (id, seq_len, pad_to) — seq_len values are tile-sub-multiples padded up to a tile-aligned length.
_PAD_SITES = [
    ("seq96_to128", 96, 128),
    ("seq128_to256", 128, 256),
    ("seq500_to512", 500, 512),
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, seq_len, pad_to",
    [pytest.param(*s, id=s[0]) for s in _PAD_SITES],
)
def test_pad(ttnn_mesh_device, reset_seeds, name, seq_len, pad_to):
    mesh = ttnn_mesh_device
    pad_len = pad_to - seq_len

    x_torch = U.torch_rand((1, 1, seq_len, U.HEAD_DIM))
    x = U.to_tt(x_torch, mesh)

    padding = [(0, 0)] * 4
    padding[2] = (0, pad_len)
    out = ttnn.experimental.quasar.pad(x, padding=padding, value=0.0)

    # F.pad pads from the last dim outward: (last_l, last_r, dim2_l, dim2_r).
    ref = F.pad(x_torch.float(), (0, 0, 0, pad_len), value=0.0)
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)

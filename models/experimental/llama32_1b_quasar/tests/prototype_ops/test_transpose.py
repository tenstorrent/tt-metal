# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.transpose``.

Model call site (modules/rope/rope_1d.py):
  * L178  cos = ttnn.transpose(cos, 1, 2)
  * L179  sin = ttnn.transpose(sin, 1, 2)

In the decode rope path the cos/sin tables are unsqueezed to 4D
[1, 1, batch, head_dim] (rope_1d.py:176-177) then transposed on dims (1, 2) to
[1, batch, 1, head_dim] before being trimmed to the real batch size.

Value-preserving permutation -> torch reference is ``torch.transpose(x, 1, 2)``;
PCC 0.999.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

# (id, shape, dim0, dim1) — rope transposes dims (1, 2) of [1, 1, batch, head_dim].
_TRANSPOSE_SITES = [
    ("rope_decode_batch1", (1, 1, 1, U.HEAD_DIM), 1, 2),
    ("rope_decode_batch32", (1, 1, U.MAX_BATCH, U.HEAD_DIM), 1, 2),
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, shape, dim0, dim1",
    [pytest.param(*s, id=s[0]) for s in _TRANSPOSE_SITES],
)
def test_transpose(ttnn_mesh_device, reset_seeds, name, shape, dim0, dim1):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)

    out = ttnn.experimental.quasar.transpose(x, dim0, dim1)

    ref = torch.transpose(x_torch.float(), dim0, dim1)
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)

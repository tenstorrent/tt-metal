# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.unsqueeze_to_4D``.

Model call sites:
  * rope_1d.py:176   cos = ttnn.unsqueeze_to_4D(cos)   # [1, batch, head_dim] -> [1, 1, batch, head_dim]
  * rope_1d.py:177   sin = ttnn.unsqueeze_to_4D(sin)
  * model.py:966/971/1013  x = ttnn.unsqueeze_to_4D(x)  # promote lower-rank tensors to rank 4

``unsqueeze_to_4D`` left-pads the shape with 1s until it is rank 4; it is
value-preserving -> torch reference is a reshape prepending leading 1s; PCC 0.999.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

# (id, in_shape) — inputs of rank 2 and 3 that the model promotes to rank 4.
_UNSQUEEZE_SITES = [
    # rope_1d.py:176 — embedding output [1, batch, head_dim]
    ("rope_3d", (1, U.MAX_BATCH, U.HEAD_DIM)),
    # generic rank-2 promotion (model.py:966/1013 style)
    ("rank2", (U.MAX_BATCH, U.DIM)),
]


def _torch_to_4d(x: torch.Tensor) -> torch.Tensor:
    shape = (1,) * (4 - x.dim()) + tuple(x.shape)
    return x.reshape(shape)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, in_shape",
    [pytest.param(*s, id=s[0]) for s in _UNSQUEEZE_SITES],
)
def test_unsqueeze_to_4D(ttnn_mesh_device, reset_seeds, name, in_shape):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(in_shape)
    x = U.to_tt(x_torch, mesh)

    out = ttnn.unsqueeze_to_4D(x)
    assert len(out.shape) == 4, f"expected rank-4 output, got shape {tuple(out.shape)}"

    ref = _torch_to_4d(x_torch.float())
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: unary SFPU ops the OFT encoder decode ran (``eltwise_sfpu`` variants).

Shape source (kernel cache): each op has two ``eltwise_sfpu`` variants whose ct[0]
(tiles per core) is 2 and 1 — 25 tiles split over 20 cores, i.e. a [1, 159, 159] bf16 TILE
tensor — except ``exp`` with 4 and 3 tiles per core (75 tiles = [1, 3, 159, 159]). All bf16,
L1 interleaved in and out (unary reader/writer accessors), HiFi4, approx off.         [H]

  * ``exp``: dims decode ``exp(dim_offsets * std + mean)``                     (126815 / 126361)
  * ``reciprocal``, ``atan``, ``gtz``, ``ltz``, ``eqz``: the pieces of the composite
    ``ttnn.atan2`` (covered as a whole in ``test_atan2.py``)
    (recip 183849 / 171905, atan 143616 / 751423, gtz 183839 / 126521,
     ltz 885732 / 863660, eqz 805151 / 989732)

There is no standalone ``relu`` kernel in the cache: every ReLU was fused into conv,
matmul or binary ops.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

DIMS = (1, 3, U.BEV, U.BEV)


def _away_from_zero(x):
    return torch.where(x.abs() < 0.1, torch.full_like(x, 0.5), x)


def _with_zeros(x):
    x = x.clone()
    x.view(-1)[::7] = 0  # comparison-to-zero ops need exact zeros in the input
    return x


# (op name, ttnn op, torch ref, input transform, exact?)
_OPS = {
    "exp": (ttnn.exp, torch.exp, lambda x: x * 0.5, False),
    "reciprocal": (ttnn.reciprocal, torch.reciprocal, _away_from_zero, False),
    "atan": (ttnn.atan, torch.atan, None, False),
    "gtz": (ttnn.gtz, lambda x: (x > 0).float(), _with_zeros, True),
    "ltz": (ttnn.ltz, lambda x: (x < 0).float(), _with_zeros, True),
    "eqz": (ttnn.eqz, lambda x: (x == 0).float(), _with_zeros, True),
}


@U.with_default_mesh()
@pytest.mark.parametrize(
    "op, shape",
    [
        pytest.param("exp", DIMS, id="exp-1x3x159x159-126815"),
        pytest.param("reciprocal", U.HEATMAP, id="reciprocal-1x159x159-183849"),
        pytest.param("atan", U.HEATMAP, id="atan-1x159x159-143616"),
        pytest.param("gtz", U.HEATMAP, id="gtz-1x159x159-183839"),
        pytest.param("ltz", U.HEATMAP, id="ltz-1x159x159-885732"),
        pytest.param("eqz", U.HEATMAP, id="eqz-1x159x159-805151"),
    ],
)
def test_unary(ttnn_mesh_device, reset_seeds, op, shape):
    mesh = ttnn_mesh_device
    tt_op, torch_op, transform, exact = _OPS[op]
    x = U.torch_rand(shape)
    if transform is not None:
        x = transform(x)
    tt_x = U.to_tt_mem(x, mesh, "l1")
    out = tt_op(tt_x, memory_config=ttnn.L1_MEMORY_CONFIG)
    ref = torch_op(x.float())
    if exact:
        U.assert_lossless(ref, out, mesh_device=mesh)
    else:
        U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: the OFT encoder decode's unary and permute calls from the **second** kernel cache
(in-tree ``tt_encoder`` at ``628a4797bf7``). That run decodes channels-last, so the shapes differ
from cache #1 (NCHW decode, ``test_unary.py`` exp on [1, 3, 159, 159]).

Shape source (kernel cache #2):
  * ``ttnn.exp(offsets * log_dim_std + log_dim_mean)`` on [1, 159, 159, 3] bf16 TILE, L1 in and
    out: ``eltwise_sfpu/688549`` (``exp_tile<0u>``, HiFi4, approx off) with
    ``reader_unary_ng/150910`` / ``writer_unary_ng/359692`` (TensorAccessor 0 = L1 interleaved,
    2048-B pages)                                                                 [H op/mem, M shape]
    Here W = 3 pads to one tile column, so the tensor is 159 x 5 = 795 tiles; cache #1 had 75.
  * ``ttnn.permute(x, (0, 2, 3, 1))`` on bf16 TILE, L1 in and out (no ``memory_config``, so the
    output inherits L1): ``reader/writer_permute_interleaved_tiled_generic``. The named compile-time
    args give C, H = W = 159 (padded 160 = 5 tiles) and 795 output tiles                     [H]
      - pos / dim offsets [1, 3, 159, 159] -> [1, 159, 159, 3]                              (539038)
      - NMS heatmap ``unsqueeze(scores, 0)`` [1, 1, 159, 159] -> [1, 159, 159, 1]           (430473)
    Cache #1 ran the heatmap permute in ROW_MAJOR (``test_permute.py``); in cache #2 the scores
    are tilized first (``to_layout(..., TILE)`` in the demo), so this is the tiled generic path.

``squeeze`` / ``unsqueeze`` are views (no kernels). The ``angle_offsets[:, i]`` and grid
slices are tile slices (``reader_unary_unpad_dims_interleaved_start_id``) and belong in
``test_slice.py``.

The permute is checked bit-exact and exp by PCC.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

DECODE = (1, U.BEV, U.BEV, 3)


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param(DECODE, id="dims-exp-1x159x159x3-688549")])
def test_exp_decode(ttnn_mesh_device, reset_seeds, shape):
    """``exp`` of the log-dimension coefficients (around log_dim_mean 0.42 / 0.48 / 1.35)."""
    mesh = ttnn_mesh_device
    x = (torch.randn(shape) * 0.1 + torch.tensor([0.42, 0.48, 1.35])).to(torch.bfloat16)
    out = ttnn.exp(U.to_tt_mem(x, mesh, "l1"), memory_config=ttnn.L1_MEMORY_CONFIG)
    U.assert_shape_dtype(out, shape=shape, dtype=ttnn.bfloat16)
    U.assert_pcc(torch.exp(x.float()), out, pcc=0.999, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 3, U.BEV, U.BEV), id="decode-offsets-tile-0231-539038"),
        pytest.param((1, 1, U.BEV, U.BEV), id="nms-heatmap-tile-0231-430473"),
    ],
)
def test_permute_decode_tiled(ttnn_mesh_device, reset_seeds, shape):
    """NCHW -> NHWC permute of TILE tensors in L1 (output memory config inherited)."""
    mesh = ttnn_mesh_device
    dims = (0, 2, 3, 1)
    x = U.torch_rand(shape)
    out = ttnn.permute(U.to_tt_mem(x, mesh, "l1"), dims)
    ref = x.permute(dims)
    U.assert_shape_dtype(out, shape=ref.shape, finite=False)
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    U.assert_lossless(ref, out, mesh_device=mesh)

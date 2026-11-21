# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.intimg`` — the OFT integral image (2D inclusive prefix sum
over H and W, per channel, channel-last).

Shape source (kernel cache): ``intimg_compute/124751`` + ``intimg_reader/117586`` +
``intimg_writer/169088``. Compile-time args carry C=256, W=160, H=48, N=1 and the CBs are
Float32: the lat8 features [1, 48, 160, 256] float32 TILE, DRAM in and out              [H].
It is the only intimg variant: the 1/16 and 1/32 OFT scales did not run on device.
The factory fixes HiFi4 with fp32 accumulation.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, U.L2_H, U.L2_W, U.OFT_C), id="lat8-1x48x160x256-124751"),
        pytest.param((1, 12, 40, U.OFT_C), id="proxy-1x12x40x256"),
    ],
)
def test_intimg(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    # lat8 is post-ReLU, so the integral image is monotone and grows to ~H*W*mean.
    x = U.rand_nonneg(shape)
    tt_x = U.to_tt(x, mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    out = ttnn.experimental.intimg(tt_x)

    ref = torch.cumsum(torch.cumsum(x.double(), dim=1), dim=2).float()
    U.assert_shape_dtype(out, shape=shape, dtype=ttnn.float32, finite=False)
    got = U.from_tt(out, mesh).reshape(shape)
    U.assert_pcc(ref, host=got, pcc=0.9999)
    # Same fp32 tolerance as tests/ttnn/unit_tests/operations/reduce/test_intimg.py (rtol 0.015).
    rel = ((got - ref).abs() / ref.abs().clamp_min(1.0)).max().item()
    assert rel < 0.015, f"integral image max relative error {rel:.3g}"

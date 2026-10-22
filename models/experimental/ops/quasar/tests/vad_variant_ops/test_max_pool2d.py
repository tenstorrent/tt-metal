# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.max_pool2d`` — the ResNet-50 stem max-pool (after the 7x7/s2 stem conv
on a 384x640 image).

Shape source (kernel cache): ``compute_pool_2d/296440``, ``reader_pool_2d/565723,689390``,
``halo_gather/209587,763006`` (pad value 0xFF80 = -inf => max pool), ``pack_untilize/618218``.
Ct args map via ``pool_multi_core_program_factory.cpp`` (reader :763-820, compute :863-905).

  * 3x3 window, stride 2, pad 1, dilation 1, ceil_mode False    [H]
  * 64 channels, bf16, output TILE layout                       [H]
  * HEIGHT_SHARDED over 20 cores, 768 output sticks per core     [H]
    => 15360 = 96x160 output sticks => 192x320 input, batch 1    [M]

Reference: ``torch.nn.functional.max_pool2d``. Max-pool is value-selecting, so the bf16 result
must match exactly; PCC 0.999 tolerates the tile padding readback.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

_KERNEL, _STRIDE, _PADDING, _DILATION = [3, 3], [2, 2], [1, 1], [1, 1]


# TILE output is what the model ran, but the tiled-output pool path is broken on Quasar:
# unimplemented in craq-sim (pacr_stride) and PCC ~0 for C=64 on RTL, identically on main.
# See tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_pool2d.py (tiled_out_*).
_QUASAR_TILED_POOL_XFAIL = (
    "Quasar max_pool2d tiled-output path is broken (pre-existing; see test_pool2d.py tiled_out_*)"
)


def _run_pool(mesh, n, c, h, w, output_layout=ttnn.TILE_LAYOUT):
    if output_layout == ttnn.TILE_LAYOUT:
        U.xfail_on_quasar(_QUASAR_TILED_POOL_XFAIL)
    x_nchw = U.torch_rand((n, c, h, w))
    x_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c).contiguous()
    x = U.to_tt(x_flat, mesh, layout=ttnn.TILE_LAYOUT)

    out = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=n,
        input_h=h,
        input_w=w,
        channels=c,
        kernel_size=_KERNEL,
        stride=_STRIDE,
        padding=_PADDING,
        dilation=_DILATION,
        ceil_mode=False,
        applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        output_layout=output_layout,
        dtype=ttnn.bfloat16,
    )

    ref = torch.nn.functional.max_pool2d(x_nchw.float(), _KERNEL, _STRIDE, _PADDING, _DILATION)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c)
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)


@U.with_mesh_l1small(U.VAD_L1_SMALL_SIZE)
@pytest.mark.parametrize("n, c, hw, h", [pytest.param(1, 64, 320, 192, id="stem-maxpool-192x320x64")])
def test_max_pool2d(ttnn_mesh_device, reset_seeds, n, c, hw, h):
    _run_pool(ttnn_mesh_device, n, c, h, hw)


@U.with_mesh_l1small(U.VAD_L1_SMALL_SIZE)
@pytest.mark.parametrize(
    "n, c, hw, h",
    # Emulator-sized proxy: same op config, 1/8-scale spatial extent.
    [pytest.param(1, 64, 40, 24, id="stem-maxpool-proxy-24x40x64")],
)
# ROW_MAJOR keeps pool coverage on Quasar while the tiled-output path is xfail there.
@pytest.mark.parametrize(
    "output_layout",
    [pytest.param(ttnn.TILE_LAYOUT, id="tile-out"), pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm-out")],
)
def test_max_pool2d_small(ttnn_mesh_device, reset_seeds, n, c, hw, h, output_layout):
    _run_pool(ttnn_mesh_device, n, c, h, hw, output_layout)

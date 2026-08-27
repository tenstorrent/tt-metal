# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.pad`` (YOLOv8, 640x640, batch 1).

The stem pads the 3-channel NCHW input up to 16 channels (conv min-channel
requirement) before the NCHW->NHWC permute:

    channel_padding_needed = 16 - C                 # 13 for C=3
    x = ttnn.pad(x, ((0,0),(0,channel_padding_needed),(0,0),(0,0)), value=0.0)

Value-preserving (pad value 0.0) -> torch reference is ``F.pad`` on the channel
dim; PCC 0.999. Uploaded ROW_MAJOR so the compare is exact.

Model call sites (branch ``origin/sdawle/yolov8_bh``):
  * models/demos/yolov8l/tt/ttnn_yolov8l.py:978
  * models/demos/yolov8s/tt/ttnn_yolov8s.py:711  (spelled ((0,0),(0,pad),(0,0),(0,0)))

The stem input is (1, 3, 640, 640); it is padded to (1, 16, 640, 640). H==W is a
spatial feature map, so this is parametrized by ``hw`` (hw=20 fits the emulator,
hw=640 is the real stem resolution and runs on real Blackhole).
"""

import pytest
import torch.nn.functional as F

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize("hw", [20, 640])
def test_pad_stem_channels(ttnn_mesh_device, reset_seeds, hw):
    """ttnn_yolov8l.py:978 / ttnn_yolov8s.py:711 — pad C: 3 -> 16."""
    mesh = ttnn_mesh_device
    pad_c = U.IMG_CH_PADDED - U.IMG_CH  # 16 - 3 = 13

    x_torch = U.torch_rand((1, U.IMG_CH, hw, hw))
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.pad(x, ((0, 0), (0, pad_c), (0, 0), (0, 0)), value=0.0)

    # F.pad pads from the last dim outward: (W_l,W_r, H_l,H_r, C_l,C_r).
    ref = F.pad(x_torch.float(), (0, 0, 0, 0, 0, pad_c), value=0.0)
    U.assert_lossless(ref, out, mesh_device=mesh)

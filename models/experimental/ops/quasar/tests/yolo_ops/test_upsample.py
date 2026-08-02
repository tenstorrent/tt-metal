# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.upsample`` — the YOLOv8 neck (FPN top-down) upsamples.

Model call sites (all scale (2, 2), default mode "nearest"):
  * yolov8l: models/demos/yolov8l/tt/ttnn_yolov8l.py
      L1029  ttnn.upsample(sppf_9, (2, 2), ...)    # SPPF out (20x20) -> 40x40
      L1067/L1070  ttnn.upsample(c2f_12, (2, 2), ...)  # C2f model.12 (40x40) -> 80x80
                   (two branches of the same logical upsample: DRAM vs sharded)
  * yolov8s: models/demos/yolov8s/tt/ttnn_yolov8s.py
      L760  ttnn.upsample(sppf_9, (2, 2), ...)     # SPPF out (20x20) -> 40x40
      L791  ttnn.upsample(c2f_12, (2, 2), ...)     # C2f model.12 (40x40) -> 80x80

Per-variant there are 2 distinct upsample sites; across both variants they dedupe to
3 distinct input shapes:

  1. (hw=20, ch=512)  SPPF-out upsample  — yolov8l AND yolov8s (identical)
       ch = SPPF cv2 out = sppf_configs.input_params[1][3] = 512
            (configs.json sppf_configs[1]=[1,1,0,512,1024]; identical l & s)
       hw = SPPF stage = 20x20 at 640 input.
  2. (hw=40, ch=512)  C2f model.12 upsample — yolov8l
       ch = C2f cv2 out = c2f_configs["model.12"].input_params[1][3] = 512
            (yolov8l configs.json model.12 cv2 = [1,1,0,512,1280])
  3. (hw=40, ch=256)  C2f model.12 upsample — yolov8s
       ch = c2f_configs["model.12"].input_params[1][3] = 256
            (yolov8s configs.json model.12 cv2 = [1,1,0,256,384])
       hw = 40x40 (upsampled SPPF level in the top-down path).

(C2f output conv = cv2 = input_params[1]; see TtC2f.__init__ ttnn_yolov8l.py:396.)

Reference: torch nearest upsample (nn.functional.interpolate, mode="nearest").
Input prep mirrors the canonical op test
tests/ttnn/unit_tests/operations/pool/test_upsample.py (NHWC row-major activation).
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

_SCALE = (2, 2)  # (scale_h, scale_w) — every neck upsample is (2, 2) nearest.


def _torch_upsample_nearest(x_nchw: torch.Tensor) -> torch.Tensor:
    """Golden: nearest upsample on NCHW, returned NHWC to match the ttnn output."""
    ref_nchw = torch.nn.functional.interpolate(x_nchw.float(), scale_factor=_SCALE, mode="nearest")
    return ref_nchw.permute(0, 2, 3, 1).contiguous()


@U.with_default_mesh()
@pytest.mark.parametrize(
    "sites, channels, hw",
    [
        # SPPF-out upsample, shared by both variants (ttnn_yolov8l.py:1029 / ttnn_yolov8s.py:760)
        pytest.param("yolov8l+yolov8s", 512, 20, id="sppf9-upsample-512x20x20"),
        # C2f model.12 upsample, yolov8l (ttnn_yolov8l.py:1067/1070)
        pytest.param("yolov8l", 512, 40, id="yolov8l-c2f12-upsample-512x40x40"),
        # C2f model.12 upsample, yolov8s (ttnn_yolov8s.py:791)
        pytest.param("yolov8s", 256, 40, id="yolov8s-c2f12-upsample-256x40x40"),
    ],
)
def test_upsample(ttnn_mesh_device, reset_seeds, sites, channels, hw):
    mesh = ttnn_mesh_device
    n, c, h, w = U.BATCH, channels, hw, hw

    # NCHW host input -> NHWC [N, H, W, C] row-major on device (the layout ttnn.upsample
    # consumes; canonical test_upsample.py:576-591).
    x_nchw = U.torch_rand((n, c, h, w))
    x = U.nhwc_to_tt(x_nchw, mesh)

    out = ttnn.upsample(x, _SCALE)  # default mode="nearest", matching the model.

    ref = _torch_upsample_nearest(x_nchw)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


# --- model-faithful: HEIGHT_SHARDED L1 input (the REAL neck upsample state). -------------
#
# The model does NOT upsample an interleaved DRAM tensor (the case above): right before
# each upsample it reshapes the feature map to (batch, out_h, out_w, C) row-major and moves
# it into a HEIGHT_SHARDED **L1** config, then upsamples in place, passing the shard config
# straight through (ttnn_yolov8l.py:1017-1029 for sppf_9, :1051-1070 for c2f_12):
#
#     nhw       = batch * out_h * out_w
#     num_cores = determine_num_cores(nhw, sppf_9.shape[2])   # width arg = out_w, NOT C
#     shardspec = create_sharded_memory_config_(shape, grid, HEIGHT, ROW_MAJOR)
#     sppf_9    = ttnn.interleaved_to_sharded(sppf_9, shardspec)
#     sppf_9    = ttnn.upsample(sppf_9, (2, 2), memory_config=sppf_9.memory_config())
#
# determine_num_cores (yolo_utils.py:17 gcd logic) with width=out_w gives one image row
# per core: sppf_9 nhw=400,w=20 -> gcd(400,20)=20 cores; c2f_12 nhw=1600,w=40 -> 40 cores.
@U.with_mesh_l1small()  # the sharded upsample program factory allocates from the L1-small region
@pytest.mark.blackhole_scale
@pytest.mark.parametrize(
    "sites, channels, hw, num_cores",
    [
        # SPPF-out upsample, HEIGHT_SHARDED L1 (ttnn_yolov8l.py:1027-1029 / ttnn_yolov8s.py:758-760)
        # determine_num_cores(400, out_w=20) = 400 // gcd(400,20) = 20 cores.
        pytest.param("yolov8l+yolov8s", 512, 20, 20, id="sppf9-sharded-512x20x20-20c"),
        # C2f model.12 upsample, HEIGHT_SHARDED L1, yolov8l (ttnn_yolov8l.py:1057-1070)
        # determine_num_cores(1600, out_w=40) = 1600 // gcd(1600,40) = 40 cores.
        pytest.param("yolov8l", 512, 40, 40, id="yolov8l-c2f12-sharded-512x40x40-40c"),
        # C2f model.12 upsample, HEIGHT_SHARDED L1, yolov8s (ttnn_yolov8s.py:789-791)
        pytest.param("yolov8s", 256, 40, 40, id="yolov8s-c2f12-sharded-256x40x40-40c"),
    ],
)
def test_upsample_sharded(ttnn_mesh_device, reset_seeds, sites, channels, hw, num_cores):
    mesh = ttnn_mesh_device
    n, c, h, w = U.BATCH, channels, hw, hw

    # NCHW host input -> NHWC (batch, H, W, C) row-major, uploaded to interleaved L1, then
    # moved into the model's HEIGHT_SHARDED L1 config (ttnn_yolov8l.py:1016-1028).
    x_nchw = U.torch_rand((n, c, h, w))
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()  # (batch, H, W, C)
    x = U.to_tt(x_nhwc, mesh, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    sharded = U.height_sharded_memcfg(mesh, num_cores, (n, h, w, c))  # skips if device < num_cores
    x = ttnn.to_memory_config(x, sharded)
    assert x.is_sharded(), "expected a HEIGHT_SHARDED L1 upsample input"

    # Upsample in place through the shard config, exactly as the model does.
    out = ttnn.upsample(x, _SCALE, memory_config=x.memory_config())

    ref = _torch_upsample_nearest(x_nchw)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)

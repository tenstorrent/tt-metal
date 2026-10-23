# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.max_pool2d`` — the YOLOv8 SPPF pooling.

Model call site (SPPF module, kernel/stride/pad/dilation copied verbatim):
  * yolov8l: models/demos/yolov8l/tt/ttnn_yolov8l.py  L529  (class TtSppf ~L508)
  * yolov8s: models/demos/yolov8s/tt/ttnn_yolov8s.py  L445  (class TtSppf ~L423)

Both variants run, in ``TtSppf.__call__``:
    ttnn.max_pool2d(input_tensor=..., batch_size=1, input_h=out_h, input_w=out_w,
                    channels=y[-1].shape[-1], kernel_size=[5, 5], stride=[1, 1],
                    padding=[2, 2], dilation=[1, 1])
applied THREE times in sequence (each pool keeps H/W: 5x5, stride 1, pad 2 is
size-preserving), on the SPPF cv1 output.

Shapes (both variants identical at the SPPF stage — SPPF cv1 out_ch = 256):
  * channels = sppf cv1 out = input_params[0][3] = 256
      configs.json sppf_configs.input_params[0] = [1,1,0,256,512]  (=[k,s,p,out,in])
      identical for yolov8l and yolov8s (both files' configs.json).
  * resolution = SPPF stage = 20x20 at 640x640 input
      backbone stride-2 chain 640->320->160->80->40->20; SPPF is model.9 (deepest).
      (traced through ttnn_yolov8l.py neck __call__ conv_0/3/5/7 stride-2 convs.)

Reference: torch.nn.functional.max_pool2d on the NCHW input. Input prep mirrors the
canonical op test tests/ttnn/nightly/unit_tests/operations/pool/test_maxpool2d.py
(row-major (1, 1, N*H*W, C) activation).
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

# SPPF pool params, copied verbatim from the model call site (identical l & s).
_KERNEL = [5, 5]
_STRIDE = [1, 1]
_PADDING = [2, 2]
_DILATION = [1, 1]


def _torch_max_pool2d(x_nchw: torch.Tensor) -> torch.Tensor:
    """Golden: F.max_pool2d on NCHW, returned as (1, 1, N*H*W_out, C) row-major NHWC."""
    ref_nchw = torch.nn.functional.max_pool2d(
        x_nchw.float(),
        kernel_size=_KERNEL,
        stride=_STRIDE,
        padding=_PADDING,
        dilation=_DILATION,
    )
    n, c, h, w = ref_nchw.shape
    return ref_nchw.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c)


@U.with_mesh_l1small()  # max_pool2d allocates from the L1-small region (default l1_small_size=0 OOMs)
@pytest.mark.parametrize(
    "variant, channels, hw",
    [
        # yolov8l SPPF: ttnn_yolov8l.py:529, configs.json sppf_configs[0]=[1,1,0,256,512]
        pytest.param("yolov8l", 256, 20, id="yolov8l-sppf-256x20x20"),
        # yolov8s SPPF: ttnn_yolov8s.py:445, configs.json sppf_configs[0]=[1,1,0,256,512]
        pytest.param("yolov8s", 256, 20, id="yolov8s-sppf-256x20x20"),
    ],
)
def test_max_pool2d(ttnn_mesh_device, reset_seeds, variant, channels, hw):
    mesh = ttnn_mesh_device
    n, c, h, w = U.BATCH, channels, hw, hw

    # NCHW host input, uploaded as row-major (1, 1, N*H*W, C) — the layout ttnn.max_pool2d
    # consumes (canonical test_maxpool2d.py:76-93).
    x_nchw = U.torch_rand((n, c, h, w))
    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c).contiguous()
    x = U.to_tt(x_nhwc_flat, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

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
    )

    ref = _torch_max_pool2d(x_nchw)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


def _torch_max_pool2d_thrice(x_nchw: torch.Tensor) -> torch.Tensor:
    """Golden for the SPPF's three stacked pools (5x5/s1/p2 is size-preserving), NHWC flat."""
    y = x_nchw.float()
    for _ in range(3):
        y = torch.nn.functional.max_pool2d(y, kernel_size=_KERNEL, stride=_STRIDE, padding=_PADDING, dilation=_DILATION)
    n, c, h, w = y.shape
    return y.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c)


# --- model-faithful: pool #2/#3 consume a HEIGHT_SHARDED L1 input (the REAL SPPF state). ---
#
# In the SPPF loop (ttnn_yolov8l.py:529 / ttnn_yolov8s.py:445) the FIRST max_pool2d consumes
# the interleaved-L1 cv1 output (the case above) but returns a HEIGHT_SHARDED L1 tensor. The
# next two pools then consume that HEIGHT_SHARDED L1 output directly (``y[-1]`` in the loop).
# We reproduce that exact state by stacking two pools: out1 (HEIGHT_SHARDED L1) is asserted
# sharded and fed as the input of pool #2 — no memory_config / applied_shard_scheme args, so
# the sharding is chosen internally exactly as the model leaves it.
@U.with_mesh_l1small()  # max_pool2d allocates from the L1-small region (default l1_small_size=0 OOMs)
@pytest.mark.blackhole_scale
@pytest.mark.parametrize(
    "variant, channels, hw",
    [
        # yolov8l SPPF: ttnn_yolov8l.py:529, configs.json sppf_configs[0]=[1,1,0,256,512]
        pytest.param("yolov8l", 256, 20, id="yolov8l-sppf-sharded-256x20x20"),
        # yolov8s SPPF: ttnn_yolov8s.py:445, configs.json sppf_configs[0]=[1,1,0,256,512]
        pytest.param("yolov8s", 256, 20, id="yolov8s-sppf-sharded-256x20x20"),
    ],
)
def test_max_pool2d_sharded(ttnn_mesh_device, reset_seeds, variant, channels, hw):
    mesh = ttnn_mesh_device
    n, c, h, w = U.BATCH, channels, hw, hw

    x_nchw = U.torch_rand((n, c, h, w))
    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c).contiguous()
    x = U.to_tt(x_nhwc_flat, mesh, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    pool_kwargs = dict(
        batch_size=n,
        input_h=h,
        input_w=w,
        channels=c,
        kernel_size=_KERNEL,
        stride=_STRIDE,
        padding=_PADDING,
        dilation=_DILATION,
    )

    # Pool #1: interleaved -> HEIGHT_SHARDED L1 (this sharded output is what pools #2/#3 eat).
    out1 = ttnn.max_pool2d(input_tensor=x, **pool_kwargs)
    assert out1.is_sharded(), "pool #1 output should be HEIGHT_SHARDED (the pool #2/#3 input state)"

    # Pool #2: consumes the HEIGHT_SHARDED L1 tensor directly, exactly as the model loop does.
    out2 = ttnn.max_pool2d(input_tensor=out1, **pool_kwargs)
    assert out2.is_sharded(), "pool #2 output should be HEIGHT_SHARDED (the pool #3 input state)"

    # Pool #3: the SPPF applies max_pool2d three times; verify the 2nd sharded output can
    # feed the 3rd invocation (catches a state/layout issue that only shows on the last pool).
    out3 = ttnn.max_pool2d(input_tensor=out2, **pool_kwargs)

    ref = _torch_max_pool2d_thrice(x_nchw)
    U.assert_pcc(ref, out3, pcc=0.99, mesh_device=mesh)

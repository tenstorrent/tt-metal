# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone de-risk for the ResNet-50 STEM max-pool on Quasar.

The stem does, right after the 7x7/s2 conv:  max_pool2d(3x3, stride 2, pad 1) over 64 channels.
This is the FIRST non-conv op the split-conv model hits, and it has two documented Quasar risks
that a full 6-hour model run would only surface late:

  1. face_r_dim=9 LLK reject -- the 3x3 window makes the reduce/tilize buffer-descriptor see a
     non-power-of-2 face_r_dim (9), which the Quasar reduce/buffer-desc validators reject (WH/BH
     accept it). Surfaces as a compile/validate assert or a hang in the pool reduce.
  2. value inflation -- an earlier Quasar max_pool2d run produced outputs ABOVE the input max
     (mean ~2x), i.e. wrong pad/init (not -inf) or a mis-tilized reduce. That is what dropped the
     pretrained resnet50 PCC to ~0.54 (stem+conv perfect, maxpool the first divergence).

So this test gates on BOTH:
  - PCC vs a torch F.max_pool2d golden (catches inflation / wrong reduce), and
  - an explicit "no value inflation" bound: device_out.max() <= input.max() + eps (max-pool can
    never exceed the input max; padding with anything > -inf would break this).

The face_r_dim=9 / reduce risks are config-driven (the 3x3 window), NOT spatial-size driven, so a
tiny single-core input reproduces them and runs fast. We also parametrize a slightly larger shape.

Run (emulator / WH, slow dispatch + forced JIT):
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_stem_maxpool.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.99

# ResNet-50 stem max-pool params (identical to the model call in ttnn_functional_resnet50.py).
KERNEL = [3, 3]
STRIDE = [2, 2]
PADDING = [1, 1]
DILATION = [1, 1]
CHANNELS = 64

# (input_h, input_w, id) -- 64 channels each. All hit the 3x3-window face_r_dim=9 path.
_SHAPES = [
    (8, 8, "8x8"),  # matches the known craq-sim repro; smallest / fastest
    (16, 16, "16x16"),
    (28, 28, "28x28"),  # stem-proportioned (post-conv spatial shrinks by /2 stages)
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_h,input_w,sid", _SHAPES, ids=[s[2] for s in _SHAPES])
def test_quasar_stem_maxpool(mesh_device, input_h, input_w, sid):
    device = mesh_device
    torch.manual_seed(0)

    batch_size = 1
    channels = CHANNELS

    # torch golden in NCHW.
    x_nchw = torch.rand((batch_size, channels, input_h, input_w), dtype=torch.float32)
    golden = torch.nn.functional.max_pool2d(
        x_nchw,
        kernel_size=KERNEL,
        stride=STRIDE,
        padding=PADDING,
        dilation=DILATION,
    )  # (N, C, oH, oW)
    out_h, out_w = golden.shape[2], golden.shape[3]

    # ttnn max_pool2d expects [1, 1, N*H*W, C] (flattened NHW, C).
    x_nhwc = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * input_h * input_w, channels)

    tensor_height = batch_size * input_h * input_w
    tensor_width = channels

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, tensor_height, tensor_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # ROW_MAJOR (the pool's natural input layout, as the model + standard max_pool2d tests use).
    # TILE_LAYOUT would force a 32-multiple shard-height constraint that non-tile-aligned N*H*W
    # (e.g. 28x28 = 784) violates on a single-core height shard -- a test-config artifact, not an op bug.
    x = ttnn.from_torch(x_nhwc.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = x.to(device, mem_config)

    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=KERNEL,
        stride=STRIDE,
        padding=PADDING,
        dilation=DILATION,
    )
    ttnn.synchronize_device(device)

    tt_out = ttnn.to_torch(ttnn.from_device(out)).float()
    # [.., N*oH*oW, C] -> NCHW to line up with the torch golden.
    tt_out = tt_out.reshape(batch_size, out_h, out_w, channels).permute(0, 3, 1, 2)

    in_max = float(x_nchw.max())
    dev_max = float(tt_out.max())
    print(
        f"[stem_maxpool {sid}] out={tuple(tt_out.shape)} golden={tuple(golden.shape)} "
        f"in_max={in_max:.4f} dev_max={dev_max:.4f} golden_max={float(golden.max()):.4f}"
    )

    # (2) value-inflation guard: max-pool output can never exceed the input max.
    assert dev_max <= in_max + 1e-2, (
        f"value inflation: device max {dev_max:.4f} > input max {in_max:.4f} "
        f"(bad pad/init -- not -inf -- or mis-tilized reduce)"
    )
    # (1)/(reduce correctness) gate.
    assert_with_pcc(golden, tt_out, pcc=PCC)

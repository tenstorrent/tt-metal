# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
HYPOTHESIS-VALIDATION test for the Quasar conv tilize<->matmul 0x19 race.

Background: the fused conv compute kernel (conv_bmm_tilize_metal2.cpp) interleaves, per height block,
tilize(block) -> matmul(block), so the compute engine ping-pongs between the tilize datacopy pipeline
and the matmul pipeline once PER height block. That transition has a Quasar MATH<->PACK DEST-handshake
race (watcher 0x19 / ERROR_TRISC1). Option C (conv_bmm_split_tilize_metal2.cpp) tilizes ALL height
blocks first, then matmuls them, so the engine transitions tilize->matmul exactly ONCE.

Option C can't run the real stem: its full per-core tilized activation is 3.67 MB, over the 1 MB uint16
DFB ring (dataflow_buffer.cpp:645 ring_trisc_units < 65536). So we can't test the hypothesis at stem
scale. This test instead uses a SMALL height-sharded conv whose WHOLE tilized activation fits the ring
(so Option C can actually run) while still having >=2 height blocks each with enough K to rotate DEST
banks (so the fused kernel still has a chance to hit the race).

Ring-fit math (the constraint the stem violated): total tilized ring units = per_core_out_height_tiles *
act_block_w_ntiles * 128, and Option C resizes ACT_TILIZED to hold ALL blocks, so that product must stay
< 65536 (=> per_core_out_height_tiles * act_block_w_ntiles < 512). Here:
  - 4x4 conv, in_channels=8  => K = 8*4*4/32 = 4 tiles (act_block_w_ntiles = 4)
  - out 32x32 = 1024 sticks = 32 out-height tiles; act_block_h_override = 128 rows = 4 tiles/block
  - per-core split over C cores: per_core_out_height_tiles = 32/C, num_blocks = (32/C)/4
      C=1: 32 tiles, ring = 32*4 = 128 tiles = 16384 units (fits), 8 blocks
      C=2: 16 tiles, ring = 64 tiles = 8192 units (fits), 4 blocks
      C=4: 8 tiles,  ring = 32 tiles = 4096 units (fits), 2 blocks
So it fits the ring and keeps >=2 blocks for any plausible emulator core count. 4x4 + packer_l1_acc +
fused bias + RELU mirror the stem's kernel path (tilize + matmul-partials + bias pack).

HOW TO READ THE RESULT (run BOTH variants):
  - fused faults/timeouts/mismatches AND split passes  => hypothesis VALIDATED: isolating the tilize
    phase avoids the race. Build Option B (tilized activation as a real tensor -> no ring limit).
  - both pass                                          => this shape is too small to trigger the race;
    scale SHAPES up (bigger out / K / act_block_h) until fused breaks, keeping the ring-fit inequality.
  - both fault                                         => the race is intrinsic to a SINGLE tilize->matmul
    transition; Option B won't help either -> escalate to the LLK team (~/llk_conv_tilize_issue.md).

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_split_tilize_hypothesis.py
"""

import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.97


def _run(mesh_device, *, use_split, out_hw, act_block_h_override):
    device = mesh_device
    torch.manual_seed(0)

    # Small stem-like conv: 4x4 / s1 / p0, in=8 (K=4 tiles), out=64, RELU + bias + packer_l1_acc.
    batch_size = 1
    in_channels = 8
    out_channels = 64
    kernel_size = (4, 4)
    stride = (1, 1)
    padding = (0, 0)
    out_h_want, out_w_want = out_hw
    input_height = out_h_want + kernel_size[0] - 1  # s1/p0 => out = in - (k-1)
    input_width = out_w_want + kernel_size[1] - 1

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((1, 1, 1, out_channels), dtype=torch.bfloat16).float()
    torch_golden = torch.relu(
        torch.nn.functional.conv2d(
            torch_input_nchw, torch_weight, bias=torch_bias.reshape(-1), stride=stride, padding=padding
        )
    )

    torch_input_nhwc = torch.permute(torch_input_nchw, (0, 2, 3, 1))
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        act_block_h_override=act_block_h_override,
        reshard_if_not_optimal=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    prev = os.environ.get("TT_METAL_QSR_CONV_SPLIT_TILIZE")
    if use_split:
        os.environ["TT_METAL_QSR_CONV_SPLIT_TILIZE"] = "1"
    elif prev is not None:
        del os.environ["TT_METAL_QSR_CONV_SPLIT_TILIZE"]
    try:
        out, [out_h, out_w], [tt_weight, tt_bias] = ttnn.experimental.quasar.conv2d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
    finally:
        if prev is None:
            os.environ.pop("TT_METAL_QSR_CONV_SPLIT_TILIZE", None)
        else:
            os.environ["TT_METAL_QSR_CONV_SPLIT_TILIZE"] = prev

    tt_out = ttnn.to_torch(ttnn.from_device(out))
    tt_out = tt_out.reshape(batch_size, out_h, out_w, tt_out.shape[-1])
    tt_out = tt_out[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))

    assert_with_pcc(torch_golden, tt_out.float(), pcc=PCC)


# Scale ladder: (out_h, out_w, act_block_h_override[rows]) -- all keep the ring-fit inequality on 1/2/4
# cores. Start small; if neither variant breaks, the larger rungs push more tiles/block through the
# tilize<->matmul transition. If a rung overflows the ring on this core count it FATALs clearly (nudge down).
_SHAPES = [
    pytest.param((32, 32), 128, id="out32x32_abh4t"),  # 32 out-h tiles, 4-tile blocks
    pytest.param((48, 48), 256, id="out48x48_abh8t"),  # bigger: more tiles/block (watch ring on 1 core)
]


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("out_hw, act_block_h_override", _SHAPES)
@pytest.mark.parametrize("use_split", [False, True], ids=["fused", "split"])
def test_conv2d_split_tilize_hypothesis(mesh_device, use_split, out_hw, act_block_h_override):
    _run(mesh_device, use_split=use_split, out_hw=out_hw, act_block_h_override=act_block_h_override)

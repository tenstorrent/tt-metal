# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
SINGLE-CORE (1x3 grid) Quasar conv2d repro for the LLK team.

WHY THIS EXISTS
---------------
The full stem conv (test_conv2d_stem.py) runs on the 2x3 emulator grid across multiple cores
(weights mcast sender + receiver), which complicates LLK debugging. The LLK team asked for a
variant that runs on a SINGLE core on the 1x3 simulator grid, so the compute pack path can be
debugged without the multi-core weights-mcast + cross-core sync.

This drives the SAME kernels as the stem conv — `conv_bmm_tilize_metal2.cpp` (tilize -> matmul ->
fuse_bias pack) via the height-sharded conv path — but with a SMALL input that fits one core's L1
(so no DRAM slicing: it goes straight through `conv2d_L1`, one core). Config mirrors the model stem:
HEIGHT_SHARDED, RELU activation, packer_l1_acc, fused bias, bf16 + LoFi.

The known Quasar fault this targets: `ERROR_TRISC1 / PACR0_TILE_INC` in the compute pack
(conv_bmm_tilize_metal2.cpp). On a single core this reproduces with no mcast/sender-receiver.

SINGLE-CORE INTENT
------------------
The conv auto-sizes its shard grid from the device compute grid. On the 1x3 sim grid (one compute
core) this naturally runs on 1 core. The input is deliberately tiny (8x8) so the per-core shard is a
handful of tiles and fits L1 even on one core. If run on a larger grid it may spread across a few
cores; run it on the 1x3 sim to guarantee single-core.

RUN (1x3 simulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_1x3.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.97


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_1x3_single_core(mesh_device):
    torch.manual_seed(0)
    device = mesh_device

    # Small stem-like conv: 3x3 stride-1 pad-1, folded-stem channel counts (32 -> 64), small 16x16 spatial.
    # Output 16x16 = 256 sticks = 8 height tiles x 2 width tiles -> multiple subblocks/height-blocks (so it
    # exercises the same tilize -> matmul -> fuse_bias pack loop as the stem), yet the per-core im2col +
    # weights + partials + output all fit one core's L1 comfortably (~a few hundred KB).
    batch_size = 1
    in_channels = 32  # groups(4) * C_aligned(8), matches the folded stem
    out_channels = 64
    input_height = input_width = 16
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)

    # --- torch golden (NCHW), with the stem RELU activation + fused bias ---
    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((1, 1, 1, out_channels), dtype=torch.bfloat16).float()
    torch_golden = torch.nn.functional.conv2d(
        torch_input_nchw, torch_weight, bias=torch_bias.reshape(-1), stride=stride, padding=padding
    )
    torch_golden = torch.relu(torch_golden)

    # --- ttnn inputs: NCHW -> NHWC row-major host activation; op flattens + shards internally ---
    torch_input_nhwc = torch.permute(torch_input_nchw, (0, 2, 3, 1))
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16)

    # Mirror the model stem conv1 config.
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
        reallocate_halo_output=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

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

    tt_out = ttnn.to_torch(ttnn.from_device(out))
    tt_out = tt_out.reshape(batch_size, out_h, out_w, tt_out.shape[-1])
    tt_out = tt_out[:, :, :, :out_channels]  # drop channel (tile) padding
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))  # NHWC -> NCHW

    assert_with_pcc(torch_golden, tt_out.float(), pcc=PCC)

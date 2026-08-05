# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Option B — UNPACK-TILIZE PROBE test.

The plain tilize path (tilize_block, used by conv_tilize_only + the standalone tilize op) 0x19s mid-stream on
Quasar even into a plain (non-borrowed) DFB — an intrinsic tilize_block DEST-bank bug. This probe runs the
SAME conv gathered activation through the DIFFERENT tilize path used by the Quasar MAXPOOL compute
(unpack_tilizeA_B_block + reduce_tile_math), which the pool uses with no 0x19.

TT_METAL_QSR_CONV_UNPACK_TILIZE=1 selects conv_unpack_tilize_probe_metal2.cpp in the sharded conv factory
(reusing the standalone-Program-A plumbing: reader gather into ACT, plain ACT_TILIZED output, no weights/
writer; plus a reduce-scalar srcB DFB). The reduce output is throwaway.

PRIMARY (and only) SIGNAL: does the op COMPLETE without a 0x19 / ERROR_TRISC1? Watch the UTPROBE DPRINT lines
(run with TT_METAL_DPRINT_CORES=all) for how many chunks the unpack-tilize path processes:
  - COMPLETES  -> the unpack-tilize path handles the conv activation; the bug is localized to tilize_block.
  - still 0x19 -> the fault is broader than tilize_block.

Run (craq-sim / emulator, slow dispatch + forced JIT, DPRINT on):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 TT_METAL_DPRINT_CORES=all \
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_unpack_tilize_probe.py
"""

import os

import pytest
import torch

import ttnn


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_unpack_tilize_probe(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    # Folded-stem conv1 params (verbatim from the model / test_conv2d_stem.py). Shape unchanged so the probe
    # processes enough blocks to reach where conv_tilize_only faults (~5 tilize_block blocks).
    batch_size = 1
    in_channels = 32  # groups(4) * C_aligned(8)
    out_channels = 64
    input_height = input_width = 115
    kernel_size = (4, 4)
    stride = (1, 1)
    padding = (0, 0)

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()
    torch_input_nhwc = torch.permute(torch_input_nchw, (0, 2, 3, 1))
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
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

    prev = os.environ.get("TT_METAL_QSR_CONV_UNPACK_TILIZE")
    os.environ["TT_METAL_QSR_CONV_UNPACK_TILIZE"] = "1"
    try:
        out, [out_h, out_w], _weights_bias = ttnn.experimental.quasar.conv2d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            bias_tensor=None,
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
            del os.environ["TT_METAL_QSR_CONV_UNPACK_TILIZE"]
        else:
            os.environ["TT_METAL_QSR_CONV_UNPACK_TILIZE"] = prev

    # PRIMARY (and only) SIGNAL: reached here without a 0x19 / ERROR_TRISC1 fault (a fault aborts the process).
    # The reduce output is throwaway — do NOT assert on values.
    tt_out = ttnn.to_torch(ttnn.from_device(out))
    assert tt_out is not None and tt_out.numel() > 0, "unpack-tilize probe returned an empty tensor"
    print(
        f"UNPACK-TILIZE PROBE COMPLETED — no 0x19. out shape={tuple(tt_out.shape)} conv_out_hw=({out_h},{out_w}) "
        f"(unpack_tilizeA_B + reduce processed the conv activation; bug localized to tilize_block)"
    )

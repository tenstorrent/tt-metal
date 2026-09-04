# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Option C (split tilize/matmul) variant of test_conv2d_stem.py.

Drives the exact folded-stem conv1 the model issues, but with TT_METAL_QSR_CONV_SPLIT_TILIZE=1 set so the
sharded conv factory selects the conv_bmm_split_tilize_metal2.cpp compute kernel. That kernel tilizes ALL
height blocks first (one contiguous tilize phase) then matmuls them, so the compute engine transitions
tilize->matmul exactly once instead of ping-ponging per height block. This is the diagnostic for the Quasar
per-block tilize<->matmul DEST-handshake race (watcher 0x19 / ERROR_TRISC1) that the fused kernel hits.

Same conv params / golden / PCC as test_conv2d_stem.py; the only difference is the env toggle. If the split
kernel avoids the 0x19 race this passes where the fused stem faults; if the ACT_TILIZED all-blocks buffer
overflows the uint16 DFB ring for this conv the factory rejects the DFB at program creation (expected data
point -- means the split buffer is too big for this shape/core-count and we fall back / narrow the gate).

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_stem_split_tilize.py
"""

import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.97


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_stem_split_tilize(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    # Folded-stem conv1 params (verbatim from the model).
    batch_size = 1
    in_channels = 32  # groups(4) * C_aligned(8)
    out_channels = 64
    input_height = input_width = 115
    kernel_size = (4, 4)
    stride = (1, 1)
    padding = (0, 0)

    # --- torch golden (NCHW), with the conv1 RELU activation ---
    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((1, 1, 1, out_channels), dtype=torch.bfloat16).float()
    torch_golden = torch.nn.functional.conv2d(
        torch_input_nchw, torch_weight, bias=torch_bias.reshape(-1), stride=stride, padding=padding
    )
    torch_golden = torch.relu(torch_golden)

    # --- ttnn inputs: NCHW -> NHWC row-major host activation ---
    torch_input_nhwc = torch.permute(torch_input_nchw, (0, 2, 3, 1))
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16)

    # conv1_config, mirroring ttnn_functional_resnet50.py.
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

    # Option C: select the split tilize/matmul compute kernel in the sharded conv factory. Set before the
    # op runs (the factory reads it via std::getenv at program creation, in-process) and restore afterwards
    # so the toggle does not leak into other tests sharing the process.
    prev = os.environ.get("TT_METAL_QSR_CONV_SPLIT_TILIZE")
    os.environ["TT_METAL_QSR_CONV_SPLIT_TILIZE"] = "1"
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
            del os.environ["TT_METAL_QSR_CONV_SPLIT_TILIZE"]
        else:
            os.environ["TT_METAL_QSR_CONV_SPLIT_TILIZE"] = prev

    tt_out = ttnn.to_torch(ttnn.from_device(out))
    tt_out = tt_out.reshape(batch_size, out_h, out_w, tt_out.shape[-1])
    tt_out = tt_out[:, :, :, :out_channels]  # drop channel (tile) padding
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))  # NHWC -> NCHW

    assert_with_pcc(torch_golden, tt_out.float(), pcc=PCC)

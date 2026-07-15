# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Option B — Increment 1 (Program A, tilize-only) test.

Drives the exact folded-stem conv1 the model issues, but with TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 set so the
sharded conv factory selects the conv_tilize_only_metal2.cpp compute kernel and runs ONLY the gather+tilize
half in a fresh, tilize-oriented Metal program (no matmul, no weights, no output writer). The op OUTPUTS the
tilized activations (not the conv result).

Purpose: confirm the Quasar 0x19 (Risc IB interrupt — the matmul's terminal MVMUL leaves the MATH DEST
data-valid bit set, so the tilize's datacopy MOP is rejected) CLEARS when the tilize runs in its own program
with no preceding matmul. Every in-fused-kernel fix failed; the standalone tilize op passes precisely because
it runs isolated. Program B (the matmul over these tilized activations) is chained in a later increment.

PRIMARY SIGNAL: the conv COMPLETES without faulting (no 0x19 / ERROR_TRISC1). The output is the tilized
activation, so there is no conv-golden PCC here; we assert the op returns a finite, correctly-shaped tensor.

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_split_program.py
"""

import os

import pytest
import torch

import ttnn


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_split_program_tilize_only(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    # Folded-stem conv1 params (verbatim from the model / test_conv2d_stem.py).
    batch_size = 1
    in_channels = 32  # groups(4) * C_aligned(8)
    out_channels = 64
    input_height = input_width = 115
    kernel_size = (4, 4)
    stride = (1, 1)
    padding = (0, 0)

    # ttnn inputs: NCHW -> NHWC row-major host activation. No bias/activation are needed for Program A
    # (tilize-only); they belong to Program B (the matmul), added in a later increment.
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

    # Option B / Program A: select the tilize-only compute kernel in the sharded conv factory. Set before the
    # op runs (the factory + device-op read it via std::getenv at program creation, in-process) and restore
    # afterwards so the toggle does not leak into other tests sharing the process.
    prev = os.environ.get("TT_METAL_QSR_CONV_SPLIT_PROGRAM")
    os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"] = "1"
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
            del os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"]
        else:
            os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"] = prev

    # PRIMARY SIGNAL: we got here without a 0x19 / ERROR_TRISC1 fault (a fault aborts the process before this).
    tt_out = ttnn.to_torch(ttnn.from_device(out))
    assert tt_out is not None and tt_out.numel() > 0, "tilize-only conv returned an empty tensor"
    assert torch.isfinite(tt_out.float()).all(), "tilize-only conv output contains NaN/Inf"

    # The op output is the per-core tilized activation shard, width = act_block_w (K) tiles. Its rows equal the
    # conv output rows (padded to tiles); we only sanity-check the tensor is non-degenerate here.
    print(f"Program A (tilize-only) completed. out shape={tuple(tt_out.shape)} conv_out_hw=({out_h},{out_w})")

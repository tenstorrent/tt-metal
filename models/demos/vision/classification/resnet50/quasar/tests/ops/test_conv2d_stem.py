# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone test for the Quasar resnet50 STEM conv exactly as the model issues it on the emulator.

WHERE IT COMES FROM
-------------------
After the C=8 fold, the model's first conv (ttnn_functional_resnet50.py: run() -> conv1) is a
4x4 / stride-1 / pad-0 conv on the folded activation:

    input  : [1, 115, 115, 32]   (N, H, W, C=groups*C_aligned = 4*8)
    weight : [64, 32, 4, 4]       (conv1 folded to 32 input channels; see custom_preprocessing)
    output : [1, 112, 112, 64]
    conv1_config: HEIGHT_SHARDED, RELU activation, packer_l1_acc, reshard_if_not_optimal=False

WHY THIS CASE SPECIFICALLY
--------------------------
On the 2-core emulator this conv's single-slice L1 footprint (~4.47 MB) exceeds the 3.7 MB L1 bank,
so Quasar `conv2d` takes the DRAM-slicing path: `conv2d_DRAM` -> `op_slicing::run_sliced_op` ->
`padded_slice` (DRAM->L1 height slice) -> `conv2d_L1` -> `slice_write` (L1->DRAM). This is the op that
failed the full-model run with "DataMovementKernel is not supported on Quasar" (PaddedSliceRMProgramFactory).
This test drives that exact path on the emulator (no emulator skip-guard: DRAM slicing is what handles
the L1 fit, so it should NOT OOM).

STATUS: needs both the Metal-2 `padded_slice` (done) AND `slice_write` (in progress) ports + the
conv2d_DRAM routing to the quasar versions. Until slice_write is ported this will FATAL at the write-back
step; that is expected and marks the next milestone.

Golden: relu(torch conv2d). bf16 + LoFi -> looser PCC.

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_stem.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.97


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_stem(mesh_device):
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

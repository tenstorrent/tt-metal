# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sheet 1 row 123 of 141 -- ttnn.conv2d, layer4_1_conv1

One op, one file. Part of the per-call-site replay of the BF16-ONLY tt-forge ResNet-50 compile;
ResNet50_Forge_Fe_bf16/ holds one of these for every one of the 141 ops in @forward.

WHERE IT COMES FROM
-------------------
torchvision ResNet-50 `layer4.1.conv1`: 2048 -> 512 channels, 1x1 kernel, stride 1, padding 0, over a 7x7
feature map, producing 7x7. One of the 53 convs in the graph.

It carries a FUSED RELU (Conv2dConfig.activation = <op_type = relu>). 33 of the 53 convs do;
the other 20 are the 16 bottleneck conv3s and the 4 downsamples, whose output feeds a residual add,
where Forge emits relu as a separate op instead.

Forge hands conv2d a channels-last flattened activation [1, 1, N*H*W, C] in ROW_MAJOR and gets a TILE
result back, so this test builds the operand row-major and asserts the result is tiled. The weight is
the RAW OIHW tensor straight from host memory: this compile runs no prepare_conv2d_weights anywhere,
so quasar.conv2d prepares it internally.

TTNN IR, verbatim from resnet50_forge_bf16_vs_quasar.xlsx sheet 1 ("Forge ops (bf16 only)"):

    %177 = "ttnn.conv2d"(%176, %arg95, %36, %53) <{batch_size = 1 : i32, compute_config =
        #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, conv2d_config =
        #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>, enable_kernel_stride_folding =
        false, config_tensors_in_dram = true>, dilation = array<i32: 1, 1>, groups = 1 : i32, in_channels = 2048
        : i32, input_height = 7 : i32, input_width = 7 : i32, kernel_size = array<i32: 1, 1>, out_channels = 512
        : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x1x49x2048xbf16,
        #ttnn_layout57>, tensor<512x2048x1x1xbf16, #ttnn_layout26>, tensor<1x1x1x512xbf16, #ttnn_layout1>,
        !ttnn.device) -> tensor<1x1x49x512xbf16, #ttnn_layout54>

Operands, verbatim from the same row:

    Activation                         1x1x49x2048    bf16   ROW_MAJOR  DRAM interleaved
    Weight                             512x2048x1x1   bf16   host       #system_memory (host)
    Bias                               1x1x1x512      bf16   host       #system_memory (host)
    Device handle                      (!ttnn.device)
    -> Result                          1x1x49x512     bf16   TILE       DRAM interleaved

Attributes:

    batch_size = 1 : i32, compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4,
        fp32_dest_acc_en = true>, conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, activation =
        <op_type = relu>, enable_kernel_stride_folding = false, config_tensors_in_dram = true>, dilation =
        array<i32: 1, 1>, groups = 1 : i32, in_channels = 2048 : i32, input_height = 7 : i32, input_width = 7 :
        i32, kernel_size = array<i32: 1, 1>, out_channels = 512 : i32, padding = array<i32: 0, 0, 0, 0>, stride
        = array<i32: 1, 1>

WHAT IT VALIDATES
-----------------
PCC >= 0.98 against torch.nn.functional.conv2d then torch.relu, plus four structural checks against the Forge
ground truth before the numbers are even looked at: the returned (out_h, out_w) is 7x7, the op's
INTERNALLY-PREPARED weight has the shape prepare_conv2d_weights would have produced
([1, 1, 2048, 512]) so the two weight-prep paths are checked to agree, the output has 49 rows, and it
landed TILE / INTERLEAVED / DRAM as the IR says.

Forge's compute config -- math_fidelity = hifi4 WITH fp32_dest_acc_en = true -- is passed through
VERBATIM, because that is the configuration the sheet records. On Quasar that flag has needed an
explicit per-DFB unpack_modes entry; if that is what fails, that is the finding, not a test defect.

This is a stride-1 1x1 conv, so it needs no halo and lowers straight onto the matmul path.

THE COMPILE
-----------
CompilerConfig() with exactly enable_optimization_passes=True and default_df_override=Float16_b,
and nothing else -- no consteval, no opt_level=2, no HiFi2, no remove_dead_values, no
max_legal_layouts. Every tensor is bf16 and DRAM INTERLEAVED, so this file pins no core range and
nothing here depends on the device grid. The same op under the OPTIMISED compile (L1, sharded,
HiFi2, pinned core ranges) is in ../ResNet50_Forge_Fe/.

RUN
---
  TT_METAL_SIMULATOR=<dir>/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 ARCH_NAME=quasar \
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/test_op123_conv2d_layer4_1_conv1.py

Status on 2026-09-04 (craq-sim, Arch.QUASAR, 8x4): FAIL -- cause A, fp32_dest_acc_en=true rejected (program_spec.cpp:1076, no unpack_modes entry for the FP32 DFB)
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# This compile pins <interleaved> #dram on every tensor -- no shard spec, no core ranges.
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _assert_quasar(device):
    """
    Refuse to report a pass unless this really ran on a Quasar part.

    Every op in this file is a ttnn.experimental.quasar op, which builds Gen2 kernels; on any other
    arch it would TT_FATAL rather than quietly produce a number, but asserting it here means a green
    tick in this file always means "green ON QUASAR" without having to go and read the run header.

    To prove the op also DISPATCHED (a device program was built and enqueued, not a host fallback),
    run the suite under the attestation plugin:
        pytest -p quasar_analysis.pytest_quasar_attest ...
    which captures the ttnn graph around every test and records the device operations underneath.
    """
    assert device.arch() == ttnn.Arch.QUASAR, (
        "this test ran on %s, not Arch.QUASAR. Open a Quasar device (TT_METAL_SIMULATOR=<dir>/"
        "libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 ARCH_NAME=quasar) -- see "
        "test_op_inventory_bf16.py::test_device_under_test_is_quasar." % device.arch()
    )


# --- the five constants test_op_inventory_bf16.py parses back off disk ---
SHEET_ROW = 123
FORGE_OP = "ttnn.conv2d"
QUASAR_OP = "quasar.conv2d"
OPERAND_SHAPES = ((1, 1, 49, 2048), (512, 2048, 1, 1), (1, 1, 1, 512))
OUTPUT_SHAPE = (1, 1, 49, 512)

IN_CHANNELS = 2048
OUT_CHANNELS = 512
INPUT_HW = 7  # both input_height and input_width
KERNEL = 1
STRIDE = 1
PADDING = 0  # symmetric on all four sides; == KERNEL // 2 for every resnet conv
FUSED_RELU = True
BATCH, GROUPS, DILATION = 1, 1, (1, 1)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_bf16_op123_conv2d(mesh_device):
    device = mesh_device
    _assert_quasar(device)
    torch.manual_seed(0)

    # ---- torch golden (NCHW) --------------------------------------------------------------------
    x_nchw = torch.randn((BATCH, IN_CHANNELS, INPUT_HW, INPUT_HW), dtype=torch.bfloat16).float()
    weight = torch.randn((OUT_CHANNELS, IN_CHANNELS // GROUPS, KERNEL, KERNEL), dtype=torch.bfloat16).float()
    bias = torch.randn((1, 1, 1, OUT_CHANNELS), dtype=torch.bfloat16).float()

    golden = torch.nn.functional.conv2d(
        x_nchw, weight, bias=bias.reshape(-1), stride=(STRIDE, STRIDE), padding=(PADDING, PADDING), dilation=DILATION
    )
    if FUSED_RELU:
        golden = torch.relu(golden)
    exp_oh, exp_ow = golden.shape[2], golden.shape[3]
    assert (exp_oh, exp_ow) == (7, 7), "torch says %dx%d, sheet 1 row 123 says 7x7" % (exp_oh, exp_ow)

    # ---- operands in Forge's exact layout: activation ROW_MAJOR in DRAM, weights on host ---------
    flat = x_nchw.to(torch.bfloat16).permute(0, 2, 3, 1).reshape(1, 1, BATCH * INPUT_HW * INPUT_HW, IN_CHANNELS)
    tt_in = ttnn.from_torch(
        flat.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
    )
    tt_w = ttnn.from_torch(weight.to(torch.bfloat16), dtype=ttnn.bfloat16)  # raw OIHW, #system_memory
    tt_b = ttnn.from_torch(bias.to(torch.bfloat16), dtype=ttnn.bfloat16)  # [1,1,1,oc], #system_memory

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if FUSED_RELU else None,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )
    # compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )

    out, [out_h, out_w], [prep_w, prep_b] = ttnn.experimental.quasar.conv2d(
        input_tensor=tt_in,
        weight_tensor=tt_w,
        bias_tensor=tt_b,
        device=device,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        batch_size=BATCH,
        input_height=INPUT_HW,
        input_width=INPUT_HW,
        kernel_size=(KERNEL, KERNEL),
        stride=(STRIDE, STRIDE),
        padding=(PADDING, PADDING, PADDING, PADDING),
        dilation=DILATION,
        groups=GROUPS,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        memory_config=DRAM,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    ttnn.synchronize_device(device)

    # ---- structural checks against the Forge ground truth ----------------------------------------
    assert (out_h, out_w) == (exp_oh, exp_ow), "op returned %dx%d, Forge IR / torch say %dx%d" % (
        out_h,
        out_w,
        exp_oh,
        exp_ow,
    )
    want_prep = (1, 1, IN_CHANNELS * KERNEL * KERNEL, OUT_CHANNELS)
    assert tuple(prep_w.shape) == want_prep, "prepared weight %s, prepare_conv2d_weights makes %s" % (
        tuple(prep_w.shape),
        want_prep,
    )
    assert tuple(prep_b.shape)[-1] >= OUT_CHANNELS, "prepared bias too narrow: %s" % (tuple(prep_b.shape),)
    assert out.shape[-1] >= OUT_CHANNELS, "output has %d channels, need >= %d" % (out.shape[-1], OUT_CHANNELS)
    assert out.shape[-2] == BATCH * exp_oh * exp_ow, "output rows %d, Forge IR says %d" % (
        out.shape[-2],
        BATCH * exp_oh * exp_ow,
    )
    mem = out.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, "not interleaved: %s" % (mem.memory_layout,)
    assert mem.buffer_type == ttnn.BufferType.DRAM, "not in DRAM: %s" % (mem.buffer_type,)
    assert out.layout == ttnn.TILE_LAYOUT, "output layout %s, the IR says the result is tiled" % (out.layout,)

    # ---- PCC -------------------------------------------------------------------------------------
    tt_out = ttnn.to_torch(ttnn.from_device(out)).reshape(BATCH, out_h, out_w, -1)[:, :, :, :OUT_CHANNELS]
    assert_with_pcc(golden, tt_out.permute(0, 3, 1, 2).float(), pcc=0.98)

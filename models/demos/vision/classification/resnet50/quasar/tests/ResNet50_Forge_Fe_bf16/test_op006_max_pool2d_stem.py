# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sheet 1 row 6 of 141 -- ttnn.max_pool2d, stem

One op, one file. Part of the per-call-site replay of the BF16-ONLY tt-forge ResNet-50 compile;
ResNet50_Forge_Fe_bf16/ holds one of these for every one of the 141 ops in @forward.

WHERE IT COMES FROM
-------------------
The stem max pool -- the only pooling op in ResNet-50 apart from the final average. 3x3 kernel,
stride 2, padding 1 over the 112x112x64 stem-conv output, halving it to 56x56.

Both the operand and the result are ROW_MAJOR here, unlike the optimised compile where the pool
output is a height-sharded tensor feeding two convs directly.

reallocate_halo_output = false is Forge's choice and is passed verbatim -- the ttnn default is True,
so leaving it out would NOT be a faithful replay.

TTNN IR, verbatim from resnet50_forge_bf16_vs_quasar.xlsx sheet 1 ("Forge ops (bf16 only)"):

    %60 = "ttnn.max_pool2d"(%59) <{batch_size = 1 : si32, ceil_mode = false, channels = 64 : si32,
        config_tensors_in_dram = true, dilation = array<i32: 1, 1>, input_height = 112 : si32, input_width = 112
        : si32, kernel_size = array<i32: 3, 3>, padding = array<i32: 1, 1>, reallocate_halo_output = false,
        stride = array<i32: 2, 2>}> : (tensor<1x1x12544x64xbf16, #ttnn_layout35>) -> tensor<1x1x3136x64xbf16,
        #ttnn_layout36>

Operands, verbatim from the same row:

    Activation                         1x1x12544x64   bf16   ROW_MAJOR  DRAM interleaved
    -> Result                          1x1x3136x64    bf16   ROW_MAJOR  DRAM interleaved

Attributes:

    batch_size = 1 : si32, ceil_mode = false, channels = 64 : si32, config_tensors_in_dram = true, dilation =
        array<i32: 1, 1>, input_height = 112 : si32, input_width = 112 : si32, kernel_size = array<i32: 3, 3>,
        padding = array<i32: 1, 1>, reallocate_halo_output = false, stride = array<i32: 2, 2>

WHAT IT VALIDATES
-----------------
PCC >= 0.999 against torch.nn.functional.max_pool2d. Max SELECTS a value, it does not accumulate one,
so bf16 in gives bf16 out with no arithmetic error and the bound can be tight.

A 3x3 pool needs the HALO, the same machinery every kernel > 1 conv needs. That makes this file the
cheapest halo test in the directory, and a useful control: if the convs fail on the halo and this one
passes, the fault is in the CONV halo path, not the halo as such.

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
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/test_op006_max_pool2d_stem.py

Status on 2026-09-04 (craq-sim, Arch.QUASAR, 8x4): PASS
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
SHEET_ROW = 6
FORGE_OP = "ttnn.max_pool2d"
QUASAR_OP = "quasar.max_pool2d"
OPERAND_SHAPES = ((1, 1, 12544, 64),)
OUTPUT_SHAPE = (1, 1, 3136, 64)

BATCH, CHANNELS = 1, 64
IN_H, IN_W = 112, 112
KERNEL, STRIDE, PADDING, DILATION = (3, 3), (2, 2), (1, 1), (1, 1)
CEIL_MODE = False
REALLOCATE_HALO_OUTPUT = False  # Forge's choice; the ttnn default is True
OUT_H, OUT_W = 56, 56


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_bf16_op006_max_pool2d(mesh_device):
    device = mesh_device
    _assert_quasar(device)
    torch.manual_seed(0)

    x_nchw = torch.randn((BATCH, CHANNELS, IN_H, IN_W), dtype=torch.bfloat16).float()
    golden = torch.nn.functional.max_pool2d(
        x_nchw, kernel_size=KERNEL, stride=STRIDE, padding=PADDING, dilation=DILATION, ceil_mode=CEIL_MODE
    )
    assert tuple(golden.shape) == (BATCH, CHANNELS, OUT_H, OUT_W), tuple(golden.shape)

    flat = x_nchw.to(torch.bfloat16).permute(0, 2, 3, 1).reshape(1, 1, BATCH * IN_H * IN_W, CHANNELS)
    tt_in = ttnn.from_torch(
        flat.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
    )

    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=tt_in,
        batch_size=BATCH,
        input_h=IN_H,
        input_w=IN_W,
        channels=CHANNELS,
        kernel_size=list(KERNEL),
        stride=list(STRIDE),
        padding=list(PADDING),
        dilation=list(DILATION),
        ceil_mode=CEIL_MODE,
        memory_config=DRAM,
        reallocate_halo_output=REALLOCATE_HALO_OUTPUT,
        dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.synchronize_device(device)

    assert out.shape[-2] == BATCH * OUT_H * OUT_W, "output rows %d, sheet 1 row 6 says %d" % (
        out.shape[-2],
        BATCH * OUT_H * OUT_W,
    )
    assert out.shape[-1] >= CHANNELS, "output has %d channels, need >= %d" % (out.shape[-1], CHANNELS)
    mem = out.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, "not interleaved: %s" % (mem.memory_layout,)
    assert mem.buffer_type == ttnn.BufferType.DRAM, "not in DRAM: %s" % (mem.buffer_type,)

    got = ttnn.to_torch(ttnn.from_device(out)).reshape(BATCH, OUT_H, OUT_W, -1)[:, :, :, :CHANNELS]
    assert_with_pcc(golden, got.permute(0, 3, 1, 2).float(), pcc=0.999)

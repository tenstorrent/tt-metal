# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sheet 1 row 140 of 141 -- ttnn.linear, fc

One op, one file. Part of the per-call-site replay of the BF16-ONLY tt-forge ResNet-50 compile;
ResNet50_Forge_Fe_bf16/ holds one of these for every one of the 141 ops in @forward.

WHERE IT COMES FROM
-------------------
The 1000-way classifier, the last op in @forward. The weight is stored K x N (2048 x 1000), so both
transposes are false, and the bias is the IR's rank-1 [1000] whose memref is 1x32 tiles -- that is
the padded 2-D layout [1, 1000], which is how it is built here.

No program_config and no core_grid: this compile leaves the matmul to pick its own. That is the
difference from the optimised compile, which pins a MatmulMultiCoreReuseMultiCast1DProgramConfig.

TTNN IR, verbatim from resnet50_forge_bf16_vs_quasar.xlsx sheet 1 ("Forge ops (bf16 only)"):

    %194 = "ttnn.linear"(%193, %arg107, %arg108) <{compute_config =
        #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false,
        transpose_b = false}> : (tensor<1x2048xbf16, #ttnn_layout59>, tensor<2048x1000xbf16, #ttnn_layout27>,
        tensor<1000xbf16, #ttnn_layout28>) -> tensor<1x1000xbf16, #ttnn_layout29>

Operands, verbatim from the same row:

    Activation                         1x2048         bf16   TILE       DRAM interleaved
    Weight                             2048x1000      bf16   TILE       DRAM interleaved
    Bias                               1000           bf16   TILE       DRAM interleaved
    -> Result                          1x1000         bf16   TILE       DRAM interleaved

Attributes:

    compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>,
        transpose_a = false, transpose_b = false

WHAT IT VALIDATES
-----------------
PCC >= 0.98 against act @ weight + bias -- a 2048-deep bf16 reduction -- plus the output shape,
TILE, INTERLEAVED and DRAM.

What is awkward about this case: M = 1 and N = 1000 are BOTH ragged. The activation is one row padded
to a 32-row tile, and the output width pads 1000 -> 1024. So it exercises the matmul's handling of a
single-tile-row activation with a non-tile-aligned N, which is where the fc has failed before
(../ops/test_linear.py, ../ops/test_fc_kspill.py).

Forge's fp32_dest_acc_en = true is passed through verbatim, because that is what the sheet records.

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
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/test_op140_linear_fc.py

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
SHEET_ROW = 140
FORGE_OP = "ttnn.linear"
QUASAR_OP = "quasar.linear"
OPERAND_SHAPES = ((1, 2048), (2048, 1000), (1000,))
OUTPUT_SHAPE = (1, 1000)

IN_FEATURES, OUT_FEATURES = 2048, 1000
ACT_SHAPE = (1, IN_FEATURES)
WEIGHT_SHAPE = (IN_FEATURES, OUT_FEATURES)
BIAS_SHAPE = (1, OUT_FEATURES)  # the IR's rank-1 [1000] in its padded 1x32-tile layout
OUT_SHAPE = (1, OUT_FEATURES)
TRANSPOSE_A = False
TRANSPOSE_B = False


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_bf16_op140_linear(mesh_device):
    device = mesh_device
    _assert_quasar(device)
    torch.manual_seed(0)

    act = torch.randn(ACT_SHAPE, dtype=torch.bfloat16)
    weight = torch.randn(WEIGHT_SHAPE, dtype=torch.bfloat16)
    bias = torch.randn(BIAS_SHAPE, dtype=torch.bfloat16)
    golden = act.float() @ weight.float() + bias.float()

    tt_act = ttnn.from_torch(act, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    tt_w = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    tt_b = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

    # compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )

    out = ttnn.experimental.quasar.linear(
        tt_act,
        tt_w,
        bias=tt_b,
        transpose_a=TRANSPOSE_A,
        transpose_b=TRANSPOSE_B,
        memory_config=DRAM,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_config,
    )
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == OUT_SHAPE, "output shape %s, sheet 1 row 140 says %s" % (tuple(out.shape), OUT_SHAPE)
    assert out.layout == ttnn.TILE_LAYOUT, "output layout %s, the IR says tiled" % (out.layout,)
    mem = out.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, "not interleaved: %s" % (mem.memory_layout,)
    assert mem.buffer_type == ttnn.BufferType.DRAM, "not in DRAM: %s" % (mem.buffer_type,)

    got = ttnn.to_torch(ttnn.from_device(out)).float()
    assert_with_pcc(golden, got, pcc=0.98)

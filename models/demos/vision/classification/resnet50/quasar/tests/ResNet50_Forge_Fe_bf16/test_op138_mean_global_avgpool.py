# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sheet 1 row 138 of 141 -- ttnn.mean, global_avgpool

One op, one file. Part of the per-call-site replay of the BF16-ONLY tt-forge ResNet-50 compile;
ResNet50_Forge_Fe_bf16/ holds one of these for every one of the 141 ops in @forward.

WHERE IT COMES FROM
-------------------
The global average pool before the classifier. dim_arg = [-2] with keep_dim reduces the flattened
spatial axis of the layer4 output -- the mean over the 49 positions of the 7x7 feature map, which is
exactly torchvision's nn.AdaptiveAvgPool2d((1, 1)).

        [1, 1, 49, 2048]  ->  [1, 1, 1, 2048]

Forge attaches its compute config here too: math_fidelity = hifi4, fp32_dest_acc_en = true.

TTNN IR, verbatim from resnet50_forge_bf16_vs_quasar.xlsx sheet 1 ("Forge ops (bf16 only)"):

    %192 = "ttnn.mean"(%191) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4,
        fp32_dest_acc_en = true>, dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xbf16,
        #ttnn_layout56>) -> tensor<1x1x1x2048xbf16, #ttnn_layout58>

Operands, verbatim from the same row:

    Activation                         1x1x49x2048    bf16   TILE       DRAM interleaved
    -> Result                          1x1x1x2048     bf16   TILE       DRAM interleaved

Attributes:

    compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg
        = [-2 : i32], keep_dim = true

WHAT IT VALIDATES
-----------------
THE GAP: the Quasar namespace binds NO reduction at all -- no sum, no mean, no max, no argmax, and
no normalization built on one.

So there is no ttnn.experimental.quasar.mean to call, and NOTHING IN THIS SUITE XFAILS. What this
file runs instead is the route that DOES exist: quasar.avg_pool2d with a 7x7 kernel over a 7x7
input, stride 1, no padding -- the same arithmetic, and the lowering a Quasar-aware compiler would
use. Forge's compute config is carried verbatim, which is worth watching: conv2d and linear are
REJECTED for fp32_dest_acc_en = true on Quasar and this op is not.

The gap itself is watched in ONE place:
    test_op_inventory_bf16.py::test_forge_ops_map_onto_the_live_quasar_build
fails the day quasar binds a reduction, which is the signal to add the direct test here.

PCC >= 0.99: a 49-term bf16 mean.

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
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/test_op138_mean_global_avgpool.py

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
SHEET_ROW = 138
FORGE_OP = "ttnn.mean"
QUASAR_OP = None  # no such op on Quasar
OPERAND_SHAPES = ((1, 1, 49, 2048),)
OUTPUT_SHAPE = (1, 1, 1, 2048)

IN_SHAPE = (1, 1, 49, 2048)
OUT_SHAPE = (1, 1, 1, 2048)
DIM_ARG = [-2]
KEEP_DIM = True
SPATIAL = 7  # 49 = 7 x 7
CHANNELS = 2048


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_bf16_op138_mean_via_avg_pool2d(mesh_device):
    """
    Sheet 1 row 138's mean over the flattened 7x7 spatial axis, lowered to a 7x7 avg_pool2d.

    Quasar binds no reduction, so this is the only route -- see the module docstring. It is a full
    device test: real operands, real PCC bound, Forge's compute config carried verbatim, no xfail.
    """
    device = mesh_device
    _assert_quasar(device)
    torch.manual_seed(0)

    host = torch.randn(IN_SHAPE, dtype=torch.bfloat16)
    golden = host.float().mean(dim=-2, keepdim=True)  # [1, 1, 1, 2048]

    # compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )

    tt_in = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)
    out = ttnn.experimental.quasar.avg_pool2d(
        input_tensor=tt_in,
        batch_size=1,
        input_h=SPATIAL,
        input_w=SPATIAL,
        channels=CHANNELS,
        kernel_size=[SPATIAL, SPATIAL],
        stride=[1, 1],
        padding=[0, 0],
        ceil_mode=False,
        count_include_pad=True,
        memory_config=DRAM,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_config,
    )
    ttnn.synchronize_device(device)

    assert out.shape[-2] == 1, "a %dx%d window left %d rows, expected 1" % (SPATIAL, SPATIAL, out.shape[-2])
    assert out.shape[-1] >= CHANNELS, "output has %d channels, need >= %d" % (out.shape[-1], CHANNELS)

    got = ttnn.to_torch(ttnn.from_device(out)).reshape(1, 1, 1, -1)[:, :, :, :CHANNELS].float()
    assert_with_pcc(golden, got, pcc=0.99)

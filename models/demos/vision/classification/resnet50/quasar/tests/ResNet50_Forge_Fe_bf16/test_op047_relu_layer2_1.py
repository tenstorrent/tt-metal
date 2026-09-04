# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sheet 1 row 47 of 141 -- ttnn.relu, layer2_1

One op, one file. Part of the per-call-site replay of the BF16-ONLY tt-forge ResNet-50 compile;
ResNet50_Forge_Fe_bf16/ holds one of these for every one of the 141 ops in @forward.

WHERE IT COMES FROM
-------------------
The relu that follows the residual add of bottleneck layer2.1 (sheet 1 row 46). Forge places it as a
SEPARATE op, with the same shape, layout and memory config as the add it consumes:

    add(conv3_out, skip)      <-- sheet 1 row 46
    relu(...)                 <-- THIS ROW, one of 16 with no Quasar equivalent

Forge DOES fuse relu into 33 of the 53 convs via Conv2dConfig.activation, and that path works on
Quasar. It is only these 16 post-add relus that have no home.

TTNN IR, verbatim from resnet50_forge_bf16_vs_quasar.xlsx sheet 1 ("Forge ops (bf16 only)"):

    %101 = "ttnn.relu"(%100) : (tensor<1x1x784x512xbf16, #ttnn_layout44>) -> tensor<1x1x784x512xbf16,
        #ttnn_layout44>

Operands, verbatim from the same row:

    Activation                         1x1x784x512    bf16   TILE       DRAM interleaved
    -> Result                          1x1x784x512    bf16   TILE       DRAM interleaved

Attributes:

    (no attributes)

WHAT IT VALIDATES
-----------------
THE GAP: ttnn.experimental.quasar binds data movement, conv2d, the pools, the matmul family and a
BINARY front-end. It binds NO plain unary activation -- no relu, sigmoid or gelu. (prelu, pow and
polyval are the only unary-with-param ops bound, and none of them is relu.)

So there is no ttnn.experimental.quasar.relu to call, and NOTHING IN THIS SUITE XFAILS. What this
file runs instead is the route that DOES exist: the add and the relu collapsed into one
quasar.add with a fused RELU activation -- exactly what the hand-written metal model already does
(resnet50Bottleneck.__call__, see ../ops/test_add.py) and what a Quasar-aware compiler would have
to emit for this pair. That is a real device test with a real PCC check, not a placeholder.

The gap itself is watched in ONE place rather than sixteen:
    test_op_inventory_bf16.py::test_forge_ops_map_onto_the_live_quasar_build
fails the day quasar binds a standalone relu, which is the signal to add the direct test here.

Inputs are torch.randn, so the clamp really clamps rather than being a no-op.

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
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/test_op047_relu_layer2_1.py

Status on 2026-09-04 (craq-sim, Arch.QUASAR, 8x4): (not run)
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
SHEET_ROW = 47
FORGE_OP = "ttnn.relu"
QUASAR_OP = None  # no such op on Quasar
OPERAND_SHAPES = ((1, 1, 784, 512),)
OUTPUT_SHAPE = (1, 1, 784, 512)

SHAPE = (1, 1, 784, 512)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_bf16_op047_relu_fused_add(mesh_device):
    """
    Sheet 1 rows 46 and 47 collapsed into one quasar.add(a, b, activations=[UnaryWithParam(RELU)]).

    Quasar has no standalone relu, so this fusion is the only route for this pair -- see the module
    docstring. It is a full device test: real operands, real PCC bound, no xfail.
    """
    device = mesh_device
    _assert_quasar(device)
    torch.manual_seed(0)

    main = torch.randn(SHAPE, dtype=torch.bfloat16)
    skip = torch.randn(SHAPE, dtype=torch.bfloat16)
    golden = torch.relu(main.float() + skip.float())

    tt_main = ttnn.from_torch(main, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    tt_skip = ttnn.from_torch(skip, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

    out = ttnn.experimental.quasar.add(
        tt_main,
        tt_skip,
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
    )
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == SHAPE, "output shape %s, sheet 1 row 47 says %s" % (tuple(out.shape), SHAPE)
    got = ttnn.to_torch(ttnn.from_device(out)).float()
    assert_with_pcc(golden, got, pcc=0.99)
    assert (got < 0).sum() == 0, "%d negative values survived the fused RELU" % int((got < 0).sum())

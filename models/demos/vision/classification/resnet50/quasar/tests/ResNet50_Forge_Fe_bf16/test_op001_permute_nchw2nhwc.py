# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sheet 1 row 1 of 141 -- ttnn.permute, nchw2nhwc

One op, one file. Part of the per-call-site replay of the BF16-ONLY tt-forge ResNet-50 compile;
ResNet50_Forge_Fe_bf16/ holds one of these for every one of the 141 ops in @forward.

WHERE IT COMES FROM
-------------------
The NCHW -> NHWC conversion of the model input, once, before the stem. torchvision hands Forge an
NCHW image; every ttnn conv wants channels-last, so Forge permutes first and flattens second
(sheet 1 row 2).

This is the hardest possible permute for a tiled layout: it moves the 3-element channel axis from
position 1 to position 3, so a 224x224 tiled face becomes a 224x3 one and every tile is rebuilt.

TTNN IR, verbatim from resnet50_forge_bf16_vs_quasar.xlsx sheet 1 ("Forge ops (bf16 only)"):

    %55 = "ttnn.permute"(%54) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16,
        #ttnn_layout30>) -> tensor<1x224x224x3xbf16, #ttnn_layout31>

Operands, verbatim from the same row:

    Activation                         1x3x224x224    bf16   TILE       DRAM interleaved
    -> Result                          1x224x224x3    bf16   TILE       DRAM interleaved

Attributes:

    permutation = array<i64: 0, 2, 3, 1>

WHAT IT VALIDATES
-----------------
THE GAP: ttnn.experimental.quasar binds `transpose` (a two-axis swap) but NO general `permute`. The
hand-written metal quasar model never needs one -- it uploads its input already channels-last and
folds it -- so this op has no counterpart there to compare against.

So there is no ttnn.experimental.quasar.permute to call, and NOTHING IN THIS SUITE XFAILS. What
this file runs instead is the route that DOES exist: 0,2,3,1 decomposed into the two adjacent swaps
quasar.transpose can express,
        [1,3,224,224] --t(1,2)--> [1,224,3,224] --t(2,3)--> [1,224,224,3]
which is what a Quasar-aware compiler would have to lower this permute into. That is a real device
test with an exact-equality check, not a placeholder.

The gap itself is watched in ONE place:
    test_op_inventory_bf16.py::test_forge_ops_map_onto_the_live_quasar_build
fails the day quasar binds a permute, which is the signal to add the direct test here.

A permute moves data, it does not compute it, so the check is EXACT equality.

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
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/test_op001_permute_nchw2nhwc.py

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
SHEET_ROW = 1
FORGE_OP = "ttnn.permute"
QUASAR_OP = None  # no such op on Quasar
OPERAND_SHAPES = ((1, 3, 224, 224),)
OUTPUT_SHAPE = (1, 224, 224, 3)

IN_SHAPE = (1, 3, 224, 224)
PERMUTATION = (0, 2, 3, 1)
OUT_SHAPE = (1, 224, 224, 3)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_bf16_op001_permute_via_transpose(mesh_device):
    """
    Sheet 1 row 1's permutation 0,2,3,1 lowered to the two adjacent swaps quasar.transpose can express.

    Quasar has no permute, so this decomposition is the only route -- see the module docstring. It is a
    full device test: real operands, exact-equality check, no xfail.
    """
    device = mesh_device
    _assert_quasar(device)
    torch.manual_seed(0)

    host = torch.randn(IN_SHAPE, dtype=torch.bfloat16)
    golden = host.permute(*PERMUTATION).contiguous()

    tt = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    tt = ttnn.experimental.quasar.transpose(tt, 1, 2, memory_config=DRAM)  # [1,3,224,224] -> [1,224,3,224]
    assert tuple(tt.shape) == (1, 224, 3, 224), "first swap gave %s" % (tuple(tt.shape),)
    tt = ttnn.experimental.quasar.transpose(tt, 2, 3, memory_config=DRAM)  # -> [1,224,224,3]
    ttnn.synchronize_device(device)

    assert tuple(tt.shape) == OUT_SHAPE, "decomposed permute gave %s, sheet 1 row 1 says %s" % (
        tuple(tt.shape),
        OUT_SHAPE,
    )
    got = ttnn.to_torch(ttnn.from_device(tt))
    assert_with_pcc(golden.float(), got.float(), pcc=0.9999)
    assert torch.equal(got.to(torch.bfloat16), golden), "decomposed permute changed %d of %d elements" % (
        int((got.to(torch.bfloat16) != golden).sum()),
        golden.numel(),
    )

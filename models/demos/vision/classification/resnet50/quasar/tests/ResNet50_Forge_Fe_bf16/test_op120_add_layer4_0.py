# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sheet 1 row 120 of 141 -- ttnn.add, layer4_0

One op, one file. Part of the per-call-site replay of the BF16-ONLY tt-forge ResNet-50 compile;
ResNet50_Forge_Fe_bf16/ holds one of these for every one of the 141 ops in @forward.

WHERE IT COMES FROM
-------------------
The residual add that closes bottleneck layer4.0: conv3's output plus the skip branch. Sheet 1 resolves
the two branches back to their common ancestor and labels the shorter path the residual/skip, so
operand 1 is the main branch and operand 2 is the skip (the downsample output on the first block of
each layer, the block input otherwise).

The op carries NO ATTRIBUTES AT ALL -- no fused activation, no memory config, no compute config --
and Forge emits the OUT-OF-PLACE ttnn.add, not add_. That is the difference that matters against the
hand-written metal model, which fuses the following relu into the add
(quasar.add_(out, ds_out, activations=[UnaryWithParam(RELU)]), see ../ops/test_add.py). Here the add
is bare and the relu is left stranded on sheet 1 row 121.

TTNN IR, verbatim from resnet50_forge_bf16_vs_quasar.xlsx sheet 1 ("Forge ops (bf16 only)"):

    %174 = "ttnn.add"(%172, %173) : (tensor<1x1x49x2048xbf16, #ttnn_layout56>, tensor<1x1x49x2048xbf16,
        #ttnn_layout56>) -> tensor<1x1x49x2048xbf16, #ttnn_layout56>

Operands, verbatim from the same row:

    Activation (main branch)           1x1x49x2048    bf16   TILE       DRAM interleaved
    Residual / skip connection         1x1x49x2048    bf16   TILE       DRAM interleaved
    -> Result                          1x1x49x2048    bf16   TILE       DRAM interleaved

Attributes:

    (no attributes)

WHAT IT VALIDATES
-----------------
PCC >= 0.99 against a plain torch add -- a bf16 elementwise add is near-exact -- plus the output
shape, TILE, INTERLEAVED and DRAM.

The height 49 is NOT tile-aligned, so both operands carry row padding and the add has to leave
the pad rows alone.

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
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/test_op120_add_layer4_0.py

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
SHEET_ROW = 120
FORGE_OP = "ttnn.add"
QUASAR_OP = "quasar.add"
OPERAND_SHAPES = ((1, 1, 49, 2048), (1, 1, 49, 2048))
OUTPUT_SHAPE = (1, 1, 49, 2048)

SHAPE = (1, 1, 49, 2048)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_bf16_op120_add(mesh_device):
    device = mesh_device
    _assert_quasar(device)
    torch.manual_seed(0)

    # operand 1 is the main branch (the conv3 output), operand 2 the residual / skip
    main = torch.randn(SHAPE, dtype=torch.bfloat16)
    skip = torch.randn(SHAPE, dtype=torch.bfloat16)
    golden = main.float() + skip.float()

    tt_main = ttnn.from_torch(main, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    tt_skip = ttnn.from_torch(skip, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

    # bare add: the IR row carries "(no attributes)" -- no fused activation, no memory config
    out = ttnn.experimental.quasar.add(tt_main, tt_skip)
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == SHAPE, "output shape %s, sheet 1 row 120 says %s" % (tuple(out.shape), SHAPE)
    assert out.layout == ttnn.TILE_LAYOUT, "output layout %s, the IR says tiled" % (out.layout,)
    mem = out.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, "not interleaved: %s" % (mem.memory_layout,)
    assert mem.buffer_type == ttnn.BufferType.DRAM, "not in DRAM: %s" % (mem.buffer_type,)

    got = ttnn.to_torch(ttnn.from_device(out)).float()
    assert_with_pcc(golden, got, pcc=0.99)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sheet 1 row 139 of 141 -- ttnn.reshape, classifier_squeeze

One op, one file. Part of the per-call-site replay of the BF16-ONLY tt-forge ResNet-50 compile;
ResNet50_Forge_Fe_bf16/ holds one of these for every one of the 141 ops in @forward.

WHERE IT COMES FROM
-------------------
Drop the pooled spatial dims before the classifier: [1, 1, 1, 2048] -> [1, 2048], between the
global average (sheet 1 row 138) and the fc (row 140).

A rank change only -- both shapes are one 32x2048 padded tile row -- so this one should be a view.

TTNN IR, verbatim from resnet50_forge_bf16_vs_quasar.xlsx sheet 1 ("Forge ops (bf16 only)"):

    %193 = "ttnn.reshape"(%192) <{shape = [1 : i32, 2048 : i32]}> : (tensor<1x1x1x2048xbf16, #ttnn_layout58>) ->
        tensor<1x2048xbf16, #ttnn_layout59>

Operands, verbatim from the same row:

    Activation                         1x1x1x2048     bf16   TILE       DRAM interleaved
    -> Result                          1x2048         bf16   TILE       DRAM interleaved

Attributes:

    shape = [1 : i32, 2048 : i32]

WHAT IT VALIDATES
-----------------
quasar.reshape is one of the four generic ops known to work unchanged on Quasar (reshape / clone /
to_memory_config / reallocate -- all layout-and-alloc, no kernel), so this is expected to pass; it is
here so the sheet's op list is covered end to end.

A reshape moves data without computing, so the check is EXACT equality against the reshaped torch
tensor, with a PCC assert alongside so a partial corruption reports a number.

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
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/test_op139_reshape_classifier_squeeze.py

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
SHEET_ROW = 139
FORGE_OP = "ttnn.reshape"
QUASAR_OP = "quasar.reshape"
OPERAND_SHAPES = ((1, 1, 1, 2048),)
OUTPUT_SHAPE = (1, 2048)

IN_SHAPE = (1, 1, 1, 2048)
OUT_SHAPE = (1, 2048)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_bf16_op139_reshape(mesh_device):
    device = mesh_device
    _assert_quasar(device)
    torch.manual_seed(0)

    host = torch.randn(IN_SHAPE, dtype=torch.bfloat16)
    golden = host.reshape(OUT_SHAPE)

    tt_in = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    out = ttnn.experimental.quasar.reshape(tt_in, list(OUT_SHAPE), memory_config=DRAM)
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == OUT_SHAPE, "output shape %s, sheet 1 row 139 says %s" % (tuple(out.shape), OUT_SHAPE)
    assert out.layout == ttnn.TILE_LAYOUT, "output layout %s, the IR says tiled" % (out.layout,)
    mem = out.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, "not interleaved: %s" % (mem.memory_layout,)
    assert mem.buffer_type == ttnn.BufferType.DRAM, "not in DRAM: %s" % (mem.buffer_type,)

    got = ttnn.to_torch(ttnn.from_device(out))
    assert_with_pcc(golden.float(), got.float(), pcc=0.9999)
    assert torch.equal(got.to(torch.bfloat16), golden), "reshape changed %d of %d elements" % (
        int((got.to(torch.bfloat16) != golden).sum()),
        golden.numel(),
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolation probe for the Quasar tilize 0x19 (ERROR_TRISC1, Risc IB interrupt / MATH datacopy MOP rejected).

Option B / Program A (conv_tilize_only_metal2.cpp) is a PURE tilize — no matmul, tilize-oriented hw_startup,
invocation byte-identical to the passing standalone tilize op, half-sync — yet it STILL hits the 0x19 in the
conv program. The standalone tilize op passes. This test removes ALL conv machinery (no reader gather, no
borrowed/resized output) and runs the plain ttnn.tilize on a simple tensor, sweeping the block WIDTH in tiles.

Goal: find out whether the fault is intrinsic to the Quasar tilize LLK at a given block width (block_width_tiles
= tensor width / 32), independent of the conv. The conv's full_inner_dim stem path tilizes a K=16-tile-wide
block (in0_block_w = 32*4*4 / 32 = 16).

  - If width_tiles=16 FAULTS here (and a narrow width passes) => the trigger is block width in the tilize LLK
    itself; this test IS the minimal LLK repro (no conv needed).
  - If ALL widths PASS here => the conv's DFB context (gathered ACT input DFB / borrowed+resized OUT) is the
    trigger, not the tilize width; the repro must keep the conv reader/output.

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_tilize_width_quasar.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("width_tiles", [4, 8, 16], ids=["w4", "w8", "w16"])
@pytest.mark.parametrize("height_tiles", [1, 4], ids=["h1", "h4"])
def test_quasar_tilize_width(mesh_device, width_tiles, height_tiles):
    device = mesh_device
    torch.manual_seed(0)

    H = height_tiles * 32
    W = width_tiles * 32  # width_tiles * 32; block_width_tiles == width_tiles in the tilize kernel
    torch_in = torch.randn((1, 1, H, W), dtype=torch.bfloat16).float()

    # Plain row-major input on device (DRAM interleaved) — no sharding, no conv reader, no borrowed output.
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # PRIMARY SIGNAL: this completes without a 0x19 / ERROR_TRISC1 (a fault aborts the process here).
    tt_tiled = ttnn.tilize(tt_in)

    tt_out = ttnn.to_torch(ttnn.from_device(tt_tiled)).float()
    assert torch.isfinite(tt_out).all(), f"tilize w{width_tiles} h{height_tiles} produced NaN/Inf"
    # to_torch untilizes back to row-major, so it should equal the input.
    assert_with_pcc(torch_in.reshape(tt_out.shape), tt_out, pcc=0.999)
    print(f"tilize width_tiles={width_tiles} height_tiles={height_tiles} PASSED (no 0x19)")

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


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("width_tiles", [4, 8, 16], ids=["w4", "w8", "w16"])
@pytest.mark.parametrize("height_tiles", [1, 4, 49], ids=["h1", "h4", "h49"])
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
    # Use the quasar tilize op explicitly: plain ttnn.tilize dispatches the GENERIC TilizeDeviceOperation,
    # whose factory builds a legacy DataMovementKernel that Quasar rejects ("DataMovementKernel is not
    # supported on Quasar"). NOTE: this DRAM-interleaved input routes to the quasar block/default factory,
    # NOT the TilizeMultiCoreShardedProgramFactory that the failing conv uses -- see test_quasar_tilize_sharded
    # below for that exact (sharded) path.
    tt_tiled = ttnn.experimental.quasar.tilize(tt_in)

    tt_out = ttnn.to_torch(ttnn.from_device(tt_tiled)).float()
    assert torch.isfinite(tt_out).all(), f"tilize w{width_tiles} h{height_tiles} produced NaN/Inf"
    # to_torch untilizes back to row-major, so it should equal the input.
    assert_with_pcc(torch_in.reshape(tt_out.shape), tt_out, pcc=0.999)
    print(f"tilize width_tiles={width_tiles} height_tiles={height_tiles} PASSED (no 0x19)")


# HEIGHT_SHARDED row-major input tilized via the quasar tilize op. ROOT CAUSE (isolated here): the borrowed-DFB
# TilizeMultiCoreShardedProgramFactory delivers correct data for only the first 64 tiles/shard, then repeats --
# a fixed 64-entry limit in the borrowed-DFB credit/tile-counter path (PROVEN: PCC == 64/num_tiles_per_shard
# exactly; tile-count driven, not block-count: h8_w16(128t/8blk) == h16_w8(128t/16blk)). The tilize LLK is fine
# (test_quasar_tilize_width, non-borrowed factory, tilizes 49 blocks correctly).
# WORKAROUND applied (tilize_device_operation.cpp can_use_sharded_optimized_factories): HEIGHT_SHARDED shards
# with > 64 tiles route to the NON-borrowed TilizeMultiCoreDefaultProgramFactory -> so ALL cases below now PASS.
# This test is therefore a REGRESSION GUARD for the reroute; without it the > 64-tile cases fail PCC == 64/N.
_SHARDED_CASES = [
    (49, 8, "h49_w8_FAILCONFIG"),  # 256ch activation, 56x56/2-core: the exact failing tilize
    (49, 4, "h49_w4"),  # same block count, narrower -> isolates width-8 vs block-count
    (8, 8, "h8_w8"),  # 64 tiles/core -> PASSES; last-known-good
    (1, 8, "h1_w8"),  # single block control
    # Bracket the corruption threshold (h8=64 tiles passes, h49=392 fails; PCC math => ~64 tiles correct).
    # per-core tile count = h*w; find where it breaks to identify the borrowed-DFB credit/tile-counter field:
    (9, 8, "h9_w8_72tiles"),  # 72 tiles -> just over 64
    (16, 8, "h16_w8_128tiles"),  # 128 tiles = 2*64
    (32, 8, "h32_w8_256tiles"),  # 256 tiles = 4*64
    (8, 16, "h8_w16_128tiles"),  # 128 tiles but only 8 blocks -> tiles-vs-blocks discriminator
]


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("h_tiles, w_tiles, tid", _SHARDED_CASES, ids=[c[-1] for c in _SHARDED_CASES])
def test_quasar_tilize_sharded(mesh_device, h_tiles, w_tiles, tid):
    """HEIGHT_SHARDED RM [1,1,shard_h*2, C] -> quasar tilize -> PCC. Isolates the conv's internal tilize."""
    # h49_w8_FAILCONFIG is an INTENTIONAL fail-config (256ch / 56x56 / 2-core: the exact failing tilize being
    # documented). On WH it OOMs L1 at this 2-core scale; on Quasar it's the tilize repro. Expected-fail by design.
    if tid == "h49_w8_FAILCONFIG":
        pytest.xfail("intentional fail-config (documents the exact failing 2-core tilize; OOMs WH L1 at this scale)")
    device = mesh_device
    torch.manual_seed(0)
    num_cores = 2
    shard_h = h_tiles * 32
    C = w_tiles * 32
    nhw = shard_h * num_cores

    grid = device.compute_with_storage_grid_size()
    if num_cores > grid.x * grid.y:
        pytest.skip(f"needs {num_cores} cores; grid has {grid.x * grid.y}")
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

    torch_in = torch.randn((1, 1, nhw, C), dtype=torch.bfloat16).float()
    in_mem = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, C),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_in = ttnn.from_torch(
        torch_in, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )
    # quasar tilize op -> TilizeMultiCoreShardedProgramFactory (the exact failing path). A 0x19 aborts here;
    # otherwise PCC vs the input (tilize is a pure layout change, values unchanged) exposes wrong tiles.
    tt_tiled = ttnn.experimental.quasar.tilize(tt_in)
    tt_out = ttnn.to_torch(ttnn.from_device(tt_tiled)).float()
    assert torch.isfinite(tt_out).all(), f"{tid}: tilize produced NaN/Inf"
    assert_with_pcc(torch_in.reshape(tt_out.shape), tt_out, pcc=0.999)
    print(f"sharded tilize {tid} (h={h_tiles} w={w_tiles} cores={num_cores}) PASSED")

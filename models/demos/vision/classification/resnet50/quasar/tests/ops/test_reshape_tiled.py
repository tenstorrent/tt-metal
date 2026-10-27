# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op repro for the Quasar (Metal-2) reshape_tiled DM->DM credit hang.

WHERE IT COMES FROM
-------------------
The Quasar conv2d DRAM-slicing path flattens its TILE-layout output with
`ttnn.experimental.quasar.reshape(...)` (conv2d.cpp:1559). A TILE-layout reshape that changes the tile
grid (e.g. [1,H,W,C] -> [1,1,H*W,C]) genuinely re-tiles data, so it routes to the Metal-2 reshape_tiled
factory:
    reader_reshape_tiled_metal2.cpp   (DM: reads mapping + input tiles, pushes to shared DFBs)
    writer_reshape_tiled_metal2.cpp   (DM: wait_front the tiles, assembles output pages, writes them)

THE HANG (see project_quasar_matmul_async_full_barrier_47797)
-------------------------------------------------------------
In the full split-conv pipeline on commit f5ed37c4ea1 this reshape wedges at program h_id=131 (well past
the matmul): watcher shows the reader DM RETURNED (firmware idle W1) while the writer DM spins forever in
`wait_front` (WFW) on one core; the other core finishes. The reader/writer page ranges are identical and
the producer/consumer dedup protocol is provably balanced, so the writer is starved even though the reader
already posted every credit -> a cross-port DM->DM DFB credit-coherence gap (the Track B / overlay-RTL
issue), not a reshape logic bug.

This test exercises reshape_tiled IN ISOLATION. If it PASSES (as the standalone sharded_to_interleaved
repro does), the pipeline hang is CUMULATIVE / cross-program credit degradation, not per-op -- the sharpest
framing for the NOC/runtime team ("each op passes alone; the pipeline loses a DM->DM credit at program
131"). If it HANGS, this is a minimal, DPRINT-able DM->DM credit-incoherence repro. 600 s timeout bounds
any wedge.

RUN (emulator, watcher; DPRINT reportedly fails on this commit so keep it off):
  TT_METAL_QSR_TC_ISOLATE=1 TT_METAL_WATCHER=10 TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_reshape_tiled.py -k small
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# (in_shape, out_shape, id): TILE-layout DRAM tensor reshaped to a DIFFERENT tile grid (real data movement
# -> reshape_tiled). out tile count > 1 so the work splits across >= 2 cores (forces the DM->DM handoff).
CASES = [
    ((1, 1, 64, 128), (1, 1, 128, 64), "small_64x128_to_128x64"),  # 8 tiles -> multi-core, fast
    ((1, 112, 112, 64), (1, 1, 12544, 64), "stem_flatten_112x112x64"),  # the conv2d.cpp:1559 output flatten
]


@pytest.mark.timeout(1200)  # bounds the DM->DM credit-coherence wedge so this repro can't hang forever
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("in_shape, out_shape, tid", CASES, ids=[c[-1] for c in CASES])
def test_quasar_reshape_tiled(mesh_device, in_shape, out_shape, tid):
    """TILE-layout DRAM reshape that re-tiles (the conv output flatten). Round-trip must PCC ~1.0."""
    torch.manual_seed(0)
    device = mesh_device

    x = torch.rand(in_shape, dtype=torch.bfloat16)

    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # TILE reshape across a changed tile grid -> reader/writer_reshape_tiled_metal2 (the hanging pair).
    out = ttnn.experimental.quasar.reshape(tt_in, ttnn.Shape(list(out_shape)))

    assert_with_pcc(x.reshape(out_shape), ttnn.to_torch(out), 0.9999)
